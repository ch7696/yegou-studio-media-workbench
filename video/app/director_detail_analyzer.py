#!/usr/bin/env python3
"""High-resolution 1fps evidence plus 20-second context/detail analysis."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import time
from pathlib import Path

import requests

from vision_core import MODEL_CONTAINER, MODEL_NAME, MODEL_URL, decodable_video_duration, ffprobe, fmt_time, metadata, number, sha256


PROMPT_PROFILE = os.environ.get(
    "DIRECTOR_PROMPT_PROFILE", "evidence-v1" if "4.7-1B" in MODEL_NAME else "legacy"
)
if PROMPT_PROFILE not in {"evidence-v1", "legacy"}:
    raise ValueError("DIRECTOR_PROMPT_PROFILE 必须为 evidence-v1 或 legacy")


def run(args: list[str], *, timeout: int = 900, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, timeout=timeout, text=True, capture_output=capture)


def docker_cp(source: str, destination: str) -> None:
    run(["docker", "cp", source, destination], timeout=1200)


def call_visual(images: list[Path], instruction: str, max_tokens: int = 3072) -> str:
    content: list[dict] = [{"type": "text", "text": instruction}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode("ascii")
                },
            }
        )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0 if PROMPT_PROFILE == "evidence-v1" else 0.25,
        "top_p": 1 if PROMPT_PROFILE == "evidence-v1" else 0.9,
        "max_tokens": max_tokens,
        "repetition_penalty": 1.06,
    }
    for attempt in range(4):
        try:
            response = requests.post(MODEL_URL, json=payload, headers={"Authorization": "Bearer x"}, timeout=600)
            if response.status_code in {502, 503, 504} and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip()
        except requests.RequestException:
            if attempt >= 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("视觉模型请求未返回结果")


def contact_prompt(start: float, end: float, frame_count: int) -> str:
    if PROMPT_PROFILE == "evidence-v1":
        return (
            f"为导演拉片概括画面证据。这是 {fmt_time(start)}–{fmt_time(end)} 的联系图，"
            f"5列、{math.ceil(frame_count / 5)}行，只有前{frame_count}格有效；从左到右、从上到下排列，"
            f"第1格是{start:g}秒，此后每格加1秒，空白格忽略。\n"
            "用一至两段中文概括主要主体、场景、动作变化，以及明显的画面范围或视角变化。"
            "只写看得清的整体变化，细小动作和文字留给高清帧；不能把所有格子假定为一个连续镜头。"
            "不判断人物身份、心理、声音或画外事件，不为每格编动作。稳定场景只写一次。"
            "不要时间码、逐格列表、标题或JSON，最多200字。"
        )
    times = "、".join(f"第{i + 1}格={start + i:.0f}秒" for i in range(frame_count))
    return f"""你在做导演拉片，不是在写绘图提示词。这是一张 {start:.0f}–{end:.0f} 秒的 5×{math.ceil(frame_count / 5)} 联系图，共 {frame_count} 张有效画面，每格是一秒一帧，时间顺序从左到右、从上到下；时间依次为：{times}。输出的所有时间码必须落在 {fmt_time(start)}–{fmt_time(end)} 范围内，不得扩展到片段之外。

请把所有画面当作一个连续镜头来观察，先概括这一段画面正在完成什么，再按照真正发生变化的节点分段叙述。不要逐格输出，不要 JSON，不要字段表，不要关键词堆叠。相邻画面没有变化时必须合并成一个时间段，并明确说明镜头保持了什么。最多写 4 个变化段；没有新变化就不要继续重复。

每个变化段必须写清：主体在画面的哪一侧/哪一高度/前中后景，人物或物体做了什么、向哪个方向运动，背景和道具具体有什么，镜头是何种景别/机位/构图/运动，光色和画面文字有哪些，以及这一变化如何承接前后镜头。方位使用“画面左/右/上/下、左中/正中/右中、前景/中景/背景”，不要凭空编造人物身份、地点和意图。看不清就说无法确认。

只输出这一段的拉片正文，每段开头写当前片段内的近似时间码；当前片段的完整范围是“{fmt_time(start)}–{fmt_time(end)}”。完成后立即结束，不要为了凑长度重复静态内容。"""


def detail_prompt(start: float, end: float, times: list[float]) -> str:
    if PROMPT_PROFILE == "evidence-v1":
        mapping = "、".join(f"第{i + 1}张={t:g}秒" for i, t in enumerate(times))
        return f"""为导演拉片记录画面证据。这{len(times)}张图按先后排列：{mapping}。
用一段简洁中文，先说明主体、所在位置及主要场景，再描述起始姿态、可见动作和最后状态。画面范围或视角有明显变化时再补充前后差异；没有证据不要判断推镜、拉镜或切镜，不把人物在画面中变大直接说成向前走。
重点写变化，稳定背景只交代一次；保留关键道具和确实可辨的文字。不要猜声音、心理、身份、品牌型号或动作意图，不写评价或气氛感想。看不清就省略，不能为了完整而补动作。不要时间码、标题、逐帧列表或JSON，最多180字。"""
    mapping = "、".join(f"第{i + 1}张={t:.0f}秒" for i, t in enumerate(times))
    return f"""你在做导演拉片。这是视频 {start:.0f}–{end:.0f} 秒内的 {len(times)} 张连续高清帧，按发送顺序排列，时间对应：{mapping}。输出的所有时间码必须落在 {fmt_time(start)}–{fmt_time(end)} 范围内，不得扩展到片段之外。

请把这几张高清帧当作一个连续短镜头来观察，不要一张图写一句话。先判断开头和结尾的状态，只在确实发生变化的地方展开；如果连续帧没有变化，只写一次并合并时间范围。用连续中文文字说明：变化发生在哪个方位，人物/动物/物体如何移动，动作的方向、幅度和前后状态，必要时再补充背景层次、道具、可读文字、材质、光线、色彩、景别、机位、构图和镜头运动。

如果多张帧之间没有新的可见变化，就合并时间范围并只说明一次“画面基本保持不变”。不能用“发生了明显变化”代替具体变化，必须写清楚前后差异。不要 JSON、不要字段列表、不要画面提示词式形容词，不要猜身份和地点。每个段落开头写当前片段内的近似时间码；当前片段的完整范围是“{fmt_time(start)}–{fmt_time(end)}”。只输出拉片正文，完成后立即结束。"""


def make_sheet(remote_frames: str, remote_sheet: str, start_sec: float, frame_count: int, columns: int, tile_w: int, tile_h: int) -> None:
    rows = max(1, math.ceil(frame_count / columns))
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    label = f"drawtext=fontfile={font}:text='%{{eif\\:n+{int(start_sec)}\\:d}}s':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.72"
    vf = (
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease," 
        f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:color=black,{label},"
        f"tile={columns}x{rows}:padding=10:margin=10:color=black"
    )
    run(
        [
            "docker",
            "exec",
            MODEL_CONTAINER,
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-start_number",
            str(int(start_sec) + 1),
            "-i",
            f"{remote_frames}/frame_%06d.jpg",
            "-frames:v",
            str(frame_count),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            remote_sheet,
        ],
        timeout=600,
    )


def frame_indices_for_interval(start_sec: float, end_sec: float, total_frames: int) -> list[int]:
    """Return only the 1fps frame indices that actually exist in an interval."""

    first = max(0, int(math.floor(start_sec + 1e-6)))
    last = min(total_frames, int(math.ceil(end_sec - 1e-9)))
    if last <= first:
        return []
    return list(range(first, last))


def analyze(source: Path, output: Path, max_duration: float | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "视觉分析配置.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "prompt_profile": PROMPT_PROFILE,
                "temperature": 0 if PROMPT_PROFILE == "evidence-v1" else 0.25,
                "sampling_fps": 1,
                "detail_group_seconds": 5,
                "context_group_seconds": 20,
                "frame_size": "original",
                "time_boundary_source": "code",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    frames_dir = output / "01_每秒高清帧"
    context_dir = output / "02_20秒联系图"
    text_dir = output / "03_模型文字分析"
    frames_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    container_video = f"/tmp/director-detail-{source_hash[:12]}.mp4"
    remote_frames = f"/tmp/director-detail-frames-{source_hash[:12]}"
    remote_sheets = f"/tmp/director-detail-sheets-{source_hash[:12]}"
    docker_cp(str(source), f"{MODEL_CONTAINER}:{container_video}")
    try:
        info = ffprobe(container_video)
        meta = metadata(info)
        declared = max(0.5, number(meta.get("duration_sec")))
        actual = decodable_video_duration(container_video)
        duration = max(0.5, min(declared, actual or declared))
        if max_duration and max_duration > 0:
            duration = min(duration, max_duration)
        run(["docker", "exec", MODEL_CONTAINER, "mkdir", "-p", remote_frames, remote_sheets], timeout=120)
        run(
            [
                "docker",
                "exec",
                MODEL_CONTAINER,
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                container_video,
                "-map",
                "0:v:0",
                "-vf",
                "fps=1",
                "-q:v",
                "2",
                f"{remote_frames}/frame_%06d.jpg",
            ],
            timeout=1200,
        )
        docker_cp(f"{MODEL_CONTAINER}:{remote_frames}/.", str(frames_dir))
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        if not frames:
            raise RuntimeError("视频没有生成可用的逐秒画面帧")
        frame_coverage = float(len(frames))
        requested_duration = duration
        if frame_coverage < duration - 1e-6:
            duration = frame_coverage
            print(
                f"逐秒帧覆盖到 {fmt_time(frame_coverage)}，安全跳过末尾 "
                f"{requested_duration - frame_coverage:.2f} 秒的无采样尾段。",
                flush=True,
            )
        (output / "测试视频信息.md").write_text(
            "\n".join(
                [
                    f"- 视频：`{source.name}`",
                    f"- 视觉模型：`{MODEL_NAME}`；提示词配置：`{PROMPT_PROFILE}`",
                    "- 时间精度：每秒采样，文档范围由程序标记；描述中的动作不等于逐帧定位的精确事件。",
                    f"- SHA256：`{source_hash}`",
                    f"- 容器声明时长：{fmt_time(declared)}",
                    f"- 可解码时长：{fmt_time(actual or declared)}",
                    f"- 逐秒帧覆盖：{fmt_time(frame_coverage)}",
                    f"- 处理范围：{'前 ' if max_duration and max_duration > 0 else '全片 '}{fmt_time(duration)}",
                    (
                        f"- 尾段处理：原始可读时长比逐秒帧覆盖多 {requested_duration - frame_coverage:.2f} 秒，"
                        "该无采样尾段已跳过，未影响前面画面分析。"
                        if requested_duration > frame_coverage + 1e-6
                        else "- 尾段处理：逐秒帧覆盖完整。"
                    ),
                    f"- 画面：{meta.get('video', {}).get('width', '未知')} × {meta.get('video', {}).get('height', '未知')}，原尺寸逐秒 JPEG",
                    "- 处理方式：每秒一张高清帧；每 20 秒一张联系图；联系图做上下文，高清帧做细节。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        with (frames_dir / "帧索引.md").open("w", encoding="utf-8") as stream:
            for index, frame in enumerate(frames):
                stream.write(f"- {fmt_time(index)}：`{frame.name}`\n")

        all_text: list[str] = []
        visual_index: list[dict[str, object]] = []
        slice_start = 0.0
        slice_id = 1
        while slice_start < duration - 1e-6:
            slice_end = min(duration, slice_start + 20.0)
            context_indices = frame_indices_for_interval(slice_start, slice_end, len(frames))
            if not context_indices:
                print(
                    f"跳过 {fmt_time(slice_start)}–{fmt_time(slice_end)}：没有对应的逐秒画面帧。",
                    flush=True,
                )
                break
            frame_count = len(context_indices)
            context_sheet_remote = f"{remote_sheets}/context_{slice_id:02d}.jpg"
            context_sheet_local = context_dir / f"联系图_{slice_id:02d}_{fmt_time(slice_start).replace(':', '-')}-{fmt_time(slice_end).replace(':', '-')}.jpg"
            make_sheet(remote_frames, context_sheet_remote, slice_start, frame_count, 5, 320, 180)
            docker_cp(f"{MODEL_CONTAINER}:{context_sheet_remote}", str(context_sheet_local))
            try:
                context_text = call_visual([context_sheet_local], contact_prompt(slice_start, slice_end, frame_count), max_tokens=512 if PROMPT_PROFILE == "evidence-v1" else 4096)
            except Exception as exc:
                context_text = f"（上下文模型调用失败：{type(exc).__name__}: {exc}）"
            context_file = text_dir / f"上下文_{slice_id:02d}_{fmt_time(slice_start).replace(':', '-')}-{fmt_time(slice_end).replace(':', '-')}.md"
            context_file.write_text(context_text + "\n", encoding="utf-8")
            all_text.append(f"# {fmt_time(slice_start)}–{fmt_time(slice_end)}：20秒上下文\n\n{context_text}\n")
            context_record: dict[str, object] = {
                "kind": "context",
                "start_sec": round(slice_start, 3),
                "end_sec": round(slice_end, 3),
                "image": str(context_sheet_local.relative_to(output)),
                "text": str(context_file.relative_to(output)),
                "detail_files": [],
            }

            detail_start = slice_start
            detail_id = 1
            while detail_start < slice_end - 1e-6:
                detail_end = min(slice_end, detail_start + 5.0)
                detail_indices = frame_indices_for_interval(detail_start, detail_end, len(frames))
                times = [float(index) for index in detail_indices]
                selected = [frames[index] for index in detail_indices]
                if not selected:
                    detail_start += 5.0
                    detail_id += 1
                    continue
                detail_file = text_dir / f"细节_{slice_id:02d}_{detail_id:02d}_{fmt_time(detail_start).replace(':', '-')}-{fmt_time(detail_end).replace(':', '-')}.md"
                fallback_local: Path | None = None
                try:
                    detail_text = call_visual(selected, detail_prompt(detail_start, detail_end, times), max_tokens=384 if PROMPT_PROFILE == "evidence-v1" else 3072)
                except Exception as first_exc:
                    fallback_remote = f"{remote_sheets}/detail_{slice_id:02d}_{detail_id:02d}.jpg"
                    fallback_local = context_dir / f"细节联系图_{slice_id:02d}_{detail_id:02d}_{fmt_time(detail_start).replace(':', '-')}-{fmt_time(detail_end).replace(':', '-')}.jpg"
                    make_sheet(remote_frames, fallback_remote, detail_start, len(selected), 3, 640, 360)
                    docker_cp(f"{MODEL_CONTAINER}:{fallback_remote}", str(fallback_local))
                    try:
                        fallback_prompt = detail_prompt(detail_start, detail_end, times)
                        if PROMPT_PROFILE == "evidence-v1":
                            fallback_prompt = (
                                f"以下只有一张3列联系图，前{len(times)}格有效，从左到右、从上到下依次对应原始帧，空白格忽略。\n"
                                + fallback_prompt
                            )
                        detail_text = call_visual([fallback_local], fallback_prompt, max_tokens=384 if PROMPT_PROFILE == "evidence-v1" else 3072)
                    except Exception as second_exc:
                        detail_text = f"（细节模型调用失败：{type(first_exc).__name__}: {first_exc}; fallback {type(second_exc).__name__}: {second_exc}）"
                detail_file.write_text(detail_text + "\n", encoding="utf-8")
                all_text.append(f"## {fmt_time(detail_start)}–{fmt_time(detail_end)}：高清细节\n\n{detail_text}\n")
                detail_files = context_record["detail_files"]
                if isinstance(detail_files, list):
                    detail_files.append(
                        {
                            "start_sec": round(detail_start, 3),
                            "end_sec": round(detail_end, 3),
                            "image": str(fallback_local.relative_to(output)) if fallback_local else "",
                            "text": str(detail_file.relative_to(output)),
                        }
                    )
                detail_start += 5.0
                detail_id += 1
            visual_index.append(context_record)
            visual_progress = min(88, 35 + round((slice_end / duration) * 53))
            print(
                f"20秒上下文完成：{slice_id}（{fmt_time(slice_start)}–{fmt_time(slice_end)}），"
                f"视觉进度={visual_progress}%",
                flush=True,
            )
            slice_start += 20.0
            slice_id += 1
        (output / "视觉分析文本汇总.md").write_text("\n".join(all_text), encoding="utf-8")
        (output / "视觉片段索引.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in visual_index) + "\n",
            encoding="utf-8",
        )
        print(f"高清逐秒/20秒上下文分析完成：{output}", flush=True)
    finally:
        for remote_path in (container_video, remote_frames, remote_sheets):
            subprocess.run(["docker", "exec", MODEL_CONTAINER, "rm", "-rf", remote_path], check=False, timeout=180, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=0.0, help="只处理前 N 秒；默认 0 表示全片")
    args = parser.parse_args()
    analyze(args.source, args.output, args.max_duration or None)


if __name__ == "__main__":
    main()
