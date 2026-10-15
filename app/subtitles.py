"""Direct subtitle recognition, editing, and video rendering jobs.

This path deliberately skips diarization and voice separation.  It is for the
simple workflow: upload media, get timestamped text, edit it, export subtitle
files, and render a hard-subtitled video when the source has a video stream.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .asr import transcribe_detailed
from .audio import extract_clean
from .config import (
    RATE,
    SUBTITLE_CHUNK_SECONDS,
    SUBTITLE_JOBS,
    SUBTITLE_WORK,
)
from .utils import digest, duration, probe, run, safe_name, save_json, to_windows_path, to_wsl_path


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
SENTENCE_END = set("。！？!?；;\n")
CLAUSE_END = set("，,：:、")
AUTO_LANGUAGES = {"auto", "automatic", "自动", "自动识别", "中英文"}
SUBTITLE_FONT_NAME = os.getenv("SUBTITLE_FONT_NAME", "SimHei").strip() or "SimHei"
_default_font_dir = Path("/mnt/c/Windows/Fonts")
SUBTITLE_FONT_DIR = Path(os.getenv("SUBTITLE_FONT_DIR", str(_default_font_dir)))
try:
    SUBTITLE_MAX_DIMENSION = max(0, int(os.getenv("SUBTITLE_MAX_DIMENSION", "1920")))
except ValueError:
    SUBTITLE_MAX_DIMENSION = 1920
SUBTITLE_VIDEO_PROFILE = os.getenv("SUBTITLE_VIDEO_PROFILE", "main").strip()
SUBTITLE_VIDEO_LEVEL = os.getenv("SUBTITLE_VIDEO_LEVEL", "").strip()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12,40}", job_id):
        raise ValueError("字幕任务编号无效")
    return SUBTITLE_JOBS / f"{job_id}.json"


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key not in {"thread", "render_thread", "source_path_internal"}
    }


def _save_job(job: dict[str, Any]) -> None:
    save_json(_job_path(str(job["job_id"])), _public_job(job))


def _update_job(job_id: str, **values: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            path = _job_path(job_id)
            if not path.exists():
                raise ValueError("字幕任务不存在")
            job = json.loads(path.read_text(encoding="utf-8"))
            JOBS[job_id] = job
        job.update(values)
        _save_job(job)
        return _public_job(job)


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            return _public_job(job)
        path = _job_path(job_id)
        if not path.exists():
            raise ValueError("字幕任务不存在")
        return json.loads(path.read_text(encoding="utf-8"))


def _clock(seconds: float, *, comma: bool = False) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _ass_clock(seconds: float) -> str:
    """Return the centisecond timestamp format required by ASS subtitles."""

    total_cs = max(0, int(round(float(seconds) * 100.0)))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _normalise_language(value: object) -> str:
    text = str(value or "").strip()
    return "auto" if text.lower() in AUTO_LANGUAGES or not text else text


def _join_token_text(left: str, right: str) -> str:
    if not left:
        return right
    if (
        left[-1].isascii()
        and right[:1].isascii()
        and left[-1].isalnum()
        and right[:1].isalnum()
    ):
        return f"{left} {right}"
    return left + right


def _build_cues(tokens: list[dict[str, Any]], duration_sec: float) -> list[dict[str, Any]]:
    """Group character/word timestamps into editable subtitle cues."""

    ordered: list[dict[str, Any]] = []
    for token in tokens:
        text = str(token.get("text", "") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(token.get("start_sec", 0.0) or 0.0))
            end = max(start + 0.04, float(token.get("end_sec", start) or start))
        except (TypeError, ValueError):
            continue
        ordered.append({"text": text, "start_sec": start, "end_sec": end})
    ordered.sort(key=lambda item: (item["start_sec"], item["end_sec"]))

    cues: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        text = str(current["text"]).strip()
        start = max(0.0, min(float(current["start_sec"]), max(0.0, duration_sec)))
        end = max(start + 0.04, min(float(current["end_sec"]), max(start + 0.04, duration_sec)))
        cues.append(
            {
                "cue_id": f"cue_{len(cues) + 1:04d}",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": text,
                "source": "qwen3-forced-aligner",
            }
        )
        current = None

    for token in ordered:
        if current is not None:
            gap = float(token["start_sec"]) - float(current["end_sec"])
            projected = _join_token_text(str(current["text"]), str(token["text"]))
            previous_text = str(current["text"])
            sentence_boundary = previous_text[-1:] in SENTENCE_END
            clause_boundary = previous_text[-1:] in CLAUSE_END
            too_long = len(projected) > 28 or float(token["end_sec"]) - float(current["start_sec"]) > 7.0
            if gap > 0.75 or sentence_boundary or (clause_boundary and gap > 0.28) or too_long:
                flush()
        if current is None:
            current = {
                "start_sec": float(token["start_sec"]),
                "end_sec": float(token["end_sec"]),
                "text": str(token["text"]),
            }
        else:
            current["end_sec"] = max(float(current["end_sec"]), float(token["end_sec"]))
            current["text"] = _join_token_text(str(current["text"]), str(token["text"]))
    flush()
    return cues


def _make_chunks(audio: Path, work_root: Path, chunk_seconds: float) -> list[tuple[Path, float]]:
    total = max(0.0, duration(audio))
    if total <= 0.0:
        raise RuntimeError("清洁音轨没有可识别的时长")
    chunk_seconds = max(30.0, min(300.0, float(chunk_seconds or SUBTITLE_CHUNK_SECONDS)))
    if total <= chunk_seconds + 0.05:
        return [(audio, 0.0)]

    chunk_root = work_root / "分段音频"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    start = 0.0
    index = 1
    while start < total - 0.04:
        length = min(chunk_seconds, total - start)
        target = chunk_root / f"part_{index:04d}.wav"
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio),
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{length:.3f}",
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            timeout=900,
        )
        if target.exists() and duration(target) >= 0.04:
            chunks.append((target, round(start, 3)))
        start += length
        index += 1
    if not chunks:
        raise RuntimeError("长音频切分失败")
    return chunks


def _transcribe_chunks(chunks: list[tuple[Path, float]], duration_sec: float, language: str) -> tuple[list[dict[str, Any]], str, str]:
    results = transcribe_detailed(
        [item[0] for item in chunks],
        return_time_stamps=True,
        raise_on_error=True,
        language=None if _normalise_language(language) == "auto" else language,
    )
    all_tokens: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    timing_source = "qwen3-forced-aligner"
    for (path, offset), result in zip(chunks, results):
        transcript = str(result.get("text", "") or "").strip()
        if transcript:
            transcript_parts.append(transcript)
        raw_tokens = list(result.get("time_stamps", []) or [])
        if not raw_tokens and transcript:
            # Keep the job usable if the ASR model is present but the aligner
            # is unavailable: one chunk-level cue is explicit and editable.
            timing_source = "chunk-fallback"
            chunk_end = min(duration_sec, offset + max(0.04, float(duration(path))))
            all_tokens.append({"text": transcript, "start_sec": offset, "end_sec": chunk_end})
            continue
        for token in raw_tokens:
            try:
                start = float(token.get("start_sec", 0.0) or 0.0) + offset
                end = float(token.get("end_sec", start) or start) + offset
            except (TypeError, ValueError):
                continue
            all_tokens.append(
                {
                    "text": str(token.get("text", "") or ""),
                    "start_sec": round(max(0.0, start), 3),
                    "end_sec": round(max(start, end), 3),
                }
            )
    return all_tokens, " ".join(transcript_parts).strip(), timing_source


def _normalise_cues(cues: list[dict[str, Any]], duration_sec: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, cue in enumerate(cues):
        text = str(cue.get("text", "") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(cue.get("start_sec", 0.0) or 0.0))
            end = min(max(start + 0.04, float(cue.get("end_sec", start) or start)), max(start + 0.04, duration_sec))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"第 {index + 1} 条字幕的时间无效") from exc
        if end <= start:
            raise ValueError(f"第 {index + 1} 条字幕的结束时间必须大于开始时间")
        base_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(cue.get("cue_id") or f"cue_{index + 1:04d}"))[:80]
        cue_id = base_id or f"cue_{index + 1:04d}"
        suffix = 2
        while cue_id in used:
            cue_id = f"{base_id}_{suffix}"
            suffix += 1
        used.add(cue_id)
        output.append(
            {
                "cue_id": cue_id,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": text,
                "source": str(cue.get("source") or "editor"),
            }
        )
    output.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["cue_id"]))
    return output


def _ass_text(value: object) -> str:
    """Make user or ASR text safe for an ASS dialogue line."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # Backslashes and braces can be interpreted as ASS override tags.  Keep
    # the visible text while preventing a subtitle edit from changing style.
    text = text.replace("\\", "").replace("{", "（").replace("}", "）")
    return text.replace("\n", r"\N")


def _write_ass(output: Path, timeline: dict[str, Any]) -> Path:
    """Write a UTF-8 ASS file used by FFmpeg/libass for hard subtitles."""

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{SUBTITLE_FONT_NAME},52,&H00FFFFFF,&H00FFFFFF,&H00000000,&HAA000000,-1,0,1,2,0,2,80,80,54,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in timeline.get("segments", []):
        text = _ass_text(cue.get("text"))
        if not text:
            continue
        lines.append(
            "Dialogue: 0,{},{},Default,,0,0,0,,{}".format(
                _ass_clock(float(cue["start_sec"])),
                _ass_clock(float(cue["end_sec"])),
                text,
            )
        )
    ass_path = output / "字幕.ass"
    ass_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ass_path


def _write_exports(output: Path, timeline: dict[str, Any]) -> dict[str, str]:
    cues = timeline.get("segments", [])
    srt_lines: list[str] = []
    vtt_lines = ["WEBVTT", ""]
    text_lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        start = float(cue["start_sec"])
        end = float(cue["end_sec"])
        text = str(cue.get("text") or "").strip()
        srt_lines.extend([str(index), f"{_clock(start, comma=True)} --> {_clock(end, comma=True)}", text, ""])
        vtt_lines.extend([f"{_clock(start)} --> {_clock(end)}", text, ""])
        text_lines.append(f"[{_clock(start)} - {_clock(end)}] {text}")
    srt_path = output / "字幕.srt"
    vtt_path = output / "字幕.vtt"
    json_path = output / "字幕.json"
    text_path = output / "字幕文本.txt"
    ass_path = _write_ass(output, timeline)
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
    text_path.write_text("\n".join(text_lines) + ("\n" if text_lines else ""), encoding="utf-8")
    save_json(json_path, timeline)
    return {
        "srt": to_windows_path(srt_path),
        "vtt": to_windows_path(vtt_path),
        "json": to_windows_path(json_path),
        "text": to_windows_path(text_path),
        "ass": to_windows_path(ass_path),
    }


def _run(job_id: str, source: Path, work_root: Path, output: Path) -> None:
    try:
        job = get_job(job_id)
        _update_job(job_id, status="running", status_label="识别中", phase="提取音轨", progress=8)
        audio = work_root / "音频" / "清洁音轨.wav"
        extract_clean(source, audio, noise_reduction=0.0, trim_leading_silence=False)
        total = max(0.04, duration(audio))
        _update_job(job_id, phase="准备时间戳识别", progress=18, duration_sec=round(total, 3))
        chunks = _make_chunks(audio, work_root, float(job.get("chunk_seconds", SUBTITLE_CHUNK_SECONDS)))
        _update_job(job_id, phase=f"识别 {len(chunks)} 个音频片段", progress=25, chunk_count=len(chunks))
        tokens, transcript, timing_source = _transcribe_chunks(
            chunks,
            total,
            str(job.get("language") or "Chinese"),
        )
        cues = _build_cues(tokens, total)
        if not cues:
            raise RuntimeError("ASR 没有返回可用的字幕时间戳")
        timeline = {
            "schema_version": "subtitle-timeline/v1",
            "source_filename": str(job.get("source_filename") or source.name),
            "source_path": str(source),
            "source_sha256": digest(source),
            "duration_sec": round(total, 3),
            "transcription": {
                "model": "Qwen3-ASR + Qwen3-ForcedAligner-0.6B",
                "language": str(job.get("language") or "Chinese"),
                "text": transcript,
                "timestamps_supported": timing_source == "qwen3-forced-aligner",
                "timing_source": timing_source,
                "chunk_count": len(chunks),
            },
            "segments": cues,
            "updated_at": _now(),
        }
        _update_job(job_id, phase="生成字幕文件", progress=88)
        files = _write_exports(output, timeline)
        save_json(work_root / "字幕.json", timeline)
        _update_job(
            job_id,
            status="done",
            status_label="识别完成",
            phase="可编辑并导出",
            progress=100,
            timeline_path=str(work_root / "字幕.json"),
            output_files=files,
            cue_count=len(cues),
            duration_sec=round(total, 3),
            timing_source=timing_source,
            logs=[f"识别完成：{len(cues)} 条字幕，覆盖 {total:.1f} 秒"],
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            status_label="识别失败",
            phase="失败",
            progress=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def start_job(
    source: Path,
    *,
    original_filename: str,
    output_dir: str,
    language: str,
    chunk_seconds: float,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    target_root = to_wsl_path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    output = target_root / f"{safe_name(Path(original_filename).stem, 70)}__字幕项目__{job_id}"
    output.mkdir(parents=True, exist_ok=True)
    work_root = SUBTITLE_WORK / job_id
    work_root.mkdir(parents=True, exist_ok=True)
    has_video = any(stream.get("codec_type") == "video" for stream in probe(source).get("streams", []))
    job = {
        "job_id": job_id,
        "kind": "subtitle_recognition",
        "status": "queued",
        "status_label": "排队中",
        "phase": "准备识别",
        "progress": 0,
        "source_filename": original_filename,
        "source_path": str(source),
        "output_path": str(output),
        "output_dir": to_windows_path(output),
        "work_path": str(work_root),
        "timeline_path": str(work_root / "字幕.json"),
        "language": _normalise_language(language),
        "render_supported": has_video,
        "render_status": "not_started",
        "render_progress": 0,
        "render_phase": "识别完成并保存字幕后生成视频" if has_video else "音频文件只能导出字幕文件",
        "chunk_seconds": max(30.0, min(300.0, float(chunk_seconds or SUBTITLE_CHUNK_SECONDS))),
        "logs": [],
        "created_at": _now(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        _save_job(job)
        thread = threading.Thread(
            target=_run,
            args=(job_id, source, work_root, output),
            daemon=True,
            name=f"subtitle-recognition-{job_id}",
        )
        job["thread"] = thread
        thread.start()
    return _public_job(job)


def get_timeline(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    path = Path(str(job.get("timeline_path") or ""))
    if not path.exists():
        raise ValueError("字幕时间线尚未生成")
    return json.loads(path.read_text(encoding="utf-8"))


def save_edits(job_id: str, cues: list[dict[str, Any]]) -> dict[str, Any]:
    job = get_job(job_id)
    timeline = get_timeline(job_id)
    duration_sec = float(timeline.get("duration_sec") or 0.0)
    normalised = _normalise_cues(cues, duration_sec)
    if not normalised:
        raise ValueError("至少保留一条有效字幕")
    timeline["segments"] = normalised
    timeline["segment_count"] = len(normalised)
    timeline["updated_at"] = _now()
    path = Path(str(job["timeline_path"]))
    save_json(path, timeline)
    files = _write_exports(Path(str(job["output_path"])), timeline)
    _update_job(
        job_id,
        status="done",
        status_label="字幕已保存",
        phase="可继续编辑或用于配音",
        progress=100,
        output_files=files,
        cue_count=len(normalised),
        updated_at=timeline["updated_at"],
        render_status="ready",
        render_progress=0,
        render_phase="字幕已保存，可以生成带字幕视频",
        render_error="",
        render_file="",
    )
    return {"ok": True, "job_id": job_id, "updated_count": len(normalised), "timeline": timeline}


def _filter_quote(value: Path) -> str:
    """Quote a path for an FFmpeg libavfilter option."""

    text = str(value).replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'")
    return f"'{text}'"


def _render_timeout(duration_sec: float) -> int:
    # CPU libx264 speed varies widely.  Keep normal clips responsive while
    # allowing long interview videos to finish in the background.
    return max(1_800, min(21_600, int(max(1.0, duration_sec) * 30.0)))


def _video_filter(source: Path, subtitle_filter: str) -> tuple[str, tuple[int, int] | None]:
    """Add a compatibility downscale only when the source is larger than configured."""

    filters = [subtitle_filter]
    dimensions: tuple[int, int] | None = None
    if SUBTITLE_MAX_DIMENSION > 0:
        metadata = probe(source)
        stream = next(
            (item for item in metadata.get("streams", []) if item.get("codec_type") == "video"),
            {},
        )
        try:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0
        if width > 0 and height > 0 and max(width, height) > SUBTITLE_MAX_DIMENSION:
            ratio = SUBTITLE_MAX_DIMENSION / float(max(width, height))
            target_width = max(2, int(round(width * ratio)) // 2 * 2)
            target_height = max(2, int(round(height * ratio)) // 2 * 2)
            filters.append(f"scale={target_width}:{target_height}")
            dimensions = (target_width, target_height)
        elif width > 0 and height > 0:
            dimensions = (width, height)
    return ",".join(filters), dimensions


def _run_render(job_id: str) -> None:
    temporary: Path | None = None
    try:
        job = get_job(job_id)
        source = Path(str(job.get("source_path") or ""))
        output_root = Path(str(job.get("output_path") or ""))
        timeline = get_timeline(job_id)
        ass_path = output_root / "字幕.ass"
        if not ass_path.is_file():
            ass_path = _write_ass(output_root, timeline)
        output = output_root / "带字幕视频.mp4"
        temporary = output_root / f".带字幕视频.{job_id}.tmp.mp4"
        _update_job(
            job_id,
            render_status="running",
            render_progress=8,
            render_phase="准备 FFmpeg 硬字幕渲染",
            render_error="",
        )
        subtitle_filter = f"subtitles={_filter_quote(ass_path)}"
        if SUBTITLE_FONT_DIR.is_dir():
            subtitle_filter += f":fontsdir={_filter_quote(SUBTITLE_FONT_DIR)}"
        video_filter, output_dimensions = _video_filter(source, subtitle_filter)
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
        ]
        if SUBTITLE_VIDEO_PROFILE:
            command.extend(["-profile:v", SUBTITLE_VIDEO_PROFILE])
        if SUBTITLE_VIDEO_LEVEL:
            command.extend(["-level:v", SUBTITLE_VIDEO_LEVEL])
        command.extend(
            [
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(temporary),
            ]
        )
        run(
            command,
            timeout=_render_timeout(float(job.get("duration_sec") or timeline.get("duration_sec") or 0.0)),
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("FFmpeg 没有生成有效的视频文件")
        rendered = probe(temporary)
        rendered_video = next(
            (item for item in rendered.get("streams", []) if item.get("codec_type") == "video"),
            None,
        )
        if not rendered_video:
            raise RuntimeError("导出文件没有可播放的视频轨道")
        os.replace(temporary, output)
        final_dimensions = (
            int(rendered_video.get("width") or 0),
            int(rendered_video.get("height") or 0),
        )
        dimensions_label = f"{final_dimensions[0]}x{final_dimensions[1]}" if all(final_dimensions) else "未知分辨率"
        _update_job(
            job_id,
            render_status="done",
            render_progress=100,
            render_phase="带字幕视频已生成",
            render_file=to_windows_path(output),
            render_error="",
            logs=[
                *(get_job(job_id).get("logs") or []),
                f"硬字幕视频已生成：{output.name}（{dimensions_label}，H.264/AAC MP4）",
            ],
        )
    except Exception as exc:
        detail = str(getattr(exc, "stderr", "") or getattr(exc, "stdout", "") or exc).strip()
        _update_job(
            job_id,
            render_status="failed",
            render_progress=0,
            render_phase="视频渲染失败",
            render_error=detail[-3000:] or f"{type(exc).__name__}: {exc}",
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def start_render(job_id: str, cues: list[dict[str, Any]]) -> dict[str, Any]:
    """Save current edits and queue an FFmpeg hard-subtitle video render."""

    job = get_job(job_id)
    if job.get("status") not in {"done"}:
        raise ValueError("字幕识别尚未完成，暂时不能生成视频")
    if job.get("render_status") in {"queued", "running"}:
        return _public_job(job)
    if not bool(job.get("render_supported", True)):
        raise ValueError("当前输入是音频文件，只有视频文件可以生成带字幕视频")
    source = Path(str(job.get("source_path") or ""))
    if not source.is_file():
        raise ValueError("原始视频文件不存在，无法生成带字幕视频")
    if not any(stream.get("codec_type") == "video" for stream in probe(source).get("streams", [])):
        raise ValueError("原始文件没有视频画面，无法生成带字幕视频")

    # The render must use exactly what is visible in the editor at click time.
    save_edits(job_id, cues)
    with JOBS_LOCK:
        job = JOBS.get(job_id) or get_job(job_id)
        if job.get("render_status") in {"queued", "running"}:
            return _public_job(job)
        _update_job(
            job_id,
            render_status="queued",
            render_progress=2,
            render_phase="排队等待视频渲染",
            render_error="",
            render_file="",
        )
        thread = threading.Thread(target=_run_render, args=(job_id,), daemon=True, name=f"subtitle-render-{job_id}")
        job = JOBS[job_id]
        job["render_thread"] = thread
        _save_job(job)
        thread.start()
        return _public_job(job)


def file_for_job(job_id: str, name: str) -> Path:
    job = get_job(job_id)
    root = Path(str(job.get("output_path") or "")).resolve()
    candidate = (root / name).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("字幕文件路径无效")
    if not candidate.is_file():
        raise ValueError("字幕文件不存在")
    return candidate
