"""Full-audio dubbing jobs: diarization, timed ASR, voice mapping and export."""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import tts
from .asr import transcribe, transcribe_detailed
from .audio import extract_clean, fit_audio
from .config import DUBBING_JOBS, DUBBING_WORK, RATE
from .direct_tts import extract_reference
from .diarization import diarize
from .utils import digest, duration, probe, safe_name, save_json, to_windows_path, to_wsl_path


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
PUNCTUATION = set("，、：,:。！？!?；;．.\n")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12,40}", job_id):
        raise ValueError("任务编号无效")
    return DUBBING_JOBS / f"{job_id}.json"


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"thread", "source_path_internal"}}


def _save_job(job: dict[str, Any]) -> None:
    save_json(_job_path(str(job["job_id"])), _public_job(job))


def _update_job(job_id: str, **values: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            path = _job_path(job_id)
            if not path.exists():
                raise ValueError("任务不存在")
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
            raise ValueError("任务不存在")
        return json.loads(path.read_text(encoding="utf-8"))


def _normalise_diarization(raw: list[list[float | int]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start", 0.0))
            end = item.get("end_sec", item.get("end", start))
            label = item.get("cluster_label", item.get("speaker", item.get("label", 0)))
        else:
            if len(item) < 3:
                continue
            start, end, label = item[0], item[1], item[2]
        try:
            start_value = max(0.0, float(start))
            end_value = max(start_value, float(end))
            label_value = int(label)
        except (TypeError, ValueError):
            continue
        if end_value - start_value < 0.05:
            continue
        output.append(
            {
                "start_sec": round(start_value, 3),
                "end_sec": round(end_value, 3),
                "cluster_label": label_value,
            }
        )
    return sorted(output, key=lambda item: (item["start_sec"], item["end_sec"], item["cluster_label"]))


def _speaker_id(index: int) -> str:
    return f"speaker_{index + 1:02d}"


def _speaker_label(index: int) -> str:
    return f"说话人 {chr(65 + index) if index < 26 else index + 1}"


def _interval_overlap(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _assign_token(
    token: dict[str, Any],
    diarization: list[dict[str, Any]],
    label_to_speaker: dict[int, str],
) -> dict[str, Any]:
    start = max(0.0, float(token.get("start_sec", 0.0) or 0.0))
    end = max(start + 0.08, float(token.get("end_sec", start) or start))
    token_duration = max(0.08, end - start)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for segment in diarization:
        overlap = _interval_overlap(start, end, segment["start_sec"], segment["end_sec"])
        if overlap > 0:
            candidates.append((overlap, segment))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        midpoint = (start + end) / 2.0
        nearest = min(
            diarization,
            key=lambda item: min(abs(midpoint - item["start_sec"]), abs(midpoint - item["end_sec"])),
            default=None,
        )
        if nearest is None:
            speaker_id = "unknown"
            confidence = 0.0
        else:
            distance = min(abs(midpoint - nearest["start_sec"]), abs(midpoint - nearest["end_sec"]))
            speaker_id = label_to_speaker.get(int(nearest["cluster_label"]), "unknown") if distance <= 0.75 else "unknown"
            confidence = max(0.0, 0.45 - distance / 2.0) if speaker_id != "unknown" else 0.0
        return {
            "text": str(token.get("text", "")),
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "speaker_id": speaker_id,
            "alignment_confidence": round(confidence, 3),
            "overlap": False,
        }
    overlap, primary = candidates[0]
    return {
        "text": str(token.get("text", "")),
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "speaker_id": label_to_speaker.get(int(primary["cluster_label"]), "unknown"),
        "alignment_confidence": round(min(1.0, overlap / token_duration), 3),
        "overlap": len(candidates) > 1,
    }


def _build_utterances(
    tokens: list[dict[str, Any]],
    diarization: list[dict[str, Any]],
    label_to_speaker: dict[int, str],
) -> list[dict[str, Any]]:
    assigned = [_assign_token(token, diarization, label_to_speaker) for token in tokens if str(token.get("text", "")).strip()]
    utterances: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        text = str(current.pop("_text", "")).strip()
        if not text:
            current = None
            return
        current["text"] = text
        current["dubbing_text"] = text
        current["original_duration_sec"] = round(current["end_sec"] - current["start_sec"], 3)
        count = max(1, int(current.pop("_confidence_count", 0)))
        current["speaker_confidence"] = round(float(current.pop("_confidence_sum", 0.0)) / count, 3)
        current["status"] = "待配音"
        current["segment_id"] = f"seg_{len(utterances) + 1:04d}"
        utterances.append(current)
        current = None

    for item in assigned:
        text = item["text"].strip()
        if not text:
            continue
        if current is not None:
            gap = item["start_sec"] - current["end_sec"]
            speaker_changed = item["speaker_id"] != current["speaker_id"]
            sentence_done = str(current.get("_text", ""))[-1:] in PUNCTUATION and gap > 0.12
            too_long = item["start_sec"] - current["start_sec"] >= 15.0
            if speaker_changed or gap > 0.85 or sentence_done or too_long:
                flush()
        if current is None:
            current = {
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "speaker_id": item["speaker_id"],
                "overlap": bool(item["overlap"]),
                "_text": text,
                "_confidence_sum": float(item["alignment_confidence"]),
                "_confidence_count": 1,
            }
        else:
            current["end_sec"] = max(float(current["end_sec"]), float(item["end_sec"]))
            current["overlap"] = bool(current["overlap"] or item["overlap"])
            current["_text"] = str(current["_text"]) + text
            current["_confidence_sum"] = float(current["_confidence_sum"]) + float(item["alignment_confidence"])
            current["_confidence_count"] = int(current["_confidence_count"]) + 1
    flush()
    return utterances


def _decorate_timestamps(tokens: list[dict[str, Any]], transcript: str) -> list[dict[str, Any]]:
    """Put punctuation from ASR text back onto aligned tokens.

    The ForcedAligner commonly returns Chinese characters as timestamp items but
    leaves punctuation in the sentence-level ASR text.  Keeping punctuation on
    the last aligned token gives the editor natural sentence boundaries without
    changing any model output or timestamps.
    """

    if not tokens or not transcript:
        return tokens
    output: list[dict[str, Any]] = []
    cursor = 0
    for original in tokens:
        item = dict(original)
        token_text = str(item.get("text", "") or "")
        comparable = token_text.strip()
        if not comparable:
            continue
        while cursor < len(transcript) and transcript[cursor].isspace():
            cursor += 1
        position = transcript.find(comparable, cursor)
        if position < 0:
            position = cursor
        end = min(len(transcript), position + len(comparable))
        punctuation = ""
        while end < len(transcript) and transcript[end] in PUNCTUATION:
            punctuation += transcript[end]
            end += 1
        if punctuation:
            item["text"] = token_text + punctuation
        cursor = max(cursor, end)
        output.append(item)
    return output


def _write_timeline_markdown(path: Path, timeline: dict[str, Any]) -> None:
    lines = [
        "# 说话人时间轴",
        "",
        f"- 原始文件：{timeline['source_filename']}",
        f"- 总时长：{timeline['duration_sec']:.3f} 秒",
        f"- 说话人数量：{len(timeline['speakers'])}",
        f"- ASR：{timeline['transcription']['model']}",
        f"- 时间戳：{'可用' if timeline['transcription']['timestamps_supported'] else '不可用'}",
        "",
        "| 时间 | 说话人 | 原文 | 置信度 | 重叠 | 状态 |",
        "|---|---|---|---:|---|---|",
    ]
    for item in timeline["segments"]:
        lines.append(
            f"| {item['start_sec']:.3f}–{item['end_sec']:.3f} | {item['speaker_label']} | "
            f"{item['text'].replace('|', '／')} | {item['speaker_confidence']:.2f} | "
            f"{'是' if item['overlap'] else '否'} | {item['status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyse_timeline(
    source: Path,
    *,
    job_id: str,
    work_root: Path,
    original_filename: str = "",
    speaker_num: int = 0,
    noise_reduction: float = 8.0,
    include_overlap: bool = True,
    language: str = "Chinese",
) -> dict[str, Any]:
    """Analyze one complete media file and create an editable dubbing timeline."""

    clean_audio = work_root / "音频" / "清洁音轨.wav"
    extract_clean(
        source,
        clean_audio,
        noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
        trim_leading_silence=False,
    )
    clean_duration = max(0.5, duration(clean_audio))
    raw_diarization = diarize(
        clean_audio,
        speaker_num if int(speaker_num) > 0 else None,
        include_overlap=bool(include_overlap),
    )
    diarization = _normalise_diarization(raw_diarization)
    labels = sorted({int(item["cluster_label"]) for item in diarization})
    label_to_speaker = {label: _speaker_id(index) for index, label in enumerate(labels)}
    speaker_items: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        speaker_id = _speaker_id(index)
        speech_duration = sum(
            max(0.0, float(item["end_sec"]) - float(item["start_sec"]))
            for item in diarization
            if int(item["cluster_label"]) == label
        )
        speaker_items.append(
            {
                "speaker_id": speaker_id,
                "speaker_label": _speaker_label(index),
                "cluster_label": label,
                "speech_duration_sec": round(speech_duration, 3),
                "reference_audio": None,
                "reference_text": "",
                "reference_text_source": "",
            }
        )

    asr_result = transcribe_detailed([clean_audio], return_time_stamps=True)[0]
    timestamps = _decorate_timestamps(
        list(asr_result.get("time_stamps", []) or []),
        str(asr_result.get("text", "") or ""),
    )
    segments = _build_utterances(timestamps, diarization, label_to_speaker)
    label_lookup = {item["speaker_id"]: item["speaker_label"] for item in speaker_items}
    for item in segments:
        item["speaker_label"] = label_lookup.get(item["speaker_id"], "未确定说话人")
    overlap_count = sum(1 for item in segments if item["overlap"])
    low_confidence_count = sum(1 for item in segments if item["speaker_confidence"] < 0.55)
    timeline = {
        "schema_version": "audio-dubbing-timeline/v1",
        "job_id": job_id,
        "source_filename": original_filename or source.name,
        "source_path": str(source),
        "source_sha256": digest(source),
        "duration_sec": round(clean_duration, 3),
        "audio_path": str(clean_audio),
        "speakers": speaker_items,
        "diarization": {
            "model": "3D-Speaker CAMPPlus + FSMN-VAD",
            "include_overlap_requested": bool(include_overlap),
            "segments": diarization,
        },
        "transcription": {
            "model": "Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B",
            "language": language,
            "text": str(asr_result.get("text", "") or ""),
            "timestamps_supported": bool(timestamps),
            "tokens": timestamps,
        },
        "segments": segments,
        "quality": {
            "segment_count": len(segments),
            "overlap_segment_count": overlap_count,
            "low_confidence_segment_count": low_confidence_count,
            "unassigned_segment_count": sum(1 for item in segments if item["speaker_id"] == "unknown"),
            "notes": [
                "speaker_confidence 是时间轴覆盖度，不是模型原生概率。",
                "原始音轨尚未做专用人声/环境声分离，首版导出会生成独立配音轨。",
            ],
        },
        "generations": [],
        "created_at": _now(),
    }
    save_json(work_root / "时间轴.json", timeline)
    _write_timeline_markdown(work_root / "说话人时间轴.md", timeline)
    return timeline


def start_analysis(
    source: Path,
    *,
    original_filename: str,
    output_dir: str,
    speaker_num: int,
    noise_reduction: float,
    include_overlap: bool,
    language: str,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    target_root = to_wsl_path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    output = target_root / f"{safe_name(Path(original_filename).stem, 70)}__全片配音项目__{job_id}"
    output.mkdir(parents=True, exist_ok=True)
    work_root = DUBBING_WORK / job_id
    job = {
        "job_id": job_id,
        "kind": "dubbing_analysis",
        "status": "queued",
        "status_label": "排队中",
        "phase": "准备分析",
        "progress": 0,
        "source_filename": original_filename,
        "source_path": str(source),
        "output_path": str(output),
        "output_dir": to_windows_path(output),
        "work_path": str(work_root),
        "timeline_path": str(work_root / "时间轴.json"),
        "speaker_num": int(speaker_num),
        "noise_reduction": float(noise_reduction),
        "include_overlap": bool(include_overlap),
        "language": language,
        "logs": [],
        "created_at": _now(),
    }
    work_root.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job_id] = job
        _save_job(job)
        thread = threading.Thread(
            target=_run_analysis,
            args=(job_id, source, work_root, output),
            daemon=True,
            name=f"dubbing-analysis-{job_id}",
        )
        job["thread"] = thread
        thread.start()
    return _public_job(job)


def _run_analysis(job_id: str, source: Path, work_root: Path, output: Path) -> None:
    try:
        job = get_job(job_id)
        _update_job(job_id, status="running", status_label="分析中", phase="提取清洁音轨", progress=5)
        timeline = analyse_timeline(
            source,
            job_id=job_id,
            work_root=work_root,
            original_filename=str(job.get("source_filename", source.name)),
            speaker_num=int(job.get("speaker_num", 0)),
            noise_reduction=float(job.get("noise_reduction", 8.0)),
            include_overlap=bool(job.get("include_overlap", True)),
            language=str(job.get("language", "Chinese")),
        )
        shutil.copy2(work_root / "时间轴.json", output / "时间轴.json")
        shutil.copy2(work_root / "说话人时间轴.md", output / "说话人时间轴.md")
        _update_job(
            job_id,
            status="done",
            status_label="分析完成",
            phase="等待绑定参考音色",
            progress=100,
            timeline_path=str(work_root / "时间轴.json"),
            speaker_count=len(timeline["speakers"]),
            segment_count=len(timeline["segments"]),
            quality=timeline["quality"],
            logs=["全片音轨分析完成", f"识别到 {len(timeline['speakers'])} 个说话人、{len(timeline['segments'])} 个台词片段"],
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            status_label="分析失败",
            phase="失败",
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_timeline(job_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    job = get_job(job_id)
    path = Path(str(job.get("timeline_path", "")))
    if not path.exists():
        raise ValueError("时间轴尚未生成")
    return job, json.loads(path.read_text(encoding="utf-8"))


def get_timeline(job_id: str) -> dict[str, Any]:
    """Return the editable timeline for an analysis job."""

    _, timeline = _load_timeline(job_id)
    return timeline


def attach_reference(
    job_id: str,
    *,
    speaker_id: str,
    source: Path,
    original_filename: str,
    start_sec: float,
    seconds: float,
    noise_reduction: float,
    trim_leading_silence: bool,
    reference_text: str,
) -> dict[str, Any]:
    job, timeline = _load_timeline(job_id)
    speaker = next((item for item in timeline["speakers"] if item["speaker_id"] == speaker_id), None)
    if speaker is None:
        raise ValueError(f"找不到说话人：{speaker_id}")
    reference_root = Path(str(job["output_path"])) / "参考音色"
    reference_root.mkdir(parents=True, exist_ok=True)
    safe_id = safe_name(speaker_id, 40)
    target = reference_root / f"{safe_id}__参考音色.wav"
    actual_seconds = extract_reference(
        source,
        target,
        start_seconds=max(0.0, float(start_sec)),
        max_seconds=max(1.0, min(30.0, float(seconds))),
        noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
        trim_leading_silence=bool(trim_leading_silence),
    )
    text = str(reference_text or "").strip()
    source_kind = "manual"
    if not text:
        detected = transcribe([target])
        text = (detected[0] if detected else "").strip()
        source_kind = "auto_asr" if text else ""
    speaker.update(
        {
            "reference_audio": str(target),
            "reference_text": text,
            "reference_text_source": source_kind,
            "reference_filename": original_filename,
            "reference_duration_sec": round(actual_seconds, 3),
        }
    )
    save_json(Path(str(job["timeline_path"])), timeline)
    shutil.copy2(Path(str(job["timeline_path"])), Path(str(job["output_path"])) / "时间轴.json")
    return {
        "ok": True,
        "job_id": job_id,
        "speaker_id": speaker_id,
        "speaker_label": speaker["speaker_label"],
        "reference_audio": to_windows_path(target),
        "reference_text": text,
        "reference_text_source": source_kind,
        "reference_duration_sec": round(actual_seconds, 3),
    }


def _apply_segment_edits(
    timeline: dict[str, Any],
    edits: list[dict[str, Any]],
    selected_ids: list[str] | None,
) -> list[dict[str, Any]]:
    edit_map = {str(item.get("segment_id")): item for item in edits if item.get("segment_id")}
    selected = set(selected_ids or [])
    output: list[dict[str, Any]] = []
    for original in timeline["segments"]:
        if selected and original["segment_id"] not in selected:
            continue
        item = dict(original)
        patch = edit_map.get(item["segment_id"], {})
        for key in ("text", "dubbing_text", "speaker_id", "start_sec", "end_sec"):
            if key in patch and patch[key] is not None:
                item[key] = patch[key]
        item["text"] = str(item.get("text", "") or "").strip()
        item["dubbing_text"] = str(item.get("dubbing_text", item["text"]) or item["text"]).strip()
        try:
            item["start_sec"] = max(0.0, float(item["start_sec"]))
            item["end_sec"] = float(item["end_sec"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"片段 {item['segment_id']} 的时间无效") from exc
        if item["end_sec"] <= item["start_sec"]:
            raise ValueError(f"片段 {item['segment_id']} 的结束时间必须大于开始时间")
        if not item["dubbing_text"]:
            raise ValueError(f"片段 {item['segment_id']} 的配音文本为空")
        output.append(item)
    if not output:
        raise ValueError("没有可生成的台词片段")
    return output


def _clamp_speed(value: float) -> float:
    return max(0.5, min(2.0, float(value)))


def _synthesise_segment(
    model: Any,
    segment: dict[str, Any],
    speaker: dict[str, Any],
    *,
    output_root: Path,
    mode: str,
    base_speed: float,
    adaptive_speed: bool,
    fit_to_slot: bool,
    prompt_instruction: str,
    text_frontend: bool,
    seed: int,
) -> dict[str, Any]:
    reference_audio = Path(str(speaker.get("reference_audio") or ""))
    if not reference_audio.exists():
        raise ValueError(f"{speaker['speaker_label']} 尚未绑定参考音频")
    reference_text = str(speaker.get("reference_text") or "").strip()
    if mode == "zero_shot" and not reference_text:
        raise ValueError(f"{speaker['speaker_label']} 缺少参考台词，请在时间轴中补充")
    target_duration = max(0.5, float(segment["end_sec"]) - float(segment["start_sec"]))
    segment_root = output_root / "配音片段"
    segment_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{segment['segment_id']}__{safe_name(speaker['speaker_label'], 24)}"
    raw_path = segment_root / f"{prefix}__原始.wav"
    fitted_path = segment_root / f"{prefix}__适配.wav"
    effective_speed = _clamp_speed(base_speed)
    generation_seed = max(0, int(seed) + int(segment["segment_id"].split("_")[-1]))

    audio, raw_seconds = tts.synthesize(
        model,
        mode=mode,
        text=segment["dubbing_text"],
        reference_text=reference_text,
        reference_audio=reference_audio,
        speed=effective_speed,
        prompt_instruction=prompt_instruction,
        text_frontend=text_frontend,
        seed=generation_seed,
    )
    tts.save_audio(raw_path, audio, model.sample_rate)
    initial_seconds = raw_seconds
    if adaptive_speed and target_duration > 0:
        proposed = _clamp_speed(effective_speed * raw_seconds / target_duration)
        if abs(proposed - effective_speed) >= 0.06:
            effective_speed = proposed
            audio, raw_seconds = tts.synthesize(
                model,
                mode=mode,
                text=segment["dubbing_text"],
                reference_text=reference_text,
                reference_audio=reference_audio,
                speed=effective_speed,
                prompt_instruction=prompt_instruction,
                text_frontend=text_frontend,
                seed=generation_seed,
            )
            tts.save_audio(raw_path, audio, model.sample_rate)
    if fit_to_slot:
        _, final_seconds = fit_audio(raw_path, fitted_path, target_duration)
        output_audio = fitted_path
    else:
        final_seconds = raw_seconds
        output_audio = raw_path
    ratio = raw_seconds / target_duration if target_duration else 0.0
    flags: list[str] = []
    if ratio < 0.8 or ratio > 1.25:
        flags.append("自然生成时长偏差较大")
    if effective_speed in {0.5, 2.0} and abs(ratio - 1.0) > 0.15:
        flags.append("speed 已达到边界")
    return {
        "segment_id": segment["segment_id"],
        "speaker_id": segment["speaker_id"],
        "speaker_label": speaker["speaker_label"],
        "start_sec": round(float(segment["start_sec"]), 3),
        "end_sec": round(float(segment["end_sec"]), 3),
        "target_duration_sec": round(target_duration, 3),
        "text": segment["dubbing_text"],
        "initial_duration_sec": round(initial_seconds, 3),
        "raw_duration_sec": round(raw_seconds, 3),
        "final_duration_sec": round(final_seconds, 3),
        "speed": round(effective_speed, 3),
        "audio_path": str(output_audio),
        "quality_flags": flags,
    }


def _render_voice_track(items: list[dict[str, Any]], duration_sec: float, target: Path) -> None:
    sample_count = max(1, int(math.ceil(float(duration_sec) * RATE)))
    track = np.zeros(sample_count, dtype=np.float32)
    for item in items:
        path = Path(item["audio_path"])
        data, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if int(rate) != RATE:
            raise ValueError(f"配音片段采样率不是 {RATE}Hz：{path.name}")
        start = max(0, int(round(float(item["start_sec"]) * RATE)))
        if start >= sample_count:
            continue
        stop = min(sample_count, start + len(data))
        track[start:stop] += np.asarray(data[: stop - start], dtype=np.float32)
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track /= peak / 0.96
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), track, RATE, subtype="PCM_16")


def _mux_video(source: Path, voice_track: Path, target: Path) -> Path | None:
    streams = probe(source).get("streams", [])
    if not any(item.get("codec_type") == "video" for item in streams):
        return None
    from .utils import run

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(voice_track),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(target),
        ],
        900,
    )
    return target if target.exists() else None


def start_generation(
    job_id: str,
    *,
    edits: list[dict[str, Any]],
    selected_ids: list[str] | None,
    mode: str,
    speed: float,
    adaptive_speed: bool,
    fit_to_slot: bool,
    prompt_instruction: str,
    text_frontend: bool,
    seed: int,
    export_video: bool,
) -> dict[str, Any]:
    job, timeline = _load_timeline(job_id)
    segments = _apply_segment_edits(timeline, edits, selected_ids)
    request = {
        "edits": edits,
        "selected_ids": selected_ids,
        "mode": mode,
        "speed": float(speed),
        "adaptive_speed": bool(adaptive_speed),
        "fit_to_slot": bool(fit_to_slot),
        "prompt_instruction": prompt_instruction,
        "text_frontend": bool(text_frontend),
        "seed": int(seed),
        "export_video": bool(export_video),
    }
    generation_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        job = JOBS.get(job_id, job)
        job.update(
            {
                "status": "queued_generation",
                "status_label": "等待配音",
                "phase": "检查参考音色",
                "progress": 0,
                "generation_id": generation_id,
                "generation_request": request,
                "generation_segment_count": len(segments),
                "logs": [],
            }
        )
        _save_job(job)
        thread = threading.Thread(
            target=_run_generation,
            args=(job_id, generation_id, segments, timeline, request),
            daemon=True,
            name=f"dubbing-generation-{generation_id}",
        )
        job["thread"] = thread
        thread.start()
    return _public_job(job)


def _run_generation(
    job_id: str,
    generation_id: str,
    segments: list[dict[str, Any]],
    timeline: dict[str, Any],
    request: dict[str, Any],
) -> None:
    try:
        job = get_job(job_id)
        speaker_map = {item["speaker_id"]: item for item in timeline["speakers"]}
        for segment in segments:
            if segment["speaker_id"] not in speaker_map:
                raise ValueError(f"片段 {segment['segment_id']} 没有关联的说话人")
        output_root = Path(str(job["output_path"])) / f"生成_{generation_id}"
        output_root.mkdir(parents=True, exist_ok=True)
        generated: list[dict[str, Any]] = []
        with tts.TTS_LOCK:
            model = tts.get_tts()
            for index, segment in enumerate(segments):
                _update_job(
                    job_id,
                    status="generating",
                    status_label="逐段生成中",
                    phase=f"生成 {index + 1}/{len(segments)}",
                    progress=int(index / max(1, len(segments)) * 85),
                )
                generated.append(
                    _synthesise_segment(
                        model,
                        segment,
                        speaker_map[segment["speaker_id"]],
                        output_root=output_root,
                        mode=str(request["mode"]),
                        base_speed=float(request["speed"]),
                        adaptive_speed=bool(request["adaptive_speed"]),
                        fit_to_slot=bool(request["fit_to_slot"]),
                        prompt_instruction=str(request["prompt_instruction"]),
                        text_frontend=bool(request["text_frontend"]),
                        seed=int(request["seed"]),
                    )
                )
        voice_track = output_root / "整段配音人声.wav"
        _render_voice_track(generated, float(timeline["duration_sec"]), voice_track)
        video_path = None
        if bool(request["export_video"]):
            video_path = _mux_video(Path(str(job["source_path"])), voice_track, output_root / "配音视频.mp4")
        quality = {
            "segment_count": len(generated),
            "duration_warnings": sum(1 for item in generated if item["quality_flags"]),
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "speaker_label": item["speaker_label"],
                    "target_duration_sec": item["target_duration_sec"],
                    "estimated_duration_sec": item["raw_duration_sec"],
                    "final_duration_sec": item["final_duration_sec"],
                    "speed": item["speed"],
                    "flags": item["quality_flags"],
                }
                for item in generated
            ],
            "notes": ["当前版本输出独立配音人声轨；视频导出会替换原音轨，暂不保留原始环境声和 BGM。"],
        }
        generation = {
            "generation_id": generation_id,
            "created_at": _now(),
            "request": request,
            "segments": generated,
            "voice_track": str(voice_track),
            "video": str(video_path) if video_path else None,
            "quality": quality,
        }
        timeline.setdefault("generations", []).append(generation)
        timeline["last_generation"] = generation
        timeline_path = Path(str(job["timeline_path"]))
        save_json(timeline_path, timeline)
        shutil.copy2(timeline_path, Path(str(job["output_path"])) / "时间轴.json")
        _update_job(
            job_id,
            status="done",
            status_label="配音完成",
            phase="导出完成",
            progress=100,
            generation_id=generation_id,
            generation=generation,
            quality=quality,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            status_label="配音失败",
            phase="失败",
            error=f"{type(exc).__name__}: {exc}",
            generation_id=generation_id,
        )


def file_for_job(job_id: str, name: str) -> Path:
    job = get_job(job_id)
    root = Path(str(job["output_path"])).resolve()
    target = (root / name).resolve()
    if root not in target.parents and target != root:
        raise ValueError("文件路径无效")
    if not target.is_file():
        raise ValueError("任务文件不存在")
    return target
