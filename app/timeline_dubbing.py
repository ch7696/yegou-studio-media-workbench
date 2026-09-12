"""Queued timeline dubbing for videos without a usable original soundtrack."""

from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from . import tts
from .asr import transcribe
from .audio import fit_audio
from .config import DUBBING_JOBS, DUBBING_WORK
from .dubbing import _mux_video, _render_voice_track
from .reference import prepare_reference
from .timeline import parse_timeline
from .utils import duration, safe_name, save_json, to_windows_path, to_wsl_path


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12,40}", str(job_id)):
        raise ValueError("任务编号无效")
    return DUBBING_JOBS / f"{job_id}.json"


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "thread"}


def _save(job: dict[str, Any]) -> None:
    save_json(_job_path(str(job["job_id"])), _public(job))


def _update(job_id: str, **values: Any) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            path = _job_path(job_id)
            if not path.exists():
                raise ValueError("任务不存在")
            job = json.loads(path.read_text(encoding="utf-8"))
            JOBS[job_id] = job
        job.update(values)
        _save(job)
        return _public(job)


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            return _public(job)
        path = _job_path(job_id)
        if not path.exists():
            raise ValueError("任务不存在")
        return json.loads(path.read_text(encoding="utf-8"))


def file_for_job(job_id: str, name: str) -> Path:
    job = get_job(job_id)
    root = Path(str(job.get("output_path") or "")).resolve()
    target = (root / name).resolve()
    if root not in target.parents and target != root:
        raise ValueError("文件路径无效")
    if not target.is_file():
        raise ValueError("任务文件不存在")
    return target


def _clamp_speed(value: float) -> float:
    return max(0.5, min(2.0, float(value)))


def _schedule_segments(generated: list[dict[str, Any]]) -> list[str]:
    """Place one-voice clips without creating unintelligible self-overlap.

    Subtitle cues can describe a title card for one second while containing a
    sentence that naturally takes much longer to speak.  Keeping that speech
    at a natural rate is preferable to mixing two copies of the same voice.
    The original cue times remain in ``start_sec`` / ``end_sec``; the actual
    render position is recorded separately for review.
    """

    warnings: list[str] = []
    cursor = 0.0
    for item in generated:
        original_start = float(item.get("start_sec", 0.0) or 0.0)
        render_start = max(original_start, cursor)
        item["render_start_sec"] = round(render_start, 3)
        shift = render_start - original_start
        if shift > 0.01:
            item["timing_shift_sec"] = round(shift, 3)
            warnings.append(
                f"片段 {item.get('segment_id', '?')} 为避免人声叠音顺延 {shift:.2f}s"
            )
        else:
            item["timing_shift_sec"] = 0.0
        cursor = render_start + max(0.0, float(item.get("final_duration_sec", 0.0) or 0.0))
    return warnings


def _usable_segments(timeline: dict[str, Any], video_duration: float) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    warnings = list(timeline.get("warnings") or [])
    for item in timeline.get("segments", []):
        try:
            start = max(0.0, min(float(video_duration), float(item["start_sec"])))
            end = max(0.0, min(float(video_duration), float(item["end_sec"])))
        except (KeyError, TypeError, ValueError):
            warnings.append("跳过了一个时间无效的字幕片段")
            continue
        text = str(item.get("dubbing_text") or item.get("text") or "").strip()
        if not text or end <= start:
            warnings.append(f"跳过片段 {item.get('segment_id', '?')}：文本为空或不在视频范围内")
            continue
        if end - start < 0.25:
            warnings.append(f"片段 {item.get('segment_id', '?')} 只有 {end - start:.2f}s，暂不生成")
            continue
        normalized = dict(item)
        normalized.update(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": text,
                "dubbing_text": text,
                "speaker_id": str(item.get("speaker_id") or "default"),
            }
        )
        output.append(normalized)
    if not output:
        raise ValueError("时间线没有落在视频范围内的可配音片段")
    return output, warnings


def _synthesise_segment(
    model: Any,
    segment: dict[str, Any],
    *,
    index: int,
    output_root: Path,
    reference_audio: Path,
    reference_text: str,
    mode: str,
    speed: float,
    adaptive_speed: bool,
    fit_to_slot: bool,
    prompt_instruction: str,
    text_frontend: bool,
    seed: int,
    prosody_tags: bool,
    voice_gain_db: float,
) -> dict[str, Any]:
    target_duration = max(0.25, float(segment["end_sec"]) - float(segment["start_sec"]))
    segment_root = output_root / "配音片段"
    segment_root.mkdir(parents=True, exist_ok=True)
    prefix = f"{segment.get('segment_id', f'T{index + 1:04d}')}__{safe_name(segment['dubbing_text'], 36)}"
    raw_path = segment_root / f"{prefix}__原始.wav"
    fitted_path = segment_root / f"{prefix}__适配.wav"
    effective_speed = _clamp_speed(speed)
    # Reusing the selected seed for every cue keeps timbre and prosody more
    # stable across a batch.  The text itself still drives each generation.
    generation_seed = max(0, int(seed))

    def generate() -> tuple[Any, float]:
        audio, seconds = tts.synthesize(
            model,
            mode=mode,
            text=segment["dubbing_text"],
            reference_text=reference_text,
            reference_audio=reference_audio,
            speed=effective_speed,
            prompt_instruction=prompt_instruction,
            text_frontend=text_frontend,
            seed=generation_seed,
            prosody_tags=prosody_tags,
        )
        return tts.apply_gain(audio, voice_gain_db), seconds

    audio, raw_seconds = generate()
    tts.save_audio(raw_path, audio, model.sample_rate)
    initial_seconds = raw_seconds
    if adaptive_speed and target_duration >= 0.5:
        proposed = _clamp_speed(effective_speed * raw_seconds / target_duration)
        # Keep the automatic correction close to the user's requested speed;
        # exact subtitle fitting belongs to the optional post-process guard.
        proposed = max(effective_speed * 0.85, min(effective_speed * 1.15, proposed))
        proposed = _clamp_speed(proposed)
        if abs(proposed - effective_speed) >= 0.06:
            effective_speed = proposed
            audio, raw_seconds = generate()
            tts.save_audio(raw_path, audio, model.sample_rate)
    ratio = raw_seconds / target_duration if target_duration else 0.0
    flags: list[str] = []
    timing_mode = "natural"
    if fit_to_slot and 0.80 <= ratio <= 1.25:
        _, final_seconds = fit_audio(raw_path, fitted_path, target_duration)
        output_audio = fitted_path
        timing_mode = "gentle_fit"
    else:
        # Extreme atempo conversion is the main source of the user's
        # fast/slow and unintelligible clips.  Keep the original model output
        # when the cue duration is too far from natural speech.
        final_seconds = raw_seconds
        output_audio = raw_path
        if fit_to_slot:
            flags.append("为保护音质未强行适配字幕时长")
    if ratio < 0.8 or ratio > 1.25:
        flags.append("自然生成时长偏差较大")
    if effective_speed in {0.5, 2.0} and abs(ratio - 1.0) > 0.15:
        flags.append("speed 已达到边界")
    return {
        "segment_id": segment.get("segment_id", f"T{index + 1:04d}"),
        "speaker_id": segment.get("speaker_id", "default"),
        "start_sec": round(float(segment["start_sec"]), 3),
        "end_sec": round(float(segment["end_sec"]), 3),
        "target_duration_sec": round(target_duration, 3),
        "text": segment["dubbing_text"],
        "synthesis_text": tts.prepare_tts_text(segment["dubbing_text"], prosody_tags=prosody_tags),
        "initial_duration_sec": round(initial_seconds, 3),
        "raw_duration_sec": round(raw_seconds, 3),
        "final_duration_sec": round(final_seconds, 3),
        "timing_mode": timing_mode,
        "speed": round(effective_speed, 3),
        "prosody_tags": bool(prosody_tags),
        "voice_gain_db": round(float(voice_gain_db), 2),
        "audio_path": str(output_audio),
        "quality_flags": flags,
    }


def _run(job_id: str) -> None:
    try:
        job = get_job(job_id)
        source = Path(str(job["source_path"]))
        timeline_path = Path(str(job["timeline_source"]))
        reference_source = Path(str(job["reference_source"]))
        work_root = Path(str(job["work_path"]))
        output_root = Path(str(job["output_path"]))
        options = dict(job.get("options") or {})
        _update(job_id, status="running", status_label="读取时间线", phase="读取字幕时间线", progress=5)
        imported = parse_timeline(timeline_path)
        video_duration = duration(source)
        if video_duration <= 0:
            raise ValueError("无法读取视频时长")
        segments, timeline_warnings = _usable_segments(imported, video_duration)

        _update(job_id, status="running", status_label="准备参考音色", phase="清洁并筛选参考音频", progress=12)
        reference_audio = work_root / "参考音频.wav"
        preparation = prepare_reference(
            reference_source,
            reference_audio,
            work_root=work_root / "reference",
            start_seconds=float(options.get("reference_start", 0.0)),
            scan_seconds=float(options.get("reference_scan_seconds", 60.0)),
            target_seconds=float(options.get("reference_target_seconds", 8.0)),
            noise_reduction=float(options.get("noise_reduction", 6.0)),
            trim_leading_silence=bool(options.get("trim_leading_silence", True)),
            auto_select=bool(options.get("auto_select_reference", True)),
            detect_speakers=bool(options.get("detect_reference_speakers", True)),
        )
        exported_reference = output_root / "参考音频.wav"
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference_audio, exported_reference)
        imported_copy = output_root / f"输入时间线{timeline_path.suffix.lower() or '.json'}"
        shutil.copy2(timeline_path, imported_copy)

        mode = str(options.get("mode", "cross_lingual"))
        reference_text = str(options.get("reference_text") or "").strip()
        reference_text_source = "manual" if reference_text else ""
        if mode == "zero_shot" and not reference_text:
            _update(job_id, phase="自动识别参考台词", progress=18)
            detected = transcribe([reference_audio])
            reference_text = (detected[0] if detected else "").strip()
            reference_text_source = "auto_asr"
            if not reference_text:
                raise ValueError("Zero-shot 没有识别到参考台词，请填写参考音频中的实际台词")

        generated: list[dict[str, Any]] = []
        _update(job_id, status="generating", status_label="逐段生成中", phase=f"生成 0/{len(segments)}", progress=20)
        with tts.TTS_LOCK:
            model = tts.get_tts()
            for index, segment in enumerate(segments):
                _update(
                    job_id,
                    phase=f"生成 {index + 1}/{len(segments)}",
                    progress=20 + int(index / max(1, len(segments)) * 65),
                )
                generated.append(
                    _synthesise_segment(
                        model,
                        segment,
                        index=index,
                        output_root=output_root / f"生成_{job_id}",
                        reference_audio=reference_audio,
                        reference_text=reference_text,
                        mode=mode,
                        speed=float(options.get("speed", 1.0)),
                        adaptive_speed=bool(options.get("adaptive_speed", False)),
                        fit_to_slot=bool(options.get("fit_to_slot", False)),
                        prompt_instruction=str(options.get("prompt_instruction") or tts.DEFAULT_PROMPT_INSTRUCTION),
                        text_frontend=bool(options.get("text_frontend", True)),
                        seed=int(options.get("seed", 0)),
                        prosody_tags=bool(options.get("prosody_tags", True)),
                        voice_gain_db=float(options.get("voice_gain_db", 6.0)),
                    )
                )

        timing_warnings = _schedule_segments(generated)
        timeline_warnings.extend(timing_warnings)
        generation_root = output_root / f"生成_{job_id}"
        voice_track = generation_root / "时间线配音.wav"
        _render_voice_track(generated, video_duration, voice_track)
        video_path = None
        if bool(options.get("export_video", True)):
            _update(job_id, phase="封装配音视频", progress=92)
            video_path = _mux_video(source, voice_track, generation_root / "时间线配音视频.mp4")

        timeline_manifest = {
            "schema_version": "voice-workbench-silent-dubbing/v1",
            "source_video": str(source),
            "source_video_filename": job["source_filename"],
            "source_timeline": str(timeline_path),
            "imported_timeline_copy": str(imported_copy),
            "source_timeline_format": imported.get("source_format"),
            "video_duration_sec": round(video_duration, 3),
            "reference_audio": str(exported_reference),
            "reference_text": reference_text,
            "reference_text_source": reference_text_source,
            "reference_preparation": preparation,
            "options": options,
            "warnings": timeline_warnings + list((preparation.get("quality") or {}).get("warnings") or []),
            "imported_segments": segments,
            "segments": generated,
            "created_at": _now(),
        }
        save_json(output_root / "时间线.json", timeline_manifest)
        quality = {
            "segment_count": len(generated),
            "duration_warning_count": sum(1 for item in generated if item["quality_flags"]),
            "warnings": timeline_manifest["warnings"],
            "notes": [
                "本分支使用导入的字幕时间作为配音时间轴，视频原本没有可保留的音轨。",
                "默认保留 CosyVoice 自然语速；只有自然时长接近字幕槽（0.80–1.25 倍）时才做温和适配。",
                "字幕槽严重短于文字时，片段会按自然语速播放，并在必要时顺延后续片段，避免快进和人声叠音。",
                "参考音色会先经过约 8 秒候选筛选；导出的参考音频可单独试听和复核。",
            ],
        }
        generation = {
            "generation_id": job_id,
            "created_at": _now(),
            "voice_track": str(voice_track),
            "video": str(video_path) if video_path else None,
            "reference_audio": str(exported_reference),
            "segments": generated,
            "quality": quality,
        }
        save_json(output_root / "配音结果.json", {"generation": generation, "timeline": timeline_manifest})
        _update(
            job_id,
            status="done",
            status_label="配音完成",
            phase="导出完成",
            progress=100,
            timeline_path=str(output_root / "时间线.json"),
            generation=generation,
            quality=quality,
        )
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            status_label="配音失败",
            phase="失败",
            progress=100,
            error=f"{type(exc).__name__}: {exc}",
        )


def start_job(
    *,
    job_id: str,
    source: Path,
    timeline_source: Path,
    reference_source: Path,
    original_filename: str,
    output_dir: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Create a queued silent-video job and return immediately."""

    if str(options.get("mode", "cross_lingual")) not in {"zero_shot", "cross_lingual", "instruct2"}:
        raise ValueError("不支持的 CosyVoice3 模式")
    imported = parse_timeline(timeline_source)
    target_root = to_wsl_path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    output = target_root / f"{safe_name(Path(original_filename).stem, 70)}__时间线配音项目__{job_id}"
    output.mkdir(parents=True, exist_ok=True)
    work_root = DUBBING_WORK / job_id / "timeline"
    work_root.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id,
        "kind": "silent_timeline_dubbing",
        "status": "queued",
        "status_label": "排队中",
        "phase": "等待处理",
        "progress": 0,
        "source_filename": original_filename,
        "source_path": str(source),
        "timeline_source": str(timeline_source),
        "reference_source": str(reference_source),
        "output_path": str(output),
        "output_dir": to_windows_path(output),
        "work_path": str(work_root),
        "options": options,
        "input_segment_count": len(imported["segments"]),
        "logs": [],
        "created_at": _now(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        _save(job)
        thread = threading.Thread(target=_run, args=(job_id,), daemon=True, name=f"timeline-dubbing-{job_id}")
        job["thread"] = thread
        thread.start()
    return _public(job)
