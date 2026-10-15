"""Environment-audio extraction and one-click background-video export."""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import extract_clean
from .config import ENVIRONMENT_JOBS, RATE
from .dubbing import _mux_video
from .separation import separate_audio, unload_separation
from .utils import duration, probe, safe_name, save_json, to_windows_path, to_wsl_path


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12,40}", str(job_id)):
        raise ValueError("任务编号无效")
    return ENVIRONMENT_JOBS / f"{job_id}.json"


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


def write_silence_track(target: Path, seconds: float) -> Path:
    """Create a correctly sized silent environment track for silent videos."""

    sample_count = max(1, int(math.ceil(max(0.1, float(seconds)) * RATE)))
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.zeros(sample_count, dtype=np.float32), RATE, subtype="PCM_16")
    return target


def _audio_streams(info: dict[str, Any]) -> list[dict[str, Any]]:
    return [stream for stream in info.get("streams", []) if stream.get("codec_type") == "audio"]


def prepare_environment_stems(
    source: Path,
    work_root: Path,
    *,
    noise_reduction: float = 8.0,
) -> dict[str, Any]:
    """Prepare clean, vocal, and environment tracks for a media file.

    A video without an audio stream receives a silent environment track so the
    same mixer can be used for both silent-video dubbing and voiced-video
    replacement.  A voiced source must pass through Demucs; on separation
    failure this function raises instead of silently returning the original
    speech as "background".
    """

    info = probe(source)
    total_seconds = max(0.1, duration(source))
    audio_root = work_root / "音频"
    clean_path = audio_root / "清洁混合音轨.wav"
    vocal_path = audio_root / "人声.wav"
    background_path = audio_root / "环境音.wav"
    audio_root.mkdir(parents=True, exist_ok=True)

    if not _audio_streams(info):
        write_silence_track(vocal_path, total_seconds)
        write_silence_track(background_path, total_seconds)
        return {
            "available": False,
            "mode": "silent_source",
            "model": "none",
            "error": "输入视频没有原始音轨，环境音按静音处理",
            "clean_path": "",
            "vocal_path": str(vocal_path),
            "background_path": str(background_path),
            "sample_rate": RATE,
            "duration_sec": round(total_seconds, 3),
        }

    extract_clean(
        source,
        clean_path,
        noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
        trim_leading_silence=False,
    )
    try:
        result = separate_audio(clean_path, vocal_path, background_path)
    finally:
        # Demucs and TTS share the 16GB GPU.  The separator is only needed for
        # this stage and must not remain resident while CosyVoice runs.
        unload_separation()
    result.update(
        {
            "mode": "demucs_two_stem",
            "clean_path": str(clean_path),
            "vocal_path": str(vocal_path),
            "background_path": str(background_path),
            "duration_sec": round(max(0.1, duration(background_path)), 3),
        }
    )
    return result


def _copy_stem(source: str | Path, target: Path) -> bool:
    path = Path(str(source or ""))
    if not path.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return True


def _apply_gain(source: Path, target: Path, gain: float) -> None:
    data, rate = sf.read(str(source), dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) * max(0.0, min(2.0, float(gain)))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.98:
        data *= 0.96 / peak
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), data, int(rate), subtype="PCM_16")


def _run(job_id: str) -> None:
    try:
        job = get_job(job_id)
        source = Path(str(job["source_path"]))
        output_root = Path(str(job["output_path"]))
        work_root = Path(str(job["work_path"]))
        options = dict(job.get("options") or {})
        output_root.mkdir(parents=True, exist_ok=True)
        _update(job_id, status="running", status_label="处理中", phase="检查媒体", progress=3)

        info = probe(source)
        if not any(stream.get("codec_type") == "video" for stream in info.get("streams", [])):
            raise ValueError("输入文件没有视频画面")
        media_duration = duration(source)
        if media_duration <= 0:
            raise ValueError("无法读取视频时长")

        original_root = output_root / "原始材料"
        original_root.mkdir(parents=True, exist_ok=True)
        original_copy = original_root / safe_name(str(job.get("source_filename") or source.name), 120)
        shutil.copy2(source, original_copy)

        _update(job_id, phase="分离人声与环境音", progress=12)
        stems = prepare_environment_stems(
            source,
            work_root,
            noise_reduction=float(options.get("noise_reduction", 8.0)),
        )
        source_background = Path(str(stems.get("background_path") or ""))
        if not source_background.is_file():
            raise ValueError("没有得到可用的环境音轨")

        audio_root = output_root / "音频"
        clean_export = audio_root / "清洁混合音轨.wav"
        vocal_export = audio_root / "人声.wav"
        background_export = audio_root / "背景声.wav"
        _copy_stem(stems.get("clean_path", ""), clean_export)
        _copy_stem(stems.get("vocal_path", ""), vocal_export)
        _apply_gain(source_background, background_export, float(options.get("background_gain", 1.0)))

        _update(job_id, phase="封装仅环境音视频", progress=75)
        video_target = output_root / "仅环境音视频.mp4"
        video_path = _mux_video(source, background_export, video_target)
        if video_path is None:
            raise ValueError("输入文件没有视频画面，无法导出环境音视频")

        manifest = {
            "schema_version": "voice-workbench-environment-audio/v1",
            "job_id": job_id,
            "source_filename": job.get("source_filename", source.name),
            "source_path": str(source),
            "source_duration_sec": round(media_duration, 3),
            "stems": {
                **stems,
                "background_path": str(background_export),
                "background_gain": float(options.get("background_gain", 1.0)),
            },
            "outputs": {
                "original_media": str(original_copy),
                "clean_audio": str(clean_export) if clean_export.is_file() else "",
                "vocal_audio": str(vocal_export) if vocal_export.is_file() else "",
                "environment_audio": str(background_export),
                "environment_video": str(video_path),
            },
            "created_at": _now(),
        }
        save_json(output_root / "环境音.json", manifest)
        quality = {
            "separation_model": stems.get("model", "none"),
            "source_had_audio": stems.get("mode") != "silent_source",
            "environment_audio_ready": True,
            "notes": [
                "最终视频只使用环境音轨，原始人声没有进入输出混音。",
                "环境音来自 HDEMUCS 的 drums、bass、other 三路合成；两位说话人同时讲话时仍属于同一人声 stem。",
                "输出视频使用 H.264/AAC 兼容封装，便于 Windows 播放。",
            ],
        }
        _update(
            job_id,
            status="done",
            status_label="完成",
            phase="导出完成",
            progress=100,
            manifest_path=str(output_root / "环境音.json"),
            output_files={
                "environment_video": str(video_path),
                "environment_audio": str(background_export),
                "vocal_audio": str(vocal_export) if vocal_export.is_file() else "",
                "clean_audio": str(clean_export) if clean_export.is_file() else "",
                "manifest": str(output_root / "环境音.json"),
            },
            quality=quality,
        )
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            status_label="失败",
            phase="失败",
            progress=100,
            error=f"{type(exc).__name__}: {exc}",
        )


def start_job(
    *,
    job_id: str,
    source: Path,
    original_filename: str,
    output_dir: str,
    noise_reduction: float,
    background_gain: float,
) -> dict[str, Any]:
    target_root = to_wsl_path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    output = target_root / f"{safe_name(Path(original_filename).stem, 70)}__环境音提取__{job_id}"
    output.mkdir(parents=True, exist_ok=True)
    work_root = Path(str(ENVIRONMENT_JOBS.parent / "work" / job_id))
    work_root.mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id,
        "kind": "environment_audio",
        "status": "queued",
        "status_label": "排队中",
        "phase": "等待处理",
        "progress": 0,
        "source_filename": original_filename,
        "source_path": str(source),
        "output_path": str(output),
        "output_dir": to_windows_path(output),
        "work_path": str(work_root),
        "options": {
            "noise_reduction": max(0.0, min(30.0, float(noise_reduction))),
            "background_gain": max(0.0, min(2.0, float(background_gain))),
        },
        "logs": [],
        "created_at": _now(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        _save(job)
        thread = threading.Thread(target=_run, args=(job_id,), daemon=True, name=f"environment-audio-{job_id}")
        job["thread"] = thread
        thread.start()
    return _public(job)
