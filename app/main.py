#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import mimetypes
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import torch

from . import asr, dubbing, separation, timeline_dubbing, tts
from .asr import transcribe
from .assets import save_card
from .audio import fit_audio
from .config import (
    DEFAULT_OUTPUT,
    DIARIZATION_BACKEND,
    DIRECT_TTS_MANIFESTS,
    DIRECT_TTS_UPLOADS,
    DIRECT_TTS_WORK,
    DUBBING_UPLOADS,
    DUBBING_WORK,
    GITHUB_REPOSITORY_URL,
    MANIFESTS,
    RESULTS,
    UPLOADS,
    ensure_directories,
)
from .diarization import backend_status
from .identification import analyze_file
from .reference import prepare_reference
from .schemas import DubbingGenerateRequest, DubbingTimelineSaveRequest, TTSGenerateRequest
from .timeline import parse_timeline
from .utils import digest, load_manifest, probe, safe_name, save_json, to_windows_path, to_wsl_path


ensure_directories()
app = FastAPI(title="野构 Studio 声像制片台")
MEMORY_RELEASE_LOCK = threading.Lock()


def _memory_status() -> dict[str, Any]:
    """Return model/cache state without loading any model."""

    status: dict[str, Any] = {
        "webui_online": True,
        "tts_loaded": tts.TTS_MODEL is not None,
        "separation_loaded": separation.SEPARATION_MODEL_INSTANCE is not None,
        "asr_service_configured": bool(asr.ASR_SERVICE_URL),
        "diarization_model_lifecycle": "任务结束后自动释放",
    }
    if not torch.cuda.is_available():
        status["cuda"] = {"available": False}
        return status
    try:
        status["cuda"] = {
            "available": True,
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        }
    except Exception as exc:
        status["cuda"] = {"available": True, "error": str(exc)}
    return status


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("/app/web/index.html").read_text(encoding="utf-8")


@app.get("/assets", response_class=HTMLResponse)
def assets_page() -> str:
    return Path("/app/web/assets.html").read_text(encoding="utf-8")


@app.get("/tts", response_class=HTMLResponse)
def direct_tts_page() -> str:
    return Path("/app/web/tts.html").read_text(encoding="utf-8")


@app.get("/studio", response_class=HTMLResponse)
def dubbing_studio_page() -> str:
    return Path("/app/web/studio.html").read_text(encoding="utf-8")


@app.get("/timeline", response_class=HTMLResponse)
def timeline_dubbing_page() -> str:
    return Path("/app/web/timeline.html").read_text(encoding="utf-8")


@app.get("/studio.js")
def dubbing_studio_script() -> FileResponse:
    return FileResponse(Path("/app/web/studio.js"), media_type="application/javascript")


@app.get("/console.css")
def console_stylesheet() -> FileResponse:
    return FileResponse(Path("/app/web/console.css"), media_type="text/css")


@app.get("/brand-logo.png")
def brand_logo() -> FileResponse:
    return FileResponse(Path("/app/web/brand-logo.png"), media_type="image/png")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "identification": "Sortformer / 3D-Speaker + Qwen3-ASR",
        "diarization_backend": DIARIZATION_BACKEND,
        "diarization_backends": backend_status(),
        "tts": "Fun-CosyVoice3-0.5B-2512",
        "tts_modes": ["zero_shot", "cross_lingual", "instruct2"],
        "tts_loaded": tts.TTS_MODEL is not None,
        "separation_loaded": separation.SEPARATION_MODEL_INSTANCE is not None,
        "dubbing": True,
    }


@app.get("/api/system/memory")
def system_memory() -> dict[str, Any]:
    """Expose the 7866 workbench's model and CUDA cache state."""

    return {"ok": True, **_memory_status()}


@app.get("/api/system/config")
def system_config() -> dict[str, Any]:
    """Expose deployment-provided defaults to the browser UI."""

    return {
        "ok": True,
        "default_output": to_windows_path(DEFAULT_OUTPUT),
        "github_url": GITHUB_REPOSITORY_URL,
    }


@app.post("/api/system/release-memory")
def release_memory() -> dict[str, Any]:
    """Release cached models while keeping the 7866 FastAPI process alive."""

    if not MEMORY_RELEASE_LOCK.acquire(False):
        raise HTTPException(status_code=409, detail="显存释放正在进行，请稍后再试")
    tts_acquired = False
    released: list[str] = []
    warnings: list[str] = []
    try:
        # TTS generation holds this lock for the complete synthesis/export
        # operation.  Do not unload a model while a request is using it.
        tts_acquired = tts.TTS_LOCK.acquire(False)
        if not tts_acquired:
            raise HTTPException(status_code=409, detail="当前有 TTS 任务正在执行，完成后再释放显存")
        # Separation holds its lock for the complete Demucs pass.
        if separation.SEPARATION_LOCK.locked():
            raise HTTPException(status_code=409, detail="当前有人声分离任务正在执行，完成后再释放显存")

        if tts.TTS_MODEL is not None:
            tts.unload_tts()
            released.append("CosyVoice3")
        if separation.SEPARATION_MODEL_INSTANCE is not None:
            separation.unload_separation()
            released.append("HDEMUCS")

        # ASR is intentionally isolated in port 18763.  Ask that service to
        # drop its persistent model too, while keeping the service itself up.
        if asr.ASR_SERVICE_URL:
            try:
                remote = asr.release_remote_model()
                if remote.get("released"):
                    released.extend(str(item) for item in remote["released"])
                warnings.extend(str(item) for item in remote.get("warnings", []))
            except Exception as exc:
                warnings.append(f"独立 ASR 服务未能释放：{exc}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        message = "7866 本地显存已释放，WebUI 保持在线"
        if warnings:
            message += "；部分独立服务需要稍后重试"
        return {
            "ok": True,
            "status": "released" if not warnings else "partial",
            "message": message,
            "released": released,
            "warnings": warnings,
            **_memory_status(),
        }
    finally:
        if tts_acquired:
            tts.TTS_LOCK.release()
        MEMORY_RELEASE_LOCK.release()


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    output_dir: str = Form(""),
    speaker_num: int = Form(0),
    sample_seconds: float = Form(10.0),
    noise_reduction: float = Form(12.0),
    trim_leading_silence: bool = Form(False),
) -> dict[str, Any]:
    original = safe_name(file.filename or "未命名音视频", 120)
    suffix = Path(original).suffix.lower() or ".bin"
    source_id = uuid.uuid4().hex
    source = UPLOADS / f"{source_id}{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("wb") as stream:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    try:
        # Probe here to give a clear error before loading GPU models.
        probe(source)
        return analyze_file(
            source,
            source_id=source_id,
            original_filename=original,
            output_dir=output_dir or str(DEFAULT_OUTPUT),
            speaker_num=max(0, int(speaker_num)),
            sample_seconds=float(sample_seconds),
            noise_reduction=float(noise_reduction),
            trim_leading_silence=trim_leading_silence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"多人资产处理失败：{exc}") from exc


@app.post("/api/dubbing/analyze", status_code=202)
async def start_dubbing_analysis(
    media: UploadFile = File(...),
    output_dir: str = Form(""),
    speaker_num: int = Form(0),
    noise_reduction: float = Form(8.0),
    include_overlap: bool = Form(True),
    diarization_backend: str = Form(""),
    language: str = Form("Chinese"),
) -> dict[str, Any]:
    """Queue full-media diarization plus word-level ASR alignment."""

    original = safe_name(media.filename or "未命名音视频", 120)
    suffix = Path(original).suffix.lower() or ".bin"
    upload_id = uuid.uuid4().hex
    source = DUBBING_UPLOADS / f"{upload_id}{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("wb") as stream:
        while True:
            chunk = await media.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    try:
        probe(source)
        return dubbing.start_analysis(
            source,
            original_filename=original,
            output_dir=output_dir or str(DEFAULT_OUTPUT),
            speaker_num=max(0, int(speaker_num)),
            noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
            include_overlap=bool(include_overlap),
            diarization_backend=str(diarization_backend or ""),
            language=str(language or "Chinese"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"全片时间轴任务创建失败：{exc}") from exc


def _dubbing_job_response(job: dict[str, Any]) -> dict[str, Any]:
    """Add browser URLs for completed generated files without changing the manifest."""

    response = dict(job)
    output_root = Path(str(response.get("output_path", "")))
    generation = response.get("generation")
    if isinstance(generation, dict):
        urls: dict[str, str] = {}
        for key in ("voice_track", "mix_track", "video"):
            value = generation.get(key)
            if not value:
                continue
            try:
                relative = Path(str(value)).resolve().relative_to(output_root.resolve())
            except ValueError:
                continue
            urls[key] = f"/api/dubbing/jobs/{response['job_id']}/files/{relative.as_posix()}"
        if urls:
            response["generation_urls"] = urls
    return response


@app.get("/api/dubbing/jobs/{job_id}")
def dubbing_job(job_id: str) -> dict[str, Any]:
    try:
        return _dubbing_job_response(dubbing.get_job(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/dubbing/jobs/{job_id}/timeline")
def dubbing_timeline(job_id: str) -> dict[str, Any]:
    try:
        return dubbing.get_timeline(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/dubbing/jobs/{job_id}/timeline")
def save_dubbing_timeline(job_id: str, request: DubbingTimelineSaveRequest) -> dict[str, Any]:
    try:
        edits = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in request.edits
        ]
        return dubbing.save_timeline_edits(job_id, edits)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"时间轴暂存失败：{exc}") from exc


@app.post("/api/dubbing/jobs/{job_id}/speakers/{speaker_id}/reference")
async def dubbing_reference(
    job_id: str,
    speaker_id: str,
    reference_media: UploadFile = File(...),
    reference_start: float = Form(0.0),
    reference_seconds: float = Form(10.0),
    noise_reduction: float = Form(0.0),
    trim_leading_silence: bool = Form(True),
    reference_text: str = Form(""),
) -> dict[str, Any]:
    """Attach a manually chosen audio/video excerpt to one diarized speaker."""

    try:
        dubbing.get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    original = safe_name(reference_media.filename or "未命名参考媒体", 120)
    suffix = Path(original).suffix.lower() or ".bin"
    upload_id = uuid.uuid4().hex
    source = DUBBING_WORK / job_id / "reference-uploads" / f"{upload_id}{suffix}"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("wb") as stream:
        while True:
            chunk = await reference_media.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    try:
        probe(source)
        return dubbing.attach_reference(
            job_id,
            speaker_id=speaker_id,
            source=source,
            original_filename=original,
            start_sec=max(0.0, float(reference_start)),
            seconds=max(1.0, min(30.0, float(reference_seconds))),
            noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
            trim_leading_silence=bool(trim_leading_silence),
            reference_text=str(reference_text or "").strip(),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"参考音色绑定失败：{exc}") from exc


@app.post("/api/dubbing/jobs/{job_id}/generate", status_code=202)
def start_dubbing_generation(job_id: str, request: DubbingGenerateRequest) -> dict[str, Any]:
    try:
        edits = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in request.edits
        ]
        return dubbing.start_generation(
            job_id,
            edits=edits,
            selected_ids=request.selected_ids,
            target_speaker_id=request.target_speaker_id,
            mode=request.mode,
            speed=request.speed,
            adaptive_speed=request.adaptive_speed,
            fit_to_slot=request.fit_to_slot,
            prompt_instruction=request.prompt_instruction,
            text_frontend=request.text_frontend,
            seed=request.seed,
            export_video=request.export_video,
            target_original_volume=request.target_original_volume,
            other_speaker_volume=request.other_speaker_volume,
            background_volume=request.background_volume,
            voice_gain_db=request.voice_gain_db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量配音任务创建失败：{exc}") from exc


@app.get("/api/dubbing/jobs/{job_id}/files/{name:path}")
def dubbing_file(job_id: str, name: str) -> FileResponse:
    try:
        path = dubbing.file_for_job(job_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


async def _save_upload(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)


def _timeline_job_response(job: dict[str, Any]) -> dict[str, Any]:
    response = dict(job)
    output_value = str(response.get("output_path") or "").strip()
    if not output_value:
        return response
    output_root = Path(output_value)
    values: dict[str, Any] = {}
    generation = response.get("generation")
    if isinstance(generation, dict):
        values.update(generation)
    if response.get("timeline_path"):
        values["timeline"] = response["timeline_path"]
    urls: dict[str, str] = {}
    for key in ("reference_audio", "voice_track", "video", "timeline"):
        value = values.get(key)
        if not value:
            continue
        try:
            relative = Path(str(value)).resolve().relative_to(output_root.resolve())
        except (OSError, ValueError):
            continue
        urls[key] = f"/api/timeline/jobs/{response['job_id']}/files/{relative.as_posix()}"
    if urls:
        response["generation_urls"] = urls
    return response


@app.post("/api/timeline/jobs", status_code=202)
async def start_timeline_dubbing(
    video: UploadFile = File(...),
    timeline: UploadFile = File(...),
    reference_media: UploadFile = File(...),
    output_dir: str = Form(""),
    mode: str = Form("zero_shot"),
    reference_text: str = Form(""),
    prompt_instruction: str = Form(tts.DEFAULT_PROMPT_INSTRUCTION),
    reference_start: float = Form(0.0),
    reference_scan_seconds: float = Form(60.0),
    reference_target_seconds: float = Form(8.0),
    auto_select_reference: bool = Form(True),
    detect_reference_speakers: bool = Form(True),
    noise_reduction: float = Form(6.0),
    trim_leading_silence: bool = Form(True),
    speed: float = Form(1.0),
    adaptive_speed: bool = Form(False),
    fit_to_slot: bool = Form(False),
    voice_gain_db: float = Form(6.0),
    text_frontend: bool = Form(True),
    seed: int = Form(0),
    prosody_tags: bool = Form(True),
    export_video: bool = Form(True),
) -> dict[str, Any]:
    """Queue TTS for an imported SRT/VTT/JSON timeline and a silent video."""

    if mode not in {"zero_shot", "cross_lingual", "instruct2"}:
        raise HTTPException(status_code=400, detail="不支持的 CosyVoice3 模式")
    original_video = safe_name(video.filename or "未命名视频", 120)
    video_suffix = Path(original_video).suffix.lower() or ".mp4"
    original_timeline = safe_name(timeline.filename or "时间线.json", 120)
    timeline_suffix = Path(original_timeline).suffix.lower() or ".json"
    original_reference = safe_name(reference_media.filename or "参考音频.wav", 120)
    reference_suffix = Path(original_reference).suffix.lower() or ".wav"
    job_id = uuid.uuid4().hex[:12]
    input_root = DUBBING_WORK / job_id / "input"
    source = input_root / f"视频{video_suffix}"
    timeline_source = input_root / f"时间线{timeline_suffix}"
    reference_source = input_root / f"参考音色{reference_suffix}"
    await _save_upload(video, source)
    await _save_upload(timeline, timeline_source)
    await _save_upload(reference_media, reference_source)
    try:
        video_probe = probe(source)
        if not any(stream.get("codec_type") == "video" for stream in video_probe.get("streams", [])):
            raise ValueError("导入文件中没有视频画面")
        probe(reference_source)
        imported = parse_timeline(timeline_source)
        numeric = {
            "reference_start": max(0.0, float(reference_start)),
            "reference_scan_seconds": max(1.0, min(300.0, float(reference_scan_seconds))),
            "reference_target_seconds": max(1.0, min(30.0, float(reference_target_seconds))),
            "noise_reduction": max(0.0, min(30.0, float(noise_reduction))),
            "speed": float(speed),
            "voice_gain_db": float(voice_gain_db),
            "seed": int(seed),
        }
        if not 0.5 <= numeric["speed"] <= 2.0:
            raise ValueError("speed 应在 0.5 到 2.0 之间")
        if not -12.0 <= numeric["voice_gain_db"] <= 18.0:
            raise ValueError("配音响度增益应在 -12 到 18 dB 之间")
        if not 0 <= numeric["seed"] <= 2_147_483_647:
            raise ValueError("seed 超出范围")
        options = {
            **numeric,
            "mode": mode,
            "reference_text": str(reference_text or "").strip()[:2000],
            "prompt_instruction": str(prompt_instruction or tts.DEFAULT_PROMPT_INSTRUCTION).strip()[:500],
            "auto_select_reference": bool(auto_select_reference),
            "detect_reference_speakers": bool(detect_reference_speakers),
            "trim_leading_silence": bool(trim_leading_silence),
            "adaptive_speed": bool(adaptive_speed),
            "fit_to_slot": bool(fit_to_slot),
            "text_frontend": bool(text_frontend),
            "prosody_tags": bool(prosody_tags),
            "export_video": bool(export_video),
        }
        job = timeline_dubbing.start_job(
            job_id=job_id,
            source=source,
            timeline_source=timeline_source,
            reference_source=reference_source,
            original_filename=original_video,
            output_dir=output_dir or str(DEFAULT_OUTPUT),
            options=options,
        )
        response = _timeline_job_response(job)
        response["timeline_preview"] = imported["segments"][:8]
        response["timeline_warnings"] = imported.get("warnings", [])
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"时间线配音任务创建失败：{exc}") from exc


@app.get("/api/timeline/jobs/{job_id}")
def timeline_dubbing_job(job_id: str) -> dict[str, Any]:
    try:
        return _timeline_job_response(timeline_dubbing.get_job(job_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/timeline/jobs/{job_id}/files/{name:path}")
def timeline_dubbing_file(job_id: str, name: str) -> FileResponse:
    try:
        path = timeline_dubbing.file_for_job(job_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/speaker/{asset_id}/sample")
def sample(asset_id: str) -> FileResponse:
    path = Path(load_manifest(MANIFESTS, asset_id)["sample_audio"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="参考样本不存在")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _normalize_reference_text(value: object) -> str:
    """Keep ASR output usable as the exact CosyVoice prompt transcript."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    return text[:2000]


def _transcribe_reference_audio(reference_audio: Path) -> tuple[str, dict[str, Any]]:
    """Transcribe one prepared prompt clip and retain ASR provenance."""

    results = asr.transcribe_detailed(
        [reference_audio],
        return_time_stamps=False,
        raise_on_error=True,
    )
    item = results[0] if results else {}
    text = _normalize_reference_text(item.get("text") if isinstance(item, dict) else "")
    if not text:
        raise RuntimeError("Qwen3-ASR 没有输出可用的参考台词，请换一段人声更清楚的媒体")
    return text, {
        "model": str(item.get("model") or "") if isinstance(item, dict) else "",
        "language": str(item.get("language") or "") if isinstance(item, dict) else "",
        "audio": str(reference_audio),
    }


@app.post("/api/tts/reference-transcript")
async def reference_transcript(
    reference_media: UploadFile = File(...),
    reference_start: float = Form(0.0),
    reference_seconds: float = Form(60.0),
    reference_target_seconds: float = Form(8.0),
    auto_select_reference: bool = Form(True),
    detect_reference_speakers: bool = Form(True),
    noise_reduction: float = Form(6.0),
    trim_leading_silence: bool = Form(True),
) -> dict[str, Any]:
    """Prepare a prompt clip and fill the Zero-shot transcript with Qwen3-ASR."""

    original = safe_name(reference_media.filename or "未命名参考媒体", 120)
    suffix = Path(original).suffix.lower() or ".bin"
    job_id = uuid.uuid4().hex
    source = DIRECT_TTS_UPLOADS / f"{job_id}{suffix}"
    with source.open("wb") as stream:
        while True:
            chunk = await reference_media.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)

    try:
        probe(source)
        reference_start = max(0.0, float(reference_start))
        reference_seconds = max(1.0, min(300.0, float(reference_seconds)))
        reference_target_seconds = max(1.0, min(30.0, float(reference_target_seconds)))
        noise_reduction = max(0.0, min(30.0, float(noise_reduction)))
        work_root = DIRECT_TTS_WORK / f"transcript-{job_id}"
        reference_audio = work_root / "参考音频.wav"
        preparation = prepare_reference(
            source,
            reference_audio,
            work_root=work_root,
            start_seconds=reference_start,
            scan_seconds=reference_seconds,
            target_seconds=reference_target_seconds,
            noise_reduction=noise_reduction,
            trim_leading_silence=trim_leading_silence,
            auto_select=auto_select_reference,
            detect_speakers=detect_reference_speakers,
        )
        text, reference_asr = _transcribe_reference_audio(reference_audio)
        preparation_manifest = {
            "source_sha256": digest(source),
            "reference_audio": str(reference_audio),
            "preparation": preparation,
            "reference_text": text,
            "reference_text_source": "auto_asr",
            "reference_asr": reference_asr,
        }
        save_json(work_root / "准备.json", preparation_manifest)
        return {
            "ok": True,
            "reference_text": text,
            "reference_text_source": "auto_asr",
            "reference_duration_sec": preparation["reference_duration_sec"],
            "reference_preparation": preparation,
            "reference_asr": reference_asr,
            "prepared_reference_id": job_id,
            "message": "参考台词已自动填入，可直接使用或手工修改",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"自动识别参考台词失败：{exc}") from exc


@app.post("/api/tts/direct")
async def direct_tts(
    reference_media: UploadFile = File(...),
    output_dir: str = Form(""),
    mode: str = Form("zero_shot"),
    reference_text: str = Form(""),
    prompt_instruction: str = Form(tts.DEFAULT_PROMPT_INSTRUCTION),
    prosody_tags: bool = Form(True),
    target_text: str = Form(...),
    reference_start: float = Form(0.0),
    reference_seconds: float = Form(60.0),
    reference_target_seconds: float = Form(8.0),
    auto_select_reference: bool = Form(True),
    detect_reference_speakers: bool = Form(True),
    noise_reduction: float = Form(6.0),
    trim_leading_silence: bool = Form(True),
    speed: float = Form(1.0),
    output_gain_db: float = Form(6.0),
    text_frontend: bool = Form(True),
    target_duration: float = Form(5.0),
    fit_duration: bool = Form(False),
    seed: int = Form(0),
    prepared_reference_id: str = Form(""),
) -> dict[str, Any]:
    """Direct TTS workbench: media -> clean prompt clip -> CosyVoice3.

    The reference preparation stage may use the configured diarization backend
    for long or multi-speaker inputs. In zero-shot mode a missing reference
    transcript is filled by Qwen3-ASR; the WebUI presents it for confirmation.
    """

    if mode not in {"zero_shot", "cross_lingual", "instruct2"}:
        raise HTTPException(status_code=400, detail="不支持的 CosyVoice3 模式")
    target_text = str(target_text or "").strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="目标文本不能为空")
    if len(target_text) > 2000:
        raise HTTPException(status_code=400, detail="目标文本不能超过 2000 个字符")
    reference_text = str(reference_text or "").strip()
    prompt_instruction = str(prompt_instruction or tts.DEFAULT_PROMPT_INSTRUCTION).strip()
    try:
        speed = float(speed)
        reference_start = max(0.0, float(reference_start))
        reference_seconds = max(1.0, min(300.0, float(reference_seconds)))
        reference_target_seconds = max(1.0, min(30.0, float(reference_target_seconds)))
        noise_reduction = max(0.0, min(30.0, float(noise_reduction)))
        target_duration = float(target_duration)
        seed = int(seed)
        output_gain_db = float(output_gain_db)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed 应在 0.5 到 2.0 之间")
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("seed 超出范围")
        if not 0.5 <= target_duration <= 120.0:
            raise ValueError("目标时长应在 0.5 到 120 秒之间")
        if not -12.0 <= output_gain_db <= 18.0:
            raise ValueError("输出响度增益应在 -12 到 18 dB 之间")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    original = safe_name(reference_media.filename or "未命名参考媒体", 120)
    suffix = Path(original).suffix.lower() or ".bin"
    job_id = uuid.uuid4().hex
    source = DIRECT_TTS_UPLOADS / f"{job_id}{suffix}"
    with source.open("wb") as stream:
        while True:
            chunk = await reference_media.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)

    try:
        source_sha256 = digest(source)
        target_root = to_wsl_path(output_dir or str(DEFAULT_OUTPUT))
        target_root.mkdir(parents=True, exist_ok=True)
        output_root = target_root / f"{safe_name(Path(original).stem, 60)}__直接TTS__{job_id[:8]}"
        output_root.mkdir(parents=True, exist_ok=True)
        work_root = DIRECT_TTS_WORK / job_id
        reference_audio = work_root / "参考音频.wav"
        reused_preparation = False
        prepared_reference_text = ""
        prepared_reference_asr: dict[str, Any] = {}
        preparation: dict[str, Any]
        if re.fullmatch(r"[a-f0-9]{32}", str(prepared_reference_id or "")):
            saved_path = DIRECT_TTS_WORK / f"transcript-{prepared_reference_id}" / "准备.json"
            if saved_path.exists():
                try:
                    saved_preparation = json.loads(saved_path.read_text(encoding="utf-8"))
                    saved_audio = Path(str(saved_preparation.get("reference_audio") or ""))
                    if saved_preparation.get("source_sha256") == source_sha256 and saved_audio.exists():
                        shutil.copy2(saved_audio, reference_audio)
                        preparation = dict(saved_preparation.get("preparation") or {})
                        preparation["reference_audio"] = str(reference_audio)
                        prepared_reference_text = _normalize_reference_text(
                            saved_preparation.get("reference_text")
                        )
                        saved_asr = saved_preparation.get("reference_asr")
                        if isinstance(saved_asr, dict):
                            prepared_reference_asr = dict(saved_asr)
                        reused_preparation = True
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    reused_preparation = False
        if not reused_preparation:
            preparation = prepare_reference(
                source,
                reference_audio,
                work_root=work_root,
                start_seconds=reference_start,
                scan_seconds=reference_seconds,
                target_seconds=reference_target_seconds,
                noise_reduction=noise_reduction,
                trim_leading_silence=trim_leading_silence,
                auto_select=auto_select_reference,
                detect_speakers=detect_reference_speakers,
            )
        exported_reference = output_root / "参考音频.wav"
        shutil.copy2(reference_audio, exported_reference)

        reference_text = _normalize_reference_text(reference_text)
        reference_asr: dict[str, Any] = {}
        reference_text_source = "manual" if reference_text else ""
        if mode == "zero_shot":
            if not reference_text and prepared_reference_text:
                reference_text = prepared_reference_text
                reference_asr = prepared_reference_asr
                reference_text_source = "auto_asr"
            elif not reference_text:
                reference_text, reference_asr = _transcribe_reference_audio(reference_audio)
                reference_text_source = "auto_asr"
            elif prepared_reference_text and reference_text == prepared_reference_text:
                # The browser sends the text returned by the preparation call;
                # preserve its provenance instead of labeling it as manual.
                reference_asr = prepared_reference_asr
                reference_text_source = "auto_asr"

        preparation_warnings = list((preparation.get("quality") or {}).get("warnings") or [])
        quality_warning = None
        if mode == "zero_shot":
            prompt_text = tts.prompt_marker(reference_text, prompt_instruction)
            if len(target_text) < 0.5 * len(prompt_text):
                quality_warning = "官方前端提示：目标文本相对参考台词过短，可能影响音色和韵律。"
        warnings = preparation_warnings + ([quality_warning] if quality_warning else [])

        with tts.TTS_LOCK:
            model = tts.get_tts()
            audio, raw_seconds = tts.synthesize(
                model,
                mode=mode,
                text=target_text,
                reference_text=reference_text,
                reference_audio=reference_audio,
                speed=speed,
                prompt_instruction=prompt_instruction,
                text_frontend=text_frontend,
                seed=seed,
                prosody_tags=prosody_tags,
            )
            audio = tts.apply_gain(audio, output_gain_db)
            raw_path = work_root / "原始生成.wav"
            tts.save_audio(raw_path, audio, model.sample_rate)
            if fit_duration:
                fitted = work_root / "严格适配时长.wav"
                _, final_seconds = fit_audio(raw_path, fitted, target_duration)
                duration_label = f"目标{target_duration:g}秒"
            else:
                fitted = raw_path
                final_seconds = raw_seconds
                duration_label = "自然时长"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = output_root / f"试音__{safe_name(target_text, 48)}__{mode}__{duration_label}__{stamp}.wav"
        shutil.copy2(fitted, exported)
        manifest = {
            "schema_version": "direct-tts-job/v1",
            "job_id": job_id,
            "source_filename": original,
            "source_path": str(source),
            "source_sha256": source_sha256,
            "reference_audio": str(exported_reference),
            "reference_start_sec": reference_start,
            "reference_duration_sec": preparation["reference_duration_sec"],
            "reference_scan_seconds": reference_seconds,
            "reference_target_seconds": reference_target_seconds,
            "auto_select_reference": auto_select_reference,
            "detect_reference_speakers": detect_reference_speakers,
            "reference_preparation": preparation,
            "prepared_reference_id": prepared_reference_id if reused_preparation else "",
            "reference_text": reference_text,
            "reference_text_source": reference_text_source,
            "reference_asr": reference_asr,
            "mode": mode,
            "prompt_instruction": prompt_instruction,
            "prosody_tags": bool(prosody_tags),
            "target_text": target_text,
            "speed": speed,
            "output_gain_db": output_gain_db,
            "seed": seed,
            "text_frontend": text_frontend,
            "fit_duration": fit_duration,
            "target_duration_sec": target_duration,
            "raw_duration_sec": round(raw_seconds, 3),
            "final_duration_sec": round(final_seconds, 3),
            "output_root": str(output_root),
            "output": str(exported),
            "identification": False,
            "asr": False,
            "diarization": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(DIRECT_TTS_MANIFESTS / f"{job_id}.json", manifest)
        return {
            "ok": True,
            "job_id": job_id,
            "message": f"直接 TTS 完成：自然时长 {raw_seconds:.2f}s",
            "output_path": to_windows_path(exported),
            "reference_path": to_windows_path(exported_reference),
            "output_url": f"/api/tts/direct/{job_id}/output",
            "reference_url": f"/api/tts/direct/{job_id}/reference",
            "mode": mode,
            "raw_duration_sec": round(raw_seconds, 3),
            "final_duration_sec": round(final_seconds, 3),
            "warning": "\n".join(warnings) if warnings else None,
            "reference_text": reference_text,
            "reference_text_source": reference_text_source,
            "reference_asr": reference_asr,
            "output_gain_db": output_gain_db,
            "reference_preparation": preparation,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"直接 TTS 失败：{exc}") from exc


def _direct_job_file(job_id: str, field: str) -> FileResponse:
    manifest = load_manifest(DIRECT_TTS_MANIFESTS, job_id)
    path = Path(manifest[field])
    if not path.exists():
        raise HTTPException(status_code=404, detail="直接 TTS 文件不存在")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/tts/direct/{job_id}/output")
def direct_tts_output(job_id: str) -> FileResponse:
    return _direct_job_file(job_id, "output")


@app.get("/api/tts/direct/{job_id}/reference")
def direct_tts_reference(job_id: str) -> FileResponse:
    return _direct_job_file(job_id, "reference_audio")


def _generate(request: TTSGenerateRequest) -> dict[str, Any]:
    manifest = load_manifest(MANIFESTS, request.speaker_asset_id)
    reference_audio = Path(manifest["sample_audio"])
    reference_text = request.reference_text.strip() or str(manifest.get("reference_text") or "").strip()
    if not reference_audio.exists():
        raise HTTPException(status_code=404, detail="参考样本不存在")
    if request.mode == "zero_shot" and not reference_text:
        raise HTTPException(status_code=400, detail="Zero-shot 模式需要参考台词，请填写参考音频的实际台词")

    quality_warning = None
    if request.mode == "zero_shot":
        prompt_text = tts.prompt_marker(reference_text, request.prompt_instruction)
        if len(request.text.strip()) < 0.5 * len(prompt_text):
            quality_warning = "官方前端提示：目标文本相对参考台词过短，可能影响音色和韵律；可换更短的参考样本或提供更长的目标文本。"

    with tts.TTS_LOCK:
        try:
            model = tts.get_tts()
            audio, raw_seconds = tts.synthesize(
                model,
                mode=request.mode,
                text=request.text.strip(),
                reference_text=reference_text,
                reference_audio=reference_audio,
                speed=request.speed,
                prompt_instruction=request.prompt_instruction,
                text_frontend=request.text_frontend,
                seed=request.seed,
                prosody_tags=request.prosody_tags,
            )
            audio = tts.apply_gain(audio, request.output_gain_db)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = RESULTS / f"{request.speaker_asset_id}__{stamp}__{request.mode}__原始.wav"
            tts.save_audio(raw_path, audio, model.sample_rate)
            if request.fit_duration:
                fitted = RESULTS / f"{request.speaker_asset_id}__{stamp}__目标{request.target_duration:g}秒.wav"
                _, final_seconds = fit_audio(raw_path, fitted, request.target_duration)
            else:
                fitted = raw_path
                final_seconds = raw_seconds

            root = Path(manifest["asset_root"])
            duration_label = f"目标{request.target_duration:g}秒" if request.fit_duration else "自然时长"
            exported = root / (
                f"试音__{safe_name(request.text, 42)}__{request.mode}__{duration_label}__{stamp}.wav"
            )
            shutil.copy2(fitted, exported)
            item = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "text": request.text.strip(),
                "reference_text": reference_text,
                "prompt_instruction": request.prompt_instruction,
                "prosody_tags": bool(request.prosody_tags),
                "target_duration_sec": request.target_duration,
                "speed": request.speed,
                "output_gain_db": request.output_gain_db,
                "seed": request.seed,
                "text_frontend": request.text_frontend,
                "fit_duration": request.fit_duration,
                "raw_duration_sec": round(raw_seconds, 3),
                "final_duration_sec": round(final_seconds, 3),
                "model": "Fun-CosyVoice3-0.5B-2512",
                "mode": request.mode,
                "output": str(exported),
                "output_sha256": digest(exported),
            }
            manifest["reference_text"] = reference_text
            manifest.setdefault("generations", []).append(item)
            manifest["latest_output"] = str(exported)
            from .utils import save_json

            save_json(MANIFESTS / f"{request.speaker_asset_id}.json", manifest)
            save_card(manifest)
            message = f"{manifest['speaker_label']} 试音完成：自然时长 {raw_seconds:.2f}s"
            if request.fit_duration:
                message += f"，已适配到 {final_seconds:.2f}s"
            return {
                "ok": True,
                "message": message,
                "output_url": f"/api/speaker/{request.speaker_asset_id}/latest",
                "output_path": to_windows_path(exported),
                "mode": request.mode,
                "final_duration_sec": round(final_seconds, 3),
                "output_gain_db": request.output_gain_db,
                "warning": quality_warning,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"试音失败：{exc}") from exc


@app.post("/api/tts/generate")
def generate_tts(request: TTSGenerateRequest) -> dict[str, Any]:
    return _generate(request)


@app.post("/api/generate", include_in_schema=False)
def generate_legacy(request: TTSGenerateRequest) -> dict[str, Any]:
    """Compatibility alias for the previous single-purpose page."""

    return _generate(request)


@app.get("/api/speaker/{asset_id}/latest")
def latest(asset_id: str) -> FileResponse:
    path = Path(load_manifest(MANIFESTS, asset_id).get("latest_output", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="还没有试音结果")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7865)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
