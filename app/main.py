#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import tts
from .assets import save_card
from .audio import fit_audio
from .config import (
    DEFAULT_OUTPUT,
    DIRECT_TTS_MANIFESTS,
    DIRECT_TTS_UPLOADS,
    DIRECT_TTS_WORK,
    MANIFESTS,
    RESULTS,
    UPLOADS,
    ensure_directories,
)
from .direct_tts import extract_reference
from .identification import analyze_file
from .schemas import TTSGenerateRequest
from .utils import digest, load_manifest, probe, safe_name, save_json, to_windows_path, to_wsl_path


ensure_directories()
app = FastAPI(title="多人参考音色资产工作台")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("/app/web/index.html").read_text(encoding="utf-8")


@app.get("/tts", response_class=HTMLResponse)
def direct_tts_page() -> str:
    return Path("/app/web/tts.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "identification": "3D-Speaker + Qwen3-ASR",
        "tts": "Fun-CosyVoice3-0.5B-2512",
        "tts_modes": ["zero_shot", "cross_lingual", "instruct2"],
        "tts_loaded": tts.TTS_MODEL is not None,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    output_dir: str = Form("F:\\voice"),
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


@app.get("/api/speaker/{asset_id}/sample")
def sample(asset_id: str) -> FileResponse:
    path = Path(load_manifest(MANIFESTS, asset_id)["sample_audio"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="参考样本不存在")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.post("/api/tts/direct")
async def direct_tts(
    reference_media: UploadFile = File(...),
    output_dir: str = Form("F:\\voice"),
    mode: str = Form("cross_lingual"),
    reference_text: str = Form(""),
    prompt_instruction: str = Form("You are a helpful assistant."),
    target_text: str = Form(...),
    reference_start: float = Form(0.0),
    reference_seconds: float = Form(10.0),
    noise_reduction: float = Form(0.0),
    trim_leading_silence: bool = Form(True),
    speed: float = Form(1.0),
    text_frontend: bool = Form(True),
    target_duration: float = Form(5.0),
    fit_duration: bool = Form(False),
    seed: int = Form(0),
) -> dict[str, Any]:
    """Direct TTS workbench: media -> prompt clip -> CosyVoice3.

    This endpoint intentionally does not call diarization or ASR. In
    zero-shot mode the reference transcript is supplied by the user because
    CosyVoice3's official prompt contains it.
    """

    if mode not in {"zero_shot", "cross_lingual", "instruct2"}:
        raise HTTPException(status_code=400, detail="不支持的 CosyVoice3 模式")
    target_text = str(target_text or "").strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="目标文本不能为空")
    if len(target_text) > 2000:
        raise HTTPException(status_code=400, detail="目标文本不能超过 2000 个字符")
    reference_text = str(reference_text or "").strip()
    prompt_instruction = str(prompt_instruction or "You are a helpful assistant.").strip()
    if mode == "zero_shot" and not reference_text:
        raise HTTPException(status_code=400, detail="Zero-shot 模式需要手工填写参考音频台词")

    try:
        speed = float(speed)
        reference_start = max(0.0, float(reference_start))
        reference_seconds = max(1.0, min(30.0, float(reference_seconds)))
        noise_reduction = max(0.0, min(30.0, float(noise_reduction)))
        target_duration = float(target_duration)
        seed = int(seed)
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed 应在 0.5 到 2.0 之间")
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("seed 超出范围")
        if not 0.5 <= target_duration <= 120.0:
            raise ValueError("目标时长应在 0.5 到 120 秒之间")
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
        target_root = to_wsl_path(output_dir or str(DEFAULT_OUTPUT))
        target_root.mkdir(parents=True, exist_ok=True)
        output_root = target_root / f"{safe_name(Path(original).stem, 60)}__直接TTS__{job_id[:8]}"
        output_root.mkdir(parents=True, exist_ok=True)
        work_root = DIRECT_TTS_WORK / job_id
        reference_audio = work_root / "参考音频.wav"
        actual_reference_seconds = extract_reference(
            source,
            reference_audio,
            start_seconds=reference_start,
            max_seconds=reference_seconds,
            noise_reduction=noise_reduction,
            trim_leading_silence=trim_leading_silence,
        )
        exported_reference = output_root / "参考音频.wav"
        shutil.copy2(reference_audio, exported_reference)

        quality_warning = None
        if mode == "zero_shot":
            prompt_text = tts.prompt_marker(reference_text, prompt_instruction)
            if len(target_text) < 0.5 * len(prompt_text):
                quality_warning = "官方前端提示：目标文本相对参考台词过短，可能影响音色和韵律。"

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
            )
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
            "source_sha256": digest(source),
            "reference_audio": str(exported_reference),
            "reference_start_sec": reference_start,
            "reference_duration_sec": round(actual_reference_seconds, 3),
            "reference_text": reference_text,
            "mode": mode,
            "prompt_instruction": prompt_instruction,
            "target_text": target_text,
            "speed": speed,
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
            "warning": quality_warning,
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
            )
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
                "target_duration_sec": request.target_duration,
                "speed": request.speed,
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
