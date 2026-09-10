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
from .config import DEFAULT_OUTPUT, MANIFESTS, RESULTS, UPLOADS, ensure_directories
from .identification import analyze_file
from .schemas import TTSGenerateRequest
from .utils import digest, load_manifest, probe, safe_name, to_windows_path


ensure_directories()
app = FastAPI(title="多人参考音色资产工作台")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("/app/web/index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "identification": "3D-Speaker + Qwen3-ASR",
        "tts": "Fun-CosyVoice3-0.5B-2512",
        "tts_modes": ["zero_shot", "instruct2"],
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
