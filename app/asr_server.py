from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


MODEL_DIR = os.getenv("ASR_MODEL_DIR", "/models/Qwen3-ASR-0.6B")
FORCED_ALIGNER_DIR = os.getenv("FORCED_ALIGNER_DIR", "/models/Qwen3-ForcedAligner-0.6B")
MODEL: Any = None
TIMESTAMPS_SUPPORTED = False
ALIGNER_ERROR = ""
MODEL_LOCK = threading.Lock()
TEMP_ROOT = Path(os.getenv("ASR_TEMP_DIR", "/tmp/qwen3-asr-api"))
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Qwen3-ASR isolated service")


def get_model() -> Any:
    global MODEL, TIMESTAMPS_SUPPORTED, ALIGNER_ERROR
    if MODEL is None:
        from qwen_asr import Qwen3ASRModel

        print(f"加载 Qwen3-ASR：{MODEL_DIR}", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": 1,
            "max_new_tokens": 1024,
        }
        aligner_path = Path(FORCED_ALIGNER_DIR)
        if aligner_path.is_dir():
            kwargs["forced_aligner"] = FORCED_ALIGNER_DIR
            kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device}
            print(f"加载 Qwen3-ForcedAligner：{FORCED_ALIGNER_DIR}", flush=True)
        else:
            ALIGNER_ERROR = f"找不到 ForcedAligner 模型目录：{FORCED_ALIGNER_DIR}"
        try:
            MODEL = Qwen3ASRModel.from_pretrained(MODEL_DIR, **kwargs)
            TIMESTAMPS_SUPPORTED = "forced_aligner" in kwargs
        except Exception as exc:
            if "forced_aligner" not in kwargs:
                raise
            ALIGNER_ERROR = f"ForcedAligner 加载失败，已降级为无时间戳 ASR：{exc}"
            print(ALIGNER_ERROR, flush=True)
            kwargs.pop("forced_aligner", None)
            kwargs.pop("forced_aligner_kwargs", None)
            MODEL = Qwen3ASRModel.from_pretrained(MODEL_DIR, **kwargs)
            TIMESTAMPS_SUPPORTED = False
    return MODEL


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": "Qwen3-ASR-0.6B",
        "forced_aligner": FORCED_ALIGNER_DIR,
        "loaded": MODEL is not None,
        "timestamps_supported": TIMESTAMPS_SUPPORTED,
        "aligner_error": ALIGNER_ERROR,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("Chinese"),
    return_time_stamps: bool = Form(False),
) -> dict[str, Any]:
    suffix = Path(audio.filename or "reference.wav").suffix or ".wav"
    target = TEMP_ROOT / f"{uuid.uuid4().hex}{suffix}"
    try:
        with target.open("wb") as stream:
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        with MODEL_LOCK:
            model = get_model()
            if return_time_stamps and not TIMESTAMPS_SUPPORTED:
                raise RuntimeError(ALIGNER_ERROR or "ForcedAligner 不可用，无法返回时间戳")
            results = model.transcribe(
                audio=str(target),
                language=language or "Chinese",
                return_time_stamps=return_time_stamps,
            )
        item = results[0] if isinstance(results, (list, tuple)) else results
        text = getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
        detected_language = getattr(item, "language", "") or (
            item.get("language", "") if isinstance(item, dict) else ""
        )
        raw_timestamps = getattr(item, "time_stamps", None) or (
            item.get("time_stamps") if isinstance(item, dict) else None
        )
        if isinstance(raw_timestamps, dict):
            raw_timestamps = raw_timestamps.get("items", [])
        else:
            raw_timestamps = getattr(raw_timestamps, "items", raw_timestamps) or []
        timestamps: list[dict[str, Any]] = []
        for token in raw_timestamps:
            token_text = getattr(token, "text", "") or (token.get("text", "") if isinstance(token, dict) else "")
            start = getattr(token, "start_time", 0.0) or (
                token.get("start_time", 0.0) if isinstance(token, dict) else 0.0
            )
            end = getattr(token, "end_time", start) or (
                token.get("end_time", start) if isinstance(token, dict) else start
            )
            try:
                start_value = max(0.0, float(start))
                end_value = max(start_value, float(end))
            except (TypeError, ValueError):
                continue
            if str(token_text or "").strip():
                timestamps.append(
                    {
                        "text": str(token_text),
                        "start_sec": round(start_value, 3),
                        "end_sec": round(end_value, 3),
                    }
                )
        return {
            "ok": True,
            "text": str(text or "").strip(),
            "language": str(detected_language or ""),
            "time_stamps": timestamps,
            "model": "Qwen3-ASR-0.6B",
            "forced_aligner": TIMESTAMPS_SUPPORTED,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen3-ASR 失败：{exc}") from exc
    finally:
        target.unlink(missing_ok=True)
