from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


MODEL_DIR = os.getenv("ASR_MODEL_DIR", "/models/Qwen3-ASR-0.6B")
MODEL: Any = None
MODEL_LOCK = threading.Lock()
TEMP_ROOT = Path(os.getenv("ASR_TEMP_DIR", "/tmp/qwen3-asr-api"))
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Qwen3-ASR isolated service")


def get_model() -> Any:
    global MODEL
    if MODEL is None:
        from qwen_asr import Qwen3ASRModel

        print(f"加载 Qwen3-ASR：{MODEL_DIR}", flush=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        MODEL = Qwen3ASRModel.from_pretrained(
            MODEL_DIR,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=1024,
        )
    return MODEL


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": "Qwen3-ASR-0.6B", "loaded": MODEL is not None}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("Chinese")) -> dict[str, Any]:
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
            results = model.transcribe(audio=str(target), language=language or "Chinese", return_time_stamps=False)
        item = results[0] if isinstance(results, (list, tuple)) else results
        text = getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
        return {"ok": True, "text": str(text or "").strip(), "model": "Qwen3-ASR-0.6B"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qwen3-ASR 失败：{exc}") from exc
    finally:
        target.unlink(missing_ok=True)
