from __future__ import annotations

import gc
import os
import re
import threading
from pathlib import Path

import torch

from .config import ASR_DIR


ASR_LOCK = threading.Lock()
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "").strip().rstrip("/")


def _unload(model: object) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(paths: list[Path]) -> list[str]:
    """Transcribe through the isolated Qwen3-ASR service or local fallback."""

    if not paths:
        return []
    if ASR_SERVICE_URL:
        return _transcribe_remote(paths)
    with ASR_LOCK:
        model = None
        try:
            from qwen_asr import Qwen3ASRModel

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = Qwen3ASRModel.from_pretrained(
                ASR_DIR,
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=1,
                max_new_tokens=1024,
            )
            output: list[str] = []
            for path in paths:
                results = model.transcribe(audio=str(path), language="Chinese", return_time_stamps=False)
                item = results[0] if isinstance(results, (list, tuple)) else results
                value = getattr(item, "text", "") or (item.get("text", "") if isinstance(item, dict) else "")
                output.append(re.sub(r"\s+", " ", str(value or "")).strip())
            return output
        except Exception as exc:
            # Identification should still return usable audio assets when ASR
            # is unavailable; the UI lets the user fill the reference text.
            print(f"参考台词自动识别失败：{exc}", flush=True)
            return [""] * len(paths)
        finally:
            if model is not None:
                _unload(model)


def _transcribe_remote(paths: list[Path]) -> list[str]:
    """Call Qwen3-ASR without importing it into the CosyVoice process."""

    try:
        import requests

        output: list[str] = []
        for path in paths:
            with path.open("rb") as stream:
                response = requests.post(
                    f"{ASR_SERVICE_URL}/transcribe",
                    files={"audio": (path.name, stream, "audio/wav")},
                    data={"language": "Chinese"},
                    timeout=600,
                )
            response.raise_for_status()
            value = response.json().get("text", "")
            output.append(re.sub(r"\s+", " ", str(value or "")).strip())
        return output
    except Exception as exc:
        print(f"远程 Qwen3-ASR 服务失败：{exc}", flush=True)
        return [""] * len(paths)
