from __future__ import annotations

import gc
import re
import threading
from pathlib import Path

import torch

from .config import ASR_DIR


ASR_LOCK = threading.Lock()


def _unload(model: object) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe(paths: list[Path]) -> list[str]:
    """Transcribe short speaker samples through the official Qwen3-ASR API."""

    if not paths:
        return []
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
