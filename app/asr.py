from __future__ import annotations

import gc
import os
import re
import threading
from pathlib import Path
from typing import Any

import torch

from .config import ASR_DIR, FORCED_ALIGNER_DIR


ASR_LOCK = threading.Lock()
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "").strip().rstrip("/")


def _unload(model: object) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _field(item: object, name: str, default: object = "") -> object:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _model_language(language: str | None) -> str | None:
    """Map the UI's auto option to the official Qwen3-ASR ``None`` value."""

    if language is None:
        return None
    value = str(language).strip()
    if value.lower() in {"auto", "automatic", "自动", "自动识别", "中英文"}:
        return None
    return value or "Chinese"


def _timestamp_items(item: object) -> list[dict[str, Any]]:
    raw = _field(item, "time_stamps", None)
    if raw is None:
        raw = _field(item, "timestamps", None)
    if isinstance(raw, dict):
        raw = raw.get("items", raw.get("timestamps", []))
    else:
        raw = getattr(raw, "items", raw)
    if callable(raw) or raw is None:
        return []
    output: list[dict[str, Any]] = []
    for token in raw:
        text = str(_field(token, "text", "") or "").strip()
        try:
            start = float(_field(token, "start_time", _field(token, "start_sec", 0.0)) or 0.0)
            end = float(_field(token, "end_time", _field(token, "end_sec", start)) or start)
        except (TypeError, ValueError):
            continue
        if not text:
            continue
        output.append(
            {
                "text": text,
                "start_sec": round(max(0.0, start), 3),
                "end_sec": round(max(max(0.0, start), end), 3),
            }
        )
    return output


def _normalize_result(item: object, *, include_timestamps: bool) -> dict[str, Any]:
    text = str(_field(item, "text", "") or "").strip()
    language = str(_field(item, "language", "") or "").strip()
    model = str(_field(item, "model", "") or "").strip()
    return {
        "text": re.sub(r"\s+", " ", text),
        "language": language,
        "model": model,
        "time_stamps": _timestamp_items(item) if include_timestamps else [],
    }


def transcribe(
    paths: list[Path],
    *,
    raise_on_error: bool = False,
    language: str | None = "Chinese",
) -> list[str]:
    """Transcribe through Qwen3-ASR, optionally surfacing service errors."""

    return [
        item["text"]
        for item in transcribe_detailed(
            paths,
            return_time_stamps=False,
            raise_on_error=raise_on_error,
            language=language,
        )
    ]


def transcribe_detailed(
    paths: list[Path],
    *,
    return_time_stamps: bool = True,
    raise_on_error: bool = False,
    language: str | None = "Chinese",
) -> list[dict[str, Any]]:
    """Return Qwen3-ASR text plus optional ForcedAligner token timestamps."""

    if not paths:
        return []
    if ASR_SERVICE_URL:
        return _transcribe_remote_detailed(
            paths,
            return_time_stamps=return_time_stamps,
            raise_on_error=raise_on_error,
            language=language,
        )
    with ASR_LOCK:
        model = None
        try:
            from qwen_asr import Qwen3ASRModel

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            kwargs: dict[str, Any] = {
                "dtype": dtype,
                "device_map": device,
                "max_inference_batch_size": 1,
                "max_new_tokens": 1024,
            }
            if return_time_stamps and Path(FORCED_ALIGNER_DIR).is_dir():
                kwargs["forced_aligner"] = FORCED_ALIGNER_DIR
                kwargs["forced_aligner_kwargs"] = {"dtype": dtype, "device_map": device}
            model = Qwen3ASRModel.from_pretrained(ASR_DIR, **kwargs)
            output: list[dict[str, Any]] = []
            for path in paths:
                results = model.transcribe(
                    audio=str(path),
                    language=_model_language(language),
                    return_time_stamps=return_time_stamps and "forced_aligner" in kwargs,
                )
                item = results[0] if isinstance(results, (list, tuple)) else results
                output.append(_normalize_result(item, include_timestamps=return_time_stamps))
            return output
        except Exception as exc:
            # Identification should still return usable audio assets when ASR
            # is unavailable; the UI lets the user fill the reference text.
            print(f"参考台词自动识别失败：{exc}", flush=True)
            if raise_on_error:
                raise RuntimeError(f"Qwen3-ASR 识别失败：{exc}") from exc
            return [{"text": "", "language": "", "model": "", "time_stamps": []} for _ in paths]
        finally:
            if model is not None:
                _unload(model)


def _transcribe_remote_detailed(
    paths: list[Path],
    *,
    return_time_stamps: bool,
    raise_on_error: bool = False,
    language: str | None = "Chinese",
) -> list[dict[str, Any]]:
    """Call Qwen3-ASR without importing it into the CosyVoice process."""

    import requests

    output: list[dict[str, Any]] = []
    for path in paths:
        try:
            requested_language = _model_language(language)
            data = {"language": requested_language or "auto"}
            if return_time_stamps:
                data["return_time_stamps"] = "true"
            with path.open("rb") as stream:
                response = requests.post(
                    f"{ASR_SERVICE_URL}/transcribe",
                    files={"audio": (path.name, stream, "audio/wav")},
                    data=data,
                    timeout=1800,
                )
            response.raise_for_status()
            payload = response.json()
            output.append(
                {
                    "text": re.sub(r"\s+", " ", str(payload.get("text", "") or "")).strip(),
                    "language": str(payload.get("language", "") or "").strip(),
                    "model": str(payload.get("model", "") or "").strip(),
                    "time_stamps": list(payload.get("time_stamps", payload.get("timestamps", [])) or [])
                    if return_time_stamps
                    else [],
                }
            )
        except Exception as exc:
            print(f"远程 Qwen3-ASR 服务失败（{path.name}）：{exc}", flush=True)
            if raise_on_error:
                raise RuntimeError(f"远程 Qwen3-ASR 识别失败（{path.name}）：{exc}") from exc
            output.append({"text": "", "language": "", "model": "", "time_stamps": []})
    return output


def _transcribe_remote(paths: list[Path]) -> list[str]:
    """Compatibility helper for callers that only need plain text."""

    try:
        return [item["text"] for item in _transcribe_remote_detailed(paths, return_time_stamps=False)]
    except Exception as exc:
        print(f"远程 Qwen3-ASR 服务失败：{exc}", flush=True)
        return [""] * len(paths)


def release_remote_model() -> dict[str, Any]:
    """Ask the isolated ASR service to unload its model without stopping it."""

    if not ASR_SERVICE_URL:
        return {"ok": True, "released": [], "warnings": ["未配置独立 ASR 服务"]}
    import requests

    response = requests.post(f"{ASR_SERVICE_URL}/release-memory", timeout=15)
    payload: dict[str, Any]
    try:
        raw = response.json()
        payload = raw if isinstance(raw, dict) else {}
    except ValueError:
        payload = {}
    if not response.ok:
        detail = str(payload.get("detail") or response.text or f"HTTP {response.status_code}").strip()
        raise RuntimeError(detail)
    return payload
