"""Two-stem human-voice/background separation for the dubbing workbench."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .config import RATE, SEPARATION_CHUNK_SECONDS, SEPARATION_MODEL, SEPARATION_OVERLAP_SECONDS


SEPARATION_LOCK = threading.Lock()
SEPARATION_MODEL_INSTANCE: Any = None
SEPARATION_SAMPLE_RATE = 44_100
SEPARATION_SOURCES = ("drums", "bass", "other", "vocals")


def _bundle() -> Any:
    if SEPARATION_MODEL != "HDEMUCS_HIGH_MUSDB":
        raise ValueError(f"暂不支持的人声分离模型：{SEPARATION_MODEL}")
    return torchaudio.pipelines.HDEMUCS_HIGH_MUSDB


def _get_model() -> tuple[Any, torch.device]:
    global SEPARATION_MODEL_INSTANCE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if SEPARATION_MODEL_INSTANCE is None:
        model = _bundle().get_model().to(device)
        model.eval()
        SEPARATION_MODEL_INSTANCE = model
        print(f"加载人声分离模型：{SEPARATION_MODEL}（{device}）", flush=True)
    return SEPARATION_MODEL_INSTANCE, device


def unload_separation() -> None:
    """Release the separator before CosyVoice starts using the GPU."""

    global SEPARATION_MODEL_INSTANCE
    with SEPARATION_LOCK:
        if SEPARATION_MODEL_INSTANCE is not None:
            del SEPARATION_MODEL_INSTANCE
            SEPARATION_MODEL_INSTANCE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _crossfade_window(length: int, *, fade_in: int, fade_out: int) -> torch.Tensor:
    window = torch.ones(length, dtype=torch.float32)
    if fade_in > 0:
        window[:fade_in] = torch.linspace(0.0, 1.0, fade_in)
    if fade_out > 0:
        window[-fade_out:] = torch.minimum(
            window[-fade_out:], torch.linspace(1.0, 0.0, fade_out)
        )
    return window


def separate_audio(
    source: Path,
    vocal_target: Path,
    background_target: Path,
    *,
    chunk_seconds: float = SEPARATION_CHUNK_SECONDS,
    overlap_seconds: float = SEPARATION_OVERLAP_SECONDS,
    load_audio=None,
    save_audio=None,
) -> dict[str, Any]:
    """Separate a mono/stereo mix into vocals and the remaining background.

    Hybrid Demucs returns drums, bass, other and vocals.  The workbench keeps
    vocals as one human-voice stem and sums the other three sources into the
    background stem.  This is deliberately a two-stem pass: it does not claim
    to separate two simultaneous human speakers from one another.
    """

    with SEPARATION_LOCK:
        model, device = _get_model()
        waveform, sample_rate = (load_audio or torchaudio.load)(str(source))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        if int(sample_rate) != SEPARATION_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), SEPARATION_SAMPLE_RATE)
        waveform = waveform.to(dtype=torch.float32)
        total_samples = int(waveform.shape[-1])
        if total_samples <= 0:
            raise RuntimeError("人声分离输入为空")

        chunk_samples = max(1, int(float(chunk_seconds) * SEPARATION_SAMPLE_RATE))
        overlap_samples = max(0, min(chunk_samples // 2, int(float(overlap_seconds) * SEPARATION_SAMPLE_RATE)))
        hop_samples = max(1, chunk_samples - overlap_samples)
        vocal_acc = torch.zeros(total_samples, dtype=torch.float32)
        background_acc = torch.zeros(total_samples, dtype=torch.float32)
        weights = torch.zeros(total_samples, dtype=torch.float32)

        for start in range(0, total_samples, hop_samples):
            stop = min(total_samples, start + chunk_samples)
            chunk = waveform[:, start:stop].unsqueeze(0).to(device)
            with torch.inference_mode():
                estimated = model(chunk)
            estimated = estimated[0].detach().to("cpu")
            usable = min(stop - start, int(estimated.shape[-1]))
            if usable <= 0:
                continue
            fade = min(overlap_samples, usable // 2)
            window = _crossfade_window(
                usable,
                fade_in=fade if start > 0 else 0,
                fade_out=fade if stop < total_samples else 0,
            )
            vocal = estimated[3, :, :usable].mean(dim=0)
            background = estimated[:3, :, :usable].sum(dim=0).mean(dim=0)
            vocal_acc[start : start + usable] += vocal * window
            background_acc[start : start + usable] += background * window
            weights[start : start + usable] += window
            if stop >= total_samples:
                break

        uncovered = weights <= 1e-6
        safe_weights = weights.clamp_min(1e-6)
        vocal = vocal_acc / safe_weights
        background = background_acc / safe_weights
        # If an unusual backend returns a shorter chunk, preserve the source in
        # the uncovered tail instead of producing a silent hole.
        source_mono = waveform.mean(dim=0).to("cpu")
        vocal[uncovered] = source_mono[uncovered]
        background[uncovered] = 0.0

        output = torch.stack((vocal, background))
        peak = float(output.abs().max()) if output.numel() else 0.0
        if peak > 0.98:
            output *= 0.96 / peak
        if int(SEPARATION_SAMPLE_RATE) != RATE:
            output = torchaudio.functional.resample(output, SEPARATION_SAMPLE_RATE, RATE)

        vocal_target.parent.mkdir(parents=True, exist_ok=True)
        background_target.parent.mkdir(parents=True, exist_ok=True)
        writer = save_audio or torchaudio.save
        writer(str(vocal_target), output[0:1], RATE)
        writer(str(background_target), output[1:2], RATE)
        return {
            "available": True,
            "model": SEPARATION_MODEL,
            "sources": list(SEPARATION_SOURCES),
            "kept_source": "vocals",
            "background_sources": ["drums", "bass", "other"],
            "device": str(device),
            "sample_rate": RATE,
            "duration_sec": round(float(output.shape[-1]) / RATE, 3),
            "chunk_seconds": float(chunk_seconds),
            "overlap_seconds": float(overlap_seconds),
        }
