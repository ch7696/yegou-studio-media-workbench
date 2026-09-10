from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio

from .config import COSYVOICE_REPO, MODEL_DIR


TTS_MODEL: Any = None
TTS_LOCK = threading.Lock()


def get_tts() -> Any:
    global TTS_MODEL
    if TTS_MODEL is None:
        if COSYVOICE_REPO not in sys.path:
            sys.path.insert(0, COSYVOICE_REPO)
        matcha = f"{COSYVOICE_REPO}/third_party/Matcha-TTS"
        if matcha not in sys.path:
            sys.path.insert(0, matcha)
        from cosyvoice.cli.cosyvoice import AutoModel

        print(f"加载 CosyVoice3：{MODEL_DIR}", flush=True)
        TTS_MODEL = AutoModel(model_dir=MODEL_DIR, fp16=True)
    return TTS_MODEL


def prompt_marker(reference_text: str, instruction: str = "You are a helpful assistant.") -> str:
    """Official CosyVoice3 zero-shot prompt layout."""

    instruction = str(instruction or "You are a helpful assistant.").replace("<|endofprompt|>", "").strip()
    reference_text = str(reference_text or "").replace("<|endofprompt|>", "").strip()
    return f"{instruction}<|endofprompt|>{reference_text}"


def instruction_marker(instruction: str) -> str:
    instruction = str(instruction or "You are a helpful assistant.")
    instruction = instruction.replace("<|endofprompt|>", "").strip()
    return f"{instruction}<|endofprompt|>"


def _audio_chunks(stream: Iterator[dict[str, Any]], sample_rate: int) -> tuple[torch.Tensor, float]:
    chunks: list[torch.Tensor] = []
    for item in stream:
        audio = item["tts_speech"].detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("CosyVoice3 没有生成音频")
    audio = torch.cat(chunks, dim=1)
    return audio, float(audio.shape[-1]) / float(sample_rate)


def synthesize(
    model: Any,
    *,
    mode: str,
    text: str,
    reference_text: str,
    reference_audio: Path,
    speed: float,
    prompt_instruction: str,
    text_frontend: bool,
) -> tuple[torch.Tensor, float]:
    """Call CosyVoice3 using the exact input shape of each official mode.

    CosyVoice3's official example uses direct prompt audio for both modes. The
    ``add_zero_shot_spk`` cache shortcut belongs to the CosyVoice2 example and
    is not interchangeable here: this model's 24 kHz frontend needs the
    prompt text/audio on the actual inference call.
    """

    if mode == "zero_shot":
        if not reference_text.strip():
            raise ValueError("Zero-shot 模式需要参考音频对应的参考台词")
        stream = model.inference_zero_shot(
            text,
            prompt_marker(reference_text, prompt_instruction),
            str(reference_audio),
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
    elif mode == "instruct2":
        stream = model.inference_instruct2(
            text,
            instruction_marker(prompt_instruction),
            str(reference_audio),
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
    else:
        raise ValueError(f"不支持的 CosyVoice3 模式：{mode}")

    return _audio_chunks(stream, model.sample_rate)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate)


def unload_tts() -> None:
    global TTS_MODEL
    if TTS_MODEL is not None:
        del TTS_MODEL
        TTS_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
