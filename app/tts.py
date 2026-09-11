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
        # Follow the official CosyVoice3 example.  The previous fp16 override
        # was an assumption made for VRAM usage, but it is not the official
        # default and can make this model unstable on some CUDA/container
        # combinations.  The 0.5B model fits the user's 16 GB card at the
        # official default precision.
        TTS_MODEL = AutoModel(model_dir=MODEL_DIR)
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


def set_seed(seed: int) -> None:
    """Use the same seed hook as the official CosyVoice WebUI."""

    if COSYVOICE_REPO not in sys.path:
        sys.path.insert(0, COSYVOICE_REPO)
    from cosyvoice.utils.common import set_all_random_seed

    set_all_random_seed(int(seed))


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
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Call CosyVoice3 using the exact input shape of each official mode.

    CosyVoice3's official example uses direct prompt audio for all prompt modes. The
    ``add_zero_shot_spk`` cache shortcut belongs to the CosyVoice2 example and
    is not interchangeable here: this model's 24 kHz frontend needs the
    prompt text/audio on the actual inference call.
    """

    set_seed(seed)
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
    elif mode == "cross_lingual":
        # CosyVoice3's official cross-lingual/fine-grained path has no
        # reference transcript. Its text input carries the required prompt
        # prefix and the target text together.
        stream = model.inference_cross_lingual(
            f"{instruction_marker(prompt_instruction)}{text}",
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


def apply_gain(audio: torch.Tensor, gain_db: float, *, peak_limit: float = 0.97) -> torch.Tensor:
    """Raise generated speech while preventing digital clipping.

    CosyVoice's output level is intentionally conservative.  Apply this after
    synthesis and before duration fitting/mixing so the same control affects
    direct previews and batch dubbing.
    """

    amplified = audio.float() * float(10.0 ** (float(gain_db) / 20.0))
    if amplified.numel():
        peak = float(amplified.abs().max().item())
        if peak > float(peak_limit) > 0:
            amplified = amplified * (float(peak_limit) / peak)
    return amplified


def unload_tts() -> None:
    global TTS_MODEL
    if TTS_MODEL is not None:
        del TTS_MODEL
        TTS_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
