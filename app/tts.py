from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path
from typing import Any, Iterator

import torch
import torchaudio

from . import models
from .config import COSYVOICE_REPO, MODEL_DIR


TTS_MODEL: Any = None
TTS_LOCK = threading.Lock()
# 已加载实例当前应用的变体 id（未加载时为 None；与 TTS_MODEL 同生命周期）。
TTS_VARIANT_ID: str | None = None

# CosyVoice3's official examples use this English prefix and then put the
# actual expression request after it.  Keeping one default across the direct,
# timeline, and full dubbing paths prevents the UI from silently falling back
# to a flat generic instruction.
DEFAULT_PROMPT_INSTRUCTION = (
    "You are a helpful assistant. 请用自然、像真人交流一样的语气说话，"
    "保持正常语速和自然的情绪起伏。"
)

# These three emotional phrases are present in CosyVoice3's own instruct list;
# the remaining presets are deliberately short so the model can still follow
# the reference voice instead of being pushed into a theatrical style.
EXPRESSION_PRESETS = {
    "natural": DEFAULT_PROMPT_INSTRUCTION,
    "cheerful": "You are a helpful assistant. 请非常开心地说一句话。",
    "sad": "You are a helpful assistant. 请非常伤心地说一句话。",
    "angry": "You are a helpful assistant. 请非常生气地说一句话。",
    "warm": "You are a helpful assistant. 请用温暖、亲切、像真人交流一样的语气说话。",
    "serious": "You are a helpful assistant. 请用沉稳、认真、清晰的语气说话。",
    "soft": "You are a helpful assistant. Please say a sentence in a very soft voice.",
    "loud": "You are a helpful assistant. Please say a sentence as loudly as possible.",
}

_CONTROL_TOKENS = (
    "[breath]", "[quick_breath]", "[laughter]", "[cough]", "[sigh]",
    "[vocalized-noise]", "[lipsmack]", "<strong>", "</strong>",
)


def normalize_symbol_reading(text: str) -> str:
    """Turn math signs into words before CosyVoice tokenization.

    The subtitle text stored in the timeline remains unchanged.  This copy is
    only used for synthesis because bare ``+`` and ``-`` are easy for a text
    frontend/tokenizer to ignore.  A leading sign before a number is read as
    positive/negative; an operator between values is read as plus/minus.
    """

    value = (
        str(text or "")
        .replace("＋", "+")
        .replace("－", "-")
        .replace("−", "-")
        .replace("±", "正负")
        .replace("+/−", "加减")
        .replace("+/-", "加减")
    )
    if not value or ("+" not in value and "-" not in value):
        return value

    def nearest_non_space(position: int, step: int) -> str:
        while 0 <= position < len(value) and value[position].isspace():
            position += step
        return value[position] if 0 <= position < len(value) else ""

    sign_context = set("([{（【《「『=：:，,；;。！？!?")
    output: list[str] = []
    for index, char in enumerate(value):
        if char == "+":
            previous = nearest_non_space(index - 1, -1)
            following = nearest_non_space(index + 1, 1)
            raw_previous = value[index - 1] if index else ""
            leading_sign = (
                following.isdigit()
                and (
                    not previous
                    or previous in sign_context
                    or (raw_previous.isspace() and not previous.isascii())
                )
            )
            word = "正" if leading_sign else "加"
            output.append(word)
        elif char == "-":
            previous = nearest_non_space(index - 1, -1)
            following = nearest_non_space(index + 1, 1)
            raw_previous = value[index - 1] if index else ""
            leading_sign = (
                following.isdigit()
                and (
                    not previous
                    or previous in sign_context
                    or (raw_previous.isspace() and not previous.isascii())
                )
            )
            word = "负" if leading_sign else "减"
            output.append(word)
        else:
            output.append(char)
    return "".join(output)


def _apply_variant(model: Any, variant: dict[str, Any]) -> None:
    """把变体权重原地加载进已初始化的 CosyVoice3 实例。

    ``AutoModel`` 只会按官方默认加载 ``llm.pt``；RL 等变体需要在初始化后显式
    覆盖 LLM 权重。``CosyVoice3Model.load(llm, flow, hift)`` 支持在活实例上
    重新 ``load_state_dict``，因此无需卸载整个模型。
    """

    llm_path = models.variant_llm_path(variant)
    if not llm_path.is_file():
        raise FileNotFoundError(f"变体权重不存在：{llm_path}")
    print(f"加载 TTS 变体权重：{llm_path}", flush=True)
    model.model.load(str(llm_path), f"{MODEL_DIR}/flow.pt", f"{MODEL_DIR}/hift.pt")


def get_tts() -> Any:
    global TTS_MODEL, TTS_VARIANT_ID
    if TTS_MODEL is None:
        if COSYVOICE_REPO not in sys.path:
            sys.path.insert(0, COSYVOICE_REPO)
        matcha = f"{COSYVOICE_REPO}/third_party/Matcha-TTS"
        if matcha not in sys.path:
            sys.path.insert(0, matcha)
        from cosyvoice.cli.cosyvoice import AutoModel

        variant = models.get_tts_variant(models.active_tts_variant_id())
        if variant is None:
            variant = dict(models.TTS_VARIANTS[0])
        print(f"加载 CosyVoice3：{MODEL_DIR}（变体：{variant['label']}）", flush=True)
        # Follow the official CosyVoice3 example.  The previous fp16 override
        # was an assumption made for VRAM usage, but it is not the official
        # default and can make this model unstable on some CUDA/container
        # combinations.  The 0.5B model fits the user's 16 GB card at the
        # official default precision.
        TTS_MODEL = AutoModel(model_dir=MODEL_DIR)
        # AutoModel 已加载 llm.pt；非基础变体在此覆盖 LLM 权重。
        if str(variant.get("llm") or "llm.pt") != "llm.pt":
            _apply_variant(TTS_MODEL, variant)
        TTS_VARIANT_ID = str(variant["id"])
    return TTS_MODEL


def active_model_label() -> str:
    """当前（或已保存的）TTS 主模型显示名，供 health / 清单记录使用。"""

    variant = models.get_tts_variant(TTS_VARIANT_ID or models.active_tts_variant_id())
    if variant is None:
        variant = dict(models.TTS_VARIANTS[0])
    return str(variant["label"])


def switch_variant(variant_id: str) -> dict[str, Any]:
    """切换 TTS 主模型变体；已加载时原地覆盖权重，失败自动回滚。

    调用方必须持有 ``TTS_LOCK``（所有合成路径在同一把锁下串行）。
    """

    global TTS_VARIANT_ID
    variant = models.get_tts_variant(variant_id)
    if variant is None:
        raise ValueError(f"未知的 TTS 变体：{variant_id}")
    if not models.variant_is_available(variant):
        raise FileNotFoundError(f"变体权重不存在：{models.variant_llm_path(variant)}")

    previous_id = TTS_VARIANT_ID or models.active_tts_variant_id()
    if TTS_MODEL is not None and previous_id != variant_id:
        try:
            _apply_variant(TTS_MODEL, variant)
        except Exception:
            # 原地回滚到旧变体，保证服务保持可用。
            previous = models.get_tts_variant(previous_id)
            if previous is not None:
                try:
                    _apply_variant(TTS_MODEL, previous)
                except Exception:
                    pass
            raise
    TTS_VARIANT_ID = str(variant["id"])
    models.save_settings({"tts_variant": str(variant["id"])})
    return {
        "variant": variant,
        "previous": previous_id,
        "loaded": TTS_MODEL is not None,
    }


def prompt_marker(reference_text: str, instruction: str = DEFAULT_PROMPT_INSTRUCTION) -> str:
    """Official CosyVoice3 zero-shot prompt layout."""

    instruction = str(instruction or DEFAULT_PROMPT_INSTRUCTION).replace("<|endofprompt|>", "").strip()
    reference_text = str(reference_text or "").replace("<|endofprompt|>", "").strip()
    return f"{instruction}<|endofprompt|>{reference_text}"


def instruction_marker(instruction: str) -> str:
    instruction = str(instruction or DEFAULT_PROMPT_INSTRUCTION)
    instruction = instruction.replace("<|endofprompt|>", "").strip()
    return f"{instruction}<|endofprompt|>"


def add_light_prosody_tags(text: str) -> str:
    """Add a few official ``[breath]`` controls at internal clause breaks.

    This does not rewrite subtitle text.  It only adds CosyVoice3 control
    tokens after internal punctuation, following the fine-grained-control
    example in the official repository.  A hard cap keeps short subtitle cues
    from becoming over-acted.
    """

    value = str(text or "").strip()
    if not value or any(token in value for token in _CONTROL_TOKENS):
        return value

    # Avoid treating a closing quote/bracket as the beginning of a new clause.
    closing = " \t\r\n，。！？；：、,.!?;:…—-)]）】》」』”’\"'"
    output: list[str] = []
    inserted = 0
    index = 0
    while index < len(value):
        if value.startswith("……", index):
            punctuation = "……"
            index += 2
        else:
            punctuation = value[index]
            index += 1
        output.append(punctuation)
        if (
            punctuation in {"，", ",", "；", ";", "：", ":", "…", "……"}
            and index < len(value)
            and value[index] not in closing
            and inserted < 3
        ):
            output.append("[breath]")
            inserted += 1
    return "".join(output)


def prepare_tts_text(text: str, *, prosody_tags: bool = True) -> str:
    """Prepare the model-facing text while preserving the original subtitle."""

    value = normalize_symbol_reading(text)
    return add_light_prosody_tags(value) if prosody_tags else value.strip()


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
    prosody_tags: bool = True,
) -> tuple[torch.Tensor, float]:
    """Call CosyVoice3 using the exact input shape of each official mode.

    CosyVoice3's official example uses direct prompt audio for all prompt modes. The
    ``add_zero_shot_spk`` cache shortcut belongs to the CosyVoice2 example and
    is not interchangeable here: this model's 24 kHz frontend needs the
    prompt text/audio on the actual inference call.
    """

    set_seed(seed)
    tts_text = prepare_tts_text(text, prosody_tags=prosody_tags)
    # The official frontend automatically bypasses text normalization when it
    # sees SSML-like control markers.  Do the same for [breath] in instruct2
    # and zero-shot, where the target text itself has no endofprompt marker.
    has_control_tokens = any(token in tts_text for token in _CONTROL_TOKENS)
    effective_text_frontend = bool(text_frontend) and not has_control_tokens
    if mode == "zero_shot":
        if not reference_text.strip():
            raise ValueError("Zero-shot 模式需要参考音频对应的参考台词")
        stream = model.inference_zero_shot(
            tts_text,
            prompt_marker(reference_text, prompt_instruction),
            str(reference_audio),
            stream=False,
            speed=speed,
            text_frontend=effective_text_frontend,
        )
    elif mode == "cross_lingual":
        # CosyVoice3's official cross-lingual/fine-grained path has no
        # reference transcript. Its text input carries the required prompt
        # prefix and the target text together.
        stream = model.inference_cross_lingual(
            f"{instruction_marker(prompt_instruction)}{tts_text}",
            str(reference_audio),
            stream=False,
            speed=speed,
            text_frontend=effective_text_frontend,
        )
    elif mode == "instruct2":
        stream = model.inference_instruct2(
            tts_text,
            instruction_marker(prompt_instruction),
            str(reference_audio),
            stream=False,
            speed=speed,
            text_frontend=effective_text_frontend,
        )
    else:
        raise ValueError(f"不支持的 CosyVoice3 模式：{mode}")

    return _audio_chunks(stream, model.sample_rate)


def save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio, sample_rate)


def apply_gain(audio: torch.Tensor, gain_db: float, *, peak_limit: float = 0.97) -> torch.Tensor:
    """Normalize generated speech to a predictable level and apply user gain.

    Peak-only amplification leaves quiet speech quiet when a single transient
    reaches the limiter. Use active-sample RMS as the practical local loudness
    proxy, then apply a peak ceiling. The default +6 dB setting targets about
    -20 dBFS RMS, which is suitable for a standalone speech preview.
    """

    output = audio.float()
    if not output.numel():
        return output
    peak = float(output.abs().max().item())
    active_threshold = max(1.0e-4, peak * 0.05)
    active = output.abs() >= active_threshold
    if bool(active.any()):
        rms = float(torch.sqrt(torch.mean(torch.square(output[active]))).item())
        current_db = 20.0 * float(torch.log10(torch.tensor(max(rms, 1.0e-8))).item())
        target_db = -26.0 + float(gain_db)
        output = output * float(10.0 ** ((target_db - current_db) / 20.0))
    peak = float(output.abs().max().item())
    if peak > float(peak_limit) > 0:
        output = output * (float(peak_limit) / peak)
    return output


def unload_tts() -> None:
    global TTS_MODEL, TTS_VARIANT_ID
    TTS_VARIANT_ID = None
    if TTS_MODEL is not None:
        del TTS_MODEL
        TTS_MODEL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
