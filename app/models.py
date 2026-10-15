"""TTS 主模型变体注册表与选择持久化。

同一个模型目录（``config.MODEL_DIR``）下可以有多个 LLM 权重文件，例如官方
基础版 ``llm.pt`` 与强化学习版 ``llm.rl.pt``。本模块登记可用变体、记录用户
选择、给出变体可用性状态；真正的权重切换动作由 ``app.tts`` 执行。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import MODEL_DIR, STATE_DIR
from .utils import save_json


SETTINGS_PATH = STATE_DIR / "settings.json"

# TTS 变体注册表：新增权重文件（同一模型目录）时在此登记即可。
TTS_VARIANTS: list[dict[str, Any]] = [
    {
        "id": "base",
        "label": "Fun-CosyVoice3-0.5B-2512",
        "llm": "llm.pt",
        "note": "官方标准权重（默认）",
    },
    {
        "id": "rl",
        "label": "Fun-CosyVoice3-0.5B-2512-RL",
        "llm": "llm.rl.pt",
        "note": "强化学习后训练权重：内容准确率更高，音色相似度略降",
    },
]


def load_settings() -> dict[str, Any]:
    """读取 /state 下的设置 JSON；损坏或缺失时返回空设置。"""

    try:
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_settings(values: dict[str, Any]) -> dict[str, Any]:
    """合并写入设置并返回完整设置。"""

    data = load_settings()
    data.update(values)
    save_json(SETTINGS_PATH, data)
    return data


def get_tts_variant(variant_id: str) -> dict[str, Any] | None:
    for variant in TTS_VARIANTS:
        if variant["id"] == variant_id:
            return dict(variant)
    return None


def active_tts_variant_id() -> str:
    """当前选择的变体 id；设置缺失或失效时回退到第一个变体。"""

    settings = load_settings()
    value = str(settings.get("tts_variant") or "").strip()
    if get_tts_variant(value) is not None:
        return value
    return str(TTS_VARIANTS[0]["id"])


def variant_llm_path(variant: dict[str, Any]) -> Path:
    return Path(MODEL_DIR) / str(variant.get("llm") or "llm.pt")


def variant_is_available(variant: dict[str, Any]) -> bool:
    return variant_llm_path(variant).is_file()


def tts_variant_status(*, active_id: str, loaded_variant: str | None) -> list[dict[str, Any]]:
    """变体列表 + 每个变体的可用/当前/已加载状态。"""

    status: list[dict[str, Any]] = []
    for variant in TTS_VARIANTS:
        status.append(
            {
                "id": variant["id"],
                "label": variant["label"],
                "note": variant["note"],
                "llm": variant["llm"],
                "available": variant_is_available(variant),
                "active": variant["id"] == active_id,
                "loaded": loaded_variant is not None and variant["id"] == loaded_variant,
            }
        )
    return status
