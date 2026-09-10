from __future__ import annotations

import os
from pathlib import Path


STATE_DIR = Path(os.getenv("VOICE_STATE_DIR", "/state"))
UPLOADS = STATE_DIR / "uploads"
WORK = STATE_DIR / "speaker-work"
RESULTS = STATE_DIR / "speaker-results"
MANIFESTS = STATE_DIR / "speaker-manifests"

DEFAULT_OUTPUT = Path(os.getenv("VOICE_DEFAULT_OUTPUT", "/mnt/f/voice"))
MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "/models/Fun-CosyVoice3-0.5B")
ASR_DIR = os.getenv("QWEN_ASR_MODEL_DIR", "/models/Qwen3-ASR-0.6B")
DIAR_REPO = os.getenv("THREE_D_SPEAKER_REPO", "/opt/3D-Speaker")
COSYVOICE_REPO = os.getenv("COSYVOICE_REPO", "/opt/CosyVoice")
MODEL_CACHE = os.getenv("MODELSCOPE_CACHE", "/root/.cache/modelscope")
RATE = 24_000


def ensure_directories() -> None:
    for directory in (UPLOADS, WORK, RESULTS, MANIFESTS, DEFAULT_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
