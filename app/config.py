from __future__ import annotations

import os
from pathlib import Path


STATE_DIR = Path(os.getenv("VOICE_STATE_DIR", "/state"))
UPLOADS = STATE_DIR / "uploads"
WORK = STATE_DIR / "speaker-work"
RESULTS = STATE_DIR / "speaker-results"
MANIFESTS = STATE_DIR / "speaker-manifests"
DIRECT_TTS_STATE = STATE_DIR / "direct-tts"
DIRECT_TTS_UPLOADS = DIRECT_TTS_STATE / "uploads"
DIRECT_TTS_WORK = DIRECT_TTS_STATE / "work"
DIRECT_TTS_MANIFESTS = DIRECT_TTS_STATE / "manifests"

DEFAULT_OUTPUT = Path(os.getenv("VOICE_DEFAULT_OUTPUT", "/mnt/f/voice"))
MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "/models/Fun-CosyVoice3-0.5B")
ASR_DIR = os.getenv("QWEN_ASR_MODEL_DIR", "/models/Qwen3-ASR-1.7B")
FORCED_ALIGNER_DIR = os.getenv("QWEN_FORCED_ALIGNER_DIR", "/models/Qwen3-ForcedAligner-0.6B")
DIAR_REPO = os.getenv("THREE_D_SPEAKER_REPO", "/opt/3D-Speaker")
COSYVOICE_REPO = os.getenv("COSYVOICE_REPO", "/opt/CosyVoice")
MODEL_CACHE = os.getenv("MODELSCOPE_CACHE", "/root/.cache/modelscope")
RATE = 24_000
SEPARATION_MODEL = os.getenv("VOICE_SEPARATION_MODEL", "HDEMUCS_HIGH_MUSDB")
SEPARATION_CHUNK_SECONDS = float(os.getenv("VOICE_SEPARATION_CHUNK_SECONDS", "30"))
SEPARATION_OVERLAP_SECONDS = float(os.getenv("VOICE_SEPARATION_OVERLAP_SECONDS", "0.5"))

DUBBING_STATE = STATE_DIR / "dubbing"
DUBBING_UPLOADS = DUBBING_STATE / "uploads"
DUBBING_WORK = DUBBING_STATE / "work"
DUBBING_JOBS = DUBBING_STATE / "jobs"


def ensure_directories() -> None:
    for directory in (
        UPLOADS,
        WORK,
        RESULTS,
        MANIFESTS,
        DIRECT_TTS_UPLOADS,
        DIRECT_TTS_WORK,
        DIRECT_TTS_MANIFESTS,
        DUBBING_UPLOADS,
        DUBBING_WORK,
        DUBBING_JOBS,
        DEFAULT_OUTPUT,
    ):
        directory.mkdir(parents=True, exist_ok=True)
