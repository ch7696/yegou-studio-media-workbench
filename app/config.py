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
GITHUB_REPOSITORY_URL = (
    os.getenv("GITHUB_REPOSITORY_URL")
    or ""
).strip()
MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "/models/Fun-CosyVoice3-0.5B")
ASR_DIR = os.getenv("QWEN_ASR_MODEL_DIR", "/models/Qwen3-ASR-1.7B")
FORCED_ALIGNER_DIR = os.getenv("QWEN_FORCED_ALIGNER_DIR", "/models/Qwen3-ForcedAligner-0.6B")
DIAR_REPO = os.getenv("THREE_D_SPEAKER_REPO", "/opt/3D-Speaker")
COSYVOICE_REPO = os.getenv("COSYVOICE_REPO", "/opt/CosyVoice")
MODEL_CACHE = os.getenv("MODELSCOPE_CACHE", "/root/.cache/modelscope")
# V3 keeps 3D-Speaker as the explicit A/B fallback and uses NVIDIA's
# four-speaker Streaming Sortformer by default for the dubbing pipeline.
DIARIZATION_BACKEND = os.getenv("DIARIZATION_BACKEND", "sortformer").strip().lower()
HF_ACCESS_TOKEN = (
    os.getenv("HF_ACCESS_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HF_TOKEN")
    or ""
).strip()
SORTFORMER_MODEL = os.getenv(
    "SORTFORMER_MODEL",
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
).strip()
SORTFORMER_ACTIVITY_THRESHOLD = float(os.getenv("SORTFORMER_ACTIVITY_THRESHOLD", "0.50"))
SORTFORMER_OVERLAP_THRESHOLD = float(os.getenv("SORTFORMER_OVERLAP_THRESHOLD", "0.50"))
SPEAKER_ASSIGNMENT_STRATEGY = os.getenv("SPEAKER_ASSIGNMENT_STRATEGY", "posterior").strip().lower()
SPEAKER_MIN_CONFIDENCE = float(os.getenv("SPEAKER_MIN_CONFIDENCE", "0.55"))
SPEAKER_AMBIGUITY_MARGIN = float(os.getenv("SPEAKER_AMBIGUITY_MARGIN", "0.12"))
SORTFORMER_CHUNK_LEN = int(os.getenv("SORTFORMER_CHUNK_LEN", "340"))
SORTFORMER_CHUNK_RIGHT_CONTEXT = int(os.getenv("SORTFORMER_CHUNK_RIGHT_CONTEXT", "40"))
SORTFORMER_FIFO_LEN = int(os.getenv("SORTFORMER_FIFO_LEN", "40"))
SORTFORMER_SPKCACHE_UPDATE_PERIOD = int(os.getenv("SORTFORMER_SPKCACHE_UPDATE_PERIOD", "300"))
RATE = 24_000
SEPARATION_MODEL = os.getenv("VOICE_SEPARATION_MODEL", "HDEMUCS_HIGH_MUSDB")
SEPARATION_CHUNK_SECONDS = float(os.getenv("VOICE_SEPARATION_CHUNK_SECONDS", "30"))
SEPARATION_OVERLAP_SECONDS = float(os.getenv("VOICE_SEPARATION_OVERLAP_SECONDS", "0.5"))

DUBBING_STATE = STATE_DIR / "dubbing"
DUBBING_UPLOADS = DUBBING_STATE / "uploads"
DUBBING_WORK = DUBBING_STATE / "work"
DUBBING_JOBS = DUBBING_STATE / "jobs"
ENVIRONMENT_STATE = STATE_DIR / "environment"
ENVIRONMENT_UPLOADS = ENVIRONMENT_STATE / "uploads"
ENVIRONMENT_WORK = ENVIRONMENT_STATE / "work"
ENVIRONMENT_JOBS = ENVIRONMENT_STATE / "jobs"
SUBTITLE_STATE = STATE_DIR / "subtitles"
SUBTITLE_UPLOADS = SUBTITLE_STATE / "uploads"
SUBTITLE_WORK = SUBTITLE_STATE / "work"
SUBTITLE_JOBS = SUBTITLE_STATE / "jobs"
SUBTITLE_CHUNK_SECONDS = float(os.getenv("SUBTITLE_CHUNK_SECONDS", "240"))
VIDEO_EXPORT_CODEC = os.getenv("VOICE_EXPORT_VIDEO_CODEC", "libx264").strip() or "libx264"
VIDEO_EXPORT_PRESET = os.getenv("VOICE_EXPORT_VIDEO_PRESET", "veryfast").strip() or "veryfast"
VIDEO_EXPORT_CRF = os.getenv("VOICE_EXPORT_VIDEO_CRF", "18").strip() or "18"


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
        ENVIRONMENT_UPLOADS,
        ENVIRONMENT_WORK,
        ENVIRONMENT_JOBS,
        SUBTITLE_UPLOADS,
        SUBTITLE_WORK,
        SUBTITLE_JOBS,
        DEFAULT_OUTPUT,
    ):
        directory.mkdir(parents=True, exist_ok=True)
