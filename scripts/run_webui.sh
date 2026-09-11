#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANAGED_DIR="/home/administrator/ragflow-managed"
STATE_DIR="$MANAGED_DIR/state/speaker-webui"
MODEL_DIR="/home/administrator/models/Fun-CosyVoice3-0.5B"
ASR_MODEL_NAME="${ASR_MODEL_NAME:-Qwen3-ASR-1.7B}"
ASR_DIR="${ASR_DIR:-/home/administrator/models/${ASR_MODEL_NAME}}"
ASR_CONTAINER_DIR="/models/${ASR_MODEL_NAME}"
ALIGNER_DIR="/home/administrator/models/Qwen3-ForcedAligner-0.6B"
DIAR_DIR="/home/administrator/3D-Speaker"
COSY_DIR="/home/administrator/cosyvoice/CosyVoice"
CACHE_DIR="$MANAGED_DIR/state/cosyvoice-cache"
IMAGE="ragflow-cosyvoice-speaker:0.0.2"
CONTAINER="ragflow-model-speaker-webui"
PORT="${SPEAKER_WEBUI_PORT:-7866}"
ASR_API_PORT="${ASR_API_PORT:-18763}"

mkdir -p "$STATE_DIR" "$CACHE_DIR"

ASR_MODEL_NAME="$ASR_MODEL_NAME" ASR_MODEL_DIR="$ASR_DIR" bash "$PROJECT_DIR/scripts/run_asr_api.sh"

if [[ "${BUILD_IMAGE:-0}" == "1" ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$PROJECT_DIR/Dockerfile" "$PROJECT_DIR"
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --shm-size=4g \
  -p "$PORT:7865" \
  --add-host=host.docker.internal:host-gateway \
  -w /app \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e ASR_SERVICE_URL="http://host.docker.internal:${ASR_API_PORT}" \
  -e VOICE_STATE_DIR=/state \
  -e VOICE_DEFAULT_OUTPUT=/mnt/f/voice \
  -e COSYVOICE_MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
  -e QWEN_ASR_MODEL_DIR="$ASR_CONTAINER_DIR" \
  -e QWEN_FORCED_ALIGNER_DIR=/models/Qwen3-ForcedAligner-0.6B \
  -e THREE_D_SPEAKER_REPO=/opt/3D-Speaker \
  -e COSYVOICE_REPO=/opt/CosyVoice \
  -v "$PROJECT_DIR:/app:ro" \
  -v "$STATE_DIR:/state" \
  -v "$CACHE_DIR:/root/.cache" \
  -v "$MODEL_DIR:/models/Fun-CosyVoice3-0.5B:ro" \
  -v "$ASR_DIR:$ASR_CONTAINER_DIR:ro" \
  -v "$ALIGNER_DIR:/models/Qwen3-ForcedAligner-0.6B:ro" \
  -v "$DIAR_DIR:/opt/3D-Speaker:ro" \
  -v "$COSY_DIR:/opt/CosyVoice:ro" \
  -v /mnt:/mnt \
  --entrypoint python3 "$IMAGE" \
  -m app.main --host 0.0.0.0 --port 7865
