#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_MODEL_NAME="${ASR_MODEL_NAME:-Qwen3-ASR-1.7B}"
ASR_HOST_DIR="${ASR_MODEL_DIR:-/home/administrator/models/${ASR_MODEL_NAME}}"
ASR_CONTAINER_DIR="/models/${ASR_MODEL_NAME}"
ALIGNER_DIR="/home/administrator/models/Qwen3-ForcedAligner-0.6B"
IMAGE="ragflow-qwen-asr:0.0.6"
CONTAINER="ragflow-model-asr-api"
PORT="${ASR_API_PORT:-18763}"

if [[ ! -d "$ASR_HOST_DIR" ]]; then
  echo "找不到 ASR 模型目录：$ASR_HOST_DIR"
  echo "可用 ASR_MODEL_NAME=Qwen3-ASR-0.6B 回退到旧模型。"
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "找不到独立 ASR 镜像：$IMAGE"
  exit 1
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --gpus all \
  --ipc=host \
  --shm-size=4g \
  -p "$PORT:7865" \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e ASR_MODEL_DIR="$ASR_CONTAINER_DIR" \
  -e ASR_MODEL_NAME="$ASR_MODEL_NAME" \
  -e FORCED_ALIGNER_DIR=/models/Qwen3-ForcedAligner-0.6B \
  -v "$PROJECT_DIR:/app:ro" \
  -v "$ASR_HOST_DIR:$ASR_CONTAINER_DIR:ro" \
  -v "$ALIGNER_DIR:/models/Qwen3-ForcedAligner-0.6B:ro" \
  -w /app \
  --entrypoint python3 \
  "$IMAGE" \
  -m uvicorn app.asr_server:app --host 0.0.0.0 --port 7865

echo "${ASR_MODEL_NAME} 服务已启动：http://127.0.0.1:${PORT}"
