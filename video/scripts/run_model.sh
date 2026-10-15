#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

REGISTRY="$REPO_ROOT/config/models.json"
CONTAINER_NAME="${MINICPM_CONTAINER:-vision-minicpm}"
PORT_VALUE="${MINICPM_PORT:-8002}"
MODEL_ID="${MODEL_ID:-}"
IMAGE_VALUE="${MINICPM_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.26.0}"
HEALTH_URL_VALUE="${MINICPM_HEALTH_URL:-http://127.0.0.1:${PORT_VALUE}/health}"
STARTUP_TIMEOUT_VALUE="${MINICPM_STARTUP_TIMEOUT:-600}"

# ---- 读取注册表：输出 模型id/显示名/权重目录/服务名/max-len/显存比/最大图数/额外参数 ----
# 退出码：0=成功；3=注册表缺失或条目不存在（回退内置配置）；4=条目存在但权重未下载（直接报错）
set +e
REGISTRY_FIELDS="$(python3 - "$REGISTRY" "$MODEL_ID" <<'PY'
import json, os, sys

registry_path, requested = sys.argv[1], sys.argv[2]
try:
    with open(registry_path, encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    sys.exit(3)

models = data.get("models") or []
model_id = requested or str(data.get("default") or "")
entry = next((m for m in models if m.get("id") == model_id), None)
if entry is None:
    sys.exit(3)

link = str(entry.get("model_link") or "")
model_path = ""
if link:
    candidate = os.path.normpath(os.path.join(registry_path, "..", "..", link))
    if os.path.isdir(candidate):
        model_path = os.path.realpath(candidate)
    else:
        root = os.environ.get("MODEL_ROOT") or os.path.normpath(
            os.path.join(registry_path, "..", "..", "models")
        )
        model_path = os.path.join(root, os.path.basename(link.rstrip("/")))
if not os.path.isdir(model_path):
    print("该模型尚未下载：%s" % (model_path or link), file=sys.stderr)
    sys.exit(4)

args = entry.get("vllm_args") or {}
print(entry["id"])
print(entry.get("display_name") or entry["id"])
print(model_path)
print(entry.get("served_name") or entry["id"])
print(args.get("max_model_len", 32768))
print(args.get("gpu_memory_utilization", 0.90))
print(args.get("max_images", 8))
print(" ".join(entry.get("extra_args") or []))
PY
)"
REGISTRY_STATUS=$?
set -e

if [[ $REGISTRY_STATUS -eq 4 ]]; then
  echo "所选视觉主模型尚未下载。先运行：$REPO_ROOT/scripts/fetch_model.sh --id <模型id>" >&2
  exit 1
fi

if [[ $REGISTRY_STATUS -ne 0 ]]; then
  echo "注册表不可用，回退到内置 MiniCPM 配置。" >&2
  MODEL_ID="minicpm-v-4_5-gptq"
  DISPLAY_NAME="${MINICPM_MODEL:-MiniCPM-V-4_5-GPTQ}"
  MODEL_PATH="$REPO_ROOT/models/MiniCPM-V-4_5-GPTQ"
  SERVED_NAME="${MINICPM_MODEL:-MiniCPM-V-4_5-GPTQ}"
  REG_MAX_LEN="32768"
  REG_GPU_MEM="0.90"
  REG_MAX_IMAGES="8"
  EXTRA_ARGS="--trust-remote-code"
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "模型目录不存在：$MODEL_PATH，请先运行 scripts/link_models.sh" >&2
    exit 1
  fi
else
  mapfile -t FIELDS <<< "$REGISTRY_FIELDS"
  MODEL_ID="${FIELDS[0]}"
  DISPLAY_NAME="${FIELDS[1]}"
  MODEL_PATH="${FIELDS[2]}"
  SERVED_NAME="${FIELDS[3]}"
  REG_MAX_LEN="${FIELDS[4]}"
  REG_GPU_MEM="${FIELDS[5]}"
  REG_MAX_IMAGES="${FIELDS[6]}"
  EXTRA_ARGS="${FIELDS[7]:-}"
fi

# 显式环境变量 > 注册表 > 内置默认
MAX_MODEL_LEN_VALUE="${MINICPM_MAX_MODEL_LEN:-$REG_MAX_LEN}"
GPU_MEMORY_VALUE="${MINICPM_GPU_MEMORY_UTILIZATION:-$REG_GPU_MEM}"
MAX_IMAGES_VALUE="${MINICPM_MAX_IMAGES:-$REG_MAX_IMAGES}"
MODEL_NAME_VALUE="${MINICPM_MODEL:-$SERVED_NAME}"

wait_for_api() {
  # 用钟表时间约束总等待，避免 curl 超时把每一轮拖长后总时长失控。
  local deadline=$(( SECONDS + STARTUP_TIMEOUT_VALUE ))
  echo "等待视觉模型 API 就绪：$HEALTH_URL_VALUE（$DISPLAY_NAME）"
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 3 "$HEALTH_URL_VALUE" >/dev/null 2>&1; then
      echo "视觉模型 API 已就绪：$HEALTH_URL_VALUE"
      return 0
    fi
    sleep 2
  done
  echo "等待视觉模型 API 超时（${STARTUP_TIMEOUT_VALUE}s）：$HEALTH_URL_VALUE" >&2
  docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
  return 1
}

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  CURRENT_MODEL_ID="$(docker inspect -f '{{index .Config.Labels "yegou.model.id"}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  if [[ -z "$CURRENT_MODEL_ID" || "$CURRENT_MODEL_ID" == "<no value>" ]]; then
    CURRENT_MODEL_ID=""
  fi
  if [[ "$CURRENT_MODEL_ID" == "$MODEL_ID" ]]; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
      wait_for_api
      echo "$DISPLAY_NAME 已运行：$CONTAINER_NAME"
      exit 0
    fi
    docker start "$CONTAINER_NAME" >/dev/null
    echo "已启动已有容器：$CONTAINER_NAME（$DISPLAY_NAME）"
    wait_for_api
    exit 0
  fi
  echo "检测到视觉主模型变化（${CURRENT_MODEL_ID:-<无标签>} -> $MODEL_ID），重建容器…"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --label "yegou.model.id=$MODEL_ID" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p "$PORT_VALUE:8000" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -v "$MODEL_PATH:/models/minicpm:ro" \
  "$IMAGE_VALUE" \
  --model /models/minicpm \
  --served-model-name "$MODEL_NAME_VALUE" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN_VALUE" \
  --gpu-memory-utilization "$GPU_MEMORY_VALUE" \
  --max-num-seqs 1 \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_VALUE}}" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key x \
  ${EXTRA_ARGS}

wait_for_api
echo "视觉模型容器已启动：$CONTAINER_NAME（$DISPLAY_NAME），API 端口：$PORT_VALUE"
