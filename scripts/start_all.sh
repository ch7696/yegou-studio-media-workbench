#!/usr/bin/env bash
# 野构工作台统一启动：音色台（7866）+ 视频分析台（7877）。
# 已在运行的服务会被跳过，不会重复重建容器。
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[YEGOU] 检查 Docker 引擎…"
if ! docker info >/dev/null 2>&1; then
  echo "[YEGOU] Docker 未运行。请先执行：wsl.exe -d Ubuntu-26.04 -u root systemctl start containerd docker" >&2
  exit 1
fi

echo "[YEGOU] 检查镜像…"
for image in "ragflow-cosyvoice-speaker:0.0.4" "ragflow-qwen-asr:0.0.6"; do
  if ! docker image inspect "$image" >/dev/null 2>&1; then
    echo "[YEGOU] 缺少镜像：$image" >&2
    exit 1
  fi
done
if ! docker image inspect "swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.26.0" >/dev/null 2>&1; then
  echo "[YEGOU] 警告：缺少视觉模型镜像，视频台的视觉任务会失败（不影响音色台）。" >&2
fi

if [[ ! -d /mnt/f ]]; then
  echo "[YEGOU] 提示：/mnt/f 未挂载，音色台默认输出目录不可用（服务仍会启动）。" >&2
fi

if curl -fsS --max-time 2 http://127.0.0.1:7866/api/health >/dev/null 2>&1; then
  echo "[YEGOU] 音色台已在运行（7866）。"
  bash "$REPO_ROOT/scripts/open_agent_workbench.sh"
else
  echo "[YEGOU] 启动音色台（7866）…"
  bash "$REPO_ROOT/scripts/start_webui_background.sh"
fi

if curl -fsS --max-time 2 http://127.0.0.1:7877/api/queue >/dev/null 2>&1; then
  echo "[YEGOU] 视频分析台已在运行（7877）。"
else
  echo "[YEGOU] 启动视频分析台（7877）…"
  bash "$REPO_ROOT/video/scripts/start_webui_background.sh"
fi

echo "[YEGOU] 音色台：http://127.0.0.1:7866  （日志 /tmp/yegou-voice-asset-workbench.log）"
echo "[YEGOU] 视频台：http://127.0.0.1:7877  （日志 /tmp/yegou-media-analysis-webui.log）"
echo "[YEGOU] 视觉模型按任务需要启动；两个台各有「模型」页，可切换启动的主模型。"
