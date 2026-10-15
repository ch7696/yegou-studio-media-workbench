#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
# 合并进 voice-asset-workbench 后：产物与运行状态固定落在本仓（Python 侧不读 .env，只能由 shell 传入）。
export MEDIA_OUTPUT_ROOT="${MEDIA_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export WEBUI_STATE="${WEBUI_STATE:-$REPO_ROOT/state}"
exec python3 app/media_analysis_webui.py --host "${WEBUI_HOST:-0.0.0.0}" --port "${WEBUI_PORT:-7877}"
