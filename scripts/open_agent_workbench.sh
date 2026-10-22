#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash scripts/start_agent_gateway.sh
exec python3 -m app.agent_cli service start
