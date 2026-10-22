#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
agent_state="${YEGOU_AGENT_STATE:-/home/administrator/ragflow-managed/state/agent-gateway}"
mkdir -p "$agent_state"
exec 9>"$agent_state/launcher.lock"
flock -x 9
if curl -fsS --max-time 2 http://127.0.0.1:7868/api/agent/health >/dev/null 2>&1; then
    exit 0
fi
if systemctl --user is-enabled yegou-agent.service >/dev/null 2>&1; then
    systemctl --user start yegou-agent.service
else
cd "$repo_root"
nohup python3 -m app.agent_service --idle-seconds "${YEGOU_AGENT_IDLE_SECONDS:-300}" >>"$agent_state/gateway.log" 2>&1 < /dev/null 9>&- &
fi
for ((attempt=0; attempt<30; attempt++)); do
    if curl -fsS --max-time 1 http://127.0.0.1:7868/api/agent/health >/dev/null 2>&1; then
        exit 0
    fi
    sleep 0.5
done
tail -n 30 "$agent_state/gateway.log" >&2
exit 1
