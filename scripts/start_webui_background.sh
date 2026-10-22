#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_file="${YEGOU_WEBUI_LOG:-/tmp/yegou-voice-asset-workbench.log}"

if ! touch "$log_file" 2>/dev/null; then
    printf '[YEGOU] cannot create log file: %s\n' "$log_file" >&2
    exit 1
fi

printf '[YEGOU] launcher started: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"

if ! cd "$repo_root"; then
    printf '[YEGOU] cannot cd to %s\n' "$repo_root" >> "$log_file"
    exit 1
fi

nohup bash ./scripts/open_agent_workbench.sh >>"$log_file" 2>&1 < /dev/null &
printf '[YEGOU] WebUI process started: %s\n' "$!" >> "$log_file"
exit 0
