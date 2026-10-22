#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_dir="$HOME/.config/systemd/user"
mkdir -p "$unit_dir"
cat >"$unit_dir/yegou-agent.service" <<EOF
[Unit]
Description=Yegou lightweight audio agent gateway

[Service]
Type=simple
WorkingDirectory=$repo_root
ExecStart=/usr/bin/python3 -m app.agent_service
EnvironmentFile=-$HOME/.config/yegou-agent.env
Restart=on-failure
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable yegou-agent.service
printf '%s\n' 'Installed. Stop any manually started gateway before systemctl --user start yegou-agent.service.'
