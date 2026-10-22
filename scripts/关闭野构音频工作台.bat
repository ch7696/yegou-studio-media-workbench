@echo off
setlocal
title Yegou Audio Workbench Stop
wsl.exe --distribution Ubuntu-26.04 --cd /home/administrator/voice-asset-workbench --exec python3 -m app.agent_cli service stop
if errorlevel 1 (
    echo [YEGOU] Workbench is busy or shutdown failed. See the JSON error above.
    pause
    exit /b 1
)
echo [YEGOU] Model services stopped. Agent gateway remains available.
timeout /t 3 >nul
