@echo off
setlocal EnableExtensions
title Yegou Studio - Workbench (Audio 7866 + Video 7877)

set "AUDIO_URL=http://127.0.0.1:7866"
set "VIDEO_URL=http://127.0.0.1:7877"
set "AUDIO_LOG=/tmp/yegou-voice-asset-workbench.log"
set "VIDEO_LOG=/tmp/yegou-media-analysis-webui.log"
set "WSL_DISTRO=Ubuntu-26.04"
set "WSL_START=/home/administrator/voice-asset-workbench/scripts/start_all.sh"
set "LAUNCH_LOG=%TEMP%\yegou-workbench-wsl-launch.log"

echo.
echo [YEGOU] Yegou Studio workbench: audio + video analysis
echo [YEGOU] Audio workbench: %AUDIO_URL%
echo [YEGOU] Video analysis: %VIDEO_URL%
call :check_audio
if not errorlevel 1 call :check_video
if not errorlevel 1 goto :open

echo [YEGOU] Starting local Ubuntu Docker engine...
call :ensure_docker
if errorlevel 1 goto :docker_failed

echo [YEGOU] Starting services...
if exist "%LAUNCH_LOG%" del /q "%LAUNCH_LOG%" >nul 2>&1
wsl.exe --distribution %WSL_DISTRO% --exec /bin/bash "%WSL_START%" >> "%LAUNCH_LOG%" 2>&1
if errorlevel 1 goto :wsl_failed

for /l %%I in (1,1,60) do (
    call :check_audio
    if not errorlevel 1 call :check_video
    if not errorlevel 1 goto :open
    timeout /t 1 /nobreak >nul
)

echo.
echo [YEGOU] Services did not become ready in 60 seconds.
goto :show_log

:open
start "" "%AUDIO_URL%/"
start "" "%VIDEO_URL%/"
echo.
echo [YEGOU] Audio workbench is ready: %AUDIO_URL%
echo [YEGOU] Video analysis is ready: %VIDEO_URL%
echo [YEGOU] Press any key to close this window. The services will keep running.
pause >nul
endlocal
exit /b 0

:wsl_failed
echo.
echo [YEGOU] WSL launch failed. Launcher output:
if exist "%LAUNCH_LOG%" type "%LAUNCH_LOG%"
goto :show_log

:docker_failed
echo.
echo [YEGOU] Local Ubuntu Docker engine is not ready.
echo [YEGOU] Docker diagnostics:
wsl.exe --distribution %WSL_DISTRO% --user root --exec /bin/systemctl status containerd docker --no-pager
wsl.exe --distribution %WSL_DISTRO% --exec /bin/bash -c "docker info 2>&1 | tail -5; docker image ls 2>&1 | head -10"
echo.
echo [YEGOU] Do not run Docker Desktop Clean/Purge/Reset; the images are stored in the Ubuntu WSL disk on G:.
pause
endlocal
exit /b 1

:show_log
echo.
echo [YEGOU] Audio workbench log:
wsl.exe --distribution %WSL_DISTRO% --exec /bin/bash -c "tail -n 60 %AUDIO_LOG%"
echo.
echo [YEGOU] Video analysis log:
wsl.exe --distribution %WSL_DISTRO% --exec /bin/bash -c "tail -n 60 %VIDEO_LOG%"
pause
endlocal
exit /b 1

:ensure_docker
echo [YEGOU] WSL distro: %WSL_DISTRO%
wsl.exe --distribution %WSL_DISTRO% --user root --exec /bin/systemctl start containerd > "%LAUNCH_LOG%" 2>&1
if errorlevel 1 exit /b 1
wsl.exe --distribution %WSL_DISTRO% --user root --exec /bin/systemctl start docker >> "%LAUNCH_LOG%" 2>&1
if errorlevel 1 exit /b 1
wsl.exe --distribution %WSL_DISTRO% --exec /bin/bash -c "docker info >/dev/null 2>&1"
if errorlevel 1 exit /b 1
exit /b 0

:check_audio
curl.exe -fsS --max-time 1 "%AUDIO_URL%/api/health" >nul 2>&1
exit /b %errorlevel%

:check_video
curl.exe -fsS --max-time 1 "%VIDEO_URL%/api/queue" >nul 2>&1
exit /b %errorlevel%
