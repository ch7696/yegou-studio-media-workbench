# Windows entry point. Arguments are passed as an array, never evaluated as shell code.
$ErrorActionPreference = 'Stop'
$repo = '/home/administrator/voice-asset-workbench'
& wsl.exe --distribution Ubuntu-26.04 --cd $repo --exec python3 -m app.agent_cli @args
exit $LASTEXITCODE
