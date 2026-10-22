# Native Windows production entry; uses the existing ComfyUI environment.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = $env:YEGOU_PRODUCTION_RUNTIME
if (-not $runtimePath) { $runtimePath = Join-Path $repoRoot 'configs\production.local.json' }
if (-not (Test-Path -LiteralPath $runtimePath)) { throw 'Configure configs/production.local.json first; see docs/PRODUCTION_AUDIO.md' }
$runtime = Get-Content -LiteralPath $runtimePath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $runtime.python)) { throw 'Production runtime.python does not exist' }
$env:PYTHONUTF8 = '1'
$env:YEGOU_PRODUCTION_RUNTIME = $runtimePath
Push-Location $repoRoot
try { & $runtime.python -m app.production @args; $resultCode = $LASTEXITCODE }
finally { Pop-Location }
exit $resultCode
