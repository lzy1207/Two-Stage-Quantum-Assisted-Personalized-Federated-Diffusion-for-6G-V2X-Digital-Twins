$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m qv2x demo --config configs/demo.json --output runs/demo
    if ($LASTEXITCODE -ne 0) { throw 'Demo failed; see the error above.' }
} finally { Pop-Location }
