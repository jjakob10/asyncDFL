param(
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runner = Join-Path $repo "scripts\run_collection_experiments.py"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Create .venv and install requirements-dev.txt first."
}
Push-Location -LiteralPath $repo
try {
    $command = if ($Smoke) { "smoke" } else { "run" }
    & $python $runner $command --all --run-id $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Collection experiments failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
