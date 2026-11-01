param(
    [ValidateSet("smoke", "pilot", "run")][string]$Mode = "smoke",
    [ValidateSet("mnist", "cifar10", "all")][string]$Dataset = "all",
    [string]$Scenario = "concentrated-r",
    [switch]$All,
    [int[]]$Seeds,
    [ValidateSet("M0", "M1", "M2", "M3", "M4")][string[]]$Methods = @("M0"),
    [string]$FrozenBundle,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runner = Join-Path $repo "scripts\run_johann_experiments.py"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Crie .venv e instale requirements-dev.txt antes de executar."
}
$arguments = @($runner, $Mode, "--dataset", $Dataset, "--run-id", $RunId)
$arguments += "--methods"; $arguments += $Methods
if ($FrozenBundle) { $arguments += @("--frozen-bundle", $FrozenBundle) }
if ($All) { $arguments += "--all" } else { $arguments += @("--scenario", $Scenario) }
if ($Seeds) { $arguments += "--seeds"; $arguments += $Seeds }
Push-Location -LiteralPath $repo
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Experimento falhou (código $LASTEXITCODE). Consulte results/johann." }
} finally {
    Pop-Location
}
