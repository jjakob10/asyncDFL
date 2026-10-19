param(
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss")
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runner = Join-Path $repo "scripts\run_paper_experiments.py"
$plotter = Join-Path $repo "scripts\plot_metrics.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python was not found at $python. Create the repository's .venv and install requirements-dev.txt first."
}

Push-Location -LiteralPath $repo
try {
    Write-Host "Run ID: $RunId"

    foreach ($dataset in @("mnist", "cifar10")) {
        $manifest = Join-Path $PSScriptRoot "$dataset\manifest.json"

        # The runner builds, runs, waits for completion, and removes each stack.
        Write-Host "Running all six $dataset experiments..."
        & $python $runner run --all --manifest $manifest --run-id $RunId

        if ($LASTEXITCODE -ne 0) {
            throw "Experiments failed for $dataset (exit code $LASTEXITCODE)."
        }

        foreach ($topology in @("ring", "star")) {
            foreach ($mode in @("iid", "dirichlet", "manual")) {
                $experiment = "thiago-$dataset-$topology-6-$mode"
                $results = Join-Path $repo "results\thiago\$dataset\$experiment\$RunId"
                $charts = Join-Path $repo "outputs\thiago\$dataset\$experiment\$RunId"

                Write-Host "Creating charts for $experiment..."
                & $python $plotter --show-neighbor-count $results $charts

                if ($LASTEXITCODE -ne 0) {
                    throw "Chart generation failed for $experiment (exit code $LASTEXITCODE)."
                }
            }
        }
    }

    Write-Host "Completed all 12 experiments. Charts: $(Join-Path $repo 'outputs\thiago')"
}
finally {
    Pop-Location
}
