# Starts MeetingGraph from this folder. Usage:  .\start.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found. Create it first (see README step 2)." -ForegroundColor Yellow
    exit 1
}

& $py tools\check_env.py --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment check failed; see messages above." -ForegroundColor Red
    exit 1
}
& $py -m app.main
