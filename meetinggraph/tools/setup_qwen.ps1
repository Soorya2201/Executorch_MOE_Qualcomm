# Downloads the Qwen 2.5 3B model into Ollama (one time, ~2 GB). Usage:  .\tools\setup_qwen.ps1
$ErrorActionPreference = "Stop"
$model = "qwen2.5:3b-instruct"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed (or not on PATH)." -ForegroundColor Yellow
    Write-Host "1) Download and install it from https://ollama.com/download (Windows)."
    Write-Host "2) Close and reopen this terminal, then run this script again."
    exit 1
}

Write-Host "Downloading $model (about 2 GB, one time)..." -ForegroundColor Cyan
ollama pull $model
if ($LASTEXITCODE -ne 0) { Write-Host "Download failed. Is the Ollama app running?" -ForegroundColor Red; exit 1 }

Write-Host "Quick test..." -ForegroundColor Cyan
ollama run $model "Reply with just the word: ready"
Write-Host ""
Write-Host "Done. Start MeetingGraph with:  python -m app.main" -ForegroundColor Green
Write-Host "The status bar at the bottom should say: AI model: $model via Ollama"
