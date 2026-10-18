# Run Qwen3-0.6B on the Snapdragon X Elite Hexagon NPU with ExecuTorch (qnn_llama_runner, ARM64).
# Usage (normal PowerShell window, no admin needed):
#   powershell -ExecutionPolicy Bypass -File C:\et\llm\run_qwen3_npu.ps1
#   powershell -ExecutionPolicy Bypass -File C:\et\llm\run_qwen3_npu.ps1 -Prompt "Explain NPUs in 3 sentences." -SeqLen 300
param(
    [string]$Prompt = "What is the capital of France? Answer in one sentence.",
    [int]$SeqLen = 256  # prompt + generated tokens; must be <= the export's --max_seq_len (512)
)
$ErrorActionPreference = 'Stop'
$export = 'C:\et\llm\qwen3_0_6b'
$run = 'C:\et\llm\npu-run'
$pte = Join-Path $export 'kv_llama_qnn.pte'

# 1. Wait for the export (compile step) to produce the .pte
while (-not (Test-Path $pte)) {
    Write-Host ("[{0}] waiting for export to finish: {1} not there yet..." -f (Get-Date -Format HH:mm:ss), $pte)
    Start-Sleep -Seconds 30
}
# The file appears before it's fully written; wait until its size stops changing.
do { $s1 = (Get-Item $pte).Length; Start-Sleep -Seconds 5; $s2 = (Get-Item $pte).Length } while ($s1 -ne $s2)
Write-Host ("model ready: {0} ({1:N1} MB)" -f $pte, ($s2 / 1MB)) -ForegroundColor Green

# 2. Put the model and tokenizer next to the ARM64 runner and the QNN ARM64 libraries
Copy-Item $pte, (Join-Path $export 'tokenizer.json') $run -Force

# 3. Run on the NPU. Watch Task Manager > Performance > NPU while this runs.
Set-Location $run
Write-Host "`nRunning on the NPU. Prompt: $Prompt`n" -ForegroundColor Cyan
& .\qnn_llama_runner.exe `
    --model_path kv_llama_qnn.pte `
    --tokenizer_path tokenizer.json `
    --decoder_model_version qwen3 `
    --eval_mode 0 `
    --seq_len $SeqLen `
    --temperature 0 `
    --prompt "$Prompt" `
    --output_path outputs.txt `
    --performance_output_path inference_speed.txt
Write-Host "`nrunner exit code: $LASTEXITCODE"

Write-Host "`n--- generated text (outputs.txt) ---" -ForegroundColor Cyan
Get-Content outputs.txt -ErrorAction SilentlyContinue
Write-Host "`n--- inference_speed.txt ---" -ForegroundColor Cyan
Get-Content inference_speed.txt -ErrorAction SilentlyContinue
