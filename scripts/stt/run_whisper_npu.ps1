# Speech-to-text on the Snapdragon X Elite Hexagon NPU: Whisper via ExecuTorch (qnn_whisper_runner, ARM64).
# Usage (normal PowerShell window):
#   powershell -ExecutionPolicy Bypass -File C:\et\stt\run_whisper_npu.ps1                       # 10 LibriSpeech test clips + WER
#   powershell -ExecutionPolicy Bypass -File C:\et\stt\run_whisper_npu.ps1 -Wav C:\path\talk.wav # your own audio (<= 30 s)
#   ... -Model small   (after the whisper-small export exists)
param(
    [ValidateSet('tiny', 'small')] [string]$Model = 'tiny',
    [string[]]$Wav = @(),
    [int]$LibriSpeech = 10
)
$ErrorActionPreference = 'Stop'
$export = "C:\et\stt\whisper_$Model"
$run = 'C:\et\stt\npu-run'
$pte = Join-Path $export 'whisper_qnn_16a8w.pte'
if (-not (Test-Path $pte)) { throw "No exported model at $pte yet." }

# 1. Audio -> log-mel features (CPU, x64 Python); tokenizer.json for the runner.
. C:\et\et-env.ps1
$env:HF_HUB_DISABLE_XET = '1'
$prepArgs = @('C:\et\scripts\whisper_npu_io.py', 'prep', '--run_dir', $run, '--hf_model', "openai/whisper-$Model")
if ($Wav.Count -gt 0) { $prepArgs += '--wav'; $prepArgs += $Wav } else { $prepArgs += @('--librispeech', $LibriSpeech) }
python @prepArgs
if ($LASTEXITCODE -ne 0) { throw "input preparation failed" }

# 2. Encoder + decoder on the NPU. Watch Task Manager > Performance > NPU.
Copy-Item $pte $run -Force
Set-Location $run
Get-ChildItem .\outputs -Filter 'output_*.txt' -ErrorAction SilentlyContinue | Remove-Item
$sw = [Diagnostics.Stopwatch]::StartNew()
& .\qnn_whisper_runner.exe --model_path whisper_qnn_16a8w.pte --tokenizer_json_path tokenizer.json `
    --input_list_path input_list.txt --seq_len 128 --output_folder_path outputs
$code = $LASTEXITCODE
Write-Host ("`nrunner exit code: {0}   wall time {1:N2} s" -f $code, $sw.Elapsed.TotalSeconds)

# 3. Show transcripts (+ WER when references exist).
python C:\et\scripts\whisper_npu_io.py score --run_dir $run --hf_model "openai/whisper-$Model"
