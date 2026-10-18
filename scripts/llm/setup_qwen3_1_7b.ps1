# Sets up the Qwen3-1.7B (8k context, hybrid) NPU model that MeetingGraph uses.
#
# The .pte was compiled with QAIRT 2.46, so it needs the 2.46 runtime DLLs ("Using newer context binary on old
# SDK" otherwise), and our runner built with patches 0001-0007 (0007: 3-D attention masks).
#
#   1. Put sc8380_qwen3-1_7b_ctxt-8000.pte, tokenizer.json, tokenizer_config.json in $ModelDir.
#   2. Build qnn_llama_runner (ARM64) with patches 0001-0007 applied.
#   3. .\setup_qwen3_1_7b.ps1
param(
    [string]$ModelDir = "C:\et\llm\qwen3_1_7b_ctx8k",
    [string]$RunDir = "C:\et\llm\npu-run-246",
    [string]$Runner = "C:\et\executorch\build-arm64-windows\examples\qualcomm\oss_scripts\llama\Release\qnn_llama_runner.exe",
    [string]$BackendDll = "C:\et\llm\npu-run\qnn_executorch_backend.dll",
    [string]$QairtVersion = "2.46.0.260424",
    [string]$Download = "C:\et\incoming\qairt246"
)
$ErrorActionPreference = "Stop"

$zip = Join-Path $Download "v$QairtVersion.zip"
if (-not (Test-Path $zip)) {
    New-Item -ItemType Directory -Force $Download | Out-Null
    $url = "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QairtVersion/v$QairtVersion.zip"
    Write-Host "Downloading QAIRT $QairtVersion (~1.8 GB)"
    curl.exe -sL -o $zip $url
}
$lib = Join-Path $Download "qairt\$QairtVersion\lib"
if (-not (Test-Path "$lib\aarch64-windows-msvc\QnnHtp.dll")) {
    tar.exe -xf $zip -C $Download "qairt/$QairtVersion/lib/aarch64-windows-msvc" "qairt/$QairtVersion/lib/hexagon-v73"
}

New-Item -ItemType Directory -Force $RunDir | Out-Null
Copy-Item "$lib\aarch64-windows-msvc\QnnHtp.dll", "$lib\aarch64-windows-msvc\QnnHtpV73Stub.dll",
          "$lib\aarch64-windows-msvc\QnnSystem.dll", "$lib\hexagon-v73\unsigned\libQnnHtpV73Skel.so",
          "$lib\hexagon-v73\unsigned\libQnnHtpV73.so", "$lib\hexagon-v73\unsigned\libqnnhtpv73.cat" $RunDir
Copy-Item $BackendDll $RunDir
Copy-Item $Runner (Join-Path $RunDir "qnn_llama_runner.exe")
Copy-Item $Runner (Join-Path $RunDir "qnn_llama_runner_server.exe")   # the same binary has --server (patch 0005)

Write-Host "Smoke test on the NPU..."
$out = Join-Path $env:TEMP "qwen17_smoke.txt"
& (Join-Path $RunDir "qnn_llama_runner.exe") --model_path (Join-Path $ModelDir "sc8380_qwen3-1_7b_ctxt-8000.pte") `
    --tokenizer_path (Join-Path $ModelDir "tokenizer.json") --decoder_model_version qwen3 --eval_mode 1 `
    --seq_len 200 --temperature 0 --prompt "Say hello in five words. /no_think" --output_path $out 2>$null | Out-Null
Get-Content $out
