# Export Phi-4-mini-instruct for the Snapdragon X Elite NPU: W8A16 (16a8w), kv mode, 2048-token context.
# Run AFTER rebooting (the 16-64 GB page file must be active). Plug in the charger. Takes hours.
#   powershell -ExecutionPolicy Bypass -File C:\et\llm\export_phi4mini_w8a16.ps1
$ErrorActionPreference = 'Stop'
$artifact = 'C:\et\llm\phi4mini_w8a16_2k'
$log = 'C:\et\llm\export_phi4mini_w8a16.log'

# 1. Preflight: refuse to start without enough commit memory (would crash hours in).
$os = Get-CimInstance Win32_OperatingSystem
$commitGB = $os.TotalVirtualMemorySize / 1MB
$pf = (Get-CimInstance Win32_PageFileUsage | Measure-Object AllocatedBaseSize -Sum).Sum
Write-Host ("RAM {0:N1} GB, page file {1:N0} MB, commit limit {2:N1} GB" -f ($os.TotalVisibleMemorySize / 1MB), $pf, $commitGB)
if ($pf -lt 16000) {
    throw "Page file is only $pf MB. Reboot first so the 16-64 GB page file takes effect, then rerun this script."
}

# 2. Keep the laptop awake for the whole export (released when this script exits).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED. Parsed from hex text: PS 5.1 reads 0x80000001 as a negative Int32.
[Win32.Power]::SetThreadExecutionState([Convert]::ToUInt32('80000001', 16)) | Out-Null

# 3. Environment: x64 Python, x64 QNN DLLs, UTF-8, offline (weights are already in the HF cache).
. C:\et\et-env.ps1
Remove-Item Env:CMAKE_GENERATOR, Env:CMAKE_GENERATOR_PLATFORM -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
New-Item -ItemType Directory -Force $artifact | Out-Null

Write-Host "Exporting Phi-4-mini W8A16 -> $artifact"
Write-Host "Log: $log   (follow it with: Get-Content $log -Wait -Tail 5)"
$sw = [Diagnostics.Stopwatch]::StartNew()
cmd /c "python examples\qualcomm\oss_scripts\llama\llama.py --build_folder build-x86_64-windows --soc_model SC8380XP --decoder_model phi_4_mini-w8a16 --model_mode kv --max_seq_len 2048 --temperature 0 --prompt ""What is the capital of France?"" --calib_samples C:\et\llm\calib_text_8.json --compile_only --artifact $artifact > $log 2>&1"
$code = $LASTEXITCODE
Write-Host ("export exit code {0} after {1:N1} h" -f $code, $sw.Elapsed.TotalHours)
[Win32.Power]::SetThreadExecutionState([Convert]::ToUInt32('80000000', 16)) | Out-Null  # back to normal sleep behaviour

# 4. Report.
Get-ChildItem $artifact -File | Select-Object Name, @{n = 'MB'; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
if ($code -ne 0 -or -not (Test-Path "$artifact\kv_llama_qnn.pte")) {
    Write-Host "`n--- last log lines ---"
    Get-Content $log | Where-Object { $_ -notmatch '\d+%\|' } | Select-Object -Last 25
    throw "export failed"
}
Write-Host "`nDONE: $artifact\kv_llama_qnn.pte" -ForegroundColor Green
