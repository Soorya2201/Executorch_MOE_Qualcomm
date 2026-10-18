# Supervisor for the Phi-4-mini hybrid export: up to 3 attempts. After the first successful calibration the
# exporter saves (and verifies) calibration_encodings.pt, so every retry skips the 25-minute calibration.
# Started as its own process (not a Claude Code background job), so it is not stopped on low memory.
#   powershell -ExecutionPolicy Bypass -File C:\et\llm\run_hybrid_until_done.ps1
$artifact = 'C:\et\llm\phi4mini_w8a16_hybrid_2k'
$status = 'C:\et\llm\hybrid_status.txt'
function Say($m) { "$(Get-Date -Format 'HH:mm:ss')  $m" | Tee-Object -FilePath $status -Append }

"" | Out-File $status -Encoding utf8
for ($i = 1; $i -le 3; $i++) {
    $resume = Test-Path "$artifact\calibration_encodings.pt.verified"
    Say ("attempt $i started" + $(if ($resume) { " (resuming from the calibration checkpoint)" } else { " (full run)" }))
    & powershell -NoProfile -ExecutionPolicy Bypass -File C:\et\llm\export_phi4mini_hybrid.ps1 *> "C:\et\llm\hybrid_attempt$i.out"
    Copy-Item C:\et\llm\export_phi4mini_hybrid.log "C:\et\llm\export_phi4mini_hybrid_attempt$i.log" -Force
    if (Test-Path "$artifact\hybrid_llama_qnn.pte") {
        $mb = [math]::Round((Get-Item "$artifact\hybrid_llama_qnn.pte").Length / 1MB)
        Say "DONE: hybrid_llama_qnn.pte written ($mb MB) on attempt $i"
        exit 0
    }
    $last = Get-Content C:\et\llm\export_phi4mini_hybrid.log | Where-Object { $_ -match '^\[INFO' } | Select-Object -Last 1
    Say "attempt $i ended without a .pte. Last step: $last"
    Start-Sleep 20      # let Windows reclaim the memory before the next attempt
}
Say "FAILED after 3 attempts"
exit 1
