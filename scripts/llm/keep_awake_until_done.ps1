# Keep the laptop awake (system + display, so Modern Standby does not kick in on idle) until the hybrid export
# supervisor writes DONE or FAILED, or $MaxHours pass. Closing the lid still sleeps the laptop (by design).
param([int]$MaxHours = 3)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED (parsed from hex text: PS 5.1 reads 0x80000003 as negative)
[Win32.Power]::SetThreadExecutionState([Convert]::ToUInt32('80000003', 16)) | Out-Null
$end = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $end) {
    $st = Get-Content C:\et\llm\hybrid_status.txt -Encoding Unicode -ErrorAction SilentlyContinue
    if ($st -match 'DONE|FAILED after') { break }
    Start-Sleep 30
}
[Win32.Power]::SetThreadExecutionState([Convert]::ToUInt32('80000000', 16)) | Out-Null
