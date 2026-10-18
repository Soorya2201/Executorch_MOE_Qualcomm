# Log RAM / commit usage every 15 s while an export runs (stops when the given process exits).
#   powershell -File C:\et\llm\memwatch.ps1 -Log C:\et\llm\export_hybrid_mem.csv -Seconds 15
param([string]$Log = 'C:\et\llm\export_hybrid_mem.csv', [int]$Seconds = 15, [int]$MaxHours = 4)
"time,free_ram_gb,commit_used_gb,commit_limit_gb,top_python_commit_gb" | Out-File -Encoding utf8 $Log
$end = (Get-Date).AddHours($MaxHours)
while ((Get-Date) -lt $end) {
    $os = Get-CimInstance Win32_OperatingSystem
    $lim = $os.TotalVirtualMemorySize / 1MB
    $used = $lim - $os.FreeVirtualMemory / 1MB
    $py = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object PagedMemorySize64 -Maximum).Maximum / 1GB
    "{0:HH:mm:ss},{1:N1},{2:N1},{3:N1},{4:N1}" -f (Get-Date), ($os.FreePhysicalMemory / 1MB), $used, $lim, $py |
        Out-File -Encoding utf8 -Append $Log
    Start-Sleep $Seconds
}
