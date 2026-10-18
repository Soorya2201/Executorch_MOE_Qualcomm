# NPU runner test matrix for C:\et\executorch\xelite-run (DeepLab v3, SC8380XP).
# Each case runs qnn_executor_runner.exe and checks exit code, log content and crash events.
$ErrorActionPreference = 'Continue'
$bundle = 'C:\et\executorch\xelite-run'
$work = 'C:\et\tests'
New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $bundle
$results = @()

function Invoke-Case($name, $runnerArgs, $expect, [scriptblock]$check, $cwd = $bundle, $exe = "$bundle\qnn_executor_runner.exe") {
    $log = "$work\$($name -replace '\W','_').log"
    $start = Get-Date
    Push-Location $cwd
    cmd /c "`"$exe`" $runnerArgs > `"$log`" 2>&1"
    $code = $LASTEXITCODE
    Pop-Location
    $crash = @(Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000; StartTime=$start} -ErrorAction SilentlyContinue |
        Where-Object { $_.Message -match 'qnn_executor_runner' }).Count
    $text = Get-Content $log -Raw
    $ok, $detail = & $check $code $text
    $script:results += [pscustomobject]@{ Case=$name; Expect=$expect; Exit=$code; Crash=$crash; Pass=($ok -and $crash -eq 0); Detail=$detail }
}

$m = '--model_path .\dlv3_qnn.pte'

Invoke-Case 'real inputs, 1 iter' "$m --input_list_path .\input_list.txt --output_folder_path .\outputs" 'exit 0, 20 outputs' {
    param($c,$t) $n = ([regex]::Matches($t,'inference took')).Count - 1; ($c -eq 0 -and $n -eq 20), "$n inferences" }

Invoke-Case 'benchmark sustained' "$m --input_list_path .\input_list.txt --output_folder_path .\outputs --warm_up 10 --iteration 100 --htp_performance_mode 1" 'exit 0, ~6.7 ms' {
    param($c,$t) $a = [regex]::Matches($t,'\d+ inference took [\d.]+ ms, avg ([\d.]+) ms') | ForEach-Object { [double]$_.Groups[1].Value }
    $mean = ($a | Measure-Object -Average).Average; ($c -eq 0 -and $mean -lt 10), ('{0:N2} ms/inference' -f $mean) }

Invoke-Case 'profiling (level 2)' "$m --input_list_path .\input_list.txt --output_folder_path .\outputs --profile_level 2 --etdump_path $work\p.etdp" 'exit 0, etdump size matches' {
    param($c,$t) $logged = [int]([regex]::Match($t,'Size = (\d+)').Groups[1].Value); $disk = (Get-Item "$work\p.etdp").Length
    ($c -eq 0 -and $logged -eq $disk), "logged $logged / disk $disk bytes" }

Invoke-Case 'no input list (defaults)' "$m --input_list_path .\does_not_exist.txt --output_folder_path $work" 'exit 0, default inputs' {
    param($c,$t) ($c -eq 0 -and $t -match 'default values'), $(if($t -match 'default values'){'ran with default inputs (smoke test only)'}else{'no default-input message'}) }

Invoke-Case 'run from another folder' "--model_path $bundle\dlv3_qnn.pte --input_list_path $bundle\input_list_abs.txt --output_folder_path $work" 'exit 0 (DLLs found next to exe)' {
    param($c,$t) ($c -eq 0), $(if($c -eq 0){'loads QNN DLLs from exe folder'}else{'failed'}) } -cwd 'C:\'

Invoke-Case 'missing model file' "--model_path .\nope.pte --input_list_path .\input_list.txt --output_folder_path $work" 'clean error, no crash' {
    param($c,$t) ($c -ne 0), ('exit {0}; ' -f $c) + (($t -split "`n" | Select-String 'fail|Fail|error|Error|abort' | Select-Object -First 1) -replace '^\s+','') }

Invoke-Case 'wrong input size' "$m --input_list_path $work\bad_list.txt --output_folder_path $work" 'clean error (size mismatch)' {
    param($c,$t) ($c -ne 0 -and $t -match 'size mismatch'), $(if($t -match 'size mismatch'){'rejected: input size mismatch'}else{"exit $c"}) }

Invoke-Case 'corrupt .pte' "--model_path $work\corrupt.pte --input_list_path .\input_list.txt --output_folder_path $work" 'clean error, no crash' {
    param($c,$t) ($c -ne 0), ('exit {0}; ' -f $c) + (($t -split "`n" | Select-String 'fail|Fail|error|Error|Invalid' | Select-Object -First 1) -replace '^\s+','') }

$results | Format-Table Case,Expect,Exit,Crash,Pass,Detail -AutoSize -Wrap
