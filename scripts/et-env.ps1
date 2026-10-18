# Environment for ExecuTorch v1.5.1 + QAIRT 2.37.0.250724 (per qc-executorch-guide Compute appendix).
# Usage: . C:\et\et-env.ps1
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$env:EXECUTORCH_ROOT = 'C:\et\executorch'
$env:QNN_SDK_ROOT = 'C:\Qualcomm\AIStack\QAIRT\2.37.0.250724'
& "$env:QNN_SDK_ROOT\bin\envsetup.ps1" | Out-Null
# envsetup.ps1 picks the host shell's arch (ARM64 here), but AOT export runs in x64 Python and must load the
# x86_64-windows-msvc QNN DLLs. The ARM64 runner in xelite-run loads its own co-located ARM64 DLLs instead.
$env:PATH = (($env:PATH -split ';') | Where-Object { $_ -notlike "$env:QNN_SDK_ROOT*aarch64-windows-msvc*" }) -join ';'
$env:PATH = "$env:QNN_SDK_ROOT\lib\x86_64-windows-msvc;$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc;$env:PATH"
$env:PYTHONPATH = "$env:EXECUTORCH_ROOT\..;$env:PYTHONPATH"
# On an ARM64 host the VS generator defaults to ARM64; the AOT pybind wheel must match the x64 (AMD64) Python.
# build.ps1 passes -A explicitly, so this only affects builds that don't (e.g. install_executorch.py).
$env:CMAKE_GENERATOR = 'Visual Studio 17 2022'   # CMake ignores CMAKE_GENERATOR_PLATFORM unless this is set
$env:CMAKE_GENERATOR_PLATFORM = 'x64'
$venv = "$env:EXECUTORCH_ROOT\.venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv $venv
}
. "$venv\Scripts\Activate.ps1"
Set-Location $env:EXECUTORCH_ROOT
