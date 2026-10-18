# Download a Hugging Face repo into the standard HF cache, using curl.exe for large LFS files.
# huggingface_hub's own downloader wrote 0-byte blobs / stalled on large files on this machine.
# Small files go through snapshot_download; each large file is fetched resumably with curl into
# blobs\<sha256>, checked against the LFS sha256, and linked into the snapshot with mklink.
param(
    [Parameter(Mandatory)] [string]$Repo,
    [Parameter(Mandatory)] [string]$Revision,
    # "name|sha256;name|sha256". One string, because `powershell -File` can't pass arrays.
    [Parameter(Mandatory)] [string]$LargeFilesSpec,
    # Extra glob patterns to skip, ';'-separated (e.g. duplicate weight formats).
    [string]$ExtraIgnore = ''
)
$LargeFiles = $LargeFilesSpec -split ';' | Where-Object { $_ }
$ErrorActionPreference = 'Stop'
. C:\et\et-env.ps1
$env:HF_HUB_DISABLE_XET = '1'
Set-Location C:\et

$skip = @($LargeFiles | ForEach-Object { ($_ -split '\|')[0] }) + @($ExtraIgnore -split ';' | Where-Object { $_ })
$patterns = ($skip | ForEach-Object { "'" + $_ + "'" }) -join ','
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$Repo', revision='$Revision', ignore_patterns=[$patterns]))"
if ($LASTEXITCODE -ne 0) { throw "snapshot_download (small files) failed" }

$hub = Join-Path $env:USERPROFILE ('.cache\huggingface\hub\models--' + ($Repo -replace '/', '--'))
$snap = Join-Path $hub "snapshots\$Revision"
foreach ($entry in $LargeFiles) {
    $name, $sha = $entry -split '\|'
    $blob = Join-Path $hub "blobs\$sha"
    $url = "https://huggingface.co/$Repo/resolve/$Revision/$name"
    curl.exe -L --fail --retry 10 --retry-delay 5 --retry-all-errors -C - -s -S -o $blob $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $name" }
    $actual = (Get-FileHash $blob -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $sha) { throw "sha256 mismatch for $name : $actual" }
    if (-not (Test-Path (Join-Path $snap $name))) {
        Push-Location $snap
        cmd /c "mklink `"$name`" `"..\..\blobs\$sha`"" | Out-Null
        Pop-Location
    }
    "{0}: {1:N0} bytes, sha256 OK" -f $name, (Get-Item $blob).Length
}
