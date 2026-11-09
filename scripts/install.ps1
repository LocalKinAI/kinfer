# install.ps1 — install kinfer on Windows, x64 or ARM64: the latest release (or
# $env:KINFER_VERSION), checked against its sha256, into
# %LOCALAPPDATA%\Programs\kinfer (or $env:KINFER_BIN), and onto your PATH.
#
#   irm https://raw.githubusercontent.com/LocalKinAI/kinfer/main/scripts/install.ps1 | iex
#
# kinfer runs on the CPU here. The binary is not signed, so Windows may warn the
# first time it starts.
& {
  $ErrorActionPreference = 'Stop'
  $ProgressPreference = 'SilentlyContinue'
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

  $repo = 'LocalKinAI/kinfer'
  $os = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
  switch ($os.ToString()) {
    'X64' { $arch = 'amd64'; $redist = 'x64' }
    'Arm64' { $arch = 'arm64'; $redist = 'arm64' }
    default { throw "kinfer has no release for Windows on $os" }
  }

  $version = $env:KINFER_VERSION
  if (-not $version) {
    $version = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest").tag_name
  }
  if (-not $version) { throw "no kinfer release found at https://github.com/$repo/releases" }

  $base = $env:KINFER_BASE_URL
  if (-not $base) { $base = "https://github.com/$repo/releases/download/$version" }
  $name = "kinfer-$version-windows-$arch"

  $tmp = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName())
  New-Item -ItemType Directory -Path $tmp | Out-Null
  try {
    Write-Host "downloading kinfer $version..."
    $zip = Join-Path $tmp "$name.zip"
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$name.zip" -OutFile $zip
    $sum = Join-Path $tmp "$name.zip.sha256"
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$name.zip.sha256" -OutFile $sum
    $want = ((Get-Content $sum -Raw).Trim() -split '\s+')[0]
    $got = (Get-FileHash -Algorithm SHA256 $zip).Hash
    if ($got -ne $want.ToUpper()) { throw 'the download does not match its sha256' }
    Expand-Archive -Path $zip -DestinationPath $tmp

    $bin = $env:KINFER_BIN
    if (-not $bin) { $bin = Join-Path $env:LOCALAPPDATA 'Programs\kinfer' }
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    $target = Join-Path $bin 'kinfer.exe'
    # A running kinfer.exe cannot be overwritten, but it can be moved aside.
    if (Test-Path $target) { Move-Item -Force $target "$target.old" }
    Copy-Item (Join-Path $tmp "$name\kinfer.exe") $target
    Remove-Item -Force "$target.old" -ErrorAction SilentlyContinue
  }
  finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  }

  $path = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (-not (($path -split ';') -contains $bin)) {
    [Environment]::SetEnvironmentVariable('Path', "$path;$bin".TrimStart(';'), 'User')
    Write-Host "added $bin to your PATH; open a new terminal to use it"
  }
  $env:Path = "$env:Path;$bin"

  $sys = Join-Path $env:windir 'System32'
  if (-not ((Test-Path "$sys\msvcp140.dll") -and (Test-Path "$sys\vcruntime140.dll"))) {
    Write-Warning "kinfer needs the Microsoft Visual C++ runtime: https://aka.ms/vs/17/release/vc_redist.$redist.exe"
  }

  Write-Host "installed $(& $target version) in $bin"
  Write-Host 'kinfer runs on the CPU on Windows for now. Then:  kinfer fit'
}
