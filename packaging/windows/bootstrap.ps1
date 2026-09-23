# Parcae Windows installer bootstrap (run elevated before / during Inno install).
# Checks MSVC Build Tools, CMake >= 3.25, Python >= 3.11, and optionally CUDA Toolkit.
# Missing pieces are installed via winget (download-on-demand; not bundled).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File bootstrap.ps1 -Flavor cpu|cuda|full
param(
    [ValidateSet("cpu", "cuda", "full")]
    [string]$Flavor = "cpu",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$MinCMake = [version]"3.25"
$MinPython = [version]"3.11"

function Write-Step([string]$msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Winget {
    if (-not (Test-Command "winget")) {
        throw "winget is required to install missing prerequisites. Install App Installer from the Microsoft Store, then re-run."
    }
}

function Install-WingetPackage([string]$Id, [string]$ExtraArgs = "") {
    Ensure-Winget
    Write-Step "winget install $Id"
    $args = @("install", "-e", "--id", $Id, "--accept-package-agreements", "--accept-source-agreements", "--disable-interactivity")
    if ($ExtraArgs) {
        $args += $ExtraArgs.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
    }
    & winget @args
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
        # -1978335189 = already installed
        throw "winget install failed for $Id (exit $LASTEXITCODE)"
    }
}

function Test-Msvc {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $path = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    return [bool]$path
}

function Ensure-Msvc {
    Write-Step "Checking MSVC / Visual Studio C++ tools"
    if (Test-Msvc) {
        Write-Host "MSVC C++ tools: OK"
        return
    }
    if ($SkipInstall) { throw "MSVC C++ Build Tools not found (SkipInstall set)" }
    # VS 2022 Build Tools with C++ workload (large download).
    Install-WingetPackage "Microsoft.VisualStudio.2022.BuildTools" `
        "--override --wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    if (-not (Test-Msvc)) {
        Write-Warning "MSVC may require a new shell or reboot before vswhere reports it."
    }
}

function Get-CMakeVersion {
    if (-not (Test-Command "cmake")) { return $null }
    $line = (& cmake --version | Select-Object -First 1)
    if ($line -match "(\d+\.\d+\.\d+)") { return [version]$Matches[1] }
    return $null
}

function Ensure-CMake {
    Write-Step "Checking CMake >= $MinCMake"
    $ver = Get-CMakeVersion
    if ($ver -and $ver -ge $MinCMake) {
        Write-Host "CMake $ver : OK"
        return
    }
    if ($SkipInstall) { throw "CMake >= $MinCMake not found (SkipInstall set)" }
    Install-WingetPackage "Kitware.CMake"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $ver = Get-CMakeVersion
    if (-not $ver -or $ver -lt $MinCMake) {
        Write-Warning "CMake installed but not yet on PATH in this session; open a new terminal after install."
    } else {
        Write-Host "CMake $ver : OK"
    }
}

function Get-PythonVersion {
    foreach ($py in @("python", "python3", "py")) {
        if (-not (Test-Command $py)) { continue }
        try {
            if ($py -eq "py") {
                $out = & py -3 -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
            } else {
                $out = & $py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
            }
            if ($out -match "(\d+\.\d+\.\d+)") { return [version]$Matches[1] }
        } catch { }
    }
    return $null
}

function Ensure-Python {
    Write-Step "Checking Python >= $MinPython"
    $ver = Get-PythonVersion
    if ($ver -and $ver -ge $MinPython) {
        Write-Host "Python $ver : OK"
        return
    }
    if ($SkipInstall) { throw "Python >= $MinPython not found (SkipInstall set)" }
    Install-WingetPackage "Python.Python.3.12"
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $ver = Get-PythonVersion
    if (-not $ver -or $ver -lt $MinPython) {
        Write-Warning "Python installed but may need a new shell before it is on PATH."
    } else {
        Write-Host "Python $ver : OK"
    }
}

function Ensure-Cuda {
    Write-Step "Checking CUDA Toolkit (nvcc)"
    if (Test-Command "nvcc") {
        & nvcc --version | Select-Object -First 4
        Write-Host "CUDA Toolkit: OK"
        return
    }
    if ($SkipInstall) { throw "CUDA Toolkit (nvcc) not found (SkipInstall set)" }
    Ensure-Winget
    Write-Host "Attempting winget install of NVIDIA CUDA Toolkit (large download; may need reboot)..."
    # Common winget id; if unavailable, fail with guidance.
    $ids = @("Nvidia.CUDA", "NVIDIA.CUDA")
    $ok = $false
    foreach ($id in $ids) {
        try {
            Install-WingetPackage $id
            $ok = $true
            break
        } catch {
            Write-Warning "winget package $id failed: $_"
        }
    }
    if (-not $ok) {
        throw @"
CUDA Toolkit was not found and could not be installed automatically via winget.
Install the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
then re-run this installer (Flavor=$Flavor).
"@
    }
    if (-not (Test-Command "nvcc")) {
        Write-Warning "CUDA Toolkit installed but nvcc not on PATH yet; a reboot may be required."
    }
}

Write-Step "Parcae bootstrap (Flavor=$Flavor)"
Ensure-Msvc
Ensure-CMake
Ensure-Python
if ($Flavor -eq "cuda" -or $Flavor -eq "full") {
    Ensure-Cuda
} else {
    Write-Host "Skipping CUDA check (cpu flavor)"
}
Write-Step "Bootstrap complete"
