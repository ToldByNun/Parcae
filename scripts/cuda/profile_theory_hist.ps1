#Requires -Version 5.1
<#
.SYNOPSIS
  Wrap nsys / ncu for Parcae theory fused-χ² / search-export profiling.

.DESCRIPTION
  Writes reports under -OutDir (default: docs/architecture/profiles/<Tag>).
  Auto-discovers nsys/ncu under typical NVIDIA install paths when not on PATH.

  Metric definitions and acceptance rules:
    docs/architecture/cuda-profile-theory.md

.PARAMETER Mode
  ncu | nsys | both

.PARAMETER Exe
  Path to the binary to profile (required unless -CheckToolsOnly).

.PARAMETER ExeArgs
  Arguments passed to Exe after `--`.

.PARAMETER KernelFilter
  Demangled kernel name regex for ncu (default: theory_chi2_hist_kernel).

.PARAMETER OutDir
  Directory for reports (created if missing).

.PARAMETER Tag
  Basename prefix for report files.

.PARAMETER NcuSet
  ncu --set value (default: full). Use 'none' with -NcuMetrics for a custom list.

.PARAMETER NcuMetrics
  Optional comma-separated --metrics list (overrides --set when non-empty).

.PARAMETER LaunchSkip
  Skip this many matching kernel launches before collecting (default 4 = warmups).

.PARAMETER LaunchCount
  Number of matching launches to profile (default 1).

.PARAMETER CheckToolsOnly
  Verify nsys/ncu resolvable and exit 0/1; do not run a workload.
#>
[CmdletBinding()]
param(
    [ValidateSet('ncu', 'nsys', 'both')]
    [string] $Mode = 'both',

    [string] $Exe = '',

    [string[]] $ExeArgs = @(),

    [string] $KernelFilter = 'theory_chi2_hist_kernel',

    [string] $OutDir = '',

    [string] $Tag = 'theory_hist',

    [string] $NcuSet = 'full',

    [string] $NcuMetrics = '',

    [int] $LaunchSkip = 4,

    [int] $LaunchCount = 1,

    [switch] $CheckToolsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-NvidiaTool {
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string[]] $CandidateGlobs
    )
    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    foreach ($glob in $CandidateGlobs) {
        $hits = @(Get-Item -Path $glob -ErrorAction SilentlyContinue)
        if ($hits.Count -gt 0) {
            return $hits[0].FullName
        }
    }
    return $null
}

$nsysPath = Resolve-NvidiaTool -Name 'nsys' -CandidateGlobs @(
    'C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe',
    "${env:CUDA_PATH}\bin\nsys.exe"
)
$ncuPath = Resolve-NvidiaTool -Name 'ncu' -CandidateGlobs @(
    'C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.bat',
    'C:\Program Files\NVIDIA Corporation\Nsight Compute *\ncu.exe',
    "${env:CUDA_PATH}\bin\ncu.exe"
)

function Require-Resolved {
    param([string] $Label, [string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        Write-Error "$Label not found on PATH or under Program Files\NVIDIA Corporation. Install Nsight and/or add it to PATH."
    }
}

$needNcu = ($Mode -eq 'ncu' -or $Mode -eq 'both')
$needNsys = ($Mode -eq 'nsys' -or $Mode -eq 'both')

if ($needNcu) { Require-Resolved 'ncu' $ncuPath }
if ($needNsys) { Require-Resolved 'nsys' $nsysPath }

Write-Host "nsys: $(if ($nsysPath) { $nsysPath } else { 'missing' })"
Write-Host "ncu:  $(if ($ncuPath) { $ncuPath } else { 'missing' })"

if ($CheckToolsOnly) {
    Write-Host 'Tools OK.'
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Exe)) {
    Write-Error 'Specify -Exe (path to parcae-search-cycle / parcae-bench / test binary), or use -CheckToolsOnly.'
}
if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Error "Exe not found: $Exe"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $repoRoot "docs\architecture\profiles\$Tag"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$exeFull = (Resolve-Path -LiteralPath $Exe).Path

Write-Host "OutDir: $OutDir"
Write-Host "Exe:    $exeFull"
Write-Host "Args:   $($ExeArgs -join ' ')"
Write-Host "Mode:   $Mode"
Write-Host "Kernel: $KernelFilter"

$ncuBase = Join-Path $OutDir $Tag
$nsysBase = Join-Path $OutDir "${Tag}_timeline"

if ($needNcu) {
    Write-Host '=== ncu ==='
    $ncuArgs = @(
        '--kernel-name-base', 'demangled',
        '--kernel-name', "regex:$KernelFilter",
        '--launch-skip', "$LaunchSkip",
        '--launch-count', "$LaunchCount",
        '--export', $ncuBase,
        '--force-overwrite'
    )
    if (-not [string]::IsNullOrWhiteSpace($NcuMetrics)) {
        $ncuArgs += @('--metrics', $NcuMetrics)
    } elseif ($NcuSet -ne 'none') {
        $ncuArgs += @('--set', $NcuSet)
    }
    $ncuArgs += @('--', $exeFull) + $ExeArgs
    & $ncuPath @ncuArgs
    $ncuCode = $LASTEXITCODE
    $repPath = Get-ChildItem -Path $OutDir -Filter "$Tag.ncu-rep" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($ncuCode -ne 0) {
        if ($null -ne $repPath) {
            Write-Warning "ncu exited $ncuCode but wrote $($repPath.Name) (app may have failed SLO)."
        } else {
            Write-Error "ncu exited with code $ncuCode"
        }
    }
    Write-Host "ncu report: ${ncuBase}.ncu-rep (or tool default extension)"
}

if ($needNsys) {
    Write-Host '=== nsys ==='
    & $nsysPath profile `
        -t cuda,nvtx `
        --stats=true `
        --force-overwrite=true `
        -o $nsysBase `
        -- $exeFull @ExeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nsys exited with code $LASTEXITCODE"
    }
    Write-Host "nsys report: ${nsysBase}.nsys-rep"
}

Write-Host 'Done. Record Kernel SLO vs 90% estimated_peak in docs/architecture/cuda-profile-theory.md'
exit 0
