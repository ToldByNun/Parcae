#Requires -Version 5.1
<#
.SYNOPSIS
  Wrap nsys / ncu for Parcae theory fused-χ² / search-export profiling.

.DESCRIPTION
  Writes reports under -OutDir (default: docs/architecture/profiles/<Tag>).
  Does not invent workloads — you pass -Exe and -ExeArgs (search-cycle, bench, tests).

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

.PARAMETER CheckToolsOnly
  Verify nsys/ncu on PATH and exit 0/1; do not run a workload.

.EXAMPLE
  .\scripts\cuda\profile_theory_hist.ps1 -CheckToolsOnly

.EXAMPLE
  .\scripts\cuda\profile_theory_hist.ps1 -Mode both `
    -Exe .\build-cuda\tools\Release\parcae-search-cycle.exe `
    -ExeArgs '--workspace','ws','--job','job.json','--allow-theory-uri','--backend','cuda','--allow-cuda','--data-dir','data' `
    -OutDir .\docs\architecture\profiles\baseline `
    -Tag theory_hist
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

    [switch] $CheckToolsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-CommandOnPath {
    param([Parameter(Mandatory = $true)][string] $Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Require-Tool {
    param([Parameter(Mandatory = $true)][string] $Name)
    if (-not (Test-CommandOnPath $Name)) {
        Write-Error "$Name not found on PATH. Install CUDA Nsight and ensure the Toolkit bin dir is on PATH."
    }
}

$needNcu = ($Mode -eq 'ncu' -or $Mode -eq 'both')
$needNsys = ($Mode -eq 'nsys' -or $Mode -eq 'both')

if ($needNcu) { Require-Tool 'ncu' }
if ($needNsys) { Require-Tool 'nsys' }

Write-Host "nsys: $(if (Test-CommandOnPath 'nsys') { (Get-Command nsys).Source } else { 'missing' })"
Write-Host "ncu:  $(if (Test-CommandOnPath 'ncu') { (Get-Command ncu).Source } else { 'missing' })"

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
    & ncu `
        --set full `
        --kernel-name-base demangled `
        --kernel-name "regex:$KernelFilter" `
        --export $ncuBase `
        --force-overwrite `
        -- $exeFull @ExeArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ncu exited with code $LASTEXITCODE"
    }
    Write-Host "ncu report: ${ncuBase}.ncu-rep (or tool default extension)"
}

if ($needNsys) {
    Write-Host '=== nsys ==='
    & nsys profile `
        -t cuda,nvtx,osrt `
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
