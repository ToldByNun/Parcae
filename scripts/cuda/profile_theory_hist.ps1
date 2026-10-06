#Requires -Version 5.1
<#
.SYNOPSIS
  Wrap nsys / ncu for Parcae theory fused-χ² / search-export profiling.

.DESCRIPTION
  Writes reports under -OutDir (default: docs/architecture/profiles/<Tag>).
  Auto-discovers nsys/ncu under typical NVIDIA install paths when not on PATH.

  Metric definitions and acceptance rules:
    docs/architecture/cuda-profile-theory.md

  MetricsPreset 'cache_bound' adds dram__bytes.sum + L1/L2 sector / hit-rate
  counters used by scripts/cuda/capture_cache_bound.ps1.

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
  ncu --set value (default: full). Use 'none' with -NcuMetrics / -MetricsPreset
  for a custom list.

.PARAMETER NcuMetrics
  Optional comma-separated --metrics list (overrides --set when non-empty).
  When empty, -MetricsPreset may supply the list.

.PARAMETER MetricsPreset
  none | baseline | cache_bound
  - none: use -NcuSet / -NcuMetrics only (default)
  - baseline: SM/DRAM % + stalls + duration (same list as capture_theory_*.ps1)
  - cache_bound: baseline + dram__bytes.sum + L1/L2 sector / hit-rate metrics

.PARAMETER LaunchSkip
  Skip this many matching kernel launches before collecting (default 4 = warmups).

.PARAMETER LaunchCount
  Number of matching launches to profile (default 1).

.PARAMETER ExportCsv
  After a successful ncu collection, import the .ncu-rep and write
  <OutDir>/<Tag>_metrics.csv (raw page). Soft-fails if import is unavailable.

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

    [ValidateSet('none', 'baseline', 'cache_bound')]
    [string] $MetricsPreset = 'none',

    [int] $LaunchSkip = 4,

    [int] $LaunchCount = 1,

    [switch] $ExportCsv,

    [switch] $CheckToolsOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ParcaeNcuMetricsPreset {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('baseline', 'cache_bound')]
        [string] $Name
    )
    $baseline = @(
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'launch__occupancy_limit_blocks',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__warp_issue_stalled_inst_fetch_per_warp_active.pct',
        'smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct',
        'smsp__warp_issue_stalled_exec_dependency_per_warp_active.pct',
        'gpu__time_duration.sum'
    )
    if ($Name -eq 'baseline') {
        return ($baseline -join ',')
    }
    # cache_bound: absolute DRAM bytes + L1/L2 residency evidence.
    # Some stall / sector names may be n/a on sm_120 — capture script falls back.
    $cache = $baseline + @(
        'dram__bytes.sum',
        'lts__t_sector_hit_rate.pct',
        'lts__t_sectors_srcunit_tex_lookup_hit.sum',
        'lts__t_sectors_srcunit_tex_lookup_miss.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum'
    )
    return ($cache -join ',')
}

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

# Resolve metrics: explicit -NcuMetrics wins; else preset; else --set.
$resolvedMetrics = $NcuMetrics
if ([string]::IsNullOrWhiteSpace($resolvedMetrics) -and $MetricsPreset -ne 'none') {
    $resolvedMetrics = Get-ParcaeNcuMetricsPreset -Name $MetricsPreset
    Write-Host "MetricsPreset: $MetricsPreset"
}

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
    if (-not [string]::IsNullOrWhiteSpace($resolvedMetrics)) {
        $ncuArgs += @('--metrics', $resolvedMetrics)
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

    if ($ExportCsv -and $null -ne $repPath) {
        $csvPath = Join-Path $OutDir "${Tag}_metrics.csv"
        Write-Host "=== ncu CSV export -> $csvPath ==="
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $ncuPath --import $repPath.FullName --csv --page raw 2>$null |
                Set-Content -Path $csvPath -Encoding utf8
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $csvPath) -or
                (Get-Item -LiteralPath $csvPath).Length -eq 0) {
                # Fallback page name used by some Nsight Compute builds.
                & $ncuPath --import $repPath.FullName --csv --page details 2>$null |
                    Set-Content -Path $csvPath -Encoding utf8
            }
            if ((Test-Path -LiteralPath $csvPath) -and (Get-Item -LiteralPath $csvPath).Length -gt 0) {
                Write-Host "CSV ok: $csvPath"
            } else {
                Write-Warning "ncu CSV export empty/failed for $Tag (report still valid)."
            }
        } catch {
            Write-Warning "ncu CSV export failed for ${Tag}: $_"
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
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
