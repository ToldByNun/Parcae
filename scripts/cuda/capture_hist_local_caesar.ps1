#Requires -Version 5.1
<#
.SYNOPSIS
  Capture Caesar twin after HistFast thread-local hist (cudaEvent + ncu).

.DESCRIPTION
  Fair Kernel SLO via parcae-bench --suite slo (T1) and ncu on
  caesar_chi2_histogram_decrypt_kernel. Writes under
  docs/architecture/profiles/hist_local_caesar/.

  Peak model: physical DRAM roofline 896B (BenchTierSpec). Compare % of 896B
  and ncu DRAM SoL vs prior ~400B / ~2–3% DRAM captures.
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-cuda',
    [string] $Config = 'Release'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$outDir = Join-Path $repoRoot 'docs\architecture\profiles\hist_local_caesar'
$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$bench = Join-Path $repoRoot "$BuildDir\tools\Release\parcae-bench.exe"
if (-not (Test-Path $bench)) {
    $bench = Join-Path $repoRoot "$BuildDir\Release\parcae-bench.exe"
}
if (-not (Test-Path $bench)) {
    # Multi-config tools path variants
    $candidates = @(
        (Join-Path $repoRoot "$BuildDir\tools\parcae-bench\$Config\parcae-bench.exe"),
        (Join-Path $repoRoot "$BuildDir\$Config\parcae-bench.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $bench = $c; break }
    }
}
if (-not (Test-Path $bench)) {
    Write-Error "parcae-bench not found under $BuildDir (config $Config). Build tools first."
}

Write-Host "bench: $bench"
Write-Host "out:   $outDir"

$ncuMetrics = @(
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'gpu__time_duration.sum'
) -join ','

Write-Host '=== fair cudaEvent: parcae-bench --suite slo (T1 focus via full slo) ==='
$fairJson = Join-Path $outDir 'slo_fair.json'
# --json emits parcae.tool_response.v0 on stdout (not a path argument).
& $bench --suite slo --allow-cuda --json | Set-Content -Path $fairJson -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Warning "parcae-bench slo exited $LASTEXITCODE (JSON may still be useful)."
}

# Shorter T for ncu so profiling finishes; same kernel as fair path.
$ncuArgs = @('--suite', 'slo', '--allow-cuda', '--tokens', '262144', '--repeats', '2')

Write-Host '=== ncu: caesar_chi2_histogram_decrypt_kernel ==='
$ncuOk = $true
try {
    & $profileScript -Mode ncu `
        -Exe $bench `
        -ExeArgs $ncuArgs `
        -KernelFilter 'caesar_chi2_histogram_decrypt_kernel' `
        -OutDir $outDir `
        -Tag 'caesar_hist' `
        -NcuSet 'none' `
        -NcuMetrics $ncuMetrics `
        -LaunchSkip 4 `
        -LaunchCount 1
} catch {
    $ncuOk = $false
    Write-Warning "ncu failed: $_"
}

if (-not $ncuOk) {
    Write-Warning 'ncu report missing (often ERR_NVGPUCTRPERM). cudaEvent JSON still valid.'
}

Write-Host 'Update docs/architecture/profiles/hist_local_caesar/SUMMARY.md and cuda-profile-theory.md.'
