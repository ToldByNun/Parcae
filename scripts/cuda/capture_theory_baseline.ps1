#Requires -Version 5.1
<#
.SYNOPSIS
  Capture baseline nsys/ncu + cudaEvent for TheoryChi2Batch vs Caesar/Affine.

.DESCRIPTION
  Writes binary reports under docs/architecture/profiles/baseline/ (gitignored)
  and a text SUMMARY.md (allow-listed). Also prints cudaEvent Kernel SLO via
  parcae-bench --suite theory / slo.

  See docs/architecture/cuda-profile-theory.md.
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-cuda',
    [string] $DataDir = 'data',
    # Fair kernel SLO uses T1 tokens; ncu uses a shorter T so profiling finishes.
    [int] $FairTokens = 1048576,
    [int] $FairRepeats = 8,
    [int] $NcuTokens = 262144,
    [int] $NcuRepeats = 2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$bench = Join-Path $repoRoot "$BuildDir\tools\Release\parcae-bench.exe"
if (-not (Test-Path -LiteralPath $bench)) {
    Write-Error "Missing $bench - build Release parcae-bench with PARCAE_BUILD_CUDA=ON."
}

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\baseline'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'

Write-Host '=== cudaEvent: theory suite (fair T) ==='
$theoryJson = & $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
    --campaign-grid --json --data-dir $DataDir 2>&1
$theoryJson | Set-Content -Encoding utf8 (Join-Path $outDir 'theory_fair.json')
Write-Host $theoryJson

Write-Host '=== cudaEvent: catalog T1 + F.affine ==='
$sloJson = & $bench --suite slo --extended --allow-cuda --json --data-dir $DataDir 2>&1
$sloJson | Set-Content -Encoding utf8 (Join-Path $outDir 'catalog_slo_extended.json')
& $bench --suite slo --extended --allow-cuda --data-dir $DataDir 2>&1 |
    Select-String -Pattern 'T1 |F.affine|ALL ROWS|PARCAE' |
    ForEach-Object { $_.Line }

Write-Host '=== ncu: theory_chi2_hist_kernel ==='
$ncuOk = $true
try {
    & $profileScript -Mode ncu `
        -Exe $bench `
        -ExeArgs @('--suite','theory','--allow-cuda','--tokens',"$NcuTokens",'--repeats',"$NcuRepeats",'--no-compare-catalog','--data-dir',$DataDir) `
        -KernelFilter 'theory_chi2_hist_kernel' `
        -OutDir $outDir `
        -Tag 'theory_hist' `
        -NcuSet none `
        -MetricsPreset baseline `
        -LaunchSkip 4 `
        -LaunchCount 1
} catch {
    $ncuOk = $false
    Write-Warning "ncu theory failed: $_"
}

Write-Host '=== ncu: caesar_chi2_histogram_decrypt_kernel ==='
try {
    & $profileScript -Mode ncu `
        -Exe $bench `
        -ExeArgs @('--suite','theory','--allow-cuda','--tokens',"$NcuTokens",'--repeats',"$NcuRepeats",'--data-dir',$DataDir) `
        -KernelFilter 'caesar_chi2_histogram_decrypt_kernel' `
        -OutDir $outDir `
        -Tag 'caesar_hist' `
        -NcuSet none `
        -MetricsPreset baseline `
        -LaunchSkip 4 `
        -LaunchCount 1
} catch {
    $ncuOk = $false
    Write-Warning "ncu caesar failed: $_"
}

Write-Host '=== ncu: affine_chi2_hist_kernel ==='
try {
    & $profileScript -Mode ncu `
        -Exe $bench `
        -ExeArgs @('--suite','slo','--extended','--allow-cuda','--data-dir',$DataDir) `
        -KernelFilter 'affine_chi2_hist_kernel' `
        -OutDir $outDir `
        -Tag 'affine_hist' `
        -NcuSet none `
        -MetricsPreset baseline `
        -LaunchSkip 4 `
        -LaunchCount 1
} catch {
    $ncuOk = $false
    Write-Warning "ncu affine failed: $_"
}

Write-Host '=== nsys: theory suite (NVTX stages, short T) ==='
& $profileScript -Mode nsys `
    -Exe $bench `
    -ExeArgs @('--suite','theory','--allow-cuda','--tokens',"$NcuTokens",'--repeats',"$NcuRepeats",'--campaign-grid','--data-dir',$DataDir) `
    -OutDir $outDir `
    -Tag 'theory_export'

if (-not $ncuOk) {
    Write-Warning 'ncu reports missing (often ERR_NVGPUCTRPERM). Enable GPU performance counters or run elevated; cudaEvent + nsys baseline still valid.'
}

Write-Host "Baseline artifacts under $outDir"
Write-Host 'Fill docs/architecture/cuda-profile-theory.md progress table from JSON + ncu/nsys.'
exit 0
