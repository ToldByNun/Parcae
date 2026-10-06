#Requires -Version 5.1
<#
.SYNOPSIS
  Capture nsys/ncu + cudaEvent after specialized theory hist emit (S0/S1/S2).

.DESCRIPTION
  Writes binary reports under docs/architecture/profiles/specialized/ (gitignored)
  and expects SUMMARY.md to be filled from JSON + ncu/nsys (allow-listed).

  Compares against profiles/baseline/ (commit-4 baseline). See
  docs/architecture/cuda-profile-theory.md progress table.

  Kernels:
    theory_chi2_hist_kernel          — S0 bytecode
    theory_hist_chi2_s1_lut_kernel   — S1 LUT-29
    theory_hist_chi2_s2_linear_kernel — S2 linear uchar4
    caesar_chi2_histogram_decrypt_kernel — catalog twin
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-cuda',
    [string] $DataDir = 'data',
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

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\specialized'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'

$theoryArgs = @('--suite', 'theory', '--allow-cuda', '--tokens', "$NcuTokens",
    '--repeats', "$NcuRepeats", '--data-dir', $DataDir)
$theoryArgsNoCatalog = $theoryArgs + @('--no-compare-catalog')

Write-Host '=== cudaEvent: theory suite (fair T, S0+S1+S2+twin) ==='
$theoryJson = & $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
    --campaign-grid --json --data-dir $DataDir 2>&1
$theoryJson | Set-Content -Encoding utf8 (Join-Path $outDir 'theory_fair.json')
Write-Host $theoryJson

Write-Host '=== cudaEvent: human-readable theory suite ==='
& $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
    --data-dir $DataDir 2>&1 |
    Select-String -Pattern 'T\.theory|ALL ROWS|PARCAE|pass|fail|pct_peak|checkpoint' |
    ForEach-Object { $_.Line }

$ncuOk = $true

function Invoke-NcuKernel {
    param(
        [Parameter(Mandatory = $true)][string] $Filter,
        [Parameter(Mandatory = $true)][string] $Tag,
        [Parameter(Mandatory = $true)][string[]] $Args
    )
    Write-Host "=== ncu: $Filter ($Tag) ==="
    try {
        & $profileScript -Mode ncu `
            -Exe $bench `
            -ExeArgs $Args `
            -KernelFilter $Filter `
            -OutDir $outDir `
            -Tag $Tag `
            -NcuSet none `
            -MetricsPreset baseline `
            -LaunchSkip 4 `
            -LaunchCount 1
    } catch {
        $script:ncuOk = $false
        Write-Warning "ncu $Tag failed: $_"
    }
}

Invoke-NcuKernel -Filter 'theory_chi2_hist_kernel' -Tag 's0_hist' -Args $theoryArgsNoCatalog
Invoke-NcuKernel -Filter 'theory_hist_chi2_s1_lut_kernel' -Tag 's1_lut' -Args $theoryArgsNoCatalog
Invoke-NcuKernel -Filter 'theory_hist_chi2_s2_linear_kernel' -Tag 's2_linear' -Args $theoryArgsNoCatalog
Invoke-NcuKernel -Filter 'caesar_chi2_histogram_decrypt_kernel' -Tag 'caesar_hist' -Args $theoryArgs

Write-Host '=== nsys: theory suite (S0/S1/S2 NVTX stages) ==='
# nsys may write WARNING lines to stderr (CPU context switches need admin).
# Keep those non-terminating under $ErrorActionPreference=Stop.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $profileScript -Mode nsys `
        -Exe $bench `
        -ExeArgs (@('--suite', 'theory', '--allow-cuda', '--tokens', "$NcuTokens",
            '--repeats', "$NcuRepeats", '--campaign-grid', '--data-dir', $DataDir)) `
        -OutDir $outDir `
        -Tag 'theory_specialized'
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nsys exited with code $LASTEXITCODE (cudaEvent + ncu artifacts above still valid)."
    }
} catch {
    Write-Warning "nsys failed: $_"
} finally {
    $ErrorActionPreference = $prevEap
}

if (-not $ncuOk) {
    Write-Warning 'ncu reports missing (often ERR_NVGPUCTRPERM). Enable GPU performance counters or run elevated; cudaEvent + nsys still valid.'
}

Write-Host "Specialized artifacts under $outDir"
Write-Host 'Fill docs/architecture/profiles/specialized/SUMMARY.md and append cuda-profile-theory.md progress rows.'
exit 0
