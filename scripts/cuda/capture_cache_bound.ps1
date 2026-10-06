#Requires -Version 5.1
<#
.SYNOPSIS
  Capture L2/DRAM evidence for fused-hist kernels (cache_bound plate).

.DESCRIPTION
  Quiet-plate Kernel SLO (cudaEvent) + ncu with dram__bytes.sum and L1/L2
  sector / hit-rate metrics for:

    caesar_chi2_histogram_decrypt_kernel
    atbash_chi2_hist_kernel
    theory_hist_chi2_s1_lut_kernel
    theory_hist_chi2_s2_linear_kernel
    affine_chi2_hist_kernel

  Writes under docs/architecture/profiles/cache_bound/ (binaries gitignored;
  JSON + SUMMARY.md allow-listed).

  If the full cache_bound metric list fails (unknown metric on sm_120), retries
  with a core subset (SM/DRAM % + dram__bytes.sum + duration). ERR_NVGPUCTRPERM
  soft-fails — cudaEvent JSON remains valid.

  Playbook: docs/architecture/cuda-profile-theory.md
  Fill:     docs/architecture/profiles/cache_bound/SUMMARY.md
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-cuda',
    [string] $DataDir = 'data',
    [int] $FairTokens = 1048576,
    [int] $FairRepeats = 8,
    # Shorter T for ncu wall time; same launch-skip/warmup contract as other captures.
    [int] $NcuTokens = 262144,
    [int] $NcuRepeats = 2,
    [switch] $SkipNcu,
    [switch] $SkipFair
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

function Find-Exe {
    param([string[]] $Candidates)
    foreach ($c in $Candidates) {
        if (Test-Path -LiteralPath $c) {
            return (Resolve-Path -LiteralPath $c).Path
        }
    }
    return $null
}

$bench = Find-Exe @(
    (Join-Path $repoRoot "$BuildDir\tools\Release\parcae-bench.exe"),
    (Join-Path $repoRoot "$BuildDir\tools\parcae-bench\Release\parcae-bench.exe"),
    (Join-Path $repoRoot "$BuildDir\Release\parcae-bench.exe"),
    (Join-Path $repoRoot 'build-rel-cuda\tools\Release\parcae-bench.exe'),
    (Join-Path $repoRoot 'build-cuda\tools\Release\parcae-bench.exe')
)
if (-not $bench) {
    Write-Error "Missing parcae-bench under $BuildDir (PARCAE_BUILD_CUDA=ON Release)."
}

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\cache_bound'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$logPath = Join-Path $outDir 'capture_log.txt'
Start-Transcript -Path $logPath -Force | Out-Null

$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'

# Full list (via MetricsPreset). Core fallback if an L2 metric name is rejected.
$coreMetrics = @(
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__bytes.sum',
    'launch__occupancy_limit_blocks',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'gpu__time_duration.sum'
) -join ','

$ncuOk = $true
$ncuUsedFallback = $false
$ncuPermissionDenied = $false

Write-Host "bench: $bench"
Write-Host "out:   $outDir"
Write-Host "fair T=$FairTokens  ncu T=$NcuTokens"

if (-not $SkipFair) {
    Write-Host '=== cudaEvent: theory suite (fair T, S0+S1+S2+Caesar twin) ==='
    $theoryJson = & $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
        --json --data-dir $DataDir 2>&1 | Out-String
    [System.IO.File]::WriteAllText((Join-Path $outDir 'theory_fair.json'), $theoryJson.Trim() + "`n")
    Write-Host $theoryJson

    Write-Host '=== cudaEvent: slo extended (Atbash + Affine + catalog) ==='
    $sloJson = & $bench --suite slo --extended --allow-cuda --json --data-dir $DataDir 2>&1 | Out-String
    [System.IO.File]::WriteAllText((Join-Path $outDir 'catalog_slo_extended.json'), $sloJson.Trim() + "`n")
    Write-Host $sloJson
}

$theoryArgsNcu = @(
    '--suite', 'theory', '--allow-cuda',
    '--tokens', "$NcuTokens", '--repeats', "$NcuRepeats",
    '--data-dir', $DataDir
)
$theoryArgsNoCatalog = $theoryArgsNcu + @('--no-compare-catalog')
$sloArgsNcu = @(
    '--suite', 'slo', '--extended', '--allow-cuda',
    '--tokens', "$NcuTokens", '--repeats', "$NcuRepeats",
    '--data-dir', $DataDir
)

function Test-NcuPermissionError {
    param([string] $Text)
    return ($Text -match 'ERR_NVGPUCTRPERM' -or $Text -match 'ERR_NVGPUCTR')
}

function Invoke-CacheBoundNcu {
    param(
        [Parameter(Mandatory = $true)][string] $Filter,
        [Parameter(Mandatory = $true)][string] $Tag,
        [Parameter(Mandatory = $true)][string[]] $BenchArgs
    )
    if ($SkipNcu) {
        Write-Host "=== ncu skipped: $Filter ($Tag) ==="
        return
    }

    Write-Host "=== ncu cache_bound: $Filter ($Tag) ==="
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = $null
    $code = 0
    try {
        $out = & $profileScript -Mode ncu `
            -Exe $bench `
            -ExeArgs $BenchArgs `
            -KernelFilter $Filter `
            -OutDir $outDir `
            -Tag $Tag `
            -NcuSet none `
            -MetricsPreset cache_bound `
            -ExportCsv `
            -LaunchSkip 4 `
            -LaunchCount 1 2>&1
        $code = $LASTEXITCODE
        Write-Host ($out | Out-String)
    } catch {
        $out = "$_"
        $code = 1
        Write-Warning "ncu ${Tag} threw: $_"
    } finally {
        $ErrorActionPreference = $prevEap
    }

    $text = if ($null -eq $out) { '' } else { ($out | Out-String) }
    if (Test-NcuPermissionError $text) {
        $script:ncuPermissionDenied = $true
        $script:ncuOk = $false
        Write-Warning "ncu ${Tag}: GPU performance counters denied (ERR_NVGPUCTRPERM). Enable counters or elevate; cudaEvent still valid."
        return
    }

    $rep = Get-ChildItem -Path $outDir -Filter "$Tag.ncu-rep" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($code -eq 0 -and $null -ne $rep) {
        return
    }

    # Retry with core metrics (drops L1/L2 names that may be invalid on sm_120).
    Write-Warning "ncu ${Tag} full preset failed (exit=$code) - retrying core metrics (dram__bytes.sum + SM/DRAM %)."
    $script:ncuUsedFallback = $true
    $ErrorActionPreference = 'Continue'
    try {
        $out2 = & $profileScript -Mode ncu `
            -Exe $bench `
            -ExeArgs $BenchArgs `
            -KernelFilter $Filter `
            -OutDir $outDir `
            -Tag $Tag `
            -NcuSet none `
            -NcuMetrics $coreMetrics `
            -ExportCsv `
            -LaunchSkip 4 `
            -LaunchCount 1 2>&1
        $code2 = $LASTEXITCODE
        Write-Host ($out2 | Out-String)
        $text2 = if ($null -eq $out2) { '' } else { ($out2 | Out-String) }
        if (Test-NcuPermissionError $text2) {
            $script:ncuPermissionDenied = $true
            $script:ncuOk = $false
            Write-Warning "ncu ${Tag} core retry: ERR_NVGPUCTRPERM."
            return
        }
        $rep2 = Get-ChildItem -Path $outDir -Filter "$Tag.ncu-rep" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($code2 -ne 0 -or $null -eq $rep2) {
            $script:ncuOk = $false
            Write-Warning "ncu ${Tag} core retry also failed (exit=$code2)."
        }
    } catch {
        $script:ncuOk = $false
        Write-Warning "ncu ${Tag} core retry threw: $_"
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

# Kernel map: catalog twins + specialized theory hist.
Invoke-CacheBoundNcu -Filter 'caesar_chi2_histogram_decrypt_kernel' -Tag 'caesar_hist' `
    -BenchArgs $theoryArgsNcu
Invoke-CacheBoundNcu -Filter 'atbash_chi2_hist_kernel' -Tag 'atbash_hist' `
    -BenchArgs $sloArgsNcu
Invoke-CacheBoundNcu -Filter 'theory_hist_chi2_s1_lut_kernel' -Tag 's1_lut' `
    -BenchArgs $theoryArgsNoCatalog
Invoke-CacheBoundNcu -Filter 'theory_hist_chi2_s2_linear_kernel' -Tag 's2_linear' `
    -BenchArgs $theoryArgsNoCatalog
Invoke-CacheBoundNcu -Filter 'affine_chi2_hist_kernel' -Tag 'affine_hist' `
    -BenchArgs $sloArgsNcu

# Digest: which reports / CSVs landed (machine-readable for SUMMARY fill).
$digest = [ordered]@{
    plate           = 'cache_bound'
    hardware        = 'RTX 5070 Ti sm_120'
    dram_bw_gbs     = 896.0
    fair_tokens     = $FairTokens
    ncu_tokens      = $NcuTokens
    ncu_ok          = $ncuOk
    ncu_fallback    = $ncuUsedFallback
    ncu_perm_denied = $ncuPermissionDenied
    skip_ncu        = [bool]$SkipNcu
    kernels         = @()
}
foreach ($tag in @('caesar_hist', 'atbash_hist', 's1_lut', 's2_linear', 'affine_hist')) {
    $rep = Test-Path -LiteralPath (Join-Path $outDir "$tag.ncu-rep")
    $csv = Test-Path -LiteralPath (Join-Path $outDir "${tag}_metrics.csv")
    $digest.kernels += [ordered]@{
        tag     = $tag
        ncu_rep = $rep
        csv     = $csv
    }
}
$digestPath = Join-Path $outDir 'capture_digest.json'
[System.IO.File]::WriteAllText($digestPath, (($digest | ConvertTo-Json -Depth 6) + "`n"))
Write-Host "digest: $digestPath"

if (-not $ncuOk) {
    if ($ncuPermissionDenied) {
        Write-Warning 'ncu incomplete: ERR_NVGPUCTRPERM. cudaEvent JSON under cache_bound/ still valid.'
    } else {
        Write-Warning 'ncu incomplete (metric names or tool error). See capture_log.txt; core fallback may have partial reps.'
    }
}

Stop-Transcript | Out-Null
Write-Host "Cache-bound artifacts under $outDir"
Write-Host 'Fill docs/architecture/profiles/cache_bound/SUMMARY.md from CSV / ncu UI (DRAM GB/s, L2 hit%, bytes/rune).'
Write-Host 'Append a progress row in docs/architecture/cuda-profile-theory.md.'
exit 0
