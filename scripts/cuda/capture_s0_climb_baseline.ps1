#Requires -Version 5.1
<#
.SYNOPSIS
  S0 climb baseline: TheoryChi2Batch vs Caesar twin (ncu + nsys + cudaEvent).

.DESCRIPTION
  Narrow repro for T.theory.caesar_bytecode optimization work. Writes binary
  reports under docs/architecture/profiles/s0-climb/ (gitignored) and JSON
  digests allow-listed for commit. Compare against profiles/baseline/ (2026-10-01).

  See docs/architecture/cuda-profile-theory.md § S0 climb baseline.
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

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\s0-climb'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'
$logPath = Join-Path $outDir 'capture_log.txt'
Start-Transcript -Path $logPath -Force | Out-Null

$theoryNcuArgs = @('--suite', 'theory', '--allow-cuda', '--tokens', "$NcuTokens",
    '--repeats', "$NcuRepeats", '--no-compare-catalog', '--data-dir', $DataDir)
$theoryNsysArgs = @('--suite', 'theory', '--allow-cuda', '--tokens', "$NcuTokens",
    '--repeats', "$NcuRepeats", '--campaign-grid', '--data-dir', $DataDir)
$caesarNcuArgs = @('--suite', 'theory', '--allow-cuda', '--tokens', "$NcuTokens",
    '--repeats', "$NcuRepeats", '--data-dir', $DataDir)

Write-Host '=== cudaEvent: theory suite (fair T, S0 + compare_caesar) ==='
$theoryJson = & $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
    --json --data-dir $DataDir 2>&1
$theoryJson | Set-Content -Encoding utf8 (Join-Path $outDir 'theory_fair.json')
Write-Host $theoryJson

Write-Host '=== cudaEvent: human-readable (S0 gate) ==='
& $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats `
    --data-dir $DataDir 2>&1 |
    Select-String -Pattern 'T\.theory\.caesar_bytecode|T\.theory\.compare_caesar|ALL ROWS|PARCAE|pass|fail|pct_peak' |
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

Invoke-NcuKernel -Filter 'theory_chi2_hist_kernel' -Tag 's0_hist' -Args $theoryNcuArgs
Invoke-NcuKernel -Filter 'caesar_chi2_histogram_decrypt_kernel' -Tag 'caesar_hist' -Args $caesarNcuArgs

Write-Host '=== nsys: theory suite (S0 + twin + campaign row) ==='
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & $profileScript -Mode nsys `
        -Exe $bench `
        -ExeArgs $theoryNsysArgs `
        -OutDir $outDir `
        -Tag 's0_climb'
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nsys exited with code $LASTEXITCODE (cudaEvent + ncu above still valid)."
    }
} catch {
    Write-Warning "nsys failed: $_"
} finally {
    $ErrorActionPreference = $prevEap
}

$nsysRep = Join-Path $outDir 's0_climb_timeline.nsys-rep'
if (Test-Path -LiteralPath $nsysRep) {
    $nsysCmd = Get-Command nsys -ErrorAction SilentlyContinue
    $nsysExe = if ($nsysCmd -and $nsysCmd.Path) { $nsysCmd.Path } else { $null }
    if (-not $nsysExe) {
        $hits = @(Get-Item -Path 'C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe' -ErrorAction SilentlyContinue)
        if ($hits.Count -gt 0) { $nsysExe = $hits[0].FullName }
    }
    if ($nsysExe) {
        Write-Host '=== nsys stats: cuda_gpu_kern_sum ==='
        $kernSumPath = Join-Path $outDir 'nsys_kern_sum.txt'
        & $nsysExe stats $nsysRep --report cuda_gpu_kern_sum --format table --force-export=true 2>&1 |
            Tee-Object -FilePath $kernSumPath
    }
}

Stop-Transcript | Out-Null

if (-not $ncuOk) {
    Write-Warning 'ncu reports missing (often ERR_NVGPUCTRPERM). Enable GPU performance counters; cudaEvent + nsys still valid.'
}

Write-Host "S0 climb baseline artifacts under $outDir"
Write-Host 'Update docs/architecture/profiles/s0-climb/SUMMARY.md and cuda-profile-theory.md progress table.'
exit 0
