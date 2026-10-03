#Requires -Version 5.1
<#
.SYNOPSIS
  Prove Kernel SLO vs export-path GPU idle (nsys duty-cycle).

.DESCRIPTION
  1) Fair Kernel SLO via parcae-bench --suite theory (H2D excluded from timer).
  2) nsys on multi-chunk GpuCandidateExport stress (NVTX: prepare/bind/h2d/hist/d2h).

  Writes under docs/architecture/profiles/export_duty/.
  Diagnosis: high fair Kernel SLO + low kernel wall share on export ⇒ idle is
  host/PCIe/sync gaps, not "Kernel SLO limited by PCIe".

  See docs/architecture/cuda-profile-theory.md § Export duty cycle.
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-cuda',
    [string] $Config = 'Release',
    [int] $FairTokens = 1048576,
    [int] $FairRepeats = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\export_duty'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$profileScript = Join-Path $repoRoot 'scripts\cuda\profile_theory_hist.ps1'
$logPath = Join-Path $outDir 'capture_log.txt'
Start-Transcript -Path $logPath -Force | Out-Null

function Find-Exe {
    param([string[]] $Candidates)
    foreach ($c in $Candidates) {
        if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path }
    }
    return $null
}

$bench = Find-Exe @(
    (Join-Path $repoRoot "$BuildDir\tools\Release\parcae-bench.exe"),
    (Join-Path $repoRoot "$BuildDir\tools\parcae-bench\$Config\parcae-bench.exe"),
    (Join-Path $repoRoot "$BuildDir\$Config\parcae-bench.exe")
)
if (-not $bench) {
    Write-Error "parcae-bench not found under $BuildDir. Build Release with PARCAE_BUILD_CUDA=ON."
}

$tests = Find-Exe @(
    (Join-Path $repoRoot "$BuildDir\tests\Release\parcae_tests.exe"),
    (Join-Path $repoRoot "$BuildDir\tests\$Config\parcae_tests.exe"),
    (Join-Path $repoRoot "$BuildDir\$Config\parcae_tests.exe")
)
if (-not $tests) {
    Write-Error "parcae_tests not found under $BuildDir."
}

Write-Host "bench: $bench"
Write-Host "tests: $tests"
Write-Host "out:   $outDir"

Write-Host '=== fair Kernel SLO: parcae-bench --suite theory ==='
$fairJson = Join-Path $outDir 'theory_fair.json'
& $bench --suite theory --allow-cuda --tokens $FairTokens --repeats $FairRepeats --json |
    Set-Content -Path $fairJson -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Warning "parcae-bench theory exited $LASTEXITCODE (JSON may still be useful)."
}

Write-Host '=== nsys: multi-chunk theory export duty stress ==='
$nsysOk = $true
try {
    & $profileScript -Mode nsys `
        -Exe $tests `
        -ExeArgs @('[search][export][theory][duty][cuda]', '--reporter', 'compact') `
        -OutDir $outDir `
        -Tag 'export_duty_stress'
} catch {
    $nsysOk = $false
    Write-Warning "nsys stress failed: $_"
}

# Export text stats if nsys report exists (tool may have written .nsys-rep).
$rep = Get-ChildItem -Path $outDir -Filter 'export_duty_stress*.nsys-rep' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($rep) {
    Write-Host "nsys report: $($rep.FullName)"
    $statsTxt = Join-Path $outDir 'export_duty_stress_stats.txt'
    try {
        $nsysPath = & $profileScript -Mode nsys -CheckToolsOnly 2>&1 | Out-Null
        # Resolve nsys the same way profile script does
        $nsysCmd = Get-Command nsys -ErrorAction SilentlyContinue
        if (-not $nsysCmd) {
            $hits = @(Get-Item 'C:\Program Files\NVIDIA Corporation\Nsight Systems *\target-windows-x64\nsys.exe' -ErrorAction SilentlyContinue)
            if ($hits.Count -gt 0) { $nsysExe = $hits[0].FullName } else { $nsysExe = $null }
        } else {
            $nsysExe = $nsysCmd.Source
        }
        if ($nsysExe) {
            & $nsysExe stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum `
                --format tsv `
                --force-export=true `
                $rep.FullName 2>&1 | Set-Content -Path $statsTxt -Encoding utf8
            Write-Host "stats: $statsTxt"
        }
    } catch {
        Write-Warning "nsys stats export failed: $_"
    }
} else {
    Write-Warning 'No export_duty_stress*.nsys-rep found.'
    $nsysOk = $false
}

Write-Host '=== also: short-fixture cache test under nsys (NVTX smoke) ==='
try {
    & $profileScript -Mode nsys `
        -Exe $tests `
        -ExeArgs @('[search][export][theory][cache][cuda]', '--reporter', 'compact') `
        -OutDir $outDir `
        -Tag 'export_duty_cache'
} catch {
    Write-Warning "nsys cache test failed: $_"
}

Stop-Transcript | Out-Null

if (-not $nsysOk) {
    Write-Warning 'nsys incomplete — fill SUMMARY from available JSON + any stats.'
}

Write-Host 'Update docs/architecture/profiles/export_duty/SUMMARY.md and cuda-profile-theory.md.'
Write-Host 'Done.'
