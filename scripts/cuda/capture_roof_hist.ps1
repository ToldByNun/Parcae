#Requires -Version 5.1
<#
.SYNOPSIS
  Commit 10: Caesar fat-tile warp-private roof-hist spike (cudaEvent sweep).

.DESCRIPTION
  1) Fair theory suite (S1 / compare_caesar baseline).
  2) Catch2 tile-cap sweep `[cuda][hist][roof][caesar]` (same BenchTimer protocol).
  Writes under docs/architecture/profiles/roof_hist/.
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-rel-cuda',
    [string] $Config = 'Release'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\roof_hist'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
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
$tests = Find-Exe @(
    (Join-Path $repoRoot "$BuildDir\tests\Release\parcae_tests.exe"),
    (Join-Path $repoRoot "$BuildDir\tests\$Config\parcae_tests.exe")
)
if (-not $bench) { Write-Error "parcae-bench not found under $BuildDir" }
if (-not $tests) { Write-Error "parcae_tests not found under $BuildDir" }

Write-Host "bench: $bench"
Write-Host "tests: $tests"
Write-Host "out:   $outDir"

Write-Host '=== fair Kernel SLO: parcae-bench --suite theory ==='
$fairJson = Join-Path $outDir 'theory_fair.json'
& $bench --suite theory --allow-cuda --tokens 1048576 --repeats 8 --json |
    Set-Content -Path $fairJson -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    Write-Warning "parcae-bench theory exited $LASTEXITCODE"
}

Write-Host '=== fat-tile warp-private sweep ==='
$sweepTxt = Join-Path $outDir 'tile_cap_sweep.txt'
& $tests '[cuda][hist][roof]' --reporter compact 2>&1 |
    Tee-Object -FilePath $sweepTxt
if ($LASTEXITCODE -ne 0) {
    Write-Warning "roof hist tests exited $LASTEXITCODE"
}

Stop-Transcript | Out-Null
Write-Host 'Update docs/architecture/profiles/roof_hist/SUMMARY.md and cuda-profile-theory.md.'
Write-Host 'Done.'
