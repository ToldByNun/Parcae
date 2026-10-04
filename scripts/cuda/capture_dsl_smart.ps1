#Requires -Version 5.1
<#
.SYNOPSIS
  Fair Kernel SLO capture for hand-written DSL smart customs.

.DESCRIPTION
  Runs `parcae-bench --suite dsl_smart` at fair T≥2^20 (Kernel SLO, not campaign
  wall). Writes JSON under docs/architecture/profiles/dsl_smart/.
  Update SUMMARY.md after quiet plate runs.
#>
[CmdletBinding()]
param(
    [string] $BuildDir = 'build-rel-cuda',
    [string] $Config = 'Release',
    [int] $Runs = 1,
    [int] $Tokens = 1048576,
    [int] $Repeats = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $repoRoot

$outDir = Join-Path $repoRoot 'docs\architecture\profiles\dsl_smart'
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
    (Join-Path $repoRoot "$BuildDir\$Config\parcae-bench.exe"),
    (Join-Path $repoRoot "build-cuda\tools\Release\parcae-bench.exe")
)
if (-not $bench) { Write-Error "parcae-bench not found under $BuildDir (or build-cuda)" }

Write-Host "bench: $bench"
Write-Host "out:   $outDir"
Write-Host "T=$Tokens runs=$Runs"

$benchArgs = @(
    '--suite', 'dsl_smart',
    '--allow-cuda',
    '--tokens', "$Tokens",
    '--json',
    '--data-dir', (Join-Path $repoRoot 'data')
)
if ($Repeats -gt 0) {
    $benchArgs += @('--repeats', "$Repeats")
}

for ($i = 1; $i -le $Runs; $i++) {
    $name = if ($Runs -eq 1) { 'dsl_smart_fair.json' } else { "dsl_smart_fair_run$i.json" }
    $jsonPath = Join-Path $outDir $name
    Write-Host "=== fair Kernel SLO: parcae-bench --suite dsl_smart ($name) ==="
    & $bench @benchArgs | Set-Content -Path $jsonPath -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "parcae-bench dsl_smart exited $LASTEXITCODE ($name)"
    }
    if ($Runs -gt 1 -and $i -eq $Runs) {
        Copy-Item -LiteralPath $jsonPath -Destination (Join-Path $outDir 'dsl_smart_fair.json') -Force
    }
}

Stop-Transcript | Out-Null
Write-Host 'Update docs/architecture/profiles/dsl_smart/SUMMARY.md (vs F.atbash / Caesar twin).'
Write-Host 'Done.'
