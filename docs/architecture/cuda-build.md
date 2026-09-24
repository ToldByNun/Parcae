# Local CUDA build notes

**Status:** Twins through batch + fused search-run (roadmap through commits 38–39 docs)  
**Sources:** [`Parcae/Parcae/cuda/`](../../Parcae/Parcae/cuda/)  
**CI:** GitHub Actions stays **CPU-only** (`PARCAE_BUILD_CUDA` unset / default `OFF`). GPU jobs are optional and local (or a future self-hosted runner).

## Prerequisites

| Piece | Notes |
|-------|--------|
| NVIDIA GPU + driver | Needed to *run* device tests; compile-only still needs the Toolkit |
| CUDA Toolkit | nvcc + VS Build Customizations (Windows) |
| C++20 host compiler | Same as CPU builds |
| CMake ≥ 3.25 | Root `CMakeLists.txt`; CUDA language uses **C++20** (`CMAKE_CUDA_STANDARD 20`) |

Default CMake CUDA architectures: `75;86` (override with `-DCMAKE_CUDA_ARCHITECTURES=...`).

## Visual Studio (day-to-day)

1. Install the CUDA Toolkit so  
   `$(VCTargetsPath)\BuildCustomizations\CUDA <ver>.props` exists.
2. Open [`Parcae/Parcae.slnx`](../../Parcae/Parcae.slnx).
3. Build configuration **x64** (Win32 is host-only; `.cu` files are gated on x64).
4. Toolkit prop version defaults to **13.3**. Override MSBuild property  
   `ParcaeCudaToolkitVersion` if your install differs.
5. The VS project defines `PARCAE_HAS_CUDA=1` on x64 and compiles  
   `parcae_cuda_stub.cu`, `identity_copy.cu`, …

Smoke entry: run the `Parcae` app (`main.cpp`) or Catch2 cases below after a CMake CUDA build.

## CMake (optional twin library)

CPU-default (matches CI):

```bash
cmake -S . -B build -DPARCAE_BUILD_TESTS=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

With CUDA twins:

```bash
cmake -S . -B build-cuda -A x64 ^
  -DPARCAE_BUILD_CUDA=ON ^
  -DPARCAE_BUILD_TESTS=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae_tests
```

On single-config generators, drop `-A x64` and pass `-DCMAKE_BUILD_TYPE=Release`.  
Do **not** pin `-G "Visual Studio 17 2022"` on machines that only have VS 2026 — let CMake pick the installed generator (or use `-G "Visual Studio 18 2026"`).

Targets:

| Target | Role |
|--------|------|
| `parcae_cuda` / alias `parcae::cuda` | Static twin library |
| `parcae_tests` | Links `parcae::cuda` when the target exists; gets `PARCAE_HAS_CUDA=1` |

### Catch2 tags (CUDA)

| Tag | Without Toolkit (`PARCAE_BUILD_CUDA=OFF`) | With CUDA ON + GPU |
|-----|-------------------------------------------|--------------------|
| `[cuda][buffer]` | Skip / SUCCEED marker | DeviceBuffer H2D/D2H |
| `[cuda][params]` | Host POD/JSON tests always run | same |
| `[cuda][interrupt]` | Host interrupt view always run | same |
| `[cuda][backend]` | Validation always; catalog run when CUDA ON | Full `CudaBackend` dispatch vs CPU |
| `[cuda][smoke]` | Skip / SUCCEED marker | `IdentityCopy` H2D → kernel → D2H |
| `[cuda][dsl][smoke]` | Emit-text + cuda_mirror always; device skip | `DslSmokeCaesarKernel` vs `DslIrApplicator` |
| `[cuda][parity][atbash]` | Skip / SUCCEED marker | `AtbashKernel` vs CPU `AtbashTransform::kernel` |
| `[cuda][parity][caesar]` | Skip / SUCCEED marker | `CaesarKernel` vs CPU `CaesarTransform::kernel` |
| `[cuda][parity][identity]` | Skip / SUCCEED marker | `CudaBackend` identity vs CPU `IdentityTransform` |
| `[cuda][parity][affine]` | Host inv table always; kernel skip without CUDA | `AffineKernel` + `Z29Device::inv` vs CPU |
| `[cuda][parity][vigenere]` | Skip / SUCCEED marker | `VigenereKeyKernel` vs CPU (bitmask + sorted skips) |
| `[cuda][parity][beaufort]` | Skip / SUCCEED marker | `BeaufortKeyKernel` vs CPU (involution + skips) |
| `[cuda][parity][totient]` | Skip / SUCCEED marker | `TotientPrimeStreamKernel` vs CPU (host shifts) |
| `[cuda][parity][compose]` | Skip / SUCCEED marker | `ComposeDriver` vs CPU (Koan-1 / ping-pong) |
| `[cuda][parity][golden]` | CPU locks `data/parity/` hashes; CUDA apply skipped | `CudaBackend` vs golden `output_sha256` (manifest) |
| `[cuda][parity][solved]` | CPU apply on locked fixture consumables | `CudaBackend` vs CPU `output_sha256` per solved page |
| `[cuda][score][exact]` | Skip / SUCCEED marker | `ExactMatchScore` vs CPU `ExactMatch` |
| `[cuda][score][hamming]` | Skip / SUCCEED marker | `HammingAgreementScore` vs CPU `HammingAgreement` |
| `[cuda][score][ic]` | Skip / SUCCEED marker | `IcMod29Score` vs CPU `IcMod29` |
| `[cuda][score][self_repeat]` | Skip / SUCCEED marker | `SelfRepeatRateScore` vs CPU `SelfRepeatRate` |
| `[cuda][score][chi2]` | Skip / SUCCEED marker | `Chi2EnglishGpScore` vs CPU `Chi2EnglishGp` |
| `[cuda][score][dispatch]` | Catalog / unavailable always | `CudaScore` vs `ScoreRegistry` for all Tier A ids |
| `[cuda][score][noise][suite]` | Catalog always; noise skip without CUDA | Plaintext vs LCG noise separation via `CudaScore` |
| `[cuda][batch][score]` | Skip / SUCCEED marker | `CudaBatchScore` vs serial `BatchRunner` top-k |
| `[cuda][batch][rank1]` | Skip / SUCCEED marker | CUDA batch apply + exact_match recovers known params |
| `[cuda][batch][chi2][fuse]` | Skip / SUCCEED marker | Fused Caesar χ² vs CPU `ScoreRegistry` |
| `[cuda][property]` | Skip / SUCCEED marker | Encrypt↔decrypt round-trips + CPU byte match via `CudaBackend` |

### Catch2 tags (smart DSL compiler)

Scope-aware HotLoop / OuterControl compiler (`include/parcae/dsl/`). Hosted CI
adds an explicit **`Gate [dsl-smart]`** filter (see [CI policy](#ci-policy));
existing `[dsl][examples]` / `[dsl][registry][stale]` gates stay. Device emit
smoke remains under `[cuda][dsl][smoke]` above.

| Tag | Without Toolkit (`PARCAE_BUILD_CUDA=OFF`) | Notes |
|-----|-------------------------------------------|-------|
| `[dsl][scope]` | Always — **part of Gate [dsl-smart]** | `DslExecScope` / `DslScopeAnalyzer`; also `[dsl][gate][scope]` |
| `[dsl][divergence]` | Always — **part of Gate [dsl-smart]** | `DslDivergenceGate` E033 / W011 |
| `[dsl][directive]` / `[dsl][directives]` | Always — **part of Gate [dsl-smart]** | `DslDirectiveTable` W010 / E031; `--allow-dsl-ignores` |
| `[dsl][golden]` | Always (overlaps smart filter via subtags) | Smart-compiler acceptance matrix |
| `[dsl][hostglue]` | Always | OuterControl for/while → HostGlueIr / E035 |
| `[dsl][build][select]` / `[dsl][emit][select]` | Always | HotLoop If → Select; CPU/CUDA emit |
| `[dsl][examples]` | Always — **separate CI gate** | Compile `theories/examples` |
| `[dsl][registry][stale]` | Always — **separate CI gate** | Reject stale `dsl_spec_version` fixture |

Exit freeze: [`dsl-console-exit.md`](dsl-console-exit.md). Authoring guide:
[`python-transpiler.md`](python-transpiler.md).

### Catch2 tags (console progress / dashboard)

Human stderr progress (`ConsoleDashboard`) shared by `parcae-search-cycle`,
`SearchRunConsole`, and BlindCrack. Hosted CI **`Gate [cli-progress]`** OR-filter
below. Digests / `--json` stdout must stay invariant under `--quiet`.

| Tag | Without Toolkit (`PARCAE_BUILD_CUDA=OFF`) | Notes |
|-----|-------------------------------------------|-------|
| `[cli][dashboard]` | Always — **part of Gate [cli-progress]** | Panel / lines / off; VT → lines fallback |
| `[tool][search_cycle][progress]` | Always — **part of Gate [cli-progress]** | Quiet / plain / digest determinism |
| `[search][scheduler][loop][progress]` | Always (also under `[search]` gate) | Library progress sink wiring |

Operator notes: [`search-handbook.md`](search-handbook.md) § Console progress.

### Catch2 tags (bench / diagnostics)

Umbrella CLI `parcae-bench` (`include/parcae/bench/`). Hosted CI **`Gate [bench]`**
OR-filters the **deterministic** tags below — schema, probe parse, accuracy smoke,
and report JSON stability. Absolute runes/s / fused CUDA SLO are **not** a hosted
gate (non-deterministic timing); run those locally with a CUDA build.

| Tag | Without Toolkit (`PARCAE_BUILD_CUDA=OFF`) | Notes |
|-----|-------------------------------------------|-------|
| `[bench][spec]` | Always — **part of Gate [bench]** | `BenchTierSpec` C/T/reps + peak sync vs `DslPeakSanity` |
| `[bench][probe]` | Always — **part of Gate [bench]** | Probe JSON 1.0.0 parse + spawn/timeout (`docs/spec/bench-probe.md`) |
| `[bench][accuracy]` | Always — **part of Gate [bench]** | Accuracy-suite CPU smoke; CUDA rows skip when no device |
| `[bench][report]` | Always — **part of Gate [bench]** | `BenchReport` / `BenchFormatter`; `--omit-timing` digests |
| `[bench][metric]` | Always (not in hosted gate) | Rate formulas / median-of-3 math |
| `[bench][timer]` / `[bench][timer][cuda]` | Always / CUDA when linked | Timing protocol (not hosted SLO gate) |
| `[bench][slo]` / `[bench][slo][cuda]` | CPU error path / GPU when linked | Fused SLO T1–T3 — **local/CUDA only** |
| `[bench][hardware]` | Always (not in hosted gate) | CPU vs CUDA compare; skip / require-cuda paths |

Operator / contributor suite map: [`include/parcae/bench/README.md`](../../include/parcae/bench/README.md).
CLI contract: [`docs/spec/tools.md`](../spec/tools.md) § `parcae-bench`. Throughput
recalibration: [`cuda-throughput.md`](cuda-throughput.md).

```bash
# Hosted CI matrix — Gate [bench] (deterministic only)
build/tests/Release/parcae_tests.exe "[bench][spec],[bench][probe],[bench][accuracy],[bench][report]"

# Broader local smoke (includes metric/timer/hardware; still CPU-safe)
ctest --test-dir build -C Release -R "bench|cli_bench" --output-on-failure

# Full fused SLO (CUDA build + GPU; not hosted)
build-cuda/tools/Release/parcae-bench.exe --suite slo --extended --allow-cuda
```

### Catch2 tags (search engine / scheduler)

Closed-loop search (`include/parcae/search/`, `parcae-search-cycle`). Hosted CI
gates the full filter `[search]` (see [CI policy](#ci-policy)). CUDA-linked export
parity cases skip or stub when `PARCAE_BUILD_CUDA=OFF`.

| Tag | Without Toolkit (`PARCAE_BUILD_CUDA=OFF`) | With CUDA ON + GPU |
|-----|-------------------------------------------|--------------------|
| `[search]` | **CI matrix gate** — full CPU search suite | Same + device export/parity cases run |
| `[search][job]` | Always | `SearchJob` parse / bounds / digests |
| `[search][prior]` | Always | `SearchPrior` from workspace / JSON |
| `[search][batch]` | Always | `BatchArtifact` store/load / rank contract |
| `[search][batch][limits]` | Always | Line-size / `k` caps + best-first reject |
| `[search][batch][fuzz]` | Always | `BatchOrdering` shuffle+sort determinism |
| `[search][cipher]` / `[search][cipher][resolve]` | Always | `WorkspaceCipher` fixture / `workspace_file` / path escape |
| `[search][export]` / `[search][export][cpu]` | Always (CPU export) | + fused/`[cuda]` export when linked |
| `[search][export][parity]` | CPU top-k ids always | CPU↔CUDA top-k on Tier-A (`[cuda]` / `[a-warning]`) |
| `[search][export][compose][parity]` | CPU compose smoke always | CPU vs fused Atbash∘Caesar / ComposeDriver (`[cuda]` when linked) |
| `[search][bridge]` / `[search][bridge][score]` | Always | `HypothesisBridge` ingest / score / status |
| `[search][scheduler]` | Always (CPU `run_once`) | CUDA backend rejected unless built+allowed |
| `[search][scheduler][loop]` | Always | `run_loop` budgets / stop reasons |
| `[search][scheduler][prior]` | Always | Promoted seeds + rejected exclusions on next job |
| `[search][scheduler][loop][determinism]` | Always | Two-iteration fixed-seed digest stability |
| `[search][adversarial]` | Always | Job JSON / path escape / `max_candidates` caps |
| `[search][roundtrip]` | Always | Job + prior + batch end-to-end |
| `[tool][search_cycle]` | Always (needs built CLI) | `parcae-search-cycle` status / run / omit-timing |
| `[tool][golden][cli][search_cycle]` | Always | JSON golden envelopes |
| `[tool][policy][cli][search_cycle]` | Always | AgentPolicy allow path for cycle CLI |
| `[run][search]` | Always (CPU metrics path) | SearchRun throughput / fixture eval; CUDA family parity when built |

Operator / contributor tag map also lives in
[`include/parcae/search/README.md`](../../include/parcae/search/README.md) and
[`search-handbook.md`](search-handbook.md).

```bash
# CPU CI path — smoke/buffer markers must still pass
build/tests/Release/parcae_tests.exe "[cuda]"

# Search-engine gate (hosted CI matrix — no GPU required)
build/tests/Release/parcae_tests.exe "[search]"
build/tests/Release/parcae_tests.exe "[search][scheduler]"

# Smart DSL + console progress (hosted CI matrix — Gate [dsl-smart] / [cli-progress])
build/tests/Release/parcae_tests.exe "[dsl][scope],[dsl][divergence],[dsl][directive],[dsl][directives]"
build/tests/Release/parcae_tests.exe "[cli][dashboard],[tool][search_cycle][progress]"

# Bench diagnostics (hosted CI matrix — Gate [bench]; no absolute runes/s)
build/tests/Release/parcae_tests.exe "[bench][spec],[bench][probe],[bench][accuracy],[bench][report]"

# Real device smoke (CUDA build)
build-cuda/tests/Release/parcae_tests.exe "[cuda][smoke]" -s
build-cuda/tests/Release/parcae_tests.exe "[search][export][parity]" -s
build-cuda/tools/Release/parcae-search-run.exe --backend cuda --family caesar
```

If `PARCAE_BUILD_CUDA=ON` but no nvcc is found, configure **fails** with a clear error (no silent half-build).

## CI policy

CPU-default workflows under [`.github/workflows/`](../../.github/workflows/):

| Workflow | Role |
|----------|------|
| [`ci.yml`](../../.github/workflows/ci.yml) | Matrix: Ubuntu GCC/Clang, macOS, Windows — full `ctest` + `[solved]` / `[parity]` / `[search]` / `[cuda]` host gates + **`[dsl-smart]`** / **`[cli-progress]`** / **`[bench]`** + `[dsl][examples]` / `[dsl][registry][stale]`; ASan+UBSan job; `parity-goldens` regen+byte-compare; optional self-hosted CUDA via `workflow_dispatch` |
| [`clang-format.yml`](../../.github/workflows/clang-format.yml) | `clang-format --dry-run --Werror` on `include/`, `tests/`, `tools/`, `Parcae/Parcae/cuda/` |
| [`codeql.yml`](../../.github/workflows/codeql.yml) | CodeQL C/C++ analysis (PR + weekly) |

Rules:

- **No** `-DPARCAE_BUILD_CUDA=ON` on hosted runners
- Host-side `[cuda][params]` / `[cuda][interrupt]` / `[cuda][backend]` still compile and run
- Device tests compile only as skip stubs when `PARCAE_HAS_CUDA` is unset
- GPU smoke: Actions → CI → Run workflow → `run_cuda=true` on a self-hosted Toolkit+GPU runner

## Layout reminder

```text
Parcae/Parcae/cuda/     canonical .cu / CUDA headers (VS + optional CMake)
include/parcae/         CPU reference (header-only) — unchanged
include/parcae/bench/   benchmark & diagnostics (`parcae-bench` / Gate [bench])
include/parcae/search/  closed-loop SearchScheduler / BatchArtifact / export
include/parcae/run/     SearchRun metrics / console / CUDA sweeps
```

ABI: [cuda-abi.md](cuda-abi.md) · Twins: [cuda-handoff.md](cuda-handoff.md) ·
Reference: [cuda-reference.md](cuda-reference.md) · Throughput: [cuda-throughput.md](cuda-throughput.md) ·
Roadmap: [cuda-roadmap.md](cuda-roadmap.md) · Search: [search-engine.md](search-engine.md) /
[search-handbook.md](search-handbook.md)
