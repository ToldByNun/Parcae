# Bench headers (`include/parcae/bench/`)

C++20 surface for Parcae **benchmark & diagnostics** (toolkit 0.9.0 target).

| Doc | Role |
|-----|------|
| [`docs/architecture/cuda-throughput.md`](../../../docs/architecture/cuda-throughput.md) | Operator SLO / peak recalibration |
| [`docs/spec/bench-probe.md`](../../../docs/spec/bench-probe.md) | External probe JSON 1.0.0 |
| [`docs/spec/tools.md`](../../../docs/spec/tools.md) § `parcae-bench` | CLI contract |
| [`docs/spec/agent-tools.md`](../../../docs/spec/agent-tools.md) | **Deny-list** — agents MUST NOT call `parcae-bench` |

## Style (HARD)

Top-level classes only — **no** C++ namespaces. One class per header:

```cpp
#ifndef NAME_HPP
#define NAME_HPP
class Name {
public:
  // ...
private:
  Name() = delete; // when static-only
};
#endif // NAME_HPP
```

## Header map

| Header | Class | Status |
|--------|-------|--------|
| `bench_tier_spec.hpp` | `BenchTierSpec` | Done — canonical T1–T3 C/T/reps + peak/SLO tables |
| `bench_metric.hpp` | `BenchMetric` | Done — runes/s, keys/s, median-of-3 |
| `bench_timer.hpp` | `BenchTimer` | Done — CPU steady_clock + CUDA cudaEvent protocol |
| `bench_report.hpp` | `BenchReport` | Done — unified rows + JSON `--omit-timing` |
| `bench_formatter.hpp` | `BenchFormatter` | Done — human tables (ConsoleDashboard vocab) |
| `bench_slo_suite.hpp` | `BenchSloSuite` | Done — T1–T3 (+ optional F.*/C.*) via ThroughputTiers |
| `bench_accuracy_suite.hpp` | `BenchAccuracySuite` | Done — fixture_eval / chi2 / oracle / CUDA planted |
| `bench_hardware_suite.hpp` | `BenchHardwareSuite` | Done — T1–T3 CPU vs CUDA; skip / require-cuda |
| `bench_config.hpp` | `BenchConfig` | Done — probe cmd / tiers / timeout / compare |
| `bench_probe_protocol.hpp` | `BenchProbeProtocol` | Done — probe_schema_version 1.0.0 parse |
| `bench_probe_runner.hpp` | `BenchProbeRunner` | Done — `{tier}` spawn + timeout → report |
| CLI `parcae-bench` | — | `--status`, `--suite slo\|accuracy\|hardware\|probe\|all` |

## Suites (`parcae-bench --suite …`)

| Suite | What it measures | Default / gates |
|-------|------------------|-----------------|
| **slo** | Fused CUDA throughput vs `BenchTierSpec` floors + ≥90% peak | Requires `--allow-cuda` + device; `--extended` adds F.* / C.* |
| **accuracy** | Statistical checks (`A.fixture_eval`, `A.chi2_sanity`, `A.oracle_rank`; optional CUDA `A.fused_parity` / planted) | CPU always; CUDA extras need `--allow-cuda` |
| **hardware** | Same T1–T3 IDs, CPU vs CUDA side-by-side; `gpu/cpu` in detail | CPU smoke scaled unless `--cpu-full`; CUDA: `--allow-cuda` / `--require-cuda` / `--allow-skip` |
| **probe** | External tool JSON 1.0.0 via `--probe-cmd` (`{tier}` substituted) | Requires `--probe-cmd`; timeout default 120000 ms; see `bench-probe.md` |
| **all** | accuracy → slo → hardware → probe | Probe only if `--probe-cmd`; SLO only with usable CUDA + `--allow-cuda` |

Compat: `parcae-throughput-tiers` ≡ `--suite slo --extended --allow-cuda`.

**Agent policy:** `bench` / `parcae-bench` (and `throughput-tiers` / `parcae-throughput-tiers`) are on `AgentPolicy::default_deny_binaries()` and `agents/parcae_agent/allowlist.py` `DENIED_BINARIES`. Operators run from a shell; agents use `search_cycle`.

## Sync rule

`BenchTierSpec` is the **single source of truth** for:

- Primary SLO configs (T1 / T2 / T3): `C`, `T`, `reps`, display bands, peaks
- Extended peak / floor ids (`F.*`, `C.*`)

`ThroughputTiers` and `DslPeakSanity` **delegate** to this header. Catch2
`[bench][spec]` and `[dsl][peak]` lock the numbers.
