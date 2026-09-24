# Bench headers (`include/parcae/bench/`)

C++20 surface for Parcae **benchmark & diagnostics** (toolkit 0.9.0 target).
Normative operator notes: [`docs/architecture/cuda-throughput.md`](../../../docs/architecture/cuda-throughput.md).

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

## Current scaffold

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
| CLI `parcae-bench` | — | `--status`, `--suite slo|accuracy|hardware` |

## Sync rule

`BenchTierSpec` is the **single source of truth** for:

- Primary SLO configs (T1 / T2 / T3): `C`, `T`, `reps`, display bands, peaks
- Extended peak / floor ids (`F.*`, `C.*`)

`ThroughputTiers` and `DslPeakSanity` **delegate** to this header. Catch2
`[bench][spec]` and `[dsl][peak]` lock the numbers.
