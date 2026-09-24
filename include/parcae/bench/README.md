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

## Sync rule

`BenchTierSpec` is the **single source of truth** for:

- Primary SLO configs (T1 / T2 / T3): `C`, `T`, `reps`, display bands, peaks
- Extended peak / floor ids (`F.*`, `C.*`)

`ThroughputTiers` and `DslPeakSanity` **delegate** to this header. Catch2
`[bench][spec]` and `[dsl][peak]` lock the numbers.
