# Bench & diagnostics operator handbook

**Status:** Operator guide (toolkit **0.9.0** / `parcae-bench`)  
**Canonical specs:** [`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp)  
**CLI contract:** [`docs/spec/tools.md`](../spec/tools.md) § `parcae-bench`  
**Probe wire format:** [`docs/spec/bench-probe.md`](../spec/bench-probe.md)  
**Exit freeze:** [`bench-exit.md`](bench-exit.md)  
**Throughput ceilings:** [`cuda-throughput.md`](cuda-throughput.md)

`parcae-bench` is the umbrella **verification + diagnostics** CLI: fused CUDA
SLO tiers, statistical accuracy checks, CPU↔GPU hardware compare, and optional
external JSON probes. Absolute runes/s are **local/CUDA** measurements — not a
hosted-CI absolute gate.

```text
parcae-bench --status --json
parcae-bench --suite slo|accuracy|hardware|probe|all …
```

---

## Suites at a glance

| Suite | Purpose | Hosted CI |
|-------|---------|-----------|
| **slo** | Fused CUDA T1–T3 vs `BenchTierSpec` floors + ≥90% peak (`--extended` → F.*/C.*) | **Not** absolute runes/s; device required + `--allow-cuda` |
| **accuracy** | `A.fixture_eval` / `A.chi2_sanity` / `A.oracle_rank`; optional CUDA planted/parity | Catch2 `[bench][accuracy]` in **`Gate [bench]`** |
| **hardware** | Same T1–T3 IDs, CPU vs CUDA; `gpu/cpu` in detail | Smoke locally; not in hosted gate filter |
| **probe** | External tool JSON 1.0.0 via `--probe-cmd` (`{tier}`) | Parse/spawn unit tests in **`Gate [bench]`** |
| **all** | accuracy → slo → hardware → probe | Probe only if `--probe-cmd`; SLO only with usable CUDA |

Compat: `parcae-throughput-tiers` ≡ `--suite slo --extended --allow-cuda`.

Header map: [`include/parcae/bench/README.md`](../../include/parcae/bench/README.md).

---

## Operator recipes

### Status / agent-safe digests

```bash
parcae-bench --status --json --data-dir data
parcae-bench --suite accuracy --json --omit-timing --data-dir data
```

`--omit-timing` requires `--json` and drops rate/wall fields (stable digests).

### CUDA SLO (local GPU)

```bash
cmake -S . -B build-cuda -DPARCAE_BUILD_CUDA=ON -DPARCAE_BUILD_TOOLS=ON
cmake --build build-cuda --config Release --target parcae-bench
./build-cuda/tools/Release/parcae-bench --suite slo --extended --allow-cuda --data-dir data
```

Pass rule: `BenchTierSpec::pass_tier` (SLO floor **and** ≥90% practical peak).
Recalibrate peaks via [`cuda-throughput.md`](cuda-throughput.md).

### Hardware compare (CPU smoke + optional CUDA)

```bash
# CI-friendly CPU rows (scaled smoke)
parcae-bench --suite hardware --backend cpu --json --omit-timing --data-dir data

# Side-by-side when a device is present
parcae-bench --suite hardware --backend both --allow-cuda --data-dir data

# Fail hard if CUDA missing
parcae-bench --suite hardware --backend cuda --require-cuda --data-dir data

# OK to skip CUDA rows
parcae-bench --suite hardware --backend cuda --allow-cuda --allow-skip --data-dir data
```

CUDA unavailable → rows with `status=skipped`, detail `skipped_not_built`
(does **not** fail `all_pass` unless `--require-cuda`).

### External probes

```bash
parcae-bench --suite probe \
  --probe-cmd "path/to/tool --bench-json --tier {tier}" \
  --probe-tiers T1,T2,T3 \
  --probe-timeout-ms 120000 \
  [--compare-builtin] \
  --data-dir data
```

See [`bench-probe.md`](../spec/bench-probe.md) for the 1.0.0 stdout JSON contract.
Timeouts kill the child; parse/schema failures fail the probe row.

---

## Agent policy

`bench` / `parcae-bench` (and `throughput-tiers` / `parcae-throughput-tiers`) are
**deny-listed** in `AgentPolicy` and `agents/parcae_agent/allowlist.py`. Agents
use `search_cycle` for workspace research; operators run bench from a shell.
Deny rationale: non-deterministic timing SLOs / diagnostics — see
[`agent-tools.md`](../spec/agent-tools.md).

---

## Catch2 / CI map

Hosted **`Gate [bench]`** (`.github/workflows/ci.yml`) runs only:

```text
[bench][spec],[bench][probe],[bench][accuracy],[bench][report]
```

| Tag | Hosted? | Notes |
|-----|---------|-------|
| `[bench][spec]` | yes | `BenchTierSpec` sync |
| `[bench][probe]` | yes | Probe 1.0.0 parse + spawn |
| `[bench][accuracy]` | yes | Accuracy CPU smoke |
| `[bench][report]` | yes | `--omit-timing` digests |
| `[bench][metric]` / `[bench][timer]` | no | Timing helpers |
| `[bench][slo]` / `[bench][slo][cuda]` | no | Absolute SLO / GPU |
| `[bench][hardware]` | no | CPU vs CUDA compare |

Full table: [`cuda-build.md`](cuda-build.md) § Catch2 tags (bench / diagnostics).

Local broader smoke:

```bash
ctest --test-dir build -C Release -R "bench|cli_bench" --output-on-failure
```

---

## Exit codes

| Code | Meaning |
|------|---------|
| **0** | All report rows pass (skipped CUDA accuracy/hardware rows do not fail) |
| **1** | Measurement / validation / probe failure |
| **2** | Usage / policy (missing `--allow-cuda`, bad `--suite`, missing `--probe-cmd`, …) |

---

## Related

- Exit checklist: [`bench-exit.md`](bench-exit.md)
- Release cut procedure: [`release.md`](release.md)
- Prior exit pattern: [`dsl-console-exit.md`](dsl-console-exit.md) / [`search-engine.md`](search-engine.md)
