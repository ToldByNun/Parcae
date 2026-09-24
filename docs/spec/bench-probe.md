# Spec: Bench external probe protocol v1.0.0

**Status:** Normative (toolkit bench / diagnostics)  
**Schema id:** `probe_schema_version` = `"1.0.0"`  
**Consumers:** `parcae-bench --suite probe`, `BenchProbeProtocol`, `BenchProbeRunner`  
**Related:** [`../architecture/cuda-throughput.md`](../architecture/cuda-throughput.md),
[`BenchTierSpec`](../../include/parcae/bench/bench_tier_spec.hpp),
[`tools.md`](tools.md)

This document is the **only** normative wire contract for **external** Cicada /
third-party throughput+accuracy probes. Parcae does **not** vendor foreign
repos; operators supply binaries and a `--probe-cmd` template.

## Principles

1. Parcae **orchestrates** (spawn, timeout, parse, report). The child tool owns
   measurement and accuracy claims.
2. Child **stdout** MUST be exactly **one** JSON object (UTF-8). Extra leading /
   trailing whitespace is allowed; multiple top-level values are **not**.
3. Child **stderr** is human diagnostics only — MUST NOT be required to parse
   the result.
4. `{tier}` in `--probe-cmd` is substituted with a primary tier id (`T1` / `T2` /
   `T3`) before spawn. No other placeholders are defined in 1.0.0.
5. Rates use the same definitions as internal SLO benches:
   - **runes/s** = `reps × C × T / elapsed` (setup excluded)
   - **keys/s** = `reps × C / elapsed`
6. Schema drift is mitigated by a hard `probe_schema_version` check.

## Invocation

```text
parcae-bench --suite probe \
  --probe-cmd "path/to/tool --bench-json --tier {tier}" \
  --probe-tiers T1,T2,T3 \
  --probe-timeout-ms 120000 \
  [--compare-builtin] \
  [--json] [--omit-timing] \
  --data-dir data
```

| Flag | Contract |
|------|----------|
| `--probe-cmd` | Required for `--suite probe`. Opaque command string; runner spawns via platform shell (`cmd /C` on Windows, `/bin/sh -c` on POSIX). |
| `--probe-tiers` | Comma-separated primary ids. Default: `T1,T2,T3`. Unknown ids are a usage error. |
| `--probe-timeout-ms` | Wall-clock cap per tier spawn. Default: **120000**. On timeout the child is terminated and the tier row **fails**. |
| `--compare-builtin` | After a valid probe, compare `config.{C,T,reps}` to `BenchTierSpec` for that tier and attach `gpu/cpu`-style ratio vs `slo_min` in `detail`. Config mismatch → **fail**. |
| `--suite all` | Order: accuracy → slo → hardware → probe. **Probe runs only if `--probe-cmd` is set.** |

`--data-dir` applies to Parcae context (profiles / fixtures). Forwarding it into
the child is operator-defined (include it in `--probe-cmd` if needed).

## Response JSON (stdout)

### Example (T1)

```json
{
  "probe_schema_version": "1.0.0",
  "tool_id": "example-cicada-tool",
  "tier": "T1",
  "runes_per_sec": 1.25e9,
  "keys_per_sec": 4.3e7,
  "wall_seconds": 0.412,
  "backend": "cpu",
  "accuracy": {
    "oracle_cracked": true,
    "top_rank": 1,
    "notes": ""
  },
  "config": { "C": 29, "T": 1048576, "reps": 64 }
}
```

### Required fields

| Field | Type | Rule |
|-------|------|------|
| `probe_schema_version` | string | MUST be `"1.0.0"` |
| `tool_id` | string | Non-empty |
| `tier` | string | MUST equal the substituted tier id for that spawn |
| `runes_per_sec` | number | Finite, ≥ 0 |
| `keys_per_sec` | number | Finite, ≥ 0 |
| `wall_seconds` | number | Finite, ≥ 0 |
| `backend` | string | One of `cpu`, `cuda`, `both`, `unknown` |
| `accuracy` | object | See below |
| `config` | object | See below |

### `accuracy` object

| Field | Type | Rule |
|-------|------|------|
| `oracle_cracked` | boolean | Required |
| `top_rank` | integer | Required, ≥ 0 |
| `notes` | string | Required (MAY be empty) |

### `config` object

| Field | Type | Rule |
|-------|------|------|
| `C` | integer | Candidates / key lanes; ≥ 1 |
| `T` | integer | Stream length in Index29 runes; ≥ 1 |
| `reps` | integer | Timed inner-loop repetitions; ≥ 1 |

When `--compare-builtin` is set, `C` / `T` / `reps` MUST match
`BenchTierSpec` for `tier` (T1: 29 / 1048576 / 64; T2: 4096 / 262144 / 8;
T3: 512 / 262144 / 8).

### Child process exit

A conforming tool SHOULD exit **0** when stdout carries a valid 1.0.0 object.
`BenchProbeRunner` MUST still attempt to parse stdout when the exit code is
non-zero; parse failure or timeout → failed probe row. Exit code alone does
**not** override a successfully parsed payload (operators may surface exit in
`detail`).

## Report mapping

Each tier becomes one `BenchReport::Row` with `suite = probe`:

- `name` ← `tier`
- `workload` ← `tool_id` (plus accuracy summary in `detail`)
- `backend` ← mapped from probe `backend`
- `runes_per_sec` / `keys_per_sec` / `wall_seconds` ← probe fields
- `candidates` / `tokens` / `repeats` ← `config.C` / `T` / `reps`
- `status` ← `pass` if parse (+ optional builtin compare) succeeds; else `fail`

Human tables use the existing **PROBE** section in `BenchFormatter`.

## Skip / exit codes

| Situation | Behavior |
|-----------|----------|
| `--suite probe` without `--probe-cmd` | Usage error (exit **2**) |
| `--suite all` without `--probe-cmd` | Omit probe leg (not a failure) |
| Timeout / spawn failure / invalid JSON / schema reject | Probe row **fail**; suite exit **1** if any fail |
| All probe rows pass | Exit **0** |

Hardware `--allow-skip` / `skipped_not_built` does **not** apply to probes.

## Non-goals (1.0.0)

- No JSON request body on stdin
- No multi-object NDJSON streams
- No vendored third-party tool binaries
- No hosted-CI absolute runes/s gates for external tools
