# Spec: Theory Artifact Manifest

**Status:** Normative  
**Schema id:** `parcae.theory_artifact.v0`  
**URI scheme:** `parcae://theories/<name>@<version>`  
**Related:** [dsl.md](dsl.md) (`dsl_spec_version`), [transforms.md](transforms.md),
[tools.md](tools.md),
[theory-hist-transpile.md](../architecture/theory-hist-transpile.md)
(stream twin vs fused hist Kernel SLO)

Compiled theories are **versioned, inspectable artifacts** under
`data/theories/`. They are the only form that validate/sweep/runtime dispatch
**MUST** treat as verified output of `parcae-compile`. Raw `.py` sources are
inputs, not runnable registry entries.

**Two CUDA products** may appear under one artifact URI — do **not** conflate them.
Throughput Done (≥90% shape peak) is defined on the **fused hist** product, not
the stream twin. See [theory-hist-transpile.md](../architecture/theory-hist-transpile.md).

| Product | Manifest paths (today / planned) | Producer | Consumer | Role |
|---------|----------------------------------|----------|----------|------|
| **Stream twin** | `paths.cuda_header` / `paths.cuda_source` under `emitted/` | `DslEmitCuda` @ compile | Transform / apply / smoke | Bit-identity 1D device apply |
| **Fused hist** | `paths.hist_*` under `hist/` (optional until writers land) | `TheoryHistChi2Emit` (+ future cubin tool) | Search `GpuCandidateExport` / module load | Fused decrypt+χ² Kernel SLO |

---

## Directory layout

```text
data/theories/
  <name>/
    <version>/                 # positive integer as decimal string, e.g. 1
      manifest.json            # parcae.theory_artifact.v0
      cpu_reference.*          # optional emitted CPU applicator / Transform text
      envelope.json            # optional TransformEnvelope bridge
      apply_ir.json            # optional TheoryApplyIr for TheoryDispatch
      emitted/                 # optional STREAM twin sources (DslEmitCuda)
        *Kernel.hpp
        *Kernel.cu
      hist/                    # optional FUSED-HIST search products (TheoryHistChi2Emit)
        hist_plan.json         # strategy + plans (S1/S2/…); see below
        *.hpp / *.cu           # optional specialized hist sources
        *.cubin / *.fatbin     # optional offline/NVRTC module bytes
      verify_report.json       # optional machine-readable gate log
```

| Rule | Requirement |
|------|-------------|
| `name` | MUST match `manifest.name`; recommended `[a-z][a-z0-9_]*` |
| `version` | Positive integer; MUST match URI `@<version>` and directory name |
| Encoding | UTF-8; JSON objects; LF preferred |
| Paths inside manifest | Relative to the artifact directory; MUST NOT escape via `..` |
| `emitted/` vs `hist/` | **MUST NOT** mix roles: stream kernels stay under `emitted/`; fused-χ² hist under `hist/` |
| Missing `hist/` | Allowed — search MAY classify/emit at runtime and soft-fallback S0 ([theory-hist-transpile.md](../architecture/theory-hist-transpile.md)) |

Runtime trees under `data/theories/` **SHOULD** be gitignored except committed
golden/example artifacts explicitly allow-listed by the project.

---

## URI scheme

```text
parcae://theories/<name>@<version>
```

| Part | Rule |
|------|------|
| Scheme | MUST be `parcae` |
| Path | MUST be `theories/<name>@<version>` |
| `name` | Non-empty; MUST equal `manifest.name` |
| `version` | Decimal positive integer; MUST equal `manifest.version` |

Examples:

```text
parcae://theories/quadratic_polynomial_stream@1
parcae://theories/my_affine@3
```

Resolvers **MUST** map the URI to
`data/theories/<name>/<version>/manifest.json` (or a configured theories root
with the same layout). Unknown URI → hard error.

---

## Manifest — `parcae.theory_artifact.v0`

File: `data/theories/<name>/<version>/manifest.json`

```json
{
  "schema": "parcae.theory_artifact.v0",
  "uri": "parcae://theories/quadratic_polynomial_stream@1",
  "name": "quadratic_polynomial_stream",
  "version": 1,
  "dsl_spec_version": "1.0.0",
  "compiler_version": "0.9.0",
  "source_path": "theories/examples/new_math_example.py",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "tier": "B",
  "family": "keyed_stream",
  "structural_claim": "Speculative. …",
  "params": [
    { "name": "c2", "min": 0, "max": 28 },
    { "name": "c1", "min": 0, "max": 28 },
    { "name": "c0", "min": 0, "max": 28 }
  ],
  "primitives": ["poly2_mod29"],
  "verification": {
    "mode": "exhaustive",
    "passed": true,
    "seed": null,
    "completed_utc": "2026-09-20T03:00:00Z"
  },
  "fusion": {
    "status": "n/a"
  },
  "paths": {
    "cpu_reference": "cpu_reference.hpp",
    "cuda_header": "emitted/QuadraticPolynomialStreamKernel.hpp",
    "cuda_source": "emitted/QuadraticPolynomialStreamKernel.cu",
    "envelope_template": "envelope.json",
    "apply_ir": "apply_ir.json",
    "verify_report": "verify_report.json",
    "hist_plan": null,
    "hist_header": null,
    "hist_source": null,
    "hist_module": null
  },
  "hist": null,
  "sweep": null,
  "interrupts": {
    "mode": "policy_method"
  }
}
```

### Required fields

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.theory_artifact.v0` |
| `uri` | MUST equal `parcae://theories/{name}@{version}` |
| `name` | MUST equal parent `name` directory |
| `version` | Positive integer; MUST equal version directory |
| `dsl_spec_version` | SemVer string; see [dsl.md](dsl.md) policy |
| `compiler_version` | Toolkit version string that produced the artifact |
| `source_sha256` | Lowercase hex SHA-256 of the exact source bytes compiled |
| `tier` | `"A"` \| `"B"` \| `"C"` |
| `family` | DSL family id from [dsl.md](dsl.md) |
| `params` | Array of `{name, min, max}` (may be empty for compose-only shells) |
| `primitives` | Array of primitive name strings (may be empty if only composing catalog ids) |
| `verification` | Object; see below |
| `fusion` | Object; see below |
| `paths` | Object; keys MAY be null if not emitted |

### Optional / conditional fields

| Field | Rule |
|-------|------|
| `source_path` | Repo-relative path of the `.py` input when known |
| `dsl_ignores_applied` | Optional array of honored `#ignore DSL_FLAG` names (review / digest) |
| `structural_claim` | **Required** in manifest when `tier` is `B` or `C` (non-empty string) |
| `sweep` | `null` or a sweep config object (see below) |
| `interrupts` | Documents interrupt mode: `policy_method` \| `none_by_design` \| `elementwise_default` |
| `hist` | `null` or fused-hist summary object (see § `hist` below); independent of stream `emitted/` |

### `paths` — stream twin vs fused hist

Existing keys (stream / apply product):

| Key | Product | Meaning |
|-----|---------|---------|
| `cpu_reference` | CPU apply text | Optional host Transform / applicator source |
| `cuda_header` | **Stream twin** | Façade header under `emitted/` (`DslEmitCuda`) |
| `cuda_source` | **Stream twin** | Kernel `.cu` under `emitted/` |
| `envelope_template` | Envelope bridge | See § Envelope bridge |
| `apply_ir` | CPU/CUDA IR apply | `apply_ir.json` for `TheoryDispatch` |
| `verify_report` | Verify log | Optional |

Additive keys for **fused hist** (search χ²). Writers **MAY** omit them or set
`null` until hist persistence ships. Loaders **MUST** treat missing keys as
absent (forward-compatible with older manifests):

| Key | Product | Meaning |
|-----|---------|---------|
| `hist_plan` | Fused hist | Relative path to `hist/hist_plan.json` (strategy + plans) |
| `hist_header` | Fused hist | Optional specialized hist façade `.hpp` under `hist/` |
| `hist_source` | Fused hist | Optional specialized hist `.cu` under `hist/` |
| `hist_module` | Fused hist | Optional `.cubin` / `.fatbin` (or documented module blob) under `hist/` |

Rules:

- `paths.cuda_*` **MUST NOT** point into `hist/`; `paths.hist_*` **MUST NOT** point into `emitted/`.
- Presence of stream `cuda_source` does **not** imply a specialized search hist kernel exists.
- Presence of `hist_plan` without `hist_module` is valid: search MAY use in-lib twins (S1/S2) keyed by the plan, or soft-fallback S0.
- Kernel SLO Done is never inferred from stream-twin build success alone.

### `hist` (optional top-level summary)

When non-null, a compact digest of the fused-hist classify/emit result for tools
and humans (full detail lives in `hist_plan.json` when present):

```json
{
  "intended_strategy": "S3_scalar_inline",
  "emitted_strategy": "S0_bytecode",
  "specialized": false,
  "reason": "skeleton: S3 classified but emit not implemented; fallback S0",
  "shape_peak_tier": "T.theory.caesar_bytecode"
}
```

| Field | Rule |
|-------|------|
| `intended_strategy` | Classify result string (e.g. `S0_bytecode`, `S1_lut29`, `S2_linear`, `S3_scalar_inline`, …) |
| `emitted_strategy` | Strategy search will prefer if specialized; else soft-fallback id (often `S0_bytecode`) |
| `specialized` | `true` only when a specialized hist launch path is available for this artifact (in-lib plan and/or module) |
| `reason` | Non-empty diagnostic when `specialized` is false or intended ≠ emitted |
| `shape_peak_tier` | Optional `BenchTierSpec` id for PRIMARY ≥90% gate of the **emitted** strategy |

`parcae-compile` **SHOULD** eventually write `hist` + `paths.hist_plan` when
`TheoryHistChi2Emit` runs at compile time. Until then, `hist: null` is the
shipping default; search still classifies at runtime via `TheoryExportCache`.

### `hist/hist_plan.json` (`parcae.theory_hist_plan.v0`)

When `paths.hist_plan` is set, the file **MUST** be JSON:

```json
{
  "schema": "parcae.theory_hist_plan.v0",
  "theory_uri": "parcae://theories/quadratic_polynomial_stream@1",
  "intended_strategy": "S3_scalar_inline",
  "emitted_strategy": "S0_bytecode",
  "specialized": false,
  "cipher_var": "x",
  "reason": "…",
  "s1_lut": null,
  "s2_linear": null,
  "s3": null,
  "s4_autokey": null
}
```

| Field | Rule |
|-------|------|
| `schema` | MUST be `parcae.theory_hist_plan.v0` |
| `theory_uri` | MUST equal manifest `uri` |
| `intended_strategy` / `emitted_strategy` / `specialized` / `reason` | Same semantics as top-level `hist` |
| `cipher_var` | HotLoop cipher binding name (default `"x"`) |
| `s1_lut` | `null` or `{ "param_names": ["…"] }` matching `TheoryHistChi2Emit::S1LutPlan` |
| `s2_linear` | `null` or `{ "b0_name", "b1_name", "cipher_minus_ks": bool }` |
| `s3` / `s4_autokey` | `null` until those strategies persist plans |

Plan files are **search/hist** metadata — not a substitute for `apply_ir.json`.

### `verification`

| Field | Rule |
|-------|------|
| `mode` | `"exhaustive"` \| `"fuzz"` |
| `passed` | MUST be `true` for a written ready artifact; `parcae-compile` MUST NOT write ready artifacts with `passed: false` |
| `seed` | `null` for exhaustive; for fuzz MUST be the fixed seed used (`0xC1CADA` for v0 default) |
| `completed_utc` | RFC 3339 UTC timestamp |

### `fusion`

| `status` | Meaning |
|----------|---------|
| `n/a` | Not a compose chain |
| `fused` | Fused kernel selected; fused ≥ staged on gate bench |
| `fallback_staged` | Staged compose path selected because fused &lt; staged |

### `sweep` (when non-null)

Sweep metadata **MUST NOT** run automatically at compile time. It configures
`parcae-sweep` only.

| Field | Rule |
|-------|------|
| `theory` | MUST equal `name` |
| `corpus` | Corpus id; **MUST NOT** be a solved-oracle corpus used as a silent default for Tier B/C discovery claims |
| `param_grid` | JSON object describing grids (see below) |
| `record_metrics` | Array of metric names |
| `compare_against` | **Required** when `tier` is `B` or `C` (e.g. research anchor reference) |

Loaders **MUST** reject Tier B/C artifacts whose `sweep` object is present but
omits `compare_against`.

#### `param_grid` (v0)

Object keyed by theory param names. Every declared theory param **MUST** appear;
unknown keys **MUST** be rejected. Each value is one of:

| Form | Meaning |
|------|---------|
| `"full"` | Inclusive range of the declared param `min..max` |
| `[int, …]` | Explicit integer list (each in declared domain) |
| `{"min": i, "max": j}` | Inclusive sub-range within declared domain |
| `{"values": [int, …]}` | Same as an explicit list |

`parcae-sweep` expands the cartesian product (optional `--limit` truncation).
Apply/score of candidates is out of scope for the plan-only CLI until
TheoryDispatch (`TheoryDispatch` / `apply_ir.json`).

---

## Spec / artifact compatibility (hard)

Machines consuming artifacts **MUST** enforce [dsl.md](dsl.md) § `dsl_spec_version`
policy:

1. Persist `dsl_spec_version` at compile time (never omit).
2. On load/validate/sweep: MAJOR mismatch with running `DslSpecVersion::current`
   → **reject** with an explicit re-compile instruction.
3. Catalog listings **MUST** expose `stale_spec: true` when MAJOR mismatches.
4. Changing only `source_sha256` without bumping `version` **MUST NOT** occur for
   published URIs; new bytes ⇒ new `version` directory or overwrite only under
   explicit local rebuild of the same `@version` (tools **SHOULD** warn).

---

## Envelope bridge

When `paths.envelope_template` is set, the file **MUST** be a valid transform
envelope per [transforms.md](transforms.md) **or** a documented extension that
references `parcae://` theory ids. Tools that only understand catalog
`transform_id` values **MUST** fail clearly if the envelope cannot be lowered to
the frozen catalog — no silent no-op decode.

### Extension: theory URI as `transform_id`

```json
{
  "transform_id": "parcae://theories/quadratic_polynomial_stream@1",
  "direction": "decrypt",
  "params": { "c2": 0, "c1": 0, "c0": 0 }
}
```

| Rule | Requirement |
|------|-------------|
| `transform_id` | Catalog id **or** `parcae://theories/<name>@<version>` |
| `params` | For theory URIs: every declared artifact param MUST be present as an integer in its declared `[min,max]` |
| Catalog lower | `TheoryEnvelopeBridge::to_catalog_envelope()` succeeds only for frozen catalog ids; theory URIs **MUST** error mentioning TheoryDispatch |
| Compile | `parcae-compile` **SHOULD** emit `envelope.json` with the artifact URI and param mins as the default binding |
| Apply IR | `parcae-compile` **SHOULD** emit `apply_ir.json` (`parcae.theory_apply_ir.v0`) for `TheoryDispatch` |
| Runtime | `TheoryDispatch` applies catalog envelopes via `ApplyTransform` and theory URIs via `apply_ir.json` + `DslIrApplicator` |

Header: `include/parcae/dsl/theory_envelope_bridge.hpp` (`TheoryEnvelopeBridge`),
`theory_apply_ir.hpp`, `theory_dispatch.hpp`.

---

## Tool contracts (summary)

| Tool | Obligation |
|------|------------|
| `parcae-compile` | Write manifest with verification passed; embed versions; hard fail on gate failure; emit **stream** twins under `emitted/` when CUDA emit succeeds; **SHOULD** (when wired) also write `hist` / `paths.hist_*` without failing compile if only hist soft-falls to S0 |
| `parcae-validate` | Re-check gates and/or manifest integrity; enforce `dsl_spec_version`; if `paths.hist_*` set, check path safety + `hist_plan.json` schema when present |
| `parcae-sweep` | Read sweep metadata; reject stale spec; never invent solved-corpus defaults |
| `parcae-catalog --theories` | List URIs + `stale_spec` |
| Search (`parcae-search-cycle`) | Prefer artifact hist plan/module when present; else runtime `TheoryHistChi2Emit`; soft-fallback S0; Kernel SLO Done per [theory-hist-transpile.md](../architecture/theory-hist-transpile.md) |

Agent-facing CLIs **SHOULD** emit `parcae.tool_response.v0` on success and
failure ([tools.md](tools.md), [agent-tools.md](agent-tools.md)).

---

## Non-goals

- Storing ciphertext or locked fixture payloads inside theory artifacts
- Treating stub-package execution as verification
- Mutating `data/fixtures/` from theory tools
- Treating stream-twin (`emitted/*Kernel.cu`) throughput as theory search Kernel SLO Done
- Requiring `hist_module` for S1/S2 while in-lib twins + `hist_plan` suffice
