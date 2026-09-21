# Spec: Theory Artifact Manifest

**Status:** Normative  
**Schema id:** `parcae.theory_artifact.v0`  
**URI scheme:** `parcae://theories/<name>@<version>`  
**Related:** [dsl.md](dsl.md) (`dsl_spec_version`), [transforms.md](transforms.md),
[tools.md](tools.md)

Compiled theories are **versioned, inspectable artifacts** under
`data/theories/`. They are the only form that validate/sweep/runtime dispatch
**MUST** treat as verified output of `parcae-compile`. Raw `.py` sources are
inputs, not runnable registry entries.

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
      emitted/                 # optional CUDA/C++ twin sources
        *.hpp
        *.cu
      verify_report.json       # optional machine-readable gate log
```

| Rule | Requirement |
|------|-------------|
| `name` | MUST match `manifest.name`; recommended `[a-z][a-z0-9_]*` |
| `version` | Positive integer; MUST match URI `@<version>` and directory name |
| Encoding | UTF-8; JSON objects; LF preferred |
| Paths inside manifest | Relative to the artifact directory; MUST NOT escape via `..` |

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
  "compiler_version": "0.5.0",
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
    "verify_report": "verify_report.json"
  },
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
| `structural_claim` | **Required** in manifest when `tier` is `B` or `C` (non-empty string) |
| `sweep` | `null` or a sweep config object (see below) |
| `interrupts` | Documents interrupt mode: `policy_method` \| `none_by_design` \| `elementwise_default` |

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
| `parcae-compile` | Write manifest with verification passed; embed versions; hard fail on gate failure |
| `parcae-validate` | Re-check gates and/or manifest integrity; enforce `dsl_spec_version` |
| `parcae-sweep` | Read sweep metadata; reject stale spec; never invent solved-corpus defaults |
| `parcae-catalog --theories` | List URIs + `stale_spec` |

Agent-facing CLIs **SHOULD** emit `parcae.tool_response.v0` on success and
failure ([tools.md](tools.md), [agent-tools.md](agent-tools.md)).

---

## Non-goals

- Storing ciphertext or locked fixture payloads inside theory artifacts
- Treating stub-package execution as verification
- Mutating `data/fixtures/` from theory tools
