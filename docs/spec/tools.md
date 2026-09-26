# Spec: Deterministic Tool & CLI Contracts

**Status:** Normative  
**Headers (planned):** `parcae/tool/api.hpp`  
**Binaries:** `parcae-tokenize`, `parcae-decode`, `parcae-score`, `parcae-validate`,
`parcae-catalog`, `parcae-compile`, `parcae-sweep`, `parcae-generate`, `parcae-rank`,
`parcae-hypothesis`, `parcae-parity`, `parcae-parity-gen`, `parcae-search-run`,
`parcae-bench`, `parcae-throughput-tiers` (compat)

## Principles

1. Tools are thin, deterministic wrappers over the library.
2. **No LLM / agent logic** inside tools (future agents call these APIs only).
3. Same inputs + flags ⇒ same stdout bytes (except explicitly documented timing
   fields, which MUST be omitted by default).
4. Nonzero exit status on any hard error.
5. UTF-8 in / UTF-8 out.

## Library API (`ToolApi`)

Headers: `parcae/tool/api.hpp`, `parcae/tool/context.hpp`,
`parcae/tool/transform_envelope.hpp`.

Callers construct a `Context` with the Parcae `data/` root; every API below is a
static method on `ToolApi` (or uses only envelope/indices when no profile I/O is
required).

### `tokenize`

```text
ToolApi::tokenize(ctx, source_utf8, grammar_id="rtkd-separator-grammar-v0", strict=true)
  → TokenStream | Status
```

### `apply_transform`

```text
ToolApi::apply_to_indices(span<Index29> | TokenStream, TransformEnvelope, backend=cpu)
  → vector<Index29> | Status

ToolApi::apply_and_rebuild_text(ctx, TokenStream, TransformEnvelope, backend=cpu)
  → string | Status   // preserves non-rune separators
```

`backend` is `cpu` (default) or `cuda`. CUDA requires a build with
`PARCAE_BUILD_CUDA=ON` (`PARCAE_HAS_CUDA`); otherwise the call returns an error.

When given a `TokenStream`, apply only to consumable runes. Both index-only and
rebuild-text paths are provided as above.

### `to_latin`

```text
ToolApi::to_latin(ctx, span<Index29>, label_profile="gematria-primus-v0-preferred")
  → string | Status
```

Multi-letter labels concatenated without inserted spaces (spaces come from
separators when rebuilding from a full stream). Exact join rules MUST be locked
in tests.

### `score`

```text
ToolApi::score(ctx, span<Index29>, score_id, score_version="v0", params_json?, request?, backend=cpu)
  → float | Status
```

`chi2_english_gp_v0` auto-loads `profiles/scores/english-gp-expected-v0.json`
when `request.expected_frequencies` is null. `log_bigram_gp_v0` auto-loads
`profiles/scores/english-gp-bigram-v0.json` when `request.bigram_model` is null;
if that file is missing, the Status message MUST include `model missing` (MUST
NOT abort; MUST NOT fall back to χ²). `backend=cuda` dispatches to `CudaScore`
when CUDA is linked.

### `validate_fixture`

```text
ToolApi::validate_fixture(ctx, fixture_dir_or_id, require_locked=false)
  → ValidationReport
```

`fixture_dir_or_id` may be an absolute/relative fixture directory or a solved
fixture id resolved under `data/fixtures/solved/<id>`.

`ValidationReport` MUST include:

| Field | Meaning |
|-------|---------|
| `ok` | overall pass |
| `fixture_id` | |
| `checks[]` | named check + pass/fail + message |
| `diff_excerpt` | optional, truncated |

Checks SHOULD include: manifest schema, tokenize, interrupt validation, transform
apply, plaintext compare, literal region compare, hash compare when locked.

---

## CLI contracts

Global flags (all tools):

| Flag | Meaning |
|------|---------|
| `--json` | Machine-readable JSON on stdout |
| `--strict` | Strict tokenizer / validation (default true for validate) |
| `-h` / `--help` | Help |

### `parcae-tokenize`

```text
parcae-tokenize [--json] <file|->
```

JSON shape (`--json` → `parcae.tool_response.v0`, `result`):

```json
{
  "token_count": 3,
  "consumable_count": 1,
  "tokens": [
    {
      "kind": "Rune",
      "index29": 4,
      "consumable_index": 0,
      "byte_begin": 0,
      "byte_end": 3,
      "text": "ᚱ"
    }
  ]
}
```

### `parcae-decode`

```text
parcae-decode --manifest <fixture_dir|manifest.json> [--backend cpu|cuda] [--allow-cuda] [--rebuild-text] [--json]
parcae-decode --transform-json <path> --input <file|-> [--backend cpu|cuda] [--allow-cuda] [--rebuild-text] [--json]
parcae-decode --input <file|-> --transform-id <id>
              [--direction decrypt|encrypt]
              [--params-json <json> | --key-indices <list> --key-latin <text> --shift <n>]
              [--skip-indices <list>]
              [--backend cpu|cuda] [--allow-cuda]
              [--rebuild-text]
              [--json]
```

Prints Latin plaintext (or JSON with `indices` + `latin`). With `--rebuild-text`, also
rebuilds UTF-8 preserving non-rune separators (`result.text` in JSON; human mode prints
that text instead of Latin). Method/key/skips come
from the fixture manifest, a transform envelope JSON file, or explicit flags.
`--backend cuda` requires a CUDA-linked build **and** `--allow-cuda` (AgentPolicy);
otherwise exit status **2** (`not_built` or `policy`).

### `parcae-score`

```text
parcae-score --score-id <id> --input <file|->
             [--latin|--runes|--indices] [--params-json <json>]
             [--backend cpu|cuda] [--allow-cuda]
             [--json] [--data-dir <path>]
parcae-score --list [--json] [--data-dir <path>]
```

Default input mode is `--latin` (letters only → delatinize). `--runes` tokenizes
UTF-8 Liber Primus text; `--indices` parses `0..28` integers. χ² and
`log_bigram_gp_v0` auto-load their tables from `--data-dir` (see ToolApi::score).
JSON shape:

```json
{ "score_id": "ic_mod29", "score_version": "v0", "backend": "cpu", "value": 1.0 }
```

`--backend cuda` requires `--allow-cuda`. `parcae-validate` stays CPU-only (no `--backend`).

`--list --json` emits a rich catalog under `result.scores[]` (`score_id`,
`score_version`, `order`, `arity`) plus flat `result.score_ids` for compatibility.

### `parcae-catalog`

```text
parcae-catalog [--all] [--transforms] [--scores] [--generators] [--backends]
               [--theories] [--json] [--data-dir <path>]
```

Lists agent-facing registries. With no section flags, all sections are included
(including `--theories`). `--json` emits `parcae.tool_response.v0` with
`backend: null` and a `result` object containing the selected sections
(`transforms`, `scores`/`score_ids`, `generators`/`generator_ids`, `backends`,
and when requested `theories` / `theory_uris` / `theories_dir`).

`--theories` lists compiled artifacts under `<data-dir>/theories/` via
`TheoryRegistry::list`. Each entry includes `uri`, `dsl_spec_version`,
`stale_spec` (MAJOR mismatch), `ready` (load-compatible with current toolchain),
and optional `detail`. Stale theories are **listed** (not omitted) so agents can
see recompile needs — they are marked `stale_spec: true` / `ready: false`, never
presented as ready-to-run.

### `parcae-compile`

```text
parcae-compile --status [--json] [--data-dir <path>]
parcae-compile <theory.py> [--json] [--data-dir <path>] [--allow-dsl-ignores]
```

Compiles a theory DSL `.py` source into a versioned artifact under
`data/theories/` (see [dsl.md](dsl.md), [theory-artifact.md](theory-artifact.md)).
Emits CPU/CUDA sources plus an `envelope.json` bridge
(`paths.envelope_template`) whose `transform_id` is the theory URI, and
`apply_ir.json` for `TheoryDispatch` (catalog-only tools must refuse lowering —
see Envelope bridge in theory-artifact.md).

`--status` reports toolchain / `dsl_spec_version` / AST-JSON protocol versions
and `pipeline_ready`. Compiling a `.py` file spawns `python -m parcae.dsl.ast_dump`,
runs ingest → semantic gate → divergence / host-glue → `DslBuildIr` → verify →
emit → `TheoryArtifact::store`. `--json` uses `parcae.tool_response.v0` with
`tool: "compile"`; success payloads **MAY** include `warnings[]` (**W010** /
**W011**) and `dsl_ignores_applied`.

`--allow-dsl-ignores` honors `#ignore DSL_FLAG:…` comments collected by
`ast_dump` ([dsl-ast-json.md](dsl-ast-json.md) § Directives). **Default is off:**
any ignore without the flag is **E031**. When honored, compile prints **W010** on
stderr and may record `dsl_ignores_applied` on the manifest. Portable / CI
theories **SHOULD** avoid ignores (prefer Param/const predicates → `Select`).

IDE `parcae.dsl` stubs and `ast_dump` alone are **not** substitutes for this tool —
see [dsl-stubs.md](../architecture/dsl-stubs.md). Smart-compiler scopes:
[python-transpiler.md](../architecture/python-transpiler.md) § Execution scopes.

### `parcae-sweep`

```text
parcae-sweep --theory <uri|name@version> [--limit <n>] [--json] [--data-dir <path>]
```

Expands `TheoryArtifact.sweep.param_grid` into a candidate plan
([theory-artifact.md](theory-artifact.md)). Loads via `TheoryRegistry` (stale
`dsl_spec_version` → fail). Tier B/C **MUST NOT** use a solved-oracle fixture id
as `sweep.corpus`. v0 is **plan-only** (no apply/score until TheoryDispatch).
`--limit` caps emitted candidates (default `100000`; `0` = unlimited). `--json`
uses `parcae.tool_response.v0` with `tool: "sweep"`.

### `parcae-generate`

```text
parcae-generate --generator-id <id> --input <file|->
                [--latin|--runes|--indices] [--direction decrypt|encrypt]
                [--params-json <json>] [--json] [--data-dir <path>]
parcae-generate --list [--json] [--data-dir <path>]
```

Runs `GenerateCandidates` / `GeneratorRegistry` and emits candidates. `--json`
wraps `result.candidates[]` (`TransformCandidate::to_json`) plus `count`,
`generator_id`, `direction`, `input_mode`. Default input mode is `--runes`.

### `parcae-rank`

```text
parcae-rank --candidates <file|-> --score-id <id> --k <n>
            [--params-json <json>] [--latin-max <n>] [--no-latin]
            [--backend cpu|cuda] [--allow-cuda]
            [--json] [--data-dir <path>]
```

Ranks `TransformCandidate` JSON via `RankCandidates` (stable ties:
score → `candidate_id` → `source_index`). `--candidates` accepts a generate
`--json` envelope, a `{"candidates":[…]}` object, or a bare candidate array.
`--json` emits `result` from `RankCandidates::result_to_json` (hits with
`rank`, `candidate_id`, `score`, `source_index`, `envelope`, optional `latin`
preview, plus `backend`). Pairwise scores may pass `reference` via
`--params-json`. χ² loads expected frequencies from `--data-dir` automatically.
`log_bigram_gp_v0` loads the bigram model from `--data-dir` the same way.
`--backend cuda` requires a CUDA-linked build **and** `--allow-cuda`
(AgentPolicy); otherwise exit status **2** (`not_built` or `policy`).
`RankCandidates::run(..., backend=cuda)` scores via `CudaScore` when linked;
CPU `BatchRunner` remains the default / small-N oracle.

**Round-trip (agent path):** `parcae-generate … --json` → `parcae-rank
--candidates - --score-id chi2_english_gp_v0 --k N --json` recovers Atbash on
`a-warning` without any plaintext / `reference` in the score path.

### `parcae-hypothesis`

```text
parcae-hypothesis init|propose|show|list|score|set-status …
```

HypothesisRecord I/O under `data/workspaces/<workspace_id>/` (see
[`hypothesis-workspace.md`](hypothesis-workspace.md)). Subcommands map to agent
tool names `hypothesis_init` / `hypothesis_propose` / `hypothesis_show` /
`hypothesis_list` / `hypothesis_score` / `hypothesis_set_status`.

| Subcommand | Key flags |
|------------|-----------|
| `init` | `--workspace` `--id` `[--title]` `[--method-json]` |
| `propose` | `--workspace` `--id` `--method-json\|--method-file` `[--title]` `[--rationale]` |
| `show` / `list` | `--workspace` (`show` also `--id`) |
| `score` | `--workspace` `--id` `--input` `[--runes\|--latin\|--indices]` `[--score-id]` (χ²/bigram tables auto-load via ToolApi) |
| `set-status` | `--workspace` `--id` `--status` |

`--json` wraps each result in `parcae.tool_response.v0`. `init`/`propose` create
a workspace manifest if missing. `score` applies the stored method, appends a
score entry, updates latin preview + digests, and promotes status toward
`scored`. Writes stay under the workspace and are gated by `AgentPolicy`
(fixture paths / traversal → `error.code = policy`); fixture paths are denied.

### `parcae-parity`

```text
parcae-parity check [--all|--name <golden>] [--compare-cuda]
                    [--parity-dir <path>] [--data-dir <path>] [--json]
parcae-parity dump  --name <golden> [--backend cpu|cuda]
                    [--parity-dir <path>] [--data-dir <path>]
```

Replays committed goldens under `data/parity/` (see `parcae-parity-gen`).
`check` verifies CPU digests; `--compare-cuda` also runs `CudaBackend` and
requires matching digests except `backend` (`cpu` vs `cuda`). Exit **2** if
CUDA is requested but not built.

### `parcae-search-run`

```text
parcae-search-run [--backend cpu|cuda] [--family caesar]
                  [--seed <u32>] [--stream-length <n>] [--repeats <n>]
                  [--score-id <id>] [--no-compare] [--json] [--omit-timing]
                  [--data-dir <path>]
```

AI-style **metrics** dashboard (not the workspace search loop): **throughput**
(runes/s over transform+score), **sweep scores** (parameter axis), and
**locked-fixture eval** (scorer sanity). Backed by `SearchRun` /
`SearchRunCuda` fused χ² — returns console metrics and sweep steps, **not**
`TransformCandidate` batches or `HypothesisRecord`s.
`tok_per_sec` is intentionally non-deterministic; with `--json --omit-timing` it is
omitted so agent output is replayable. Each `steps[]` entry includes replayable
`params` (plus `param_hash`). `--json` uses `parcae.tool_response.v0`.
v0 families: `caesar|atbash|atbash_caesar|affine|vigenere` (CPU: caesar only).
With `--backend cuda`, also reports CPU↔CUDA score parity unless `--no-compare`.
Exit **1** if fixture eval is not all-pass or CUDA parity fails;
exit **2** if CUDA is requested but not built.

| vs | Role |
|----|------|
| `parcae-search-cycle` | Workspace closed loop → batches / hypotheses (agent **allow**-list) |
| `parcae-search-run` | Throughput / sweep / fixture-eval **metrics** CLI (agent **deny**-list by default) |

Workspace closed-loop cycles (job → `BatchArtifact` → hypotheses) are **not**
this tool — see `parcae-search-cycle` / [`search-loop.md`](search-loop.md) and
[`search-engine.md`](../architecture/search-engine.md). Agents MUST use
`search_cycle` (allow-list) rather than `search-run` (deny-list by default)
([`agent-tools.md`](agent-tools.md) § Deny-list / `search_cycle` vs
`parcae-search-run`). Human operators MAY run it from a shell for dashboards
and CUDA parity smoke.

### `parcae-blind-crack`

```text
parcae-blind-crack [--data-dir <path>] [-h|--help]
```

Research **foothold bench** on locked Tier-A fixtures: enumerate an additive
family battery (identity / atbash / caesar / atbash_caesar / affine), rank by
`chi2_english_gp_v0` on **ciphertext only**, then oracle-check against known
plaintext. Unsolved LP2 pages `0`–`55` ship as the research workspace
`data/workspaces/_lp2_unsolved_corpus/` (not under `data/fixtures/`). This CLI
still exercises Liber-Primus-length **locked** streams that have plaintext for
the oracle check.

| vs | Role |
|----|------|
| `parcae-search-cycle` | Workspace closed loop → batches / hypotheses (agent **allow**-list) |
| `parcae-blind-crack` | Fixture battery + oracle report (agent **deny**-list by default) |

Agents MUST NOT receive `parcae-blind-crack` as a tool by default
([`agent-tools.md`](agent-tools.md) § Deny-list / `search_cycle` vs
`parcae-blind-crack`). Human operators MAY run it from a shell.

### `parcae-bench`

```text
parcae-bench --status [--json] [--data-dir <path>]
parcae-bench --suite slo [--extended] --allow-cuda
             [--json] [--omit-timing] [--data-dir <path>]
parcae-bench --suite accuracy [--allow-cuda]
             [--json] [--omit-timing] [--data-dir <path>]
parcae-bench --suite hardware [--backend cpu|cuda|both]
             [--allow-cuda|--require-cuda] [--allow-skip] [--cpu-full]
             [--json] [--omit-timing] [--data-dir <path>]
parcae-bench --suite probe --probe-cmd <cmd>
             [--probe-tiers T1,T2,T3] [--probe-timeout-ms N]
             [--compare-builtin] [--json] [--omit-timing] [--data-dir <path>]
parcae-bench --suite all [flags for each leg…]
```

Umbrella **benchmark & diagnostics** CLI (toolkit 0.9.0 target). Headers live
under [`include/parcae/bench/`](../../include/parcae/bench/README.md). Canonical
SLO constants: `BenchTierSpec`. External probe wire format:
[`bench-probe.md`](bench-probe.md). Operator throughput notes:
[`cuda-throughput.md`](../architecture/cuda-throughput.md).

| Suite | Purpose |
|-------|---------|
| `slo` | Fused CUDA T1–T3 SLO (optional `--extended` → F.* / C.*). Requires `--allow-cuda` + usable device. |
| `accuracy` | Statistical validation (CPU always; CUDA planted/parity with `--allow-cuda`). |
| `hardware` | CPU vs CUDA side-by-side for T1–T3; scaled CPU smoke by default (`--cpu-full` for full C/T/reps). CUDA skip → `skipped_not_built`; `--require-cuda` fails; `--allow-skip` OK. |
| `probe` | Spawn external JSON probes (`--probe-cmd` with `{tier}`); timeout default 120000 ms. |
| `all` | Order: accuracy → slo → hardware → probe. Probe runs **only** if `--probe-cmd` is set. SLO runs only with `--allow-cuda` when CUDA is usable. |

Shared flags: `--json` → `parcae.tool_response.v0` (`tool: "bench"`);
`--omit-timing` requires `--json` and drops rate/wall fields for stable digests.
Exit codes: **0** all rows pass, **1** fail, **2** usage/policy.

| vs | Role |
|----|------|
| `parcae-bench` | Canonical umbrella (agent **deny**-list by default) |
| `parcae-throughput-tiers` | Thin compat wrapper ≡ `--suite slo --extended --allow-cuda` (also deny-listed) |

Agents MUST NOT receive `parcae-bench` / `parcae-throughput-tiers` as tools by
default ([`agent-tools.md`](agent-tools.md) § Deny-list). Absolute runes/s are
**not** a hosted-CI gate (non-deterministic); operators MAY run suites from a
shell. Prefer `search_cycle` for workspace research.

### `parcae-throughput-tiers`

```text
parcae-throughput-tiers [--data-dir <path>] [-h|--help]
```

Compatibility binary (CUDA builds only). Equivalent to
`parcae-bench --suite slo --extended --allow-cuda`. Prefer `parcae-bench` for
new scripts (`--json` / `--omit-timing` / `--status`). Agent **deny**-list by
default (same reason as `parcae-bench`).

### `parcae-search-cycle`

```text
parcae-search-cycle --status [--json] [--data-dir <path>]
parcae-search-cycle --workspace <id> --job <file>
                    [--backend cpu|cuda] [--allow-cuda] [--iterations <n>]
                    [--json] [--omit-timing] [--created-utc <rfc3339>]
                    [--quiet | --plain-progress | --progress auto|panel|lines|off]
                    [--data-dir <path>]
parcae-search-cycle --workspace <id> --family <id> [--k <n>] [--seed <u32>]
                    [--score-id <id>] [--backend cpu|cuda] [--allow-cuda]
                    [--allow-extended-families] [--allow-theory-uri]
                    [--iterations <n>] [--json]
                    [--omit-timing] [--created-utc <rfc3339>]
                    [--quiet | --plain-progress | --progress auto|panel|lines|off]
                    [--data-dir <path>]
```

Workspace closed-loop search via `SearchScheduler` ([search-loop.md](search-loop.md)):
job → `BatchArtifact` → `HypothesisBridge`. `--status` reports toolkit version,
schema ids, CUDA build flag, and readiness (`scheduler_ready` / `run_ready`).
`--json` uses `parcae.tool_response.v0` with `tool: "search_cycle"`.

Cycle runs require `--workspace` plus either `--job` (`parcae.search_job.v0`) or
`--family` (builds a job using workspace `default_score_id`). `--score-id` may be
any registered id (including `log_bigram_gp_v0`); fused CUDA export remains
χ²-only and non-χ² jobs with `--backend cuda` MUST fall back to CPU export (see
[search-loop.md](search-loop.md)). `--backend cuda`
requires `--allow-cuda` (AgentPolicy). `--allow-extended-families` opts in
`beaufort` / `totient` / `hill_2` / `hill_3` / `ciphertext_autokey` /
`plaintext_autokey` (also settable on the job JSON). `--allow-theory-uri` opts
in family `theory` (job JSON MUST supply `param_grid.theory_uri` +
`params_list`; CPU-only). `--iterations` defaults to `1`.
`--omit-timing` requires `--json` and forces agent-safe output (no timing fields;
no `report.json`). `--json` alone also omits timing by default. `--created-utc`
fixes batch/prior timestamps for replayable digests and auto `batch_id`s.
`--with-agent` lands in a follow-up commit.

#### Console progress (stderr)

Live progress paints **stderr only** via `ConsoleDashboard`. With `--json`,
stdout remains a single `parcae.tool_response.v0` envelope; progress does **not**
pollute JSON. Digests / ranking are unchanged when progress is on or off.

| Flag | Effect |
|------|--------|
| *(default)* | `--progress auto`: TTY → live panel; pipe/CI → append-only lines |
| `--progress panel\|lines\|off` | Force mode (`off` = no progress UI) |
| `--plain-progress` | Force append-only lines |
| `--quiet` | Suppress all progress UI (agents / scripts) |

Precedence: `--quiet` > `--plain-progress` > `--progress`.

Agent ToolBridge injects `--quiet` (with `--omit-timing`) on cycle runs so
transcripts stay clean. Human operators wanting live feedback omit `--quiet`
(and may pass `--plain-progress` for CI-friendly logs). ASCII examples:
[`search-handbook.md`](../architecture/search-handbook.md) § Console progress.

### `parcae-validate`

```text
parcae-validate --id <fixture_id|path> [--require-locked] [--json] [--data-dir <path>]
parcae-validate --all [--require-locked] [--json] [--data-dir <path>]
parcae-validate --theory <uri|name@version|dir> [--json] [--data-dir <path>]
parcae-validate --theories [--json] [--data-dir <path>]
```

**Fixture mode** (`--id` / `--all`): validate solved fixtures under
`data/fixtures/solved/`. With `--all --require-locked`, only fixtures whose
`verification.status` is `locked` are selected (draft/synth fixtures are skipped).

**Theory mode** (`--theory` / `--theories`): validate compiled theory artifacts
under `data/theories/` ([theory-artifact.md](theory-artifact.md)). Checks
manifest integrity, `dsl_spec_version` compatibility (stale MAJOR → fail;
message includes re-run `parcae-compile`), `verification.passed`, that
declared `paths.*` files exist, and when `paths.envelope_template` is set that
the envelope parses (catalog or theory URI) and matches the artifact.
Does not re-run exhaustive/fuzz verify.
`--theory` accepts `parcae://theories/<name>@<ver>`, `<name>@<ver>`, or an
artifact directory / `manifest.json` path.

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | all selected fixtures / theories passed |
| 1 | validation failure |
| 2 | usage / I/O / schema error |

---

## Agent-facing stability

Library compute primitives agents SHOULD build on:

1. `tokenize`
2. `apply_transform`
3. `to_latin`
4. `score`
5. `validate_fixture`
6. `GenerateCandidates` (`from_indices` / `from_stream` / `from_source`) — `gen_*` dispatch
7. `RankCandidates` (`run` / `result_to_json`) — top-k with stable ties;
   optional `backend=cuda` via `CudaScore`. Ordering contract:
   `[tool][rank][order]` (CPU always; CUDA device compare when linked)

Plus read-only listing of `transform_id` / `score_id` / `generator_id` registries
(`parcae-catalog`, `GenerateCandidates::list_generator_ids`).

**Normative agent CLI surface** (allow-list, deny-list, `parcae.tool_response.v0`
envelope, exit codes, `parcae-agent` loop): see [`agent-tools.md`](agent-tools.md).

Agents MUST NOT receive:

- Arbitrary shell
- Mutable fixture hash rewriting
- Hidden non-deterministic “best guess” interrupt inference in reference tools

## Logging

Default: quiet success. `--json` is the structured interface. Human stderr
diagnostics MUST NOT affect stdout JSON mode.
