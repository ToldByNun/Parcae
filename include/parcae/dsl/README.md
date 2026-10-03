# Theory DSL headers (`include/parcae/dsl/`)

C++20 compiler surface for the theory DSL. Normative contracts:

- [`docs/spec/dsl.md`](../../../docs/spec/dsl.md)
- [`docs/spec/dsl-ast-json.md`](../../../docs/spec/dsl-ast-json.md)
- [`docs/spec/theory-artifact.md`](../../../docs/spec/theory-artifact.md)
- Architecture: [`docs/architecture/python-transpiler.md`](../../../docs/architecture/python-transpiler.md)

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
| `dsl_spec_version.hpp` | `DslSpecVersion` | Done (`1.0.0`) |
| `dsl_ast_json_version.hpp` | `DslAstJsonVersion` | Done (`1.1.0`, directives additive) |
| `dsl_rule_id.hpp` | `DslRuleId` | Done (stable E0xx / E1xx) |
| `dsl_diag.hpp` | `DslDiag` | Done (`path:line:col: RULE message`) |
| `dsl_ast_limits.hpp` | `DslAstLimits` | Done (v0 ceilings) |
| `dsl_ast.hpp` | `DslAstNode` / `DslAstDocument` / `DslAstDirective` | Done |
| `dsl_ast_json_ingest.hpp` | `DslAstJsonIngest` | Done (strict + limits + `directives[]`) |
| `dsl_semantic_gate.hpp` | `DslSemanticGate` | Done (whitelist + `z29_*` Call allowlist / E032 + scope-aware CF / E034) |
| `dsl_z29_builtins.hpp` | `DslZ29Builtins` | Done (`z29_*` intrinsic names incl. matmul/det/autokey_shift) |
| `dsl_divergence_gate.hpp` | `DslDivergenceGate` / `DslPredicateClass` | Done (HotLoop If → E033 / W011) |
| `dsl_directive_table.hpp` | `DslDirectiveTable` | Done (bind `#ignore` → W010; `--allow-dsl-ignores`) |
| `host_glue_ir.hpp` | `HostGlueIr` | Done (OuterControl host IR nodes) |
| `dsl_host_glue.hpp` | `DslHostGlue` | Done (for/while → HostGlueIr / E035) |
| `dsl_exec_scope.hpp` | `DslExecScope` | Done (OuterControl / HotLoop + loop depth) |
| `dsl_scope_analyzer.hpp` | `DslScopeAnalyzer` / `DslScopeMap` | Done (AST → scope map) |
| `matrix_ir.hpp` | `MatrixIr` | Done (2×2 / 3×3 `Z29Expr`; det / mul_vec expand → host `Z29Matrix{2,3}` / device twins) |
| `z29_bytecode.hpp` | `Z29Bytecode` | Done (HotLoop stack program + host eval; `op_as_u8` for device twin) |
| `z29_bytecode_device.hpp` | `Z29BytecodeDevice` | Done (host/device `eval_at` twin) |
| `theory_chi2_batch.hpp` | `TheoryChi2Batch` | Done (fused bytecode hist + χ²; S0 path via `TheoryHistChi2Launch`) |
| `theory_hist_chi2_emit.hpp` | `TheoryHistChi2Emit` | Done (S0–S3 select; S1 LUT-29 + S2 linear uchar4 emit + goldens; S3 TBD) |
| `theory_hist_chi2_launch.hpp` | `TheoryHistChi2Launch` | Done (CUDA façade: S0 bytecode / S1 LUT / S2 linear async) — under `Parcae/Parcae/cuda/` |
| `theory_hist_chi2_s1.hpp` | `TheoryHistChi2S1` | Done (LUT-29 twin) — under `Parcae/Parcae/cuda/` |
| `theory_hist_chi2_s2.hpp` | `TheoryHistChi2S2` | Done (linear uchar4 twin) — under `Parcae/Parcae/cuda/` |
| `param_ir.hpp` | `ParamIr` | Done |
| `primitive_ir.hpp` | `PrimitiveIr` | Done |
| `theory_ir.hpp` | `TheoryIr` | Done |
| `compose_ir.hpp` | `ComposeIr` | Done |
| `dsl_ir_applicator.hpp` | `DslIrApplicator` | Done (CPU apply_into) |
| `dsl_emit_cpu.hpp` | `DslEmitCpu` | Done (Transform-shaped; `Select` → `Z29::select` / branch; matmul/det/autokey) |
| `dsl_emit_cuda.hpp` | `DslEmitCuda` | Done (Z29Device; matmul/det expand; autokey → AutokeyRingDevice) |
| `dsl_verifier.hpp` | `DslVerifier` | Done (exhaustive ≤4 + fuzz + CPU↔CUDA mirror) |
| `dsl_catalog_builtins.hpp` | `DslCatalogBuiltins` | Done (`identity`/`atbash`/`caesar`/`affine` + DSL-only `matrix_mix`/`autokey_lag`; **not** `TransformId` / decode `--transform-id`) |
| `dsl_fuse.hpp` | `DslFuse` | Done (inline + emit + CPU bench; DSL-only leaves → fused emit) |
| `dsl_optimize.hpp` | `DslOptimize` | Done (const-fold + Select dead-arm + `inv` hoist; hooked in `DslCompile`) |
| `z29_expr_normalize.hpp` | `Z29ExprNormalize` | Done (algebraic HotLoop normalize for smart hist; `[dsl][normalize]` goldens) |
| `theory_shape_match.hpp` | `TheoryShapeMatch` | Done (ShapeId from normalized trees; name-irrelevant; `[dsl][shape]` goldens) |
| `dsl_compile.hpp` | `DslCompile` | Done (ast_dump → Optimize → emit/apply_ir) |
| `dsl_launch_plan.hpp` | `DslLaunchPlan` | Done (1D / HistFast 2D twin grids) |
| `dsl_peak_sanity.hpp` | `DslPeakSanity` | Done (ThroughputTiers peak / SLO) |
| `theory_uri.hpp` | `TheoryUri` | Done (`parcae://theories/<name>@<ver>`) |
| `theory_artifact.hpp` | `TheoryArtifact` | Done (writer embeds `dsl_spec_version`) |
| `theory_registry.hpp` | `TheoryRegistry` | Done (rejects stale; catalog marks `stale_spec`) |
| `theory_validate.hpp` | `TheoryValidate` | Done (manifest + dsl_spec + path files) |
| `theory_sweep.hpp` | `TheorySweep` | Done (expand `sweep.param_grid` plan) |
| `theory_envelope_bridge.hpp` | `TheoryEnvelopeBridge` | Done (catalog / theory-URI envelope) |
| `theory_apply_ir.hpp` | `TheoryApplyIr` | Done (`apply_ir.json` serialize) |
| `theory_dispatch.hpp` | `TheoryDispatch` | Done (ApplyTransform hook) |
| `dsl_build_ir.hpp` | `DslBuildIr` | Done (AST → PrimitiveIr / TheoryIr; HotLoop If → Select) |
| `dsl_compile.hpp` | `DslCompile` | Done (ast_dump → Optimize → emit → artifact) |
| `dsl.hpp` | umbrella | Done |

Tests: `[dsl][gate][forbidden]` covers dsl-ast-json.md always-forbidden kinds → stable `E031`/`E021`.
Tests: `[dsl][gate][scope]` OuterControl If/For/While OK; HotLoop For/While → E034.
Tests: `[dsl][divergence]` HotLoop If ThreadVarying → E033; Param/const → W011.
Tests: `[dsl][directives]` / `[dsl][directive]` `#ignore` binding; W010; E031 without `--allow-dsl-ignores`.
Tests: `[dsl][golden]` smart-compiler acceptance matrix (scope / divergence / ignore / compile).
Tests: `[dsl][hostglue]` OuterControl range-for / bounded while; E035 negatives.
Tests: `[dsl][build][select]` HotLoop If/IfExp → Z29Expr Select + fold.
Tests: `[dsl][emit][select]` CPU/CUDA Select mux + `prefer_branch` conditional.
Tests: `[dsl][scope]` DslExecScope + DslScopeAnalyzer OuterControl vs HotLoop.
Tests: `[dsl][ingest][fuzz]` adversarial mutations + limit rejects (no crash).
Tests: `[dsl][applicator]` IR → Index29 stream apply_into.
Tests: `[dsl][bytecode]` Z29Bytecode encode + host eval parity vs DslIrApplicator.
Tests: `[dsl][emit]` DslEmitCpu / DslEmitCuda source text.
Tests: `[dsl][oracle][poly2]` exhaustive poly2 IR vs host Z29.
Tests: `[dsl][verify]` / `[dsl][verify][gate]` DslVerifier exhaustive + fuzz +
CPU↔CUDA mirror; inv-domain + poly2 \(29^4\) hard gates.
Tests: `[dsl][fuse]` / `[dsl][fuse][koan][parity]` / `[dsl][fuse][catalog]` DslFuse +
catalog builtins (`matrix_mix` / `autokey_lag`) + Koan-1 vs ComposeTransform.
Tests: `[dsl][examples][matrix]` `theories/examples/matrix_builtins_example.py` compile.
Tests: `[dsl][emit][hist][chi2]` strategy select + S1/S2 emit; `[cuda][golden]` bytecode χ² == specialized; `[cuda][theory][edge]` top-k / Autokey→S0 / Div0 +inf / interrupt reject.
Tests: `[dsl][optimize]` DslOptimize const-fold + inv hoist.
Tests: `[dsl][launch]` DslLaunchPlan vs HistFast / 1D twin formula.
Tests: `[dsl][peak]` DslPeakSanity vs `BenchTierSpec` ceilings / SLO (incl. `T.theory.*`).
Tests: `[dsl][artifact]` TheoryArtifact writer stamps `dsl_spec_version` + store/load.
Tests: `[dsl][registry]` TheoryRegistry load rejects stale MAJOR; list marks `stale_spec`
(+ `CatalogEntry::to_json` for `parcae-catalog --theories`).
Tests: `[dsl][registry][stale]` committed `0.9.0` fixture + major-bump / recompile contract.
Tests: `[dsl][validate]` TheoryValidate manifest / dsl_spec / path / envelope content.
Tests: `[dsl][sweep]` TheorySweep param_grid expansion + stale/corpus gates.
Tests: `[dsl][compile]` DslCompile end-to-end on quadratic_polynomial_stream fixture.
Tests: `[dsl][examples][i39]` `theories/examples/new_math_example.py` compile + validate.
Tests: `[dsl][examples][i40]` `theories/examples/full_lifecycle_example.py` compose fuse + validate.
Tests: `[dsl][dispatch][i41]` TheoryDispatch catalog + theory URI apply.
Tests: `[cuda][dsl][smoke]` DslEmitCuda text + `DslSmokeCaesarKernel` (device when CUDA ON).

Authoring examples (also CI via `scripts/check-dsl-examples.sh`):
[`theories/examples/`](../../../theories/examples/) —
`matrix_builtins_example.py` (`z29_det` / `z29_matmul` / `@define_primitive`);
`param_select_example.py` (Param → Select / **W011**);
`ignore_divergent_example.py` denied without `--allow-dsl-ignores` (**E031**).
Handbook: [python-transpiler.md](../../../docs/architecture/python-transpiler.md)
§ Execution scopes. Theory fused-χ² search path (S1/S2 prefer, S0 soft-fallback;
**S0–S5 all ≥90% shape peak**):
[theory-hist-transpile.md](../../../docs/architecture/theory-hist-transpile.md),
[dsl-smart-hist.md](../../../docs/architecture/dsl-smart-hist.md)
(normalize → ShapeId; customs without presets),
[cuda-throughput.md](../../../docs/architecture/cuda-throughput.md) § Theory,
[search-handbook.md](../../../docs/architecture/search-handbook.md) § Theory URI,
[cuda-profile-theory.md](../../../docs/architecture/cuda-profile-theory.md).

## Theory fused χ² (search export)

Compile still emits CPU/CUDA transform text + bytecode. Separately,
`TheoryHistChi2Emit` classifies the decrypt HotLoop for **search** fused χ²:

| Strategy | Shape | Runtime |
|----------|-------|---------|
| S1 | `f(x; params)` only | `TheoryHistChi2S1` LUT-29 |
| S2 | `x ± (b0 + b1·i)` | `TheoryHistChi2S2` linear uchar4 |
| S0 | unmatched / Autokey / soft-fallback | `TheoryChi2Batch` bytecode interpreter |
| S3–S5 | planned (expr inline / autokey ring / poly) | see contract |

Target smart path: `Z29ExprNormalize` → `TheoryShapeMatch` (`Atbash` /
`Caesar` / `Affine` / … by **algebra**, not catalog API) → shared `HistFast`
twins — [`dsl-smart-hist.md`](../../../docs/architecture/dsl-smart-hist.md).

`GpuCandidateExport` prefers S1/S2 when the cached `HistPlan` matches; launch
failure soft-falls back to S0 with `export_backend=cuda` unchanged. Soft-fallback
still requires the **S0** fair Kernel SLO ≥90% Spec peak. Normative rules:
[`theory-hist-transpile.md`](../../../docs/architecture/theory-hist-transpile.md),
[`dsl-smart-hist.md`](../../../docs/architecture/dsl-smart-hist.md).
Artifact **stream** (`emitted/`, `paths.cuda_*`) vs **fused hist** (`hist/`,
`paths.hist_*`):
[`theory-artifact.md`](../../../docs/spec/theory-artifact.md).
Peaks / suite: `BenchTierSpec` `T.theory.*` via `parcae-bench --suite theory`.

## Status

Toolkit **1.0.0** (prior: `v0.9.0-bench`, `v0.8.0-dsl-console`,
`v0.7.0-search-engine`, `v0.5.0-theory-dsl`).
Next exit freeze (CUDA catalog parity + Pack D → toolkit **1.1.0** / `v1.1.0`):
[`cuda-catalog-parity.md`](../../../docs/architecture/cuda-catalog-parity.md),
[`dsl-pack-d.md`](../../../docs/architecture/dsl-pack-d.md).
Freeze checklist for the prior bench cut:
[`bench-exit.md`](../../../docs/architecture/bench-exit.md); smart-DSL freeze:
[`dsl-console-exit.md`](../../../docs/architecture/dsl-console-exit.md).
CI gates: `[dsl-smart]`, `[cli-progress]`, `[bench]`, `[dsl][examples]`,
`[dsl][registry][stale]`, `[falsify]`, `[research]`, plus `dsl-examples-cli` /
`dsl-stubs-pytest` jobs in `.github/workflows/ci.yml`.

Headers are part of the header-only `parcae::core` INTERFACE include tree
(`include/` via root `CMakeLists.txt`).
