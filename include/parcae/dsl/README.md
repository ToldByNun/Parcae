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
| `dsl_ast_json_version.hpp` | `DslAstJsonVersion` | Done (`1.0.0`) |
| `dsl_rule_id.hpp` | `DslRuleId` | Done (stable E0xx / E1xx) |
| `dsl_diag.hpp` | `DslDiag` | Done (`path:line:col: RULE message`) |
| `dsl_ast_limits.hpp` | `DslAstLimits` | Done (v0 ceilings) |
| `dsl_ast.hpp` | `DslAstNode` / `DslAstDocument` | Done |
| `dsl_ast_json_ingest.hpp` | `DslAstJsonIngest` | Done (strict + limits) |
| `dsl_semantic_gate.hpp` | `DslSemanticGate` | Done (whitelist + imports) |
| `z29_expr.hpp` | `Z29Expr` | Done (IR + eval via Z29) |
| `param_ir.hpp` | `ParamIr` | Done |
| `primitive_ir.hpp` | `PrimitiveIr` | Done |
| `theory_ir.hpp` | `TheoryIr` | Done |
| `compose_ir.hpp` | `ComposeIr` | Done |
| `dsl_ir_applicator.hpp` | `DslIrApplicator` | Done (CPU apply_into) |
| `dsl_emit_cpu.hpp` | `DslEmitCpu` | Done (Transform-shaped source text) |
| `dsl_emit_cuda.hpp` | `DslEmitCuda` | Done (Z29Device + Kernel façade) |
| `dsl_verifier.hpp` | `DslVerifier` | Done (exhaustive ≤4 + fuzz + CPU↔CUDA mirror) |
| `dsl_fuse.hpp` | `DslFuse` | Done (inline + emit + CPU bench gate) |
| `dsl_optimize.hpp` | `DslOptimize` | Done (const-fold + `inv` hoist) |
| `dsl_launch_plan.hpp` | `DslLaunchPlan` | Done (1D / HistFast 2D twin grids) |
| `dsl_peak_sanity.hpp` | `DslPeakSanity` | Done (ThroughputTiers peak / SLO) |
| `theory_uri.hpp` | `TheoryUri` | Done (`parcae://theories/<name>@<ver>`) |
| `theory_artifact.hpp` | `TheoryArtifact` | Done (writer embeds `dsl_spec_version`) |
| `theory_registry.hpp` | `TheoryRegistry` | Done (rejects stale `dsl_spec_version`) |
| `dsl.hpp` | umbrella | Done |

Tests: `[dsl][gate][forbidden]` covers dsl-ast-json.md forbidden kinds → stable `E031`/`E021`.
Tests: `[dsl][ingest][fuzz]` adversarial mutations + limit rejects (no crash).
Tests: `[dsl][applicator]` IR → Index29 stream apply_into.
Tests: `[dsl][emit]` DslEmitCpu / DslEmitCuda source text.
Tests: `[dsl][oracle][poly2]` exhaustive poly2 IR vs host Z29.
Tests: `[dsl][verify]` / `[dsl][verify][gate]` DslVerifier exhaustive + fuzz +
CPU↔CUDA mirror; inv-domain + poly2 \(29^4\) hard gates.
Tests: `[dsl][fuse]` / `[dsl][fuse][koan][parity]` DslFuse + Koan-1 vs ComposeTransform.
Tests: `[dsl][optimize]` DslOptimize const-fold + inv hoist.
Tests: `[dsl][launch]` DslLaunchPlan vs HistFast / 1D twin formula.
Tests: `[dsl][peak]` DslPeakSanity vs ThroughputTiers ceilings / SLO.
Tests: `[dsl][artifact]` TheoryArtifact writer stamps `dsl_spec_version` + store/load.
Tests: `[dsl][registry]` TheoryRegistry load rejects stale MAJOR; list marks `stale_spec`.
Tests: `[dsl][registry][stale]` committed `0.9.0` fixture + major-bump / recompile contract.

## Planned (later commits)

`dsl_build_ir`.

Headers are part of the header-only `parcae::core` INTERFACE include tree
(`include/` via root `CMakeLists.txt`).
