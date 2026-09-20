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
| `dsl.hpp` | umbrella | Done |

Tests: `[dsl][gate][forbidden]` covers dsl-ast-json.md forbidden kinds → stable `E031`/`E021`.

## Planned (later commits)

`dsl_build_ir`, `dsl_verifier`, `dsl_optimize`,
`dsl_fuse`, `dsl_emit_cpu`, `dsl_emit_cuda`, `theory_artifact`, `theory_registry`,
`theory_uri`.

Headers are part of the header-only `parcae::core` INTERFACE include tree
(`include/` via root `CMakeLists.txt`).
