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
| `dsl.hpp` | umbrella | Done |

## Planned (later commits)

`dsl_ast`, `dsl_ast_json_ingest`, `dsl_semantic_gate`, `z29_expr`,
`primitive_ir`, `theory_ir`, `dsl_build_ir`, `dsl_verifier`, `dsl_optimize`,
`dsl_fuse`, `dsl_emit_cpu`, `dsl_emit_cuda`, `theory_artifact`, `theory_registry`,
`theory_uri`.

Headers are part of the header-only `parcae::core` INTERFACE include tree
(`include/` via root `CMakeLists.txt`).
