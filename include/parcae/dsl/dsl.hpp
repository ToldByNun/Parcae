#ifndef DSL_HPP
#define DSL_HPP

/// Umbrella include for the theory-DSL compiler headers (scaffold).
/// Normative: docs/spec/dsl.md, docs/spec/dsl-ast-json.md, docs/spec/theory-artifact.md
/// Architecture: docs/architecture/python-transpiler.md
///
/// Implemented so far:
///   - DslSpecVersion / DslAstJsonVersion
///   - DslDiag / DslRuleId
///   - DslAst / DslAstLimits / DslAstJsonIngest
///   - DslSemanticGate
///   - Z29Expr
/// Later commits fill PrimitiveIr / TheoryIr, verify, emit, registry.

#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_ast_json_ingest.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_ast_limits.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_semantic_gate.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/z29_expr.hpp"

#endif // DSL_HPP
