#ifndef DSL_HPP
#define DSL_HPP

/// Umbrella include for the theory-DSL compiler headers (scaffold).
/// Normative: docs/spec/dsl.md, docs/spec/dsl-ast-json.md, docs/spec/theory-artifact.md
/// Architecture: docs/architecture/python-transpiler.md
///
/// Implemented so far:
///   - DslSpecVersion / DslAstJsonVersion
///   - DslDiag / DslRuleId
/// Later commits fill ingest, IR, verify, emit, registry.

#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"

#endif // DSL_HPP
