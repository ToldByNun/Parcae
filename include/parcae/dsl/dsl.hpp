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
///   - ParamIr / PrimitiveIr / TheoryIr / ComposeIr
///   - DslIrApplicator
///   - DslEmitCpu
///   - DslEmitCuda
///   - DslVerifier (exhaustive ≤4 + seeded fuzz + CPU↔CUDA mirror)
///   - DslFuse (inline + fused/staged emit + CPU bench gate)
///   - DslOptimize (const-fold + z29_inv hoist)
///   - DslLaunchPlan (twin 1D / HistFast 2D grids)
/// Later commits fill peak sanity, DslBuildIr, registry.

#include "parcae/dsl/compose_ir.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_ast_json_ingest.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_ast_limits.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_emit_cpu.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
#include "parcae/dsl/dsl_fuse.hpp"
#include "parcae/dsl/dsl_ir_applicator.hpp"
#include "parcae/dsl/dsl_launch_plan.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_semantic_gate.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/dsl_verifier.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/primitive_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"

#endif // DSL_HPP
