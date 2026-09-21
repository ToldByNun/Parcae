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
///   - DslPeakSanity (ThroughputTiers peak / SLO gate)
///   - TheoryUri / TheoryArtifact (manifest writer embeds dsl_spec_version)
///   - TheoryRegistry (rejects stale dsl_spec_version)
///   - TheoryValidate (manifest + dsl_spec + path files + envelope content)
///   - TheorySweep (expand sweep.param_grid plan)
///   - TheoryEnvelopeBridge (envelope_template ↔ TransformEnvelope / theory URI)
///   - TheoryApplyIr / TheoryDispatch (ApplyTransform hook for theory URIs)
///   - DslBuildIr / DslCompile (ast_dump → artifact)
///   - Example: theories/examples/new_math_example.py
///   - Example: theories/examples/full_lifecycle_example.py (@ComposedTheory)
/// Later commits: CI examples / docs index / version bump.

#include "parcae/dsl/compose_ir.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_ast_json_ingest.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_ast_limits.hpp"
#include "parcae/dsl/dsl_build_ir.hpp"
#include "parcae/dsl/dsl_catalog_builtins.hpp"
#include "parcae/dsl/dsl_compile.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_emit_cpu.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
#include "parcae/dsl/dsl_fuse.hpp"
#include "parcae/dsl/dsl_ir_applicator.hpp"
#include "parcae/dsl/dsl_launch_plan.hpp"
#include "parcae/dsl/dsl_optimize.hpp"
#include "parcae/dsl/dsl_peak_sanity.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/dsl_semantic_gate.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/dsl_verifier.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/primitive_ir.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_sweep.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/dsl/theory_validate.hpp"
#include "parcae/dsl/z29_expr.hpp"

#endif // DSL_HPP
