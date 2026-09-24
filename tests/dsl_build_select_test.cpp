#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_build_ir.hpp>
#include <parcae/dsl/dsl_divergence_gate.hpp>
#include <parcae/dsl/dsl_emit_cpu.hpp>
#include <parcae/dsl/dsl_emit_cuda.hpp>
#include <parcae/dsl/dsl_optimize.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace {

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" +
           R"("source_path":"theories/select_prim.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" +
           R"("ok":true,)" +
           R"("module":)" + module_json + "}";
}

[[nodiscard]] DslAstDocument ingest_or_fail(const std::string& module_json) {
    const StatusOr<DslAstDocument> doc =
        DslAstJsonIngest::parse_text(minimal_success_doc(module_json));
    REQUIRE(doc.ok());
    return doc.value();
}

[[nodiscard]] std::string primitive_module(const std::string& body_stmts_json) {
    return std::string(R"JSON({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"branchy","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":12},
          {"kind":"arg","arg":"a","lineno":3,"col_offset":15}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":)JSON") +
           body_stmts_json +
           R"JSON(,
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],
          "keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"branchy","lineno":2,"col_offset":20}},
            {"kind":"keyword","arg":"signature","value":{"kind":"Constant","value":"(x: Z29, a: Z29) -> Z29","lineno":2,"col_offset":40}}
          ]
        }],
        "returns":null
      }],
      "type_ignores":[]
    })JSON";
}

}  // namespace

TEST_CASE(
    "DslBuildIr lowers HotLoop If/else to Select",
    "[dsl][build][select]") {
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"If","lineno":4,"col_offset":4,
      "test":{"kind":"Compare","lineno":4,"col_offset":7,
        "left":{"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":7},
        "ops":["Eq"],
        "comparators":[{"kind":"Constant","value":1,"lineno":4,"col_offset":12}]},
      "body":[{
        "kind":"Return","lineno":5,"col_offset":8,
        "value":{"kind":"Constant","value":7,"lineno":5,"col_offset":15}
      }],
      "orelse":[{
        "kind":"Return","lineno":7,"col_offset":8,
        "value":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":15}
      }]
    }])"));

    REQUIRE(DslSemanticGate::check(doc).ok());
    REQUIRE(DslDivergenceGate::check_errors_only(doc).ok());

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    REQUIRE(unit.value().primitives().size() == 1);
    const Z29Expr::Ptr& body = unit.value().primitives().front().body();
    REQUIRE(body);
    REQUIRE(body->kind() == Z29Expr::Kind::Select);
    REQUIRE(body->if_true()->kind() == Z29Expr::Kind::Const);
    REQUIRE(body->if_true()->const_value() == 7);
    REQUIRE(body->if_false()->kind() == Z29Expr::Kind::Var);
    REQUIRE(body->if_false()->name() == "x");

    Z29Expr::Env env{{"x", Index29{3}}, {"a", Index29{1}}};
    REQUIRE(body->eval(env).value().value() == 7);
    env["a"] = Index29{0};
    REQUIRE(body->eval(env).value().value() == 3);
}

TEST_CASE(
    "DslBuildIr lowers HotLoop IfExp ternary to Select",
    "[dsl][build][select]") {
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"Return","lineno":4,"col_offset":4,
      "value":{
        "kind":"IfExp","lineno":4,"col_offset":11,
        "test":{"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":11},
        "body":{"kind":"Constant","value":5,"lineno":4,"col_offset":18},
        "orelse":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":25}
      }
    }])"));

    REQUIRE(DslSemanticGate::check(doc).ok());
    REQUIRE(DslDivergenceGate::check_errors_only(doc).ok());

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    const Z29Expr::Ptr& body = unit.value().primitives().front().body();
    REQUIRE(body->kind() == Z29Expr::Kind::Select);

    Z29Expr::Env env{{"x", Index29{9}}, {"a", Index29{0}}};
    REQUIRE(body->eval(env).value().value() == 9);
    env["a"] = Index29{2};
    REQUIRE(body->eval(env).value().value() == 5);
}

TEST_CASE(
    "DslBuildIr + Optimize folds const-Select from HotLoop If",
    "[dsl][build][select][optimize]") {
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"If","lineno":4,"col_offset":4,
      "test":{"kind":"Constant","value":true,"lineno":4,"col_offset":7},
      "body":[{
        "kind":"Return","lineno":5,"col_offset":8,
        "value":{"kind":"BinOp","lineno":5,"col_offset":15,"op":"Add",
          "left":{"kind":"Name","id":"x","ctx":"Load","lineno":5,"col_offset":15},
          "right":{"kind":"Constant","value":1,"lineno":5,"col_offset":19}}
      }],
      "orelse":[{
        "kind":"Return","lineno":7,"col_offset":8,
        "value":{"kind":"Constant","value":0,"lineno":7,"col_offset":15}
      }]
    }])"));

    REQUIRE(DslDivergenceGate::check_errors_only(doc).ok());
    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    REQUIRE(unit.value().primitives().front().body()->kind() == Z29Expr::Kind::Select);

    const StatusOr<Z29Expr::Ptr> folded =
        DslOptimize::const_fold(unit.value().primitives().front().body());
    REQUIRE(folded.ok());
    // Dead-arm: true → x+1
    REQUIRE(folded.value()->kind() == Z29Expr::Kind::Add);
    REQUIRE(folded.value()->left()->name() == "x");
}

TEST_CASE(
    "DslBuildIr rejects HotLoop if without else",
    "[dsl][build][select]") {
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"If","lineno":4,"col_offset":4,
      "test":{"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":7},
      "body":[{
        "kind":"Return","lineno":5,"col_offset":8,
        "value":{"kind":"Name","id":"x","ctx":"Load","lineno":5,"col_offset":15}
      }],
      "orelse":[]
    }])"));

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE_FALSE(unit.ok());
    REQUIRE(unit.status().message().find("else") != std::string::npos);
}

TEST_CASE(
    "DslBuildIr lowers elif chain to nested Select",
    "[dsl][build][select]") {
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"If","lineno":4,"col_offset":4,
      "test":{"kind":"Compare","lineno":4,"col_offset":7,
        "left":{"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":7},
        "ops":["Eq"],
        "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":12}]},
      "body":[{
        "kind":"Return","lineno":5,"col_offset":8,
        "value":{"kind":"Constant","value":1,"lineno":5,"col_offset":15}
      }],
      "orelse":[{
        "kind":"If","lineno":6,"col_offset":4,
        "test":{"kind":"Compare","lineno":6,"col_offset":9,
          "left":{"kind":"Name","id":"a","ctx":"Load","lineno":6,"col_offset":9},
          "ops":["Eq"],
          "comparators":[{"kind":"Constant","value":1,"lineno":6,"col_offset":14}]},
        "body":[{
          "kind":"Return","lineno":7,"col_offset":8,
          "value":{"kind":"Constant","value":2,"lineno":7,"col_offset":15}
        }],
        "orelse":[{
          "kind":"Return","lineno":9,"col_offset":8,
          "value":{"kind":"Constant","value":3,"lineno":9,"col_offset":15}
        }]
      }]
    }])"));

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    const Z29Expr::Ptr& body = unit.value().primitives().front().body();
    REQUIRE(body->kind() == Z29Expr::Kind::Select);
    REQUIRE(body->if_false()->kind() == Z29Expr::Kind::Select);

    Z29Expr::Env env{{"x", Index29{0}}, {"a", Index29{1}}};
    REQUIRE(body->eval(env).value().value() == 2);
    env["a"] = Index29{2};
    REQUIRE(body->eval(env).value().value() == 3);
}

TEST_CASE(
    "ThreadVarying HotLoop If → Select prefer_branch for emit",
    "[dsl][build][select][emit]") {
    // Gate would E033 without ignore; BuildIr still lowers when called directly
    // (compile path honors ignore first). prefer_branch marks divergent emit.
    const DslAstDocument doc = ingest_or_fail(primitive_module(R"([{
      "kind":"If","lineno":4,"col_offset":4,
      "test":{"kind":"Compare","lineno":4,"col_offset":7,
        "left":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":7},
        "ops":["Eq"],
        "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":12}]},
      "body":[{
        "kind":"Return","lineno":5,"col_offset":8,
        "value":{"kind":"Constant","value":1,"lineno":5,"col_offset":15}
      }],
      "orelse":[{
        "kind":"Return","lineno":7,"col_offset":8,
        "value":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":15}
      }]
    }])"));

    REQUIRE_FALSE(DslDivergenceGate::check_errors_only(doc).ok());

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    const Z29Expr::Ptr& body = unit.value().primitives().front().body();
    REQUIRE(body->kind() == Z29Expr::Kind::Select);
    REQUIRE(body->prefer_branch());

    const StatusOr<std::string> cpu = DslEmitCpu::emit_expr(body, "x", "input[i]");
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().find("?") != std::string::npos);
    REQUIRE(cpu.value().find("Z29::select(") == std::string::npos);

    const StatusOr<std::string> cuda = DslEmitCuda::emit_expr(body, "x", "in[i]");
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value().find("?") != std::string::npos);
    REQUIRE(cuda.value().find("Z29Device::select(") == std::string::npos);
}
