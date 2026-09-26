#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_build_ir.hpp>
#include <parcae/dsl/dsl_compile.hpp>
#include <parcae/dsl/dsl_directive_table.hpp>
#include <parcae/dsl/dsl_divergence_gate.hpp>
#include <parcae/dsl/dsl_emit_cpu.hpp>
#include <parcae/dsl/dsl_emit_cuda.hpp>
#include <parcae/dsl/dsl_host_glue.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <string>
#include <vector>

#ifndef PARCAE_DSL_FIXTURES_DIR
#error "PARCAE_DSL_FIXTURES_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_DIR
#error "PARCAE_PYTHON_DIR must be defined"
#endif
#ifndef PARCAE_PYTHON_EXE
#error "PARCAE_PYTHON_EXE must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path fixture_source(const char* name) {
    return std::filesystem::path(PARCAE_DSL_FIXTURES_DIR) / "sources" / name;
}

[[nodiscard]] DslCompile::Options compile_options(bool allow_ignores = false) {
    DslCompile::Options opt;
    (void)opt.set_python_exe(PARCAE_PYTHON_EXE);
    (void)opt.set_python_path(PARCAE_PYTHON_DIR);
    (void)opt.set_allow_dsl_ignores(allow_ignores);
    return opt;
}

[[nodiscard]] std::string wrap_doc(const std::string& module_json,
                                   const std::string& directives_json = "[]") {
    return std::string("{") + R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.1.0",)" + R"("source_path":"golden.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" + R"("ok":true,)" + R"("directives":)" +
           directives_json + "," + R"("module":)" + module_json + "}";
}

[[nodiscard]] DslAstDocument ingest(const std::string& module, const std::string& dirs = "[]") {
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(wrap_doc(module, dirs));
    REQUIRE(doc.ok());
    return doc.value();
}

} // namespace

// --- Acceptance matrix goldens (plan § Verification) ---

TEST_CASE("golden: OuterControl For ok / HotLoop For -> E034", "[dsl][golden][scope][gate]") {
    const DslAstDocument outer = ingest(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Call","lineno":2,"col_offset":9,
          "func":{"kind":"Name","id":"range","ctx":"Load","lineno":2,"col_offset":9},
          "args":[{"kind":"Constant","value":3,"lineno":2,"col_offset":15}],"keywords":[]},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],"type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(outer).ok());

    const DslAstDocument hot = ingest(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":8}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"For","lineno":4,"col_offset":4,
          "target":{"kind":"Name","id":"i","ctx":"Store","lineno":4,"col_offset":8},
          "iter":{"kind":"Call","lineno":4,"col_offset":13,
            "func":{"kind":"Name","id":"range","ctx":"Load","lineno":4,"col_offset":13},
            "args":[{"kind":"Constant","value":2,"lineno":4,"col_offset":19}],"keywords":[]},
          "body":[{"kind":"Pass","lineno":5,"col_offset":8}],
          "orelse":[]
        },{
          "kind":"Return","lineno":6,"col_offset":4,
          "value":{"kind":"Name","id":"x","ctx":"Load","lineno":6,"col_offset":11}
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],"type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(hot);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E034") != std::string::npos);
}

TEST_CASE("golden: HotLoop if cipher -> E033 / Param -> W011", "[dsl][golden][divergence]") {
    const auto make_if = [](const char* left_id) {
        return std::string(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":8},
          {"kind":"arg","arg":"a","lineno":3,"col_offset":11}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"If","lineno":4,"col_offset":4,
          "test":{"kind":"Compare","lineno":4,"col_offset":7,
            "left":{"kind":"Name","id":")") +
               left_id +
               R"(","ctx":"Load","lineno":4,"col_offset":7},
            "ops":["Eq"],
            "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":12}]},
          "body":[{
            "kind":"Return","lineno":5,"col_offset":8,
            "value":{"kind":"Name","id":"x","ctx":"Load","lineno":5,"col_offset":15}
          }],
          "orelse":[{
            "kind":"Return","lineno":7,"col_offset":8,
            "value":{"kind":"Name","id":"a","ctx":"Load","lineno":7,"col_offset":15}
          }]
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],"type_ignores":[]
    })";
    };

    REQUIRE_FALSE(DslDivergenceGate::check_errors_only(ingest(make_if("x"))).ok());

    const StatusOr<DslDivergenceGate::Report> ok = DslDivergenceGate::check(ingest(make_if("a")));
    REQUIRE(ok.ok());
    REQUIRE_FALSE(ok.value().empty());
    REQUIRE(ok.value().warnings().front().rule_id() == "W011");
}

TEST_CASE("golden: ignore divergent_branch -> W010 + prefer_branch emit",
          "[dsl][golden][directive][directives][divergence][emit]") {
    const std::string module = R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":8},
          {"kind":"arg","arg":"a","lineno":3,"col_offset":11}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"If","lineno":5,"col_offset":4,
          "test":{"kind":"Compare","lineno":5,"col_offset":7,
            "left":{"kind":"Name","id":"x","ctx":"Load","lineno":5,"col_offset":7},
            "ops":["Eq"],
            "comparators":[{"kind":"Constant","value":0,"lineno":5,"col_offset":12}]},
          "body":[{
            "kind":"Return","lineno":6,"col_offset":8,
            "value":{"kind":"Constant","value":1,"lineno":6,"col_offset":15}
          }],
          "orelse":[{
            "kind":"Return","lineno":8,"col_offset":8,
            "value":{"kind":"Name","id":"x","ctx":"Load","lineno":8,"col_offset":15}
          }]
        }],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],
          "keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"poly","lineno":2}},
            {"kind":"keyword","arg":"signature","value":{"kind":"Constant","value":"(x: Z29, a: Z29) -> Z29","lineno":2}}
          ]
        }],
        "returns":null
      }],"type_ignores":[]
    })";
    const DslAstDocument doc = ingest(
        module,
        R"([{"lineno":4,"flag":"divergent_branch","raw":"#ignore DSL_FLAG:divergent_branch"}])");

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE(table.ok());
    REQUIRE(DslSemanticGate::check(doc, &table.value()).ok());
    REQUIRE(DslDivergenceGate::check_errors_only(doc, "x", &table.value()).ok());
    REQUIRE(table.value().warnings().size() == 1);
    REQUIRE(table.value().warnings()[0].rule_id() == "W010");

    const StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    REQUIRE_FALSE(unit.value().primitives().empty());
    const Z29Expr::Ptr& body = unit.value().primitives().front().body();
    REQUIRE(body->kind() == Z29Expr::Kind::Select);
    REQUIRE(body->prefer_branch());

    const StatusOr<std::string> cpu = DslEmitCpu::emit_expr(body, "x", "input[i]");
    REQUIRE(cpu.ok());
    REQUIRE(cpu.value().find('?') != std::string::npos);

    const StatusOr<std::string> cuda = DslEmitCuda::emit_expr(body, "x", "in[i]");
    REQUIRE(cuda.ok());
    REQUIRE(cuda.value().find('?') != std::string::npos);

    // Both Select arms covered on a tiny stream.
    std::vector<Index29> in{Index29{0}, Index29{3}};
    std::vector<Index29> out(in.size());
    REQUIRE(DslIrApplicator::apply_into(body, "x", {{"a", Index29{9}}}, in, out).ok());
    REQUIRE(out[0].value() == 1);
    REQUIRE(out[1].value() == 3);
}

TEST_CASE("golden: host_loop_bound suppresses E035", "[dsl][golden][directive][hostglue]") {
    const DslAstDocument doc = ingest(
        R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"While","lineno":3,"col_offset":0,
        "test":{"kind":"Name","id":"keep","ctx":"Load","lineno":3,"col_offset":6},
        "body":[{"kind":"Pass","lineno":4,"col_offset":4}],
        "orelse":[]
      }],"type_ignores":[]
    })",
        R"([{"lineno":2,"flag":"host_loop_bound","raw":"#ignore DSL_FLAG:host_loop_bound"}])");

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE(table.ok());
    REQUIRE(DslHostGlue::check_errors_only(doc, &table.value()).ok());
    REQUIRE(table.value().flags_applied().front() == "host_loop_bound");
}

TEST_CASE("golden: compile smart_select_param without allow-dsl-ignores",
          "[dsl][golden][compile][select]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_golden_select_param";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    StatusOr<DslCompile::Result> result = DslCompile::compile_file(
        fixture_source("smart_select_param.py"), root, compile_options(false));
    if (!result.ok()) {
        INFO(result.status().message());
    }
    REQUIRE(result.ok());
    REQUIRE(result.value().artifacts().size() == 1);
    REQUIRE(result.value().artifacts().front().uri().name() == "smart_select_param");
    // Param if → W011 (warnings collected); no DSL ignores applied.
    REQUIRE(result.value().dsl_ignores_applied().empty());
    bool saw_w011 = false;
    for (const DslDiag& w : result.value().warnings()) {
        if (w.rule_id() == "W011") {
            saw_w011 = true;
        }
    }
    REQUIRE(saw_w011);

    const std::filesystem::path cu =
        root / "smart_select_param" / "1" / "emitted" / "SmartSelectParamKernel.cu";
    REQUIRE(std::filesystem::is_regular_file(cu));
    std::ifstream in(cu);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(text.find("Z29Device::select") != std::string::npos);

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("golden: compile ignore divergent requires --allow-dsl-ignores",
          "[dsl][golden][compile][directive][directives]") {
    REQUIRE(DslCompile::pipeline_ready(compile_options()));
    const auto root = std::filesystem::temp_directory_path() / "parcae_dsl_golden_ignore";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root, ec);

    const auto src = fixture_source("smart_ignore_divergent.py");
    REQUIRE(std::filesystem::is_regular_file(src));

    StatusOr<DslCompile::Result> denied =
        DslCompile::compile_file(src, root, compile_options(false));
    REQUIRE_FALSE(denied.ok());
    REQUIRE(denied.status().message().find("E031") != std::string::npos);
    REQUIRE(denied.status().message().find("allow-dsl-ignores") != std::string::npos);

    StatusOr<DslCompile::Result> allowed =
        DslCompile::compile_file(src, root, compile_options(true));
    if (!allowed.ok()) {
        INFO(allowed.status().message());
    }
    REQUIRE(allowed.ok());
    REQUIRE(allowed.value().dsl_ignores_applied().size() == 1);
    REQUIRE(allowed.value().dsl_ignores_applied().front() == "divergent_branch");
    bool saw_w010 = false;
    for (const DslDiag& w : allowed.value().warnings()) {
        if (w.rule_id() == "W010") {
            saw_w010 = true;
        }
    }
    REQUIRE(saw_w010);
    REQUIRE_FALSE(allowed.value().artifacts().front().dsl_ignores_applied().empty());

    const std::filesystem::path cpu = root / "smart_ignore_divergent" / "1" / "cpu_reference.hpp";
    REQUIRE(std::filesystem::is_regular_file(cpu));
    std::ifstream in(cpu);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // prefer_branch → C++ ?: rather than Z29::select
    REQUIRE(text.find("?") != std::string::npos);

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("golden: z29_det / z29_matmul / z29_autokey_shift Calls pass gate",
          "[dsl][golden][gate][matrix]") {
    const DslAstDocument ok = ingest(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,"value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_det","ctx":"Load","lineno":1,"col_offset":0},
          "args":[{"kind":"Name","id":"m","ctx":"Load","lineno":1,"col_offset":8}],
          "keywords":[]
        }
      },{
        "kind":"Expr","lineno":2,"col_offset":0,"value":{
          "kind":"Call","lineno":2,"col_offset":0,
          "func":{"kind":"Name","id":"z29_matmul","ctx":"Load","lineno":2,"col_offset":0},
          "args":[
            {"kind":"Name","id":"m","ctx":"Load","lineno":2,"col_offset":11},
            {"kind":"Name","id":"v","ctx":"Load","lineno":2,"col_offset":14}
          ],
          "keywords":[]
        }
      },{
        "kind":"Expr","lineno":3,"col_offset":0,"value":{
          "kind":"Call","lineno":3,"col_offset":0,
          "func":{"kind":"Name","id":"z29_autokey_shift","ctx":"Load","lineno":3,"col_offset":0},
          "args":[
            {"kind":"Name","id":"x","ctx":"Load","lineno":3,"col_offset":18},
            {"kind":"Name","id":"lag","ctx":"Load","lineno":3,"col_offset":21}
          ],
          "keywords":[]
        }
      }],"type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(ok).ok());

    const DslAstDocument bad = ingest(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,"value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_not_a_real_op","ctx":"Load","lineno":1,"col_offset":0},
          "args":[{"kind":"Name","id":"x","ctx":"Load","lineno":1,"col_offset":20}],
          "keywords":[]
        }
      }],"type_ignores":[]
    })");
    const Status gate = DslSemanticGate::check(bad);
    REQUIRE_FALSE(gate.ok());
    REQUIRE(gate.message().find("E032") != std::string::npos);
}
