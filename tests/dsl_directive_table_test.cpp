#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_directive_table.hpp>
#include <parcae/dsl/dsl_divergence_gate.hpp>
#include <parcae/dsl/dsl_host_glue.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/host_glue_ir.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

[[nodiscard]] std::string wrap_doc(const std::string& module_json, const std::string& directives_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.1.0",)" +
           R"("source_path":"theories/ign.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" +
           R"("ok":true,)" +
           R"("directives":)" + directives_json + "," +
           R"("module":)" + module_json + "}";
}

[[nodiscard]] DslAstDocument ingest_or_fail(
    const std::string& module_json,
    const std::string& directives_json = "[]") {
    const StatusOr<DslAstDocument> doc =
        DslAstJsonIngest::parse_text(wrap_doc(module_json, directives_json));
    REQUIRE(doc.ok());
    return doc.value();
}

[[nodiscard]] std::string hotloop_if_module() {
    return R"({
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
            "value":{"kind":"Name","id":"x","ctx":"Load","lineno":6,"col_offset":15}
          }],
          "orelse":[]
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],
      "type_ignores":[]
    })";
}

[[nodiscard]] std::string hotloop_for_module() {
    return R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":8}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"For","lineno":5,"col_offset":4,
          "target":{"kind":"Name","id":"i","ctx":"Store","lineno":5,"col_offset":8},
          "iter":{"kind":"Call","lineno":5,"col_offset":13,
            "func":{"kind":"Name","id":"range","ctx":"Load","lineno":5,"col_offset":13},
            "args":[{"kind":"Constant","value":3,"lineno":5,"col_offset":19}],
            "keywords":[]},
          "body":[{
            "kind":"Pass","lineno":6,"col_offset":8
          }],
          "orelse":[]
        },{
          "kind":"Return","lineno":7,"col_offset":4,
          "value":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":11}
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],
      "type_ignores":[]
    })";
}

[[nodiscard]] std::string unbounded_while_module() {
    return R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"While","lineno":3,"col_offset":0,
        "test":{"kind":"Name","id":"keep_going","ctx":"Load","lineno":3,"col_offset":6},
        "body":[{"kind":"Pass","lineno":4,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })";
}

}  // namespace

TEST_CASE("DslDirectiveTable rejects ignores without allow flag", "[dsl][directives][E031]") {
    const DslAstDocument doc = ingest_or_fail(
        hotloop_if_module(),
        R"([{"lineno":4,"flag":"divergent_branch","raw":"#ignore DSL_FLAG:divergent_branch"}])");

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(false);
    const StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE_FALSE(table.ok());
    REQUIRE(table.status().message().find("E031") != std::string::npos);
    REQUIRE(table.status().message().find("allow-dsl-ignores") != std::string::npos);
}

TEST_CASE("DslDirectiveTable rejects unknown flag", "[dsl][directives][E031]") {
    const DslAstDocument doc = ingest_or_fail(
        hotloop_if_module(),
        R"([{"lineno":4,"flag":"not_a_real_flag","raw":"#ignore DSL_FLAG:not_a_real_flag"}])");

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    const StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE_FALSE(table.ok());
    REQUIRE(table.status().message().find("unknown DSL_FLAG") != std::string::npos);
}

TEST_CASE(
    "divergent_branch suppresses E033 and emits W010",
    "[dsl][directives][W010][E033]") {
    const DslAstDocument doc = ingest_or_fail(
        hotloop_if_module(),
        R"([{"lineno":4,"flag":"divergent_branch","raw":"#ignore DSL_FLAG:divergent_branch"}])");

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE(table.ok());
    REQUIRE(table.value().binding_count() == 1);

    REQUIRE(DslSemanticGate::check(doc, &table.value()).ok());
    const StatusOr<DslDivergenceGate::Report> r =
        DslDivergenceGate::check(doc, "x", &table.value());
    REQUIRE(r.ok());
    REQUIRE(table.value().warnings().size() == 1);
    REQUIRE(table.value().warnings()[0].rule_id() == "W010");
    REQUIRE(table.value().warnings()[0].message().find("divergent_branch") != std::string::npos);
    REQUIRE(table.value().warnings()[0].message().find("E033") != std::string::npos);
    REQUIRE(table.value().flags_applied() == std::vector<std::string>{"divergent_branch"});
}

TEST_CASE(
    "hotloop_restriction suppresses E034 and emits W010",
    "[dsl][directives][W010][E034]") {
    const DslAstDocument doc = ingest_or_fail(
        hotloop_for_module(),
        R"([{"lineno":4,"flag":"hotloop_restriction","raw":"#ignore DSL_FLAG:hotloop_restriction"}])");

    REQUIRE_FALSE(DslSemanticGate::check(doc).ok());  // without table → E034

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE(table.ok());

    const Status st = DslSemanticGate::check(doc, &table.value());
    REQUIRE(st.ok());
    REQUIRE(table.value().warnings().size() == 1);
    REQUIRE(table.value().warnings()[0].rule_id() == "W010");
    REQUIRE(table.value().warnings()[0].message().find("E034") != std::string::npos);
}

TEST_CASE(
    "host_loop_bound suppresses E035 and emits W010",
    "[dsl][directives][W010][E035]") {
    const DslAstDocument doc = ingest_or_fail(
        unbounded_while_module(),
        R"([{"lineno":2,"flag":"host_loop_bound","raw":"#ignore DSL_FLAG:host_loop_bound"}])");

    REQUIRE_FALSE(DslHostGlue::check_errors_only(doc).ok());

    DslDirectiveTable::Options opts;
    (void)opts.set_allow_dsl_ignores(true);
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc, opts);
    REQUIRE(table.ok());

    const StatusOr<DslHostGlue::Program> prog = DslHostGlue::build(doc, &table.value());
    REQUIRE(prog.ok());
    REQUIRE(prog.value().while_count() == 1);
    REQUIRE(prog.value().root()->kind() == HostGlueIr::Kind::WhileBounded);
    REQUIRE(
        prog.value().root()->int_value() == DslDirectiveTable::host_loop_bound_asserted_max);
    REQUIRE(prog.value().root()->bound_kind() == HostGlueIr::BoundKind::HostKnown);
    REQUIRE(table.value().warnings().size() == 1);
    REQUIRE(table.value().warnings()[0].rule_id() == "W010");
    REQUIRE(table.value().flags_applied() == std::vector<std::string>{"host_loop_bound"});
}

TEST_CASE("empty directives table is a no-op", "[dsl][directives]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_if_module(), "[]");
    StatusOr<DslDirectiveTable> table = DslDirectiveTable::build(doc);
    REQUIRE(table.ok());
    REQUIRE(table.value().binding_count() == 0);
    REQUIRE_FALSE(DslDivergenceGate::check_errors_only(doc, "x", &table.value()).ok());
}
