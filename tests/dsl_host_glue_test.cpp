#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_host_glue.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/host_glue_ir.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace {

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" +
           R"("source_path":"theories/host.py",)" +
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

}  // namespace

TEST_CASE(
    "DslHostGlue lowers const range for to ForRange ConstUnroll",
    "[dsl][hostglue]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Call","lineno":2,"col_offset":9,
          "func":{"kind":"Name","id":"range","ctx":"Load","lineno":2,"col_offset":9},
          "args":[{"kind":"Constant","value":3,"lineno":2,"col_offset":15}],
          "keywords":[]},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    REQUIRE(DslSemanticGate::check(doc).ok());
    const StatusOr<DslHostGlue::Program> prog = DslHostGlue::build(doc);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().for_count() == 1);
    REQUIRE(prog.value().root()->kind() == HostGlueIr::Kind::ForRange);
    REQUIRE(prog.value().root()->bound_kind() == HostGlueIr::BoundKind::ConstUnroll);
    REQUIRE(prog.value().root()->name() == "i");
    REQUIRE(prog.value().root()->children()[1]->kind() == HostGlueIr::Kind::ConstInt);
    REQUIRE(prog.value().root()->children()[1]->int_value() == 3);
}

TEST_CASE(
    "DslHostGlue lowers Param range for to HostKnown",
    "[dsl][hostglue]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Call","lineno":2,"col_offset":9,
          "func":{"kind":"Name","id":"range","ctx":"Load","lineno":2,"col_offset":9},
          "args":[{"kind":"Name","id":"n","ctx":"Load","lineno":2,"col_offset":15}],
          "keywords":[]},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslHostGlue::Program> prog = DslHostGlue::build(doc);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().root()->bound_kind() == HostGlueIr::BoundKind::HostKnown);
    REQUIRE(prog.value().root()->children()[1]->kind() == HostGlueIr::Kind::Name);
    REQUIRE(prog.value().root()->children()[1]->name() == "n");
}

TEST_CASE(
    "DslHostGlue rejects non-range for with E035",
    "[dsl][hostglue][E035]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":2,"col_offset":9},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    const Status st = DslHostGlue::check_errors_only(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E035") != std::string::npos);
}

TEST_CASE(
    "DslHostGlue rejects unbounded while with E035",
    "[dsl][hostglue][E035]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"While","lineno":2,"col_offset":0,
        "test":{"kind":"Constant","value":true,"lineno":2,"col_offset":6},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    const Status st = DslHostGlue::check_errors_only(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E035") != std::string::npos);
    REQUIRE(st.message().find("finite bound") != std::string::npos);
}

TEST_CASE(
    "DslHostGlue accepts while i < N const as WhileBounded",
    "[dsl][hostglue]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"While","lineno":2,"col_offset":0,
        "test":{"kind":"Compare","lineno":2,"col_offset":6,
          "left":{"kind":"Name","id":"i","ctx":"Load","lineno":2,"col_offset":6},
          "ops":["Lt"],
          "comparators":[{"kind":"Constant","value":4,"lineno":2,"col_offset":10}]},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslHostGlue::Program> prog = DslHostGlue::build(doc);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().while_count() == 1);
    REQUIRE(prog.value().root()->kind() == HostGlueIr::Kind::WhileBounded);
    REQUIRE(prog.value().root()->int_value() == 4);
    REQUIRE(prog.value().root()->bound_kind() == HostGlueIr::BoundKind::ConstUnroll);
}

TEST_CASE(
    "DslHostGlue skips HotLoop for (E034 is SemanticGate)",
    "[dsl][hostglue]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"For","lineno":4,"col_offset":4,
          "target":{"kind":"Name","id":"i","ctx":"Store","lineno":4,"col_offset":8},
          "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":4,"col_offset":13},
          "body":[{"kind":"Pass","lineno":5,"col_offset":8}],
          "orelse":[]
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],
      "type_ignores":[]
    })");

    // Semantic gate rejects HotLoop For with E034.
    REQUIRE_FALSE(DslSemanticGate::check(doc).ok());
    // HostGlue skips HotLoop body → success with empty program.
    const StatusOr<DslHostGlue::Program> prog = DslHostGlue::build(doc);
    REQUIRE(prog.ok());
    REQUIRE(prog.value().for_count() == 0);
}

TEST_CASE("HostGlueIr kind strings", "[dsl][hostglue]") {
    const HostGlueIr::Ptr n = HostGlueIr::make_const_int(2);
    REQUIRE(n->kind_string() == "ConstInt");
    const HostGlueIr::Ptr f = HostGlueIr::make_for_range(
        "i",
        HostGlueIr::make_const_int(0),
        HostGlueIr::make_const_int(1),
        HostGlueIr::make_const_int(1),
        HostGlueIr::make_pass(),
        HostGlueIr::BoundKind::HostKnown);
    REQUIRE(f->bound_kind_string() == "HostKnown");
}
