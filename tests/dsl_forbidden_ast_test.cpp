#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>
#include <utility>

namespace {

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" +
           R"("source_path":"theories/x.py",)" +
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

/// Module wrapping a single top-level statement node (kind at lineno 7).
[[nodiscard]] std::string module_with_stmt(const std::string& stmt_json) {
    return std::string(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[)") +
           stmt_json + R"(],
      "type_ignores":[]
    })";
}

/// Module wrapping `Expr(value=…)` for expression-level forbidden kinds.
[[nodiscard]] std::string module_with_expr_value(const std::string& value_json) {
    return module_with_stmt(
        std::string(R"({
        "kind":"Expr","lineno":7,"col_offset":0,
        "value":)") +
        value_json + "}");
}

void require_e031_forbidden(const Status& st, const std::string& kind_substr) {
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find(DslRuleId::E031_forbidden_construct) != std::string::npos);
    REQUIRE(st.message().find(kind_substr) != std::string::npos);
    REQUIRE(st.message().find("theories/x.py:7:") != std::string::npos);
}

}  // namespace

TEST_CASE("forbidden statement kinds → stable E031", "[dsl][gate][forbidden]") {
    const auto row = GENERATE(
        table<std::string, std::string>(
            {// kind substring expected in message, statement JSON
             {"AsyncFunctionDef",
              R"({"kind":"AsyncFunctionDef","name":"f","lineno":7,"col_offset":0,
                  "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                          "kw_defaults":[],"defaults":[]},
                  "body":[{"kind":"Pass","lineno":8,"col_offset":4}],
                  "decorator_list":[],"returns":null})"},
             {"While",
              R"({"kind":"While","lineno":7,"col_offset":0,
                  "test":{"kind":"Constant","value":true,"lineno":7,"col_offset":6},
                  "body":[{"kind":"Pass","lineno":8,"col_offset":4}],"orelse":[]})"},
             {"AsyncFor",
              R"({"kind":"AsyncFor","lineno":7,"col_offset":0,
                  "target":{"kind":"Name","id":"i","ctx":"Store","lineno":7,"col_offset":10},
                  "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":7,"col_offset":15},
                  "body":[{"kind":"Pass","lineno":8,"col_offset":4}],"orelse":[]})"},
             {"With",
              R"({"kind":"With","lineno":7,"col_offset":0,
                  "items":[],"body":[{"kind":"Pass","lineno":8,"col_offset":4}]})"},
             {"AsyncWith",
              R"({"kind":"AsyncWith","lineno":7,"col_offset":0,
                  "items":[],"body":[{"kind":"Pass","lineno":8,"col_offset":4}]})"},
             {"Try",
              R"({"kind":"Try","lineno":7,"col_offset":0,
                  "body":[{"kind":"Pass","lineno":8,"col_offset":4}],
                  "handlers":[{"kind":"ExceptHandler","lineno":9,"col_offset":0,
                               "type":null,"name":null,
                               "body":[{"kind":"Pass","lineno":10,"col_offset":4}]}],
                  "orelse":[],"finalbody":[]})"},
             {"Global",
              R"({"kind":"Global","lineno":7,"col_offset":0,"names":["x"]})"},
             {"Nonlocal",
              R"({"kind":"Nonlocal","lineno":7,"col_offset":0,"names":["x"]})"},
             {"Delete",
              R"({"kind":"Delete","lineno":7,"col_offset":0,
                  "targets":[{"kind":"Name","id":"x","ctx":"Del","lineno":7,"col_offset":4}]})"},
             {"Assert",
              R"({"kind":"Assert","lineno":7,"col_offset":0,
                  "test":{"kind":"Constant","value":true,"lineno":7,"col_offset":7},
                  "msg":null})"},
             {"For",
              R"({"kind":"For","lineno":7,"col_offset":0,
                  "target":{"kind":"Name","id":"i","ctx":"Store","lineno":7,"col_offset":4},
                  "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":7,"col_offset":9},
                  "body":[{"kind":"Pass","lineno":8,"col_offset":4}],"orelse":[]})"}}));

    const std::string& kind = std::get<0>(row);
    const std::string& stmt = std::get<1>(row);
    const DslAstDocument doc = ingest_or_fail(module_with_stmt(stmt));
    require_e031_forbidden(DslSemanticGate::check(doc), kind);
}

TEST_CASE("forbidden expression kinds → stable E031", "[dsl][gate][forbidden]") {
    const auto row = GENERATE(
        table<std::string, std::string>(
            {{"Await",
              R"({"kind":"Await","lineno":7,"col_offset":0,
                  "value":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":6}})"},
             {"Yield",
              R"({"kind":"Yield","lineno":7,"col_offset":0,
                  "value":{"kind":"Constant","value":1,"lineno":7,"col_offset":6}})"},
             {"YieldFrom",
              R"({"kind":"YieldFrom","lineno":7,"col_offset":0,
                  "value":{"kind":"Name","id":"xs","ctx":"Load","lineno":7,"col_offset":11}})"},
             {"Lambda",
              R"({"kind":"Lambda","lineno":7,"col_offset":0,
                  "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                          "kw_defaults":[],"defaults":[]},
                  "body":{"kind":"Constant","value":1,"lineno":7,"col_offset":7}})"},
             {"ListComp",
              R"({"kind":"ListComp","lineno":7,"col_offset":0,
                  "elt":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":1},
                  "generators":[]})"},
             {"SetComp",
              R"({"kind":"SetComp","lineno":7,"col_offset":0,
                  "elt":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":1},
                  "generators":[]})"},
             {"DictComp",
              R"({"kind":"DictComp","lineno":7,"col_offset":0,
                  "key":{"kind":"Name","id":"k","ctx":"Load","lineno":7,"col_offset":1},
                  "value":{"kind":"Name","id":"v","ctx":"Load","lineno":7,"col_offset":3},
                  "generators":[]})"},
             {"GeneratorExp",
              R"({"kind":"GeneratorExp","lineno":7,"col_offset":0,
                  "elt":{"kind":"Name","id":"x","ctx":"Load","lineno":7,"col_offset":1},
                  "generators":[]})"}}));

    const std::string& kind = std::get<0>(row);
    const std::string& value = std::get<1>(row);
    const DslAstDocument doc = ingest_or_fail(module_with_expr_value(value));
    require_e031_forbidden(DslSemanticGate::check(doc), kind);
}

TEST_CASE("forbidden ExceptHandler alone → E031", "[dsl][gate][forbidden]") {
    // Orphan ExceptHandler (not only nested under Try) must still be rejected.
    const DslAstDocument doc = ingest_or_fail(module_with_stmt(R"({
      "kind":"ExceptHandler","lineno":7,"col_offset":0,
      "type":null,"name":null,
      "body":[{"kind":"Pass","lineno":8,"col_offset":4}]
    })"));
    require_e031_forbidden(DslSemanticGate::check(doc), "ExceptHandler");
}

TEST_CASE("forbidden arguments.vararg → E031", "[dsl][gate][forbidden]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"f","lineno":7,"col_offset":0,
        "args":{"kind":"arguments","lineno":7,"col_offset":0,
                "posonlyargs":[],"args":[],"kwonlyargs":[],"kw_defaults":[],"defaults":[],
                "vararg":{"kind":"arg","arg":"xs","lineno":7,"col_offset":6},
                "kwarg":null},
        "body":[{"kind":"Pass","lineno":8,"col_offset":4}],
        "decorator_list":[],"returns":null
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("vararg") != std::string::npos);
    REQUIRE(st.message().find("theories/x.py:7:") != std::string::npos);
}

TEST_CASE("forbidden arguments.kwarg → E031", "[dsl][gate][forbidden]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"f","lineno":7,"col_offset":0,
        "args":{"kind":"arguments","lineno":7,"col_offset":0,
                "posonlyargs":[],"args":[],"kwonlyargs":[],"kw_defaults":[],"defaults":[],
                "vararg":null,
                "kwarg":{"kind":"arg","arg":"kw","lineno":7,"col_offset":6}},
        "body":[{"kind":"Pass","lineno":8,"col_offset":4}],
        "decorator_list":[],"returns":null
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("kwarg") != std::string::npos);
}

TEST_CASE("forbidden nested ClassDef → E031 with location", "[dsl][gate][forbidden]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"outer","lineno":1,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"ClassDef","name":"Inner","lineno":7,"col_offset":4,
          "bases":[],"keywords":[],"decorator_list":[],
          "body":[{"kind":"Pass","lineno":8,"col_offset":8}]
        }],
        "decorator_list":[],"returns":null
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("ClassDef") != std::string::npos);
    REQUIRE(st.message().find("theories/x.py:7:4:") != std::string::npos);
}

TEST_CASE("forbidden Import and bad import module keep stable rule ids", "[dsl][gate][forbidden]") {
    SECTION("bare Import → E031") {
        const DslAstDocument doc = ingest_or_fail(module_with_stmt(R"({
          "kind":"Import","lineno":7,"col_offset":0,
          "names":[{"kind":"alias","name":"os","asname":null,"lineno":7,"col_offset":7}]
        })"));
        require_e031_forbidden(DslSemanticGate::check(doc), "Import");
    }
    SECTION("ImportFrom os → E021") {
        const DslAstDocument doc = ingest_or_fail(module_with_stmt(R"({
          "kind":"ImportFrom","lineno":7,"col_offset":0,"module":"os",
          "names":[{"kind":"alias","name":"path","asname":null}],"level":0
        })"));
        const Status st = DslSemanticGate::check(doc);
        REQUIRE_FALSE(st.ok());
        REQUIRE(st.message().find("E021") != std::string::npos);
        REQUIRE(st.message().find("os") != std::string::npos);
        REQUIRE(st.message().find("theories/x.py:7:") != std::string::npos);
    }
}

TEST_CASE("forbidden illegal ops/ctx → E031", "[dsl][gate][forbidden]") {
    SECTION("MatMult") {
        const DslAstDocument doc = ingest_or_fail(module_with_expr_value(R"({
          "kind":"BinOp","lineno":7,"col_offset":0,
          "left":{"kind":"Constant","value":2,"lineno":7,"col_offset":0},
          "op":"MatMult",
          "right":{"kind":"Constant","value":3,"lineno":7,"col_offset":4}
        })"));
        const Status st = DslSemanticGate::check(doc);
        REQUIRE_FALSE(st.ok());
        REQUIRE(st.message().find("E031") != std::string::npos);
        REQUIRE(st.message().find("MatMult") != std::string::npos);
    }
    SECTION("Del ctx") {
        const DslAstDocument doc = ingest_or_fail(module_with_expr_value(R"({
          "kind":"Name","id":"x","ctx":"Del","lineno":7,"col_offset":0
        })"));
        const Status st = DslSemanticGate::check(doc);
        REQUIRE_FALSE(st.ok());
        REQUIRE(st.message().find("E031") != std::string::npos);
        REQUIRE(st.message().find("Del") != std::string::npos);
    }
}

TEST_CASE("module-level ClassDef still allowed", "[dsl][gate][forbidden]") {
    const DslAstDocument doc = ingest_or_fail(module_with_stmt(R"({
      "kind":"ClassDef","name":"MyAffine","lineno":7,"col_offset":0,
      "bases":[],"keywords":[],"decorator_list":[],
      "body":[{"kind":"Pass","lineno":8,"col_offset":4}]
    })"));
    REQUIRE(DslSemanticGate::check(doc).ok());
}
