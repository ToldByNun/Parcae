#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

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

}  // namespace

TEST_CASE("DslSemanticGate accepts empty Module", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE(st.ok());
}

TEST_CASE("DslSemanticGate accepts ImportFrom parcae.dsl.math", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"ImportFrom","lineno":3,"col_offset":0,"end_lineno":3,"end_col_offset":40,
        "module":"parcae.dsl.math",
        "names":[{"kind":"alias","name":"z29_add","asname":null,"lineno":3,"col_offset":0}],
        "level":0
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(doc).ok());
}

TEST_CASE("DslSemanticGate accepts BinOp Mult primitive-style body", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":1,"col_offset":0,"end_lineno":2,"end_col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],"kw_defaults":[],
                "defaults":[],"vararg":null,"kwarg":null,
                "lineno":1,"col_offset":0,"end_lineno":1,"end_col_offset":0},
        "body":[{
          "kind":"Return","lineno":2,"col_offset":4,"end_lineno":2,"end_col_offset":14,
          "value":{
            "kind":"BinOp","lineno":2,"col_offset":11,"end_lineno":2,"end_col_offset":14,
            "left":{"kind":"Name","id":"c2","ctx":"Load","lineno":2,"col_offset":11},
            "op":"Mult",
            "right":{"kind":"Name","id":"i","ctx":"Load","lineno":2,"col_offset":14}
          }
        }],
        "decorator_list":[],
        "returns":null
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(doc).ok());
}

TEST_CASE("DslSemanticGate rejects ImportFrom numpy with E021", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"ImportFrom","lineno":3,"col_offset":1,"module":"numpy",
        "names":[{"kind":"alias","name":"array","asname":null,"lineno":3,"col_offset":1}],
        "level":0
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E021") != std::string::npos);
    REQUIRE(st.message().find("numpy") != std::string::npos);
    REQUIRE(st.message().find("theories/x.py:3:1:") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects bare Import with E031", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Import","lineno":2,"col_offset":0,
        "names":[{"kind":"alias","name":"os","asname":null,"lineno":2,"col_offset":0}]
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("Import") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects For with E031", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":5,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":5,"col_offset":4},
        "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":5,"col_offset":9},
        "body":[{"kind":"Pass","lineno":6,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("For") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects Lambda with E031", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Lambda","lineno":1,"col_offset":0,
          "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                  "kw_defaults":[],"defaults":[]},
          "body":{"kind":"Constant","value":1,"lineno":1,"col_offset":7}
        }
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("Lambda") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects nested ClassDef inside FunctionDef", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"outer","lineno":1,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"ClassDef","name":"Inner","lineno":2,"col_offset":4,
          "bases":[],"keywords":[],"decorator_list":[],
          "body":[{"kind":"Pass","lineno":3,"col_offset":8}]
        }],
        "decorator_list":[],
        "returns":null
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("ClassDef") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects relative ImportFrom", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"ImportFrom","lineno":1,"col_offset":0,"module":"parcae.dsl.math",
        "names":[{"kind":"alias","name":"z29_add","asname":null}],
        "level":1
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E021") != std::string::npos);
    REQUIRE(st.message().find("relative") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects Div operator string", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"BinOp","lineno":1,"col_offset":0,
          "left":{"kind":"Constant","value":1,"lineno":1,"col_offset":0},
          "op":"Div",
          "right":{"kind":"Constant","value":2,"lineno":1,"col_offset":4}
        }
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("Div") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects Starred with E031", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"f","ctx":"Load","lineno":1,"col_offset":0},
          "args":[{
            "kind":"Starred","lineno":1,"col_offset":2,
            "value":{"kind":"Name","id":"xs","ctx":"Load","lineno":1,"col_offset":3},
            "ctx":"Load"
          }],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("starred") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects **kwargs keyword", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"Param","ctx":"Load","lineno":1,"col_offset":0},
          "args":[],
          "keywords":[{
            "kind":"keyword","arg":null,"lineno":1,"col_offset":6,
            "value":{"kind":"Name","id":"kw","ctx":"Load","lineno":1,"col_offset":8}
          }]
        }
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("kwargs") != std::string::npos);
}
