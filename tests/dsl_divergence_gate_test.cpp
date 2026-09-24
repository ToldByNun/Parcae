#include <catch2/catch_test_macros.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_divergence_gate.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <string>

namespace {

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") + R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" + R"("source_path":"theories/x.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" + R"("ok":true,)" + R"("module":)" + module_json + "}";
}

[[nodiscard]] DslAstDocument ingest_or_fail(const std::string& module_json) {
    const StatusOr<DslAstDocument> doc =
        DslAstJsonIngest::parse_text(minimal_success_doc(module_json));
    REQUIRE(doc.ok());
    return doc.value();
}

[[nodiscard]] std::string hotloop_primitive_with_if(const std::string& test_json) {
    return std::string(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"x","lineno":3,"col_offset":8},
          {"kind":"arg","arg":"a","lineno":3,"col_offset":11}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"If","lineno":4,"col_offset":4,
          "test":)") +
           test_json +
           R"(,
          "body":[{
            "kind":"Return","lineno":5,"col_offset":8,
            "value":{"kind":"Name","id":"x","ctx":"Load","lineno":5,"col_offset":15}
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

} // namespace

TEST_CASE("DslDivergenceGate rejects HotLoop if on cipher with E033", "[dsl][divergence][E033]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_primitive_with_if(
        R"({"kind":"Compare","lineno":4,"col_offset":7,
            "left":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":7},
            "ops":["Eq"],
            "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":12}]})"));

    REQUIRE(DslSemanticGate::check(doc).ok());
    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE_FALSE(r.ok());
    REQUIRE(r.status().message().find("E033") != std::string::npos);
    REQUIRE(r.status().message().find("x") != std::string::npos);
}

TEST_CASE("DslDivergenceGate rejects HotLoop if on stream index i", "[dsl][divergence][E033]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_primitive_with_if(
        R"({"kind":"Compare","lineno":4,"col_offset":7,
            "left":{"kind":"Name","id":"i","ctx":"Load","lineno":4,"col_offset":7},
            "ops":["Eq"],
            "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":12}]})"));

    const Status st = DslDivergenceGate::check_errors_only(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E033") != std::string::npos);
    REQUIRE(st.message().find("'i'") != std::string::npos);
}

TEST_CASE("DslDivergenceGate accepts HotLoop if on Param with W011", "[dsl][divergence][W011]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_primitive_with_if(
        R"({"kind":"Compare","lineno":4,"col_offset":7,
            "left":{"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":7},
            "ops":["Eq"],
            "comparators":[{"kind":"Constant","value":1,"lineno":4,"col_offset":12}]})"));

    REQUIRE(DslSemanticGate::check(doc).ok());
    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE(r.ok());
    REQUIRE_FALSE(r.value().empty());
    REQUIRE(r.value().warnings().size() == 1);
    REQUIRE(r.value().warnings()[0].rule_id() == DslRuleId::W011_relaxed_branch);
    REQUIRE(r.value().warnings()[0].message().find("LoopInvariant") != std::string::npos);
}

TEST_CASE("DslDivergenceGate accepts const HotLoop if with W011", "[dsl][divergence][W011]") {
    const DslAstDocument doc = ingest_or_fail(
        hotloop_primitive_with_if(R"({"kind":"Constant","value":true,"lineno":4,"col_offset":7})"));

    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE(r.ok());
    REQUIRE(r.value().warnings().size() == 1);
    REQUIRE(r.value().warnings()[0].message().find("CompileTimeConstant") != std::string::npos);
}

TEST_CASE("DslDivergenceGate accepts HostFlag name with W011", "[dsl][divergence][W011]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_primitive_with_if(
        R"({"kind":"Name","id":"flag","ctx":"Load","lineno":4,"col_offset":7})"));

    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE(r.ok());
    REQUIRE(r.value().warnings()[0].message().find("HostFlag") != std::string::npos);
}

TEST_CASE("DslDivergenceGate ignores OuterControl If", "[dsl][divergence]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"If","lineno":2,"col_offset":0,
        "test":{"kind":"Name","id":"x","ctx":"Load","lineno":2,"col_offset":3},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    REQUIRE(DslSemanticGate::check(doc).ok());
    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE(r.ok());
    REQUIRE(r.value().empty());
}

TEST_CASE("DslDivergenceGate BoolOp with cipher child is E033", "[dsl][divergence][E033]") {
    const DslAstDocument doc = ingest_or_fail(hotloop_primitive_with_if(
        R"({"kind":"BoolOp","lineno":4,"col_offset":7,"op":"And",
            "values":[
              {"kind":"Name","id":"a","ctx":"Load","lineno":4,"col_offset":7},
              {"kind":"Compare","lineno":4,"col_offset":14,
               "left":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":14},
               "ops":["Eq"],
               "comparators":[{"kind":"Constant","value":0,"lineno":4,"col_offset":19}]}
            ]})"));

    const Status st = DslDivergenceGate::check_errors_only(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E033") != std::string::npos);
}

TEST_CASE("DslPredicateClass classify_expr helper", "[dsl][divergence]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{"kind":"Name","id":"x","ctx":"Load","lineno":1,"col_offset":0}
      }],
      "type_ignores":[]
    })");
    const DslAstValue* body = doc.module()->find_field("body");
    REQUIRE(body);
    const DslAstNode& expr_stmt = *body->as_array()[0].as_node();
    const DslAstNode& name = *expr_stmt.find_field("value")->as_node();
    const DslPredicateClass pred = DslDivergenceGate::classify_expr(name, "x");
    REQUIRE(pred.is_thread_varying());
    REQUIRE(pred.evidence() == "x");
}

TEST_CASE("DslDivergenceGate skips self when resolving method cipher_var",
          "[dsl][divergence][golden]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"ClassDef","name":"T","lineno":2,"col_offset":0,
        "bases":[],"keywords":[],"decorator_list":[{
          "kind":"Call","lineno":1,"col_offset":1,
          "func":{"kind":"Name","id":"Theory","ctx":"Load","lineno":1},
          "args":[],"keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"t","lineno":1}},
            {"kind":"keyword","arg":"family","value":{"kind":"Constant","value":"elementwise","lineno":1}},
            {"kind":"keyword","arg":"tier","value":{"kind":"Constant","value":"A","lineno":1}}
          ]
        }],
        "body":[{
          "kind":"FunctionDef","name":"encrypt_step","lineno":4,"col_offset":4,
          "args":{"kind":"arguments","posonlyargs":[],"args":[
            {"kind":"arg","arg":"self","lineno":4,"col_offset":22},
            {"kind":"arg","arg":"x","lineno":4,"col_offset":28}
          ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
          "body":[{
            "kind":"If","lineno":5,"col_offset":8,
            "test":{"kind":"Compare","lineno":5,"col_offset":11,
              "left":{"kind":"Attribute","lineno":5,"col_offset":11,
                "value":{"kind":"Name","id":"self","ctx":"Load","lineno":5,"col_offset":11},
                "attr":"mode","ctx":"Load"},
              "ops":["Eq"],
              "comparators":[{"kind":"Constant","value":1,"lineno":5,"col_offset":24}]},
            "body":[{
              "kind":"Return","lineno":6,"col_offset":12,
              "value":{"kind":"Name","id":"x","ctx":"Load","lineno":6,"col_offset":19}
            }],
            "orelse":[{
              "kind":"Return","lineno":8,"col_offset":12,
              "value":{"kind":"Name","id":"x","ctx":"Load","lineno":8,"col_offset":19}
            }]
          }],
          "decorator_list":[],"returns":null
        }]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslDivergenceGate::Report> r = DslDivergenceGate::check(doc);
    REQUIRE(r.ok());
    REQUIRE_FALSE(r.value().empty());
    REQUIRE(r.value().warnings().front().rule_id() == "W011");
}
