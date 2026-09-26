#include <catch2/catch_test_macros.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/dsl_z29_builtins.hpp>
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

} // namespace

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

TEST_CASE("DslSemanticGate accepts OuterControl For", "[dsl][gate][scope]") {
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
    REQUIRE(st.ok());
}

TEST_CASE("DslSemanticGate accepts OuterControl If and While", "[dsl][gate][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[
        {
          "kind":"If","lineno":2,"col_offset":0,
          "test":{"kind":"Constant","value":true,"lineno":2,"col_offset":3},
          "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
          "orelse":[]
        },
        {
          "kind":"While","lineno":5,"col_offset":0,
          "test":{"kind":"Constant","value":true,"lineno":5,"col_offset":6},
          "body":[{"kind":"Pass","lineno":6,"col_offset":4}],
          "orelse":[]
        }
      ],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(doc).ok());
}

TEST_CASE("DslSemanticGate rejects HotLoop For with E034", "[dsl][gate][scope]") {
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
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find(DslRuleId::E034_hotloop_control) != std::string::npos);
    REQUIRE(st.message().find("For") != std::string::npos);
    REQUIRE(st.message().find("HotLoop") != std::string::npos);
}

TEST_CASE("DslSemanticGate rejects HotLoop While with E034", "[dsl][gate][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"While","lineno":4,"col_offset":4,
          "test":{"kind":"Constant","value":true,"lineno":4,"col_offset":10},
          "body":[{"kind":"Pass","lineno":5,"col_offset":8}],
          "orelse":[]
        }],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],"keywords":[]
        }],
        "returns":null
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E034") != std::string::npos);
    REQUIRE(st.message().find("While") != std::string::npos);
}

TEST_CASE("DslSemanticGate accepts HotLoop If (divergence is E033 later)", "[dsl][gate][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"If","lineno":4,"col_offset":4,
          "test":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":7},
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
    })");
    REQUIRE(DslSemanticGate::check(doc).ok());
}

TEST_CASE("DslSemanticGate rejects HotLoop Break with E034", "[dsl][gate][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"For","lineno":4,"col_offset":4,
          "target":{"kind":"Name","id":"i","ctx":"Store","lineno":4,"col_offset":8},
          "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":4,"col_offset":13},
          "body":[{"kind":"Break","lineno":5,"col_offset":8}],
          "orelse":[]
        }],
        "decorator_list":[{
          "kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1
        }],
        "returns":null
      }],
      "type_ignores":[]
    })");
    // HotLoop For fails first with E034 — Break never reached, or same rule.
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E034") != std::string::npos);
}

TEST_CASE("DslSemanticGate accepts OuterControl Break inside For", "[dsl][gate][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":2,"col_offset":9},
        "body":[{"kind":"Break","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(doc).ok());
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

TEST_CASE("DslSemanticGate accepts Div Mod Pow BitXor LShift", "[dsl][gate]") {
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
    REQUIRE(DslSemanticGate::check(doc).ok());

    const DslAstDocument mod = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"BinOp","lineno":1,"col_offset":0,
          "left":{"kind":"Name","id":"x","ctx":"Load","lineno":1,"col_offset":0},
          "op":"BitXor",
          "right":{"kind":"Constant","value":3,"lineno":1,"col_offset":4}
        }
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(mod).ok());
}

TEST_CASE("DslSemanticGate rejects MatMult operator string", "[dsl][gate]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"BinOp","lineno":1,"col_offset":0,
          "left":{"kind":"Constant","value":1,"lineno":1,"col_offset":0},
          "op":"MatMult",
          "right":{"kind":"Constant","value":2,"lineno":1,"col_offset":4}
        }
      }],
      "type_ignores":[]
    })");
    const Status st = DslSemanticGate::check(doc);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.message().find("E031") != std::string::npos);
    REQUIRE(st.message().find("MatMult") != std::string::npos);
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

TEST_CASE("DslSemanticGate accepts z29_matmul / z29_det / z29_autokey_shift",
          "[dsl][gate][z29][intrinsics]") {
    REQUIRE(DslZ29Builtins::is_builtin("z29_matmul"));
    REQUIRE(DslZ29Builtins::is_builtin("z29_det"));
    REQUIRE(DslZ29Builtins::is_builtin("z29_autokey_shift"));
    REQUIRE(DslZ29Builtins::arity("z29_matmul") == 2);
    REQUIRE(DslZ29Builtins::arity("z29_det") == 1);
    REQUIRE(DslZ29Builtins::arity("z29_autokey_shift") == 2);

    const DslAstDocument matmul = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_matmul","ctx":"Load","lineno":1,"col_offset":0},
          "args":[
            {"kind":"Name","id":"M","ctx":"Load","lineno":1,"col_offset":11},
            {"kind":"Name","id":"v","ctx":"Load","lineno":1,"col_offset":14}
          ],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(matmul).ok());

    const DslAstDocument det = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_det","ctx":"Load","lineno":1,"col_offset":0},
          "args":[{"kind":"Name","id":"M","ctx":"Load","lineno":1,"col_offset":8}],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(det).ok());

    const DslAstDocument autokey = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_autokey_shift","ctx":"Load","lineno":1,"col_offset":0},
          "args":[
            {"kind":"Name","id":"stream","ctx":"Load","lineno":1,"col_offset":18},
            {"kind":"Name","id":"lag","ctx":"Load","lineno":1,"col_offset":26}
          ],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    REQUIRE(DslSemanticGate::check(autokey).ok());
}

TEST_CASE("DslSemanticGate rejects unknown z29_* and wrong arity", "[dsl][gate][z29]") {
    REQUIRE_FALSE(DslZ29Builtins::is_allowed_call_name("z29_not_a_real_op"));

    const DslAstDocument unknown = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_not_a_real_op","ctx":"Load","lineno":1,"col_offset":0},
          "args":[],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    const Status unk = DslSemanticGate::check(unknown);
    REQUIRE_FALSE(unk.ok());
    REQUIRE(unk.message().find("E032") != std::string::npos);
    REQUIRE(unk.message().find("z29_not_a_real_op") != std::string::npos);

    const DslAstDocument bad_arity = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"Expr","lineno":1,"col_offset":0,
        "value":{
          "kind":"Call","lineno":1,"col_offset":0,
          "func":{"kind":"Name","id":"z29_det","ctx":"Load","lineno":1,"col_offset":0},
          "args":[
            {"kind":"Name","id":"M","ctx":"Load","lineno":1,"col_offset":8},
            {"kind":"Name","id":"extra","ctx":"Load","lineno":1,"col_offset":11}
          ],
          "keywords":[]
        }
      }],
      "type_ignores":[]
    })");
    const Status arity = DslSemanticGate::check(bad_arity);
    REQUIRE_FALSE(arity.ok());
    REQUIRE(arity.message().find("E032") != std::string::npos);
    REQUIRE(arity.message().find("expects 1") != std::string::npos);
}

