#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_exec_scope.hpp>
#include <parcae/dsl/dsl_scope_analyzer.hpp>

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

[[nodiscard]] const DslAstNode* require_body_stmt(
    const DslAstNode& parent,
    std::size_t index) {
    const DslAstValue* body = parent.find_field("body");
    REQUIRE(body);
    REQUIRE(body->type() == DslAstValue::Type::Array);
    REQUIRE(body->as_array().size() > index);
    const DslAstValue& item = body->as_array()[index];
    REQUIRE(item.type() == DslAstValue::Type::Node);
    REQUIRE(item.as_node());
    return item.as_node().get();
}

[[nodiscard]] DslExecScope require_scope(const DslScopeMap& map, const DslAstNode* node) {
    REQUIRE(node != nullptr);
    const std::optional<DslExecScope> scope = map.get(node);
    REQUIRE(scope.has_value());
    return scope.value();
}

}  // namespace

TEST_CASE("DslExecScope defaults and enter_loop", "[dsl][scope]") {
    const DslExecScope outer;
    REQUIRE(outer.is_outer_control());
    REQUIRE_FALSE(outer.is_hot_loop());
    REQUIRE(outer.loop_depth() == 0);
    REQUIRE(outer.kind_string() == "OuterControl");

    const DslExecScope hot{DslExecScope::Kind::HotLoop};
    REQUIRE(hot.is_hot_loop());
    REQUIRE(hot.kind_string() == "HotLoop");

    const DslExecScope nested = hot.enter_loop().enter_loop();
    REQUIRE(nested.is_hot_loop());
    REQUIRE(nested.loop_depth() == 2);
    REQUIRE(nested.in_loop());
}

TEST_CASE("DslScopeAnalyzer rejects document without module", "[dsl][scope]") {
    DslAstDocument doc;
    doc.set_source_path("theories/x.py");
    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE_FALSE(map.ok());
    REQUIRE(map.status().message().find("no module") != std::string::npos);
}

TEST_CASE("DslScopeAnalyzer marks module Assign as OuterControl", "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"Assign","lineno":2,"col_offset":0,
        "targets":[{"kind":"Name","id":"FLAG","ctx":"Store","lineno":2,"col_offset":0}],
        "value":{"kind":"Constant","value":1,"lineno":2,"col_offset":7}
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* assign = require_body_stmt(*doc.module(), 0);
    const DslExecScope scope = require_scope(map.value(), assign);
    REQUIRE(scope.is_outer_control());
    REQUIRE(scope.loop_depth() == 0);
}

TEST_CASE(
    "DslScopeAnalyzer marks define_primitive body as HotLoop",
    "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"Return","lineno":4,"col_offset":4,
          "value":{"kind":"Name","id":"x","ctx":"Load","lineno":4,"col_offset":11}
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

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* fn = require_body_stmt(*doc.module(), 0);
    REQUIRE(require_scope(map.value(), fn).is_outer_control());
    const DslAstNode* ret = require_body_stmt(*fn, 0);
    const DslExecScope body = require_scope(map.value(), ret);
    REQUIRE(body.is_hot_loop());
    REQUIRE(body.loop_depth() == 0);
}

TEST_CASE(
    "DslScopeAnalyzer Theory encrypt_step HotLoop vs step_params OuterControl",
    "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"ClassDef","name":"T","lineno":3,"col_offset":0,
        "bases":[],"keywords":[],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"Theory","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],"keywords":[]
        }],
        "body":[
          {
            "kind":"FunctionDef","name":"step_params","lineno":5,"col_offset":4,
            "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                    "kw_defaults":[],"defaults":[]},
            "body":[{"kind":"Pass","lineno":6,"col_offset":8}],
            "decorator_list":[],"returns":null
          },
          {
            "kind":"FunctionDef","name":"encrypt_step","lineno":8,"col_offset":4,
            "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                    "kw_defaults":[],"defaults":[]},
            "body":[{
              "kind":"Return","lineno":9,"col_offset":8,
              "value":{"kind":"Name","id":"x","ctx":"Load","lineno":9,"col_offset":15}
            }],
            "decorator_list":[],"returns":null
          }
        ]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* cls = require_body_stmt(*doc.module(), 0);
    const DslAstNode* step_params = require_body_stmt(*cls, 0);
    const DslAstNode* encrypt = require_body_stmt(*cls, 1);
    REQUIRE(require_scope(map.value(), require_body_stmt(*step_params, 0)).is_outer_control());
    REQUIRE(require_scope(map.value(), require_body_stmt(*encrypt, 0)).is_hot_loop());
}

TEST_CASE(
    "DslScopeAnalyzer OuterControl for increases loop_depth",
    "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"For","lineno":2,"col_offset":0,
        "target":{"kind":"Name","id":"i","ctx":"Store","lineno":2,"col_offset":4},
        "iter":{"kind":"Name","id":"xs","ctx":"Load","lineno":2,"col_offset":9},
        "body":[{"kind":"Pass","lineno":3,"col_offset":4}],
        "orelse":[]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* for_node = require_body_stmt(*doc.module(), 0);
    const DslExecScope for_scope = require_scope(map.value(), for_node);
    REQUIRE(for_scope.is_outer_control());
    REQUIRE(for_scope.loop_depth() == 1);
    const DslExecScope pass_scope = require_scope(map.value(), require_body_stmt(*for_node, 0));
    REQUIRE(pass_scope.is_outer_control());
    REQUIRE(pass_scope.loop_depth() == 1);
}

TEST_CASE(
    "DslScopeAnalyzer HotLoop nested While depth accumulates",
    "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"FunctionDef","name":"poly","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                "kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"While","lineno":4,"col_offset":4,
          "test":{"kind":"Constant","value":true,"lineno":4,"col_offset":10},
          "body":[{
            "kind":"While","lineno":5,"col_offset":8,
            "test":{"kind":"Constant","value":true,"lineno":5,"col_offset":14},
            "body":[{"kind":"Pass","lineno":6,"col_offset":12}],
            "orelse":[]
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

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* fn = require_body_stmt(*doc.module(), 0);
    const DslAstNode* outer_while = require_body_stmt(*fn, 0);
    const DslAstNode* inner_while = require_body_stmt(*outer_while, 0);
    const DslAstNode* pass = require_body_stmt(*inner_while, 0);

    const DslExecScope outer_s = require_scope(map.value(), outer_while);
    REQUIRE(outer_s.is_hot_loop());
    REQUIRE(outer_s.loop_depth() == 1);

    const DslExecScope inner_s = require_scope(map.value(), inner_while);
    REQUIRE(inner_s.is_hot_loop());
    REQUIRE(inner_s.loop_depth() == 2);

    const DslExecScope pass_s = require_scope(map.value(), pass);
    REQUIRE(pass_s.is_hot_loop());
    REQUIRE(pass_s.loop_depth() == 2);
}

TEST_CASE(
    "DslScopeAnalyzer interrupt_policy body is HotLoop",
    "[dsl][scope]") {
    const DslAstDocument doc = ingest_or_fail(R"({
      "kind":"Module","lineno":1,"col_offset":0,
      "body":[{
        "kind":"ClassDef","name":"T","lineno":3,"col_offset":0,
        "bases":[],"keywords":[],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"Theory","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],"keywords":[]
        }],
        "body":[{
          "kind":"FunctionDef","name":"interrupt_policy","lineno":5,"col_offset":4,
          "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],
                  "kw_defaults":[],"defaults":[]},
          "body":[{"kind":"Pass","lineno":6,"col_offset":8}],
          "decorator_list":[],"returns":null
        }]
      }],
      "type_ignores":[]
    })");

    const StatusOr<DslScopeMap> map = DslScopeAnalyzer::analyze(doc);
    REQUIRE(map.ok());
    const DslAstNode* cls = require_body_stmt(*doc.module(), 0);
    const DslAstNode* policy = require_body_stmt(*cls, 0);
    REQUIRE(require_scope(map.value(), require_body_stmt(*policy, 0)).is_hot_loop());
}
