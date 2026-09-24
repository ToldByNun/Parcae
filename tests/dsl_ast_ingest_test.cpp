#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_ast_limits.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace {

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" +
           R"("source_path":"t.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" +
           R"("ok":true,)" +
           R"("module":)" + module_json + "}";
}

}  // namespace

TEST_CASE("DslAstJsonIngest accepts minimal Module document", "[dsl][ingest]") {
    const std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"end_lineno":null,"end_col_offset":null,"body":[],"type_ignores":[]})");

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().source_path() == "t.py");
    REQUIRE(doc.value().dsl_ast_json_version() == "1.0.0");
    REQUIRE(doc.value().module()->kind() == "Module");
    const DslAstValue* body = doc.value().module()->find_field("body");
    REQUIRE(body != nullptr);
    REQUIRE(body->type() == DslAstValue::Type::Array);
    REQUIRE(body->as_array().empty());
}

TEST_CASE("DslAstJsonIngest accepts BinOp with compact op string", "[dsl][ingest]") {
    const std::string module = R"({
      "kind":"Module","lineno":1,"col_offset":0,"end_lineno":2,"end_col_offset":0,
      "body":[{
        "kind":"FunctionDef","name":"poly","lineno":1,"col_offset":0,"end_lineno":2,"end_col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[],"kwonlyargs":[],"kw_defaults":[],"defaults":[],
                "lineno":1,"col_offset":0,"end_lineno":1,"end_col_offset":0},
        "body":[{
          "kind":"Return","lineno":2,"col_offset":4,"end_lineno":2,"end_col_offset":14,
          "value":{
            "kind":"BinOp","lineno":2,"col_offset":11,"end_lineno":2,"end_col_offset":14,
            "left":{"kind":"Name","id":"c2","ctx":"Load","lineno":2,"col_offset":11,"end_lineno":2,"end_col_offset":13},
            "op":"Mult",
            "right":{"kind":"Name","id":"i","ctx":"Load","lineno":2,"col_offset":14,"end_lineno":2,"end_col_offset":15}
          }
        }],
        "decorator_list":[],
        "returns":null
      }],
      "type_ignores":[]
    })";

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(minimal_success_doc(module));
    REQUIRE(doc.ok());
}

TEST_CASE("DslAstJsonIngest rejects unknown top-level key", "[dsl][ingest]") {
    std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    const auto pos = text.find("\"schema\"");
    REQUIRE(pos != std::string::npos);
    text.insert(pos, "\"extra\":1,");

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E100") != std::string::npos);
    REQUIRE(doc.status().message().find("extra") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects failure envelope", "[dsl][ingest]") {
    const std::string text = R"({
      "schema":"parcae.dsl_ast_json.v0",
      "ok":false,
      "error":{"kind":"syntax_error","message":"invalid syntax","lineno":1,"col_offset":0}
    })";
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("failure envelope") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects trailing garbage", "[dsl][ingest]") {
    const std::string text =
        minimal_success_doc(
            R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})") +
        "\n{\"nope\":true}";
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E111") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects oversized string field", "[dsl][ingest]") {
    const std::string huge(DslAstLimits::max_string_bytes + 1, 'a');
    const std::string module =
        std::string(R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[],"note":")") +
        huge + "\"}";
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(minimal_success_doc(module));
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E105") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects oversized list", "[dsl][ingest]") {
    std::string body = "[";
    for (std::size_t i = 0; i < DslAstLimits::max_list_length + 1; ++i) {
        if (i > 0) {
            body += ',';
        }
        body += "null";
    }
    body += "]";
    const std::string module =
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":)" + body + R"(,"type_ignores":[]})";
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(minimal_success_doc(module));
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E106") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects excessive depth", "[dsl][ingest]") {
    std::string inner = R"({"kind":"Name","id":"x","ctx":"Load","lineno":1,"col_offset":0})";
    for (std::size_t i = 0; i < DslAstLimits::max_tree_depth + 2; ++i) {
        inner = std::string(R"({"kind":"Expr","lineno":1,"col_offset":0,"value":)") + inner + "}";
    }
    const std::string module =
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[)" + inner + R"(],"type_ignores":[]})";
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(minimal_success_doc(module));
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E104") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects wrong schema version major", "[dsl][ingest]") {
    std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    const auto pos = text.find("1.0.0");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, 5, "2.0.0");
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E100") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest accepts directives array (1.1.0)", "[dsl][ingest][directives]") {
    std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    const auto insert_at = text.rfind('}');
    REQUIRE(insert_at != std::string::npos);
    text.insert(
        insert_at,
        R"(,"directives":[{"lineno":3,"flag":"divergent_branch","raw":"#ignore DSL_FLAG:divergent_branch"}])");
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE(doc.ok());
    REQUIRE(doc.value().directives().size() == 1);
    REQUIRE(doc.value().directives()[0].lineno() == 3);
    REQUIRE(doc.value().directives()[0].flag() == "divergent_branch");
    REQUIRE(
        doc.value().directives()[0].raw() == "#ignore DSL_FLAG:divergent_branch");
}

TEST_CASE("DslAstJsonIngest rejects bad directives flag", "[dsl][ingest][directives]") {
    std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    const auto insert_at = text.rfind('}');
    text.insert(
        insert_at,
        R"(,"directives":[{"lineno":1,"flag":"BadFlag","raw":"#ignore DSL_FLAG:BadFlag"}])");
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E100") != std::string::npos);
}

TEST_CASE("DslAstJsonIngest rejects JSON larger than limit", "[dsl][ingest]") {
    std::string text(DslAstLimits::max_json_bytes + 1, ' ');
    text[0] = '{';
    text[1] = '}';
    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find("E102") != std::string::npos);
}
