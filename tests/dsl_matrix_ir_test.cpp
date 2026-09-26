#include <catch2/catch_test_macros.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_build_ir.hpp>
#include <parcae/dsl/dsl_emit_cpu.hpp>
#include <parcae/dsl/dsl_semantic_gate.hpp>
#include <parcae/dsl/matrix_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/math/z29_matrix2.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] Index29 I(unsigned v) { return Index29{static_cast<std::uint8_t>(v)}; }

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") + R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" + R"("source_path":"theories/matrix_prim.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" + R"("ok":true,)" + R"("module":)" + module_json + "}";
}

[[nodiscard]] DslAstDocument ingest_or_fail(const std::string& module_json) {
    const StatusOr<DslAstDocument> doc =
        DslAstJsonIngest::parse_text(minimal_success_doc(module_json));
    REQUIRE(doc.ok());
    return doc.value();
}

[[nodiscard]] std::string name_load(const char* id, int line = 4) {
    return std::string(R"({"kind":"Name","id":")") + id +
           R"(","ctx":"Load","lineno":)" + std::to_string(line) + R"(,"col_offset":0})";
}

} // namespace

TEST_CASE("MatrixIr det_expr matches Z29Matrix2::det", "[dsl][matrix][ir]") {
    StatusOr<MatrixIr> m = MatrixIr::make({
        Z29Expr::constant(2).value(),
        Z29Expr::constant(3).value(),
        Z29Expr::constant(5).value(),
        Z29Expr::constant(7).value(),
    });
    REQUIRE(m.ok());
    REQUIRE(m.value().n() == 2);
    const Z29Expr::Ptr det = m.value().det_expr();
    Z29Expr::Env env;
    StatusOr<Index29> got = det->eval(env);
    REQUIRE(got.ok());
    const Index29 expect = Z29Matrix2{I(2), I(3), I(5), I(7)}.det();
    REQUIRE(got.value() == expect);
}

TEST_CASE("MatrixIr mul_vec_exprs matches Z29Matrix2::mul_vec", "[dsl][matrix][ir]") {
    StatusOr<MatrixIr> m = MatrixIr::make({
        Z29Expr::constant(2).value(),
        Z29Expr::constant(3).value(),
        Z29Expr::constant(5).value(),
        Z29Expr::constant(7).value(),
    });
    REQUIRE(m.ok());
    StatusOr<std::vector<Z29Expr::Ptr>> out =
        m.value().mul_vec_exprs({Z29Expr::constant(1).value(), Z29Expr::constant(4).value()});
    REQUIRE(out.ok());
    REQUIRE(out.value().size() == 2);
    Z29Expr::Env env;
    StatusOr<Index29> y0 = out.value()[0]->eval(env);
    StatusOr<Index29> y1 = out.value()[1]->eval(env);
    REQUIRE(y0.ok());
    REQUIRE(y1.ok());
    const auto expect = Z29Matrix2{I(2), I(3), I(5), I(7)}.mul_vec(I(1), I(4));
    REQUIRE(y0.value() == expect[0]);
    REQUIRE(y1.value() == expect[1]);
}

TEST_CASE("DslEmitCpu emit_expr lowers z29_det Call via Z29Matrix2", "[dsl][emit][matrix]") {
    const Z29Expr::Ptr call = Z29Expr::call(
        "z29_det", {Z29Expr::var("a"), Z29Expr::var("b"), Z29Expr::var("c"), Z29Expr::var("d")});
    const StatusOr<std::string> cpp = DslEmitCpu::emit_expr(call, "x", "input[i]");
    REQUIRE(cpp.ok());
    REQUIRE(cpp.value().find("Z29Matrix2::from_row_major") != std::string::npos);
    REQUIRE(cpp.value().find(".det()") != std::string::npos);
    REQUIRE(cpp.value().find("a") != std::string::npos);
}

TEST_CASE("DslEmitCpu emit_expr lowers flattened z29_matmul Call", "[dsl][emit][matrix]") {
    const Z29Expr::Ptr call = Z29Expr::call(
        "z29_matmul", {Z29Expr::var("a"), Z29Expr::var("b"), Z29Expr::var("c"), Z29Expr::var("d"),
                       Z29Expr::var("x0"), Z29Expr::var("x1")});
    const StatusOr<std::string> cpp = DslEmitCpu::emit_expr(call, "x", "input[i]");
    REQUIRE(cpp.ok());
    REQUIRE(cpp.value().find("Z29Matrix2::from_row_major") != std::string::npos);
    REQUIRE(cpp.value().find(".mul_vec(") != std::string::npos);
    REQUIRE(cpp.value().find("[0]") != std::string::npos);
}

TEST_CASE("DslBuildIr lowers z29_det Tuple to det tree", "[dsl][build][matrix]") {
    const std::string module = std::string(R"JSON({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"det2","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"a","lineno":3,"col_offset":12},
          {"kind":"arg","arg":"b","lineno":3,"col_offset":15},
          {"kind":"arg","arg":"c","lineno":3,"col_offset":18},
          {"kind":"arg","arg":"d","lineno":3,"col_offset":21}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"Return","lineno":4,"col_offset":4,
          "value":{
            "kind":"Call","lineno":4,"col_offset":11,
            "func":{"kind":"Name","id":"z29_det","ctx":"Load","lineno":4,"col_offset":11},
            "args":[{
              "kind":"Tuple","lineno":4,"col_offset":19,"ctx":"Load",
              "elts":[)JSON") +
                               name_load("a") + "," + name_load("b") + "," + name_load("c") + "," +
                               name_load("d") + R"JSON(]
            }],
            "keywords":[]
          }
        }],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],
          "keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"det2","lineno":2,"col_offset":20}},
            {"kind":"keyword","arg":"signature","value":{"kind":"Constant","value":"(a: Z29, b: Z29, c: Z29, d: Z29) -> Z29","lineno":2,"col_offset":40}}
          ]
        }],
        "returns":null
      }],
      "type_ignores":[]
    })JSON";

    const DslAstDocument doc = ingest_or_fail(module);
    REQUIRE(DslSemanticGate::check(doc).ok());
    StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    REQUIRE(unit.value().primitives().size() == 1);
    const Z29Expr::Ptr body = unit.value().primitives()[0].body();
    REQUIRE(body);
    REQUIRE(body->kind() != Z29Expr::Kind::Call);
    Z29Expr::Env env{{"a", I(2)}, {"b", I(3)}, {"c", I(5)}, {"d", I(7)}};
    StatusOr<Index29> got = body->eval(env);
    REQUIRE(got.ok());
    REQUIRE(got.value() == Z29Matrix2{I(2), I(3), I(5), I(7)}.det());

    const StatusOr<std::string> cpp = DslEmitCpu::emit_expr(body, "x", "input[i]");
    REQUIRE(cpp.ok());
    REQUIRE(cpp.value().find("Z29::sub") != std::string::npos);
    REQUIRE(cpp.value().find("Z29::mul") != std::string::npos);
}

TEST_CASE("DslBuildIr lowers z29_matmul(...)[i] to mul_vec component", "[dsl][build][matrix]") {
    const std::string module = std::string(R"JSON({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"hill_y0","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"a","lineno":3,"col_offset":12},
          {"kind":"arg","arg":"b","lineno":3,"col_offset":15},
          {"kind":"arg","arg":"c","lineno":3,"col_offset":18},
          {"kind":"arg","arg":"d","lineno":3,"col_offset":21},
          {"kind":"arg","arg":"x0","lineno":3,"col_offset":24},
          {"kind":"arg","arg":"x1","lineno":3,"col_offset":28}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"Return","lineno":4,"col_offset":4,
          "value":{
            "kind":"Subscript","lineno":4,"col_offset":11,"ctx":"Load",
            "value":{
              "kind":"Call","lineno":4,"col_offset":11,
              "func":{"kind":"Name","id":"z29_matmul","ctx":"Load","lineno":4,"col_offset":11},
              "args":[{
                "kind":"Tuple","lineno":4,"col_offset":22,"ctx":"Load",
                "elts":[)JSON") +
                               name_load("a") + "," + name_load("b") + "," + name_load("c") + "," +
                               name_load("d") +
                               R"MID(]
              },{
                "kind":"Tuple","lineno":4,"col_offset":40,"ctx":"Load",
                "elts":[)MID" +
                               name_load("x0") + "," + name_load("x1") + R"END(]
              }],
              "keywords":[]
            },
            "slice":{"kind":"Constant","value":0,"lineno":4,"col_offset":55}
          }
        }],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],
          "keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"hill_y0","lineno":2,"col_offset":20}},
            {"kind":"keyword","arg":"signature","value":{"kind":"Constant","value":"(a: Z29, b: Z29, c: Z29, d: Z29, x0: Z29, x1: Z29) -> Z29","lineno":2,"col_offset":40}}
          ]
        }],
        "returns":null
      }],
      "type_ignores":[]
    })END";

    const DslAstDocument doc = ingest_or_fail(module);
    REQUIRE(DslSemanticGate::check(doc).ok());
    StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE(unit.ok());
    const Z29Expr::Ptr body = unit.value().primitives()[0].body();
    Z29Expr::Env env{{"a", I(2)}, {"b", I(3)}, {"c", I(5)}, {"d", I(7)},
                     {"x0", I(1)}, {"x1", I(4)}};
    StatusOr<Index29> got = body->eval(env);
    REQUIRE(got.ok());
    REQUIRE(got.value() == Z29Matrix2{I(2), I(3), I(5), I(7)}.mul_vec(I(1), I(4))[0]);
}

TEST_CASE("DslBuildIr rejects bare z29_matmul without subscript", "[dsl][build][matrix]") {
    const std::string module = std::string(R"JSON({
      "kind":"Module","lineno":1,"col_offset":0,"body":[{
        "kind":"FunctionDef","name":"bad","lineno":3,"col_offset":0,
        "args":{"kind":"arguments","posonlyargs":[],"args":[
          {"kind":"arg","arg":"a","lineno":3,"col_offset":12},
          {"kind":"arg","arg":"b","lineno":3,"col_offset":15},
          {"kind":"arg","arg":"c","lineno":3,"col_offset":18},
          {"kind":"arg","arg":"d","lineno":3,"col_offset":21},
          {"kind":"arg","arg":"x0","lineno":3,"col_offset":24},
          {"kind":"arg","arg":"x1","lineno":3,"col_offset":28}
        ],"kwonlyargs":[],"kw_defaults":[],"defaults":[]},
        "body":[{
          "kind":"Return","lineno":4,"col_offset":4,
          "value":{
            "kind":"Call","lineno":4,"col_offset":11,
            "func":{"kind":"Name","id":"z29_matmul","ctx":"Load","lineno":4,"col_offset":11},
            "args":[{
              "kind":"Tuple","lineno":4,"col_offset":22,"ctx":"Load",
              "elts":[)JSON") +
                               name_load("a") + "," + name_load("b") + "," + name_load("c") + "," +
                               name_load("d") +
                               R"MID(]
            },{
              "kind":"Tuple","lineno":4,"col_offset":40,"ctx":"Load",
              "elts":[)MID" +
                               name_load("x0") + "," + name_load("x1") + R"END(]
            }],
            "keywords":[]
          }
        }],
        "decorator_list":[{
          "kind":"Call","lineno":2,"col_offset":1,
          "func":{"kind":"Name","id":"define_primitive","ctx":"Load","lineno":2,"col_offset":1},
          "args":[],
          "keywords":[
            {"kind":"keyword","arg":"name","value":{"kind":"Constant","value":"bad","lineno":2,"col_offset":20}},
            {"kind":"keyword","arg":"signature","value":{"kind":"Constant","value":"(a: Z29, b: Z29, c: Z29, d: Z29, x0: Z29, x1: Z29) -> Z29","lineno":2,"col_offset":40}}
          ]
        }],
        "returns":null
      }],
      "type_ignores":[]
    })END";

    const DslAstDocument doc = ingest_or_fail(module);
    REQUIRE(DslSemanticGate::check(doc).ok());
    StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc);
    REQUIRE_FALSE(unit.ok());
    REQUIRE(unit.status().message().find("vector") != std::string::npos);
}