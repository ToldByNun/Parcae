#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/compose_ir.hpp>
#include <parcae/dsl/dsl_fuse.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace {

[[nodiscard]] TheoryIr make_atbash() {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "atbash",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        Z29Expr::atbash(x),
        Z29Expr::atbash(x));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_caesar() {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "caesar",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {shift.value()},
        Z29Expr::add(x, s),
        Z29Expr::sub(x, s));
    REQUIRE(th.ok());
    return th.value();
}

}  // namespace

TEST_CASE("Z29Expr remap substitutes var", "[dsl][fuse]") {
    const Z29Expr::Ptr expr = Z29Expr::add(Z29Expr::var("x"), Z29Expr::var("shift"));
    std::unordered_map<std::string, Z29Expr::Ptr> mapping;
    mapping.emplace("x", Z29Expr::atbash(Z29Expr::var("x")));
    mapping.emplace("shift", Z29Expr::var("caesar_shift"));
    const Z29Expr::Ptr remapped = expr->remap(mapping);

    Z29Expr::Env env{{"x", Index29{3}}, {"caesar_shift", Index29{5}}};
    // atbash(3)=25; 25+5=1
    REQUIRE(remapped->eval(env).value() == Index29{1});
}

TEST_CASE("DslFuse inlines atbash then caesar", "[dsl][fuse]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_caesar",
        TheoryIr::Tier::A,
        {"atbash", "caesar"},
        {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::Result> fused =
        DslFuse::fuse_inline(compose.value(), catalog);
    REQUIRE(fused.ok());
    REQUIRE(fused.value().flattened_steps().size() == 2);
    REQUIRE_FALSE(fused.value().nested_flattened());
    REQUIRE(fused.value().theory().family() == TheoryIr::Family::Elementwise);
    REQUIRE(fused.value().theory().name() == "atbash_then_caesar");
    REQUIRE(fused.value().theory().encrypt_step());
    REQUIRE(fused.value().theory().decrypt_step());

    // Decrypt: caesar_dec(atbash(x)) = sub(atbash(x), caesar_shift)
    // Encrypt: atbash(caesar_enc(x)) = atbash(add(x, caesar_shift))
    const Index29 shift_v{3};
    Z29Expr::Env env{{"caesar_shift", shift_v}};
    const std::vector<Index29> plain{Index29{0}, Index29{7}, Index29{28}};

    std::vector<Index29> cipher(plain.size());
    REQUIRE(DslIrApplicator::apply_into(
                fused.value().theory().encrypt_step(),
                "x",
                env,
                plain,
                cipher)
                .ok());

    // Staged encrypt: caesar then atbash (reverse of decrypt stages).
    std::vector<Index29> staged = plain;
    {
        Z29Expr::Env caesar_env{{"shift", shift_v}};
        REQUIRE(DslIrApplicator::apply_into(
                    caesar.encrypt_step(), "x", caesar_env, staged, staged)
                    .ok());
        REQUIRE(DslIrApplicator::apply_into(
                    atbash.encrypt_step(), "x", {}, staged, staged)
                    .ok());
    }
    REQUIRE(cipher == staged);

    std::vector<Index29> roundtrip(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(
                fused.value().theory().decrypt_step(),
                "x",
                env,
                cipher,
                roundtrip)
                .ok());
    REQUIRE(roundtrip == plain);
}

TEST_CASE("DslFuse flattens nested compose", "[dsl][fuse]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();

    const StatusOr<ComposeIr> inner = ComposeIr::make(
        "inner_atbash", TheoryIr::Tier::A, {"atbash"});
    REQUIRE(inner.ok());

    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> outer = ComposeIr::make(
        "nested_koan",
        TheoryIr::Tier::A,
        {"inner_atbash", "caesar"},
        {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(outer.ok());

    const std::vector<TheoryIr> theories{atbash, caesar};
    const std::vector<ComposeIr> composes{inner.value()};
    const StatusOr<DslFuse::Result> fused =
        DslFuse::fuse_inline(outer.value(), theories, composes);
    REQUIRE(fused.ok());
    REQUIRE(fused.value().nested_flattened());
    REQUIRE(fused.value().flattened_steps() == std::vector<std::string>{"atbash", "caesar"});
}

TEST_CASE("DslFuse rejects unknown leaf step", "[dsl][fuse]") {
    const StatusOr<ComposeIr> compose =
        ComposeIr::make("bad", TheoryIr::Tier::A, {"missing_theory"});
    REQUIRE(compose.ok());
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose.value(), {});
    REQUIRE_FALSE(fused.ok());
    REQUIRE(fused.status().message().find("E032") != std::string::npos);
    REQUIRE(fused.status().message().find("missing_theory") != std::string::npos);
}

TEST_CASE("DslFuse rejects missing decrypt_step", "[dsl][fuse]") {
    const StatusOr<TheoryIr> bare = TheoryIr::make(
        "bare",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {});
    REQUIRE(bare.ok());
    REQUIRE_FALSE(bare.value().decrypt_step());

    const StatusOr<ComposeIr> compose =
        ComposeIr::make("needs_steps", TheoryIr::Tier::A, {"bare"});
    REQUIRE(compose.ok());
    const std::vector<TheoryIr> catalog{bare.value()};
    const StatusOr<DslFuse::Result> fused =
        DslFuse::fuse_inline(compose.value(), catalog);
    REQUIRE_FALSE(fused.ok());
    REQUIRE(fused.status().message().find("decrypt_step") != std::string::npos);
}
