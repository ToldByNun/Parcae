#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <parcae/core/index29.hpp>
#include <parcae/dsl/compose_ir.hpp>
#include <parcae/dsl/dsl_fuse.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/transform/compose_transform.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] TheoryIr make_atbash() {
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const StatusOr<TheoryIr> th = TheoryIr::make(
        "atbash", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault, {}, Z29Expr::atbash(x), Z29Expr::atbash(x));
    REQUIRE(th.ok());
    return th.value();
}

[[nodiscard]] TheoryIr make_caesar() {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    const StatusOr<TheoryIr> th =
        TheoryIr::make("caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                       Z29Expr::add(x, s), Z29Expr::sub(x, s));
    REQUIRE(th.ok());
    return th.value();
}

/// Koan-1 compose: decrypt = Atbash then Caesar(+shift) via encrypt-on-decrypt.
[[nodiscard]] ComposeIr make_koan1_compose() {
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose =
        ComposeIr::make("koan1_style", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
                        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}},
                        std::nullopt, {}, std::nullopt, std::nullopt,
                        {ComposeIr::StageDirection{"caesar", TransformDirection::Encrypt}});
    REQUIRE(compose.ok());
    return compose.value();
}

} // namespace

TEST_CASE("ComposeIr stage_directions rejects unknown step", "[dsl][fuse][koan]") {
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "bad", TheoryIr::Tier::A, {"atbash"}, {}, {}, std::nullopt, {}, std::nullopt, std::nullopt,
        {ComposeIr::StageDirection{"caesar", TransformDirection::Encrypt}});
    REQUIRE_FALSE(compose.ok());
    REQUIRE(compose.status().message().find("E032") != std::string::npos);
    REQUIRE(compose.status().message().find("stage_directions") != std::string::npos);
}

TEST_CASE("DslFuse koan1 fused decrypt matches ComposeTransform golden vector",
          "[dsl][fuse][koan][parity]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const ComposeIr compose = make_koan1_compose();
    const std::vector<TheoryIr> catalog{atbash, caesar};

    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose, catalog);
    REQUIRE(fused.ok());

    // Same golden as transform_test Compose atbash_then_caesar helper.
    const std::vector<Index29> cipher{Index29{0}, Index29{5}, Index29{28}};
    const std::vector<Index29> expect{Index29{2}, Index29{26}, Index29{3}};

    Z29Expr::Env env{{"caesar_shift", Index29{3}}};
    std::vector<Index29> out(cipher.size());
    REQUIRE(
        DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, cipher, out)
            .ok());
    REQUIRE(out == expect);

    StatusOr<std::vector<Index29>> ref =
        ComposeTransform::apply_atbash_then_caesar(cipher, 3, TransformDirection::Decrypt);
    REQUIRE(ref.ok());
    REQUIRE(out == ref.value());
}

TEST_CASE("DslFuse koan1 fused encrypt/decrypt roundtrip matches ComposeTransform",
          "[dsl][fuse][koan][parity]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const ComposeIr compose = make_koan1_compose();
    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose, catalog);
    REQUIRE(fused.ok());

    const std::uint8_t shift = 3;
    Z29Expr::Env env{{"caesar_shift", Index29{shift}}};
    const std::vector<Index29> plain{Index29{0}, Index29{7}, Index29{14}, Index29{28}, Index29{3}};

    std::vector<Index29> cipher(plain.size());
    REQUIRE(
        DslIrApplicator::apply_into(fused.value().theory().encrypt_step(), "x", env, plain, cipher)
            .ok());

    StatusOr<std::vector<Index29>> ref_cipher =
        ComposeTransform::apply_atbash_then_caesar(plain, shift, TransformDirection::Encrypt);
    REQUIRE(ref_cipher.ok());
    REQUIRE(cipher == ref_cipher.value());

    std::vector<Index29> roundtrip(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, cipher,
                                        roundtrip)
                .ok());
    REQUIRE(roundtrip == plain);

    StatusOr<std::vector<Index29>> ref_plain =
        ComposeTransform::apply_atbash_then_caesar(cipher, shift, TransformDirection::Decrypt);
    REQUIRE(ref_plain.ok());
    REQUIRE(roundtrip == ref_plain.value());
}

TEST_CASE("DslFuse koan1 fused matches ComposeTransform for all shifts 0..28",
          "[dsl][fuse][koan][parity]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const ComposeIr compose = make_koan1_compose();
    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose, catalog);
    REQUIRE(fused.ok());

    const std::vector<Index29> stream{Index29{0}, Index29{1}, Index29{10}, Index29{15},
                                      Index29{28}};

    for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
        Z29Expr::Env env{{"caesar_shift", Index29{shift}}};
        std::vector<Index29> fused_out(stream.size());
        REQUIRE(DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, stream,
                                            fused_out)
                    .ok());

        StatusOr<std::vector<Index29>> ref =
            ComposeTransform::apply_atbash_then_caesar(stream, shift, TransformDirection::Decrypt);
        REQUIRE(ref.ok());
        REQUIRE(fused_out == ref.value());
    }
}

TEST_CASE("DslFuse koan1 staged recipe includes caesar encrypt direction",
          "[dsl][fuse][koan][parity]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const ComposeIr compose = make_koan1_compose();
    const std::vector<TheoryIr> catalog{atbash, caesar};

    const StatusOr<std::string> recipe = DslFuse::emit_staged_recipe_json(
        compose, catalog, {}, {{"caesar_shift", static_cast<std::uint8_t>(3)}});
    REQUIRE(recipe.ok());
    REQUIRE(recipe.value().find("\"transform_id\": \"atbash\"") != std::string::npos);
    REQUIRE(recipe.value().find("\"transform_id\": \"caesar\"") != std::string::npos);
    REQUIRE(recipe.value().find("\"direction\": \"encrypt\"") != std::string::npos);
    REQUIRE(recipe.value().find("\"shift\": 3") != std::string::npos);

    // Recipe must be accepted by ComposeTransform and match fused decrypt.
    const nlohmann::json params = nlohmann::json::parse(recipe.value());
    const std::vector<Index29> cipher{Index29{0}, Index29{5}, Index29{28}};
    StatusOr<std::vector<Index29>> via_recipe =
        ComposeTransform{}.apply(cipher, params, TransformDirection::Decrypt);
    REQUIRE(via_recipe.ok());

    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose, catalog);
    REQUIRE(fused.ok());
    Z29Expr::Env env{{"caesar_shift", Index29{3}}};
    std::vector<Index29> fused_out(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, cipher,
                                        fused_out)
                .ok());
    REQUIRE(via_recipe.value() == fused_out);
}

TEST_CASE("DslFuse koan1 staged cuda façade sets CudaDir::Encrypt on caesar", "[dsl][fuse][koan]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const ComposeIr compose = make_koan1_compose();
    const std::vector<TheoryIr> catalog{atbash, caesar};

    const StatusOr<DslFuse::EmitBundle> bundle =
        DslFuse::emit_compose(compose, catalog, {}, DslFuse::FusionStatus::FallbackStaged,
                              {{"caesar_shift", static_cast<std::uint8_t>(3)}});
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().staged_cuda_header().find("CudaFamilyId::Caesar") != std::string::npos);
    REQUIRE(bundle.value().staged_cuda_header().find("CudaDir::Encrypt") != std::string::npos);
}
