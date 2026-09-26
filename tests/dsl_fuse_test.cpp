#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/dsl/compose_ir.hpp>
#include <parcae/dsl/dsl_catalog_builtins.hpp>
#include <parcae/dsl/dsl_fuse.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/interrupt/policy.hpp>
#include <string>
#include <unordered_map>
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

} // namespace

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
        "atbash_then_caesar", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose.value(), catalog);
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
    REQUIRE(
        DslIrApplicator::apply_into(fused.value().theory().encrypt_step(), "x", env, plain, cipher)
            .ok());

    // Staged encrypt: caesar then atbash (reverse of decrypt stages).
    std::vector<Index29> staged = plain;
    {
        Z29Expr::Env caesar_env{{"shift", shift_v}};
        REQUIRE(DslIrApplicator::apply_into(caesar.encrypt_step(), "x", caesar_env, staged, staged)
                    .ok());
        REQUIRE(DslIrApplicator::apply_into(atbash.encrypt_step(), "x", {}, staged, staged).ok());
    }
    REQUIRE(cipher == staged);

    std::vector<Index29> roundtrip(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, cipher,
                                        roundtrip)
                .ok());
    REQUIRE(roundtrip == plain);
}

TEST_CASE("DslFuse flattens nested compose", "[dsl][fuse]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();

    const StatusOr<ComposeIr> inner =
        ComposeIr::make("inner_atbash", TheoryIr::Tier::A, {"atbash"});
    REQUIRE(inner.ok());

    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> outer = ComposeIr::make(
        "nested_koan", TheoryIr::Tier::A, {"inner_atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(outer.ok());

    const std::vector<TheoryIr> theories{atbash, caesar};
    const std::vector<ComposeIr> composes{inner.value()};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(outer.value(), theories, composes);
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
    const StatusOr<TheoryIr> bare =
        TheoryIr::make("bare", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {});
    REQUIRE(bare.ok());
    REQUIRE_FALSE(bare.value().decrypt_step());

    const StatusOr<ComposeIr> compose = ComposeIr::make("needs_steps", TheoryIr::Tier::A, {"bare"});
    REQUIRE(compose.ok());
    const std::vector<TheoryIr> catalog{bare.value()};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose.value(), catalog);
    REQUIRE_FALSE(fused.ok());
    REQUIRE(fused.status().message().find("decrypt_step") != std::string::npos);
}

TEST_CASE("DslFuse emit_compose fused selects fused headers", "[dsl][fuse][emit]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_caesar", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    DslFuse::ParamValues values{{"caesar_shift", static_cast<std::uint8_t>(3)}};
    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::EmitBundle> bundle =
        DslFuse::emit_compose(compose.value(), catalog, {}, DslFuse::FusionStatus::Fused, values);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().status() == DslFuse::FusionStatus::Fused);
    REQUIRE(bundle.value().status_str() == "fused");
    REQUIRE(bundle.value().selected_cpu_header() == bundle.value().fused_cpu_header());
    REQUIRE(bundle.value().selected_cuda_header() == bundle.value().fused_cuda_header());
    REQUIRE(bundle.value().fused_cpu_header().find("AtbashThenCaesarTransform") !=
            std::string::npos);
    REQUIRE(bundle.value().fused_cuda_header().find("AtbashThenCaesarKernel") != std::string::npos);
    REQUIRE(bundle.value().fused_cuda_cu().find("Z29Device::") != std::string::npos);
    REQUIRE(bundle.value().staged_recipe_json().find("\"transform_id\": \"atbash\"") !=
            std::string::npos);
    REQUIRE(bundle.value().staged_recipe_json().find("\"shift\": 3") != std::string::npos);
    REQUIRE(bundle.value().staged_cpu_header().find("StagedTransform") != std::string::npos);
    REQUIRE(bundle.value().staged_cpu_header().find("ComposeTransform") != std::string::npos);
    REQUIRE(bundle.value().staged_cuda_header().find("StagedKernel") != std::string::npos);
    REQUIRE(bundle.value().staged_cuda_header().find("ComposeDriver::apply_host") !=
            std::string::npos);
    REQUIRE(bundle.value().staged_cuda_header().find("CudaFamilyId::Atbash") != std::string::npos);
    REQUIRE(bundle.value().staged_cuda_header().find("caesar_shift") != std::string::npos);
}

TEST_CASE("DslFuse emit_compose fallback_staged selects staged headers", "[dsl][fuse][emit]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_caesar", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::EmitBundle> bundle =
        DslFuse::emit_compose(compose.value(), catalog, {}, DslFuse::FusionStatus::FallbackStaged,
                              {{"caesar_shift", static_cast<std::uint8_t>(7)}});
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().status_str() == "fallback_staged");
    REQUIRE(bundle.value().selected_cpu_header() == bundle.value().staged_cpu_header());
    REQUIRE(bundle.value().selected_cuda_header() == bundle.value().staged_cuda_header());
    REQUIRE(bundle.value().staged_cpu_header().find("fusion_status = \"fallback_staged\"") !=
            std::string::npos);
    REQUIRE(bundle.value().staged_recipe_json().find("\"shift\": 7") != std::string::npos);
}

TEST_CASE("DslFuse fusion_status_str", "[dsl][fuse]") {
    REQUIRE(DslFuse::fusion_status_str(DslFuse::FusionStatus::Fused) == "fused");
    REQUIRE(DslFuse::fusion_status_str(DslFuse::FusionStatus::FallbackStaged) == "fallback_staged");
}

TEST_CASE("DslFuse choose_status follows fused >= staged rule", "[dsl][fuse][bench]") {
    REQUIRE(DslFuse::choose_status(100.0, 90.0) == DslFuse::FusionStatus::Fused);
    REQUIRE(DslFuse::choose_status(100.0, 100.0) == DslFuse::FusionStatus::Fused);
    REQUIRE(DslFuse::choose_status(90.0, 100.0) == DslFuse::FusionStatus::FallbackStaged);
}

TEST_CASE("DslFuse bench_cpu returns positive rates and status", "[dsl][fuse][bench]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_caesar", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::BenchReport> bench = DslFuse::bench_cpu(
        compose.value(), catalog, {}, {{"caesar_shift", static_cast<std::uint8_t>(3)}},
        /*stream_len=*/1024,
        /*reps=*/8);
    REQUIRE(bench.ok());
    REQUIRE(bench.value().fused_elems_per_sec() > 0.0);
    REQUIRE(bench.value().staged_elems_per_sec() > 0.0);
    REQUIRE(bench.value().status() == DslFuse::choose_status(bench.value().fused_elems_per_sec(),
                                                             bench.value().staged_elems_per_sec()));
    REQUIRE(bench.value().detail().find("cpu_bench") != std::string::npos);
}

TEST_CASE("DslFuse emit_compose_auto attaches bench report", "[dsl][fuse][bench]") {
    const TheoryIr atbash = make_atbash();
    const TheoryIr caesar = make_caesar();
    const StatusOr<ParamIr> shift = ParamIr::make("caesar_shift", 0, 28);
    REQUIRE(shift.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_caesar", TheoryIr::Tier::A, {"atbash", "caesar"}, {shift.value()},
        {ComposeIr::StepParamBinding{"caesar", "shift", "caesar_shift"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, caesar};
    const StatusOr<DslFuse::EmitBundle> bundle = DslFuse::emit_compose_auto(
        compose.value(), catalog, {}, {{"caesar_shift", static_cast<std::uint8_t>(3)}},
        /*stream_len=*/512,
        /*reps=*/4);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().bench().has_value());
    REQUIRE(bundle.value().status() == bundle.value().bench()->status());
    REQUIRE(bundle.value().status_str() == DslFuse::fusion_status_str(bundle.value().status()));
    if (bundle.value().status() == DslFuse::FusionStatus::Fused) {
        REQUIRE(bundle.value().selected_cpu_header() == bundle.value().fused_cpu_header());
    } else {
        REQUIRE(bundle.value().selected_cpu_header() == bundle.value().staged_cpu_header());
    }
}

TEST_CASE("DslCatalogBuiltins lookup covers all()", "[dsl][fuse][catalog]") {
    const auto catalog = DslCatalogBuiltins::all();
    REQUIRE(catalog.size() == 6);
    for (const TheoryIr& th : catalog) {
        const auto found = DslCatalogBuiltins::lookup(th.name());
        REQUIRE(found.has_value());
        REQUIRE(found->name() == th.name());
    }
    REQUIRE_FALSE(DslCatalogBuiltins::lookup("missing_theory").has_value());
    REQUIRE(DslCatalogBuiltins::is_staged_catalog_id("caesar"));
    REQUIRE_FALSE(DslCatalogBuiltins::is_staged_catalog_id("matrix_mix"));
    REQUIRE_FALSE(DslCatalogBuiltins::is_staged_catalog_id("autokey_lag"));
}

TEST_CASE("DslFuse inlines catalog matrix_mix (DSL-only leaf)", "[dsl][fuse][catalog]") {
    const TheoryIr atbash = DslCatalogBuiltins::atbash();
    const TheoryIr mix = DslCatalogBuiltins::matrix_mix();
    const StatusOr<ParamIr> a = ParamIr::make("a", 0, 28);
    const StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
    const StatusOr<ParamIr> c = ParamIr::make("c", 0, 28);
    const StatusOr<ParamIr> d = ParamIr::make("d", 0, 28);
    REQUIRE(a.ok());
    REQUIRE(b.ok());
    REQUIRE(c.ok());
    REQUIRE(d.ok());
    const StatusOr<ComposeIr> compose = ComposeIr::make(
        "atbash_then_mix", TheoryIr::Tier::A, {"atbash", "matrix_mix"},
        {a.value(), b.value(), c.value(), d.value()},
        {ComposeIr::StepParamBinding{"matrix_mix", "a", "a"},
         ComposeIr::StepParamBinding{"matrix_mix", "b", "b"},
         ComposeIr::StepParamBinding{"matrix_mix", "c", "c"},
         ComposeIr::StepParamBinding{"matrix_mix", "d", "d"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{atbash, mix};
    const StatusOr<DslFuse::Result> fused = DslFuse::fuse_inline(compose.value(), catalog);
    REQUIRE(fused.ok());
    REQUIRE(fused.value().flattened_steps() == std::vector<std::string>{"atbash", "matrix_mix"});

    // det([[2,3],[5,7]]) = 28; encrypt: atbash(x + 28) after mix then atbash order…
    // Decrypt chain: matrix_mix_dec(atbash(x)) = atbash(x) - det
    Z29Expr::Env env{{"a", Index29{2}}, {"b", Index29{3}}, {"c", Index29{5}}, {"d", Index29{7}}};
    const std::vector<Index29> cipher{Index29{10}, Index29{0}};
    std::vector<Index29> out(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(fused.value().theory().decrypt_step(), "x", env, cipher,
                                        out)
                .ok());
    REQUIRE(out.size() == 2);

    const StatusOr<DslFuse::EmitBundle> bundle =
        DslFuse::emit_compose(compose.value(), catalog, {}, DslFuse::FusionStatus::FallbackStaged);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().status() == DslFuse::FusionStatus::Fused);
    REQUIRE(bundle.value().staged_recipe_json().empty());
    REQUIRE(bundle.value().fused_cpu_header().find("AtbashThenMixTransform") != std::string::npos);
    REQUIRE(bundle.value().fused_cuda_cu().find("Z29Device::") != std::string::npos);
}

TEST_CASE("DslFuse catalog autokey_lag emit includes AutokeyRing", "[dsl][fuse][catalog][autokey]") {
    const TheoryIr lag = DslCatalogBuiltins::autokey_lag();
    const StatusOr<ParamIr> L = ParamIr::make("lag", 1, 28);
    REQUIRE(L.ok());
    const StatusOr<ComposeIr> compose =
        ComposeIr::make("just_autokey", TheoryIr::Tier::A, {"autokey_lag"}, {L.value()},
                        {ComposeIr::StepParamBinding{"autokey_lag", "lag", "lag"}});
    REQUIRE(compose.ok());

    const std::vector<TheoryIr> catalog{lag};
    const StatusOr<DslFuse::EmitBundle> bundle =
        DslFuse::emit_compose(compose.value(), catalog, {}, DslFuse::FusionStatus::Fused);
    REQUIRE(bundle.ok());
    REQUIRE(bundle.value().status() == DslFuse::FusionStatus::Fused);
    REQUIRE(bundle.value().fused_cpu_header().find("AutokeyRing::shift") != std::string::npos);
    REQUIRE(bundle.value().fused_cuda_cu().find("AutokeyRingDevice::shift") != std::string::npos);
    REQUIRE(bundle.value().fused_cuda_cu().find("autokey_ring_device.hpp") != std::string::npos);

    Z29Expr::Env env{{"lag", Index29{2}}};
    const std::vector<Index29> cipher{Index29{5}, Index29{6}, Index29{7}, Index29{8}};
    std::vector<Index29> out(cipher.size());
    REQUIRE(DslIrApplicator::apply_into(lag.decrypt_step(), "x", env, cipher, out).ok());
    // i<2 → key 0; i=2 key=cipher[0]=5 → 7-5=2; i=3 key=6 → 8-6=2
    REQUIRE(out[0] == Index29{5});
    REQUIRE(out[1] == Index29{6});
    REQUIRE(out[2] == Index29{2});
    REQUIRE(out[3] == Index29{2});
}
