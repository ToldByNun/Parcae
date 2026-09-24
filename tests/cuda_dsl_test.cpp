#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <parcae/dsl/dsl_emit_cuda.hpp>
#include <parcae/dsl/dsl_ir_applicator.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <string>
#include <vector>

namespace {

[[nodiscard]] TheoryIr make_dsl_smoke_caesar() {
    const StatusOr<ParamIr> shift = ParamIr::make("shift", 0, 28);
    REQUIRE(shift.ok());
    const Z29Expr::Ptr x = Z29Expr::var("x");
    const Z29Expr::Ptr s = Z29Expr::var("shift");
    const StatusOr<TheoryIr> theory =
        TheoryIr::make("dsl_smoke_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                       TheoryIr::InterruptMode::ElementwiseDefault, {shift.value()},
                       Z29Expr::add(x, s), Z29Expr::sub(x, s));
    REQUIRE(theory.ok());
    return theory.value();
}

[[nodiscard]] std::vector<Index29> stream_of(std::initializer_list<std::uint8_t> vals) {
    std::vector<Index29> out;
    out.reserve(vals.size());
    for (std::uint8_t v : vals) {
        out.push_back(Index29{v});
    }
    return out;
}

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> out;
    out.reserve(indices.size());
    for (Index29 index : indices) {
        out.push_back(index.value());
    }
    return out;
}

[[nodiscard]] std::vector<Index29> from_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        out.push_back(Index29{value});
    }
    return out;
}

} // namespace

TEST_CASE("DslEmitCuda smoke theory text matches golden twin shape", "[cuda][dsl][smoke]") {
    const TheoryIr theory = make_dsl_smoke_caesar();

    const StatusOr<std::string> header = DslEmitCuda::emit_theory_header(theory);
    REQUIRE(header.ok());
    REQUIRE(header.value().find("class DslSmokeCaesarKernel") != std::string::npos);
    REQUIRE(header.value().find("theory_id = \"dsl_smoke_caesar\"") != std::string::npos);
    REQUIRE(header.value().find("launch_device(") != std::string::npos);
    REQUIRE(header.value().find("apply_host(") != std::string::npos);
    REQUIRE(header.value().find("#include \"../params.hpp\"") != std::string::npos);

    const StatusOr<std::string> cu = DslEmitCuda::emit_theory_cu(theory);
    REQUIRE(cu.ok());
    REQUIRE(cu.value().find("#include \"DslSmokeCaesarKernel.hpp\"") != std::string::npos);
    REQUIRE(cu.value().find("#include \"../z29_device.hpp\"") != std::string::npos);
    REQUIRE(cu.value().find("__global__ void dsl_smoke_caesar_kernel(") != std::string::npos);
    REQUIRE(cu.value().find("Z29Device::add(in[i], shift)") != std::string::npos);
    REQUIRE(cu.value().find("Z29Device::sub(in[i], shift)") != std::string::npos);
    REQUIRE(cu.value().find("kThreadsPerBlock = 256") != std::string::npos);
    REQUIRE(cu.value().find("DslSmokeCaesarKernel::launch_device(") != std::string::npos);
    REQUIRE(cu.value().find("DslSmokeCaesarKernel::apply_host(") != std::string::npos);
}

TEST_CASE("DslEmitCuda smoke encrypt/decrypt cuda_mirror matches CPU applicator",
          "[cuda][dsl][smoke]") {
    const TheoryIr theory = make_dsl_smoke_caesar();
    const auto input = stream_of({0, 1, 14, 27, 28});
    const nlohmann::json params{{"shift", 5}};

    StatusOr<std::vector<Index29>> cpu =
        DslIrApplicator::apply(theory, input, params, TransformDirection::Encrypt);
    REQUIRE(cpu.ok());

    for (std::size_t i = 0; i < input.size(); ++i) {
        Z29Expr::Env env;
        env["x"] = input[i];
        env["shift"] = Index29{5};
        StatusOr<Index29> mirrored = theory.encrypt_step()->eval_cuda_mirror(env);
        REQUIRE(mirrored.ok());
        REQUIRE(mirrored.value() == cpu.value()[i]);
    }

    StatusOr<std::vector<Index29>> back =
        DslIrApplicator::apply(theory, cpu.value(), params, TransformDirection::Decrypt);
    REQUIRE(back.ok());
    REQUIRE(back.value() == input);
}

#if defined(PARCAE_HAS_CUDA)

#include "dsl_smoke_caesar_kernel.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

TEST_CASE("CUDA DSL smoke kernel parity vs DslIrApplicator", "[cuda][dsl][smoke]") {
    REQUIRE(ParcaeCuda::available());

    const TheoryIr theory = make_dsl_smoke_caesar();
    const auto input = stream_of({0, 1, 14, 27, 28, 3, 9});
    const std::uint8_t shift = 7;
    const nlohmann::json params{{"shift", static_cast<int>(shift)}};

    StatusOr<std::vector<Index29>> cpu_enc =
        DslIrApplicator::apply(theory, input, params, TransformDirection::Encrypt);
    REQUIRE(cpu_enc.ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_enc(host_in.size(), 0xFFu);
    REQUIRE(DslSmokeCaesarKernel::apply_host(host_in, host_enc, shift, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_enc) == cpu_enc.value());

    StatusOr<std::vector<Index29>> cpu_dec =
        DslIrApplicator::apply(theory, cpu_enc.value(), params, TransformDirection::Decrypt);
    REQUIRE(cpu_dec.ok());

    std::vector<std::uint8_t> host_dec(host_enc.size());
    REQUIRE(DslSmokeCaesarKernel::apply_host(host_enc, host_dec, shift, CudaDir::Decrypt).ok());
    REQUIRE(from_bytes(host_dec) == cpu_dec.value());
    REQUIRE(host_dec == host_in);
}

TEST_CASE("CUDA DSL smoke kernel empty and null guards", "[cuda][dsl][smoke]") {
    REQUIRE(ParcaeCuda::available());
    REQUIRE(DslSmokeCaesarKernel::launch_device(nullptr, nullptr, 0, 0, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(
        DslSmokeCaesarKernel::launch_device(nullptr, nullptr, 1, 0, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(DslSmokeCaesarKernel::apply_host({}, {1}, 0, CudaDir::Encrypt).ok());
}

#else

TEST_CASE("CUDA DSL device smoke skipped (PARCAE_HAS_CUDA unset)", "[cuda][dsl][smoke]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise DslSmokeCaesarKernel");
}

#endif
