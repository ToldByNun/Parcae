#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae/core/index29.hpp"
#include "parcae/math/z29_matrix2.hpp"
#include "parcae/math/z29_matrix3.hpp"
#include "parcae/transform/hill2_transform.hpp"
#include "parcae/transform/hill3_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hill2_kernel.hpp"
#include "hill3_kernel.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace {

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

[[nodiscard]] std::array<std::uint8_t, 4> matrix2_bytes(const Z29Matrix2& m) {
    return {m.a().value(), m.b().value(), m.c().value(), m.d().value()};
}

[[nodiscard]] std::array<std::uint8_t, 9> matrix3_bytes(const Z29Matrix3& m) {
    std::array<std::uint8_t, 9> out{};
    for (std::size_t i = 0; i < 9; ++i) {
        out[i] = m.row_major()[i].value();
    }
    return out;
}

} // namespace

TEST_CASE("CUDA hill2 parity vs CPU fixed encrypt decrypt", "[cuda][parity][hill]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix2 matrix{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    const auto matrix_bytes = matrix2_bytes(matrix);
    const std::vector<Index29> input{Index29{4}, Index29{6}, Index29{0}, Index29{1},
                                     Index29{28}, Index29{14}};

    std::vector<Index29> cpu_enc(input.size());
    REQUIRE(Hill2Transform::kernel(input, cpu_enc, matrix, TransformDirection::Encrypt).ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_enc(host_in.size(), 0xFFu);
    REQUIRE(Hill2Kernel::apply_host(host_in, host_enc, matrix_bytes, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_enc) == cpu_enc);

    std::vector<Index29> cpu_dec(input.size());
    REQUIRE(Hill2Transform::kernel(cpu_enc, cpu_dec, matrix, TransformDirection::Decrypt).ok());

    std::vector<std::uint8_t> host_dec(host_enc.size());
    REQUIRE(Hill2Kernel::apply_host(host_enc, host_dec, matrix_bytes, CudaDir::Decrypt).ok());
    REQUIRE(from_bytes(host_dec) == cpu_dec);
    REQUIRE(host_dec == host_in);
}

TEST_CASE("CUDA hill3 parity vs CPU fixed encrypt decrypt", "[cuda][parity][hill]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix3 matrix{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                            Index29{4}, Index29{5}, Index29{6}, Index29{0}};
    const auto matrix_bytes = matrix3_bytes(matrix);
    const std::vector<Index29> input{Index29{2}, Index29{3}, Index29{4}, Index29{0},
                                     Index29{1}, Index29{28}, Index29{7}, Index29{8},
                                     Index29{9}};

    std::vector<Index29> cpu_enc(input.size());
    REQUIRE(Hill3Transform::kernel(input, cpu_enc, matrix, TransformDirection::Encrypt).ok());

    const std::vector<std::uint8_t> host_in = to_bytes(input);
    std::vector<std::uint8_t> host_enc(host_in.size(), 0xFFu);
    REQUIRE(Hill3Kernel::apply_host(host_in, host_enc, matrix_bytes, CudaDir::Encrypt).ok());
    REQUIRE(from_bytes(host_enc) == cpu_enc);

    std::vector<Index29> cpu_dec(input.size());
    REQUIRE(Hill3Transform::kernel(cpu_enc, cpu_dec, matrix, TransformDirection::Decrypt).ok());

    std::vector<std::uint8_t> host_dec(host_enc.size());
    REQUIRE(Hill3Kernel::apply_host(host_enc, host_dec, matrix_bytes, CudaDir::Decrypt).ok());
    REQUIRE(from_bytes(host_dec) == cpu_dec);
    REQUIRE(host_dec == host_in);
}

TEST_CASE("CUDA hill2/hill3 random round-trip and in-place", "[cuda][parity][hill]") {
    REQUIRE(ParcaeCuda::available());

    std::mt19937 rng(0xC11A11u);
    std::uniform_int_distribution<int> dist(0, 28);

    {
        const Z29Matrix2 matrix{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
        const auto matrix_bytes = matrix2_bytes(matrix);

        std::vector<Index29> input;
        input.reserve(258);
        for (std::size_t i = 0; i < 258; ++i) {
            input.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        std::vector<Index29> cpu_out(input.size());
        REQUIRE(Hill2Transform::kernel(input, cpu_out, matrix, TransformDirection::Encrypt).ok());

        std::vector<std::uint8_t> host = to_bytes(input);
        std::vector<std::uint8_t> host_out(host.size());
        REQUIRE(Hill2Kernel::apply_host(host, host_out, matrix_bytes, CudaDir::Encrypt).ok());
        REQUIRE(from_bytes(host_out) == cpu_out);

        std::vector<std::uint8_t> round_trip(host_out.size());
        REQUIRE(Hill2Kernel::apply_host(host_out, round_trip, matrix_bytes, CudaDir::Decrypt).ok());
        REQUIRE(round_trip == host);

        REQUIRE(Hill2Kernel::apply_host(host, host, matrix_bytes, CudaDir::Encrypt).ok());
        REQUIRE(from_bytes(host) == cpu_out);
    }

    {
        const Z29Matrix3 matrix{Index29{1}, Index29{2}, Index29{3}, Index29{0}, Index29{1},
                                Index29{4}, Index29{5}, Index29{6}, Index29{0}};
        const auto matrix_bytes = matrix3_bytes(matrix);

        std::vector<Index29> input;
        input.reserve(258);
        for (std::size_t i = 0; i < 258; ++i) {
            input.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }

        std::vector<Index29> cpu_out(input.size());
        REQUIRE(Hill3Transform::kernel(input, cpu_out, matrix, TransformDirection::Encrypt).ok());

        std::vector<std::uint8_t> host = to_bytes(input);
        std::vector<std::uint8_t> host_out(host.size());
        REQUIRE(Hill3Kernel::apply_host(host, host_out, matrix_bytes, CudaDir::Encrypt).ok());
        REQUIRE(from_bytes(host_out) == cpu_out);

        std::vector<std::uint8_t> round_trip(host_out.size());
        REQUIRE(Hill3Kernel::apply_host(host_out, round_trip, matrix_bytes, CudaDir::Decrypt).ok());
        REQUIRE(round_trip == host);

        REQUIRE(Hill3Kernel::apply_host(host, host, matrix_bytes, CudaDir::Encrypt).ok());
        REQUIRE(from_bytes(host) == cpu_out);
    }
}

TEST_CASE("CUDA hill2 uchar4 lengths and async", "[cuda][parity][hill]") {
    REQUIRE(ParcaeCuda::available());

    const Z29Matrix2 matrix{Index29{2}, Index29{3}, Index29{5}, Index29{7}};
    const auto matrix_bytes = matrix2_bytes(matrix);
    StatusOr<DeviceBuffer<std::uint8_t>> device_matrix =
        DeviceBuffer<std::uint8_t>::from_host(matrix_bytes);
    REQUIRE(device_matrix.ok());

    const std::size_t lengths[] = {2u, 4u, 6u, 28u, 258u};
    for (std::size_t n : lengths) {
        std::vector<Index29> input;
        input.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            input.push_back(Index29{static_cast<std::uint8_t>(i % 29u)});
        }
        std::vector<Index29> cpu(n);
        REQUIRE(Hill2Transform::kernel(input, cpu, matrix, TransformDirection::Encrypt).ok());

        const std::vector<std::uint8_t> host_in = to_bytes(input);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        REQUIRE(device_in.ok());
        StatusOr<DeviceBuffer<std::uint8_t>> device_out =
            DeviceBuffer<std::uint8_t>::allocate(n);
        REQUIRE(device_out.ok());

        REQUIRE(Hill2Kernel::launch_device_async(device_in.value().data(),
                                                 device_out.value().data(), n,
                                                 device_matrix.value().data(), CudaDir::Encrypt)
                    .ok());
        REQUIRE(CudaError::to_status(cudaDeviceSynchronize(), "hill2 async sync").ok());

        std::vector<std::uint8_t> host_out(n);
        REQUIRE(device_out.value().copy_to_host(host_out).ok());
        REQUIRE(from_bytes(host_out) == cpu);
    }
}

TEST_CASE("CUDA hill rejects odd length singular matrix and size mismatch",
          "[cuda][parity][hill]") {
    REQUIRE(ParcaeCuda::available());

    const std::array<std::uint8_t, 4> hill2_ok{2, 3, 5, 7};
    const std::array<std::uint8_t, 4> hill2_singular{1, 2, 2, 4};
    const std::array<std::uint8_t, 9> hill3_ok{1, 2, 3, 0, 1, 4, 5, 6, 0};
    const std::array<std::uint8_t, 9> hill3_singular{1, 2, 3, 2, 4, 6, 0, 1, 0};

    REQUIRE(Hill2Kernel::launch_device(nullptr, nullptr, 0, nullptr, CudaDir::Encrypt).ok());
    REQUIRE(Hill3Kernel::launch_device(nullptr, nullptr, 0, nullptr, CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> odd2{1, 2, 3};
    std::vector<std::uint8_t> out2(3);
    REQUIRE_FALSE(Hill2Kernel::apply_host(odd2, out2, hill2_ok, CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> odd3{1, 2, 3, 4};
    std::vector<std::uint8_t> out3(4);
    REQUIRE_FALSE(Hill3Kernel::apply_host(odd3, out3, hill3_ok, CudaDir::Encrypt).ok());

    std::vector<std::uint8_t> even{1, 2};
    REQUIRE_FALSE(Hill2Kernel::apply_host(even, even, hill2_singular, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(Hill2Kernel::apply_host(even, even, hill2_singular, CudaDir::Decrypt).ok());

    std::vector<std::uint8_t> triple{1, 2, 3};
    REQUIRE_FALSE(Hill3Kernel::apply_host(triple, triple, hill3_singular, CudaDir::Encrypt).ok());
    REQUIRE_FALSE(Hill3Kernel::apply_host(triple, triple, hill3_singular, CudaDir::Decrypt).ok());

    std::vector<std::uint8_t> in{1, 2};
    std::vector<std::uint8_t> short_out(1);
    REQUIRE_FALSE(Hill2Kernel::apply_host(in, short_out, hill2_ok, CudaDir::Encrypt).ok());

    StatusOr<DeviceBuffer<std::uint8_t>> device_matrix =
        DeviceBuffer<std::uint8_t>::from_host(hill2_ok);
    REQUIRE(device_matrix.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_buf =
        DeviceBuffer<std::uint8_t>::allocate(3);
    REQUIRE(device_buf.ok());
    REQUIRE_FALSE(Hill2Kernel::launch_device_async(device_buf.value().data(),
                                                   device_buf.value().data(), 3,
                                                   device_matrix.value().data(), CudaDir::Encrypt)
                      .ok());

    StatusOr<DeviceBuffer<std::uint8_t>> device_matrix3 =
        DeviceBuffer<std::uint8_t>::from_host(hill3_ok);
    REQUIRE(device_matrix3.ok());
    StatusOr<DeviceBuffer<std::uint8_t>> device_buf4 =
        DeviceBuffer<std::uint8_t>::allocate(4);
    REQUIRE(device_buf4.ok());
    REQUIRE_FALSE(Hill3Kernel::launch_device_async(device_buf4.value().data(),
                                                   device_buf4.value().data(), 4,
                                                   device_matrix3.value().data(), CudaDir::Encrypt)
                      .ok());
}

#else

TEST_CASE("CUDA hill skipped (PARCAE_HAS_CUDA unset)", "[cuda][parity][hill]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise Hill2Kernel / Hill3Kernel");
}

#endif
