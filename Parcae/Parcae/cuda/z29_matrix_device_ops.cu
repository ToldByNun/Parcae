#include "z29_matrix_device_ops.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "z29_matrix2_device.hpp"
#include "z29_matrix3_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void matrix2_inverse_kernel(const std::uint8_t* matrices_in, std::uint8_t* matrices_out,
                                       std::uint8_t* singular, std::size_t count) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t* in = matrices_in + i * Z29MatrixDeviceOps::kMatrix2Entries;
    std::uint8_t* out = matrices_out + i * Z29MatrixDeviceOps::kMatrix2Entries;
    Z29Matrix2Device::try_inverse(in, out, singular + i);
}

__global__ void matrix3_inverse_kernel(const std::uint8_t* matrices_in, std::uint8_t* matrices_out,
                                       std::uint8_t* singular, std::size_t count) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t* in = matrices_in + i * Z29MatrixDeviceOps::kMatrix3Entries;
    std::uint8_t* out = matrices_out + i * Z29MatrixDeviceOps::kMatrix3Entries;
    Z29Matrix3Device::try_inverse(in, out, singular + i);
}

__global__ void matrix2_mul_vec_kernel(const std::uint8_t* matrices, const std::uint8_t* vecs_in,
                                       std::uint8_t* vecs_out, std::size_t count) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t* m = matrices + i * Z29MatrixDeviceOps::kMatrix2Entries;
    const std::uint8_t* v = vecs_in + i * Z29MatrixDeviceOps::kVec2;
    std::uint8_t* out = vecs_out + i * Z29MatrixDeviceOps::kVec2;
    Z29Matrix2Device::mul_vec(m, v[0], v[1], out);
}

__global__ void matrix3_mul_vec_kernel(const std::uint8_t* matrices, const std::uint8_t* vecs_in,
                                       std::uint8_t* vecs_out, std::size_t count) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t* m = matrices + i * Z29MatrixDeviceOps::kMatrix3Entries;
    const std::uint8_t* v = vecs_in + i * Z29MatrixDeviceOps::kVec3;
    std::uint8_t* out = vecs_out + i * Z29MatrixDeviceOps::kVec3;
    Z29Matrix3Device::mul_vec(m, v[0], v[1], v[2], out);
}

[[nodiscard]] int blocks_for(std::size_t count) {
    return static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                            static_cast<std::size_t>(kThreadsPerBlock));
}

} // namespace

Status Z29MatrixDeviceOps::launch_device_inverse2(const std::uint8_t* device_matrices_in,
                                                 std::uint8_t* device_matrices_out,
                                                 std::uint8_t* device_singular,
                                                 std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_matrices_in == nullptr || device_matrices_out == nullptr ||
        device_singular == nullptr) {
        return Status::error("Z29MatrixDeviceOps::launch_device_inverse2 null device pointer");
    }
    matrix2_inverse_kernel<<<blocks_for(count), kThreadsPerBlock>>>(
        device_matrices_in, device_matrices_out, device_singular, count);
    Status launch =
        CudaError::to_status(cudaGetLastError(), "Z29MatrixDeviceOps::launch_device_inverse2");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "Z29MatrixDeviceOps::launch_device_inverse2 sync");
}

Status Z29MatrixDeviceOps::apply_host_inverse2(std::span<const std::uint8_t> matrices_in,
                                               std::span<std::uint8_t> matrices_out,
                                               std::span<std::uint8_t> singular) {
    if (matrices_in.size() != matrices_out.size()) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse2 matrix size mismatch");
    }
    if (matrices_in.size() % kMatrix2Entries != 0) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse2 matrices not multiple of 4");
    }
    const std::size_t count = matrices_in.size() / kMatrix2Entries;
    if (singular.size() != count) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse2 singular size mismatch");
    }
    if (count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> d_in = DeviceBuffer<std::uint8_t>::from_host(matrices_in);
    if (!d_in.ok()) {
        return d_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_out =
        DeviceBuffer<std::uint8_t>::allocate(matrices_out.size());
    if (!d_out.ok()) {
        return d_out.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_sing = DeviceBuffer<std::uint8_t>::allocate(count);
    if (!d_sing.ok()) {
        return d_sing.status();
    }

    Status launched =
        launch_device_inverse2(d_in.value().data(), d_out.value().data(), d_sing.value().data(),
                              count);
    if (!launched.ok()) {
        return launched;
    }
    Status copy_out = d_out.value().copy_to_host(matrices_out);
    if (!copy_out.ok()) {
        return copy_out;
    }
    return d_sing.value().copy_to_host(singular);
}

Status Z29MatrixDeviceOps::launch_device_inverse3(const std::uint8_t* device_matrices_in,
                                                 std::uint8_t* device_matrices_out,
                                                 std::uint8_t* device_singular,
                                                 std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_matrices_in == nullptr || device_matrices_out == nullptr ||
        device_singular == nullptr) {
        return Status::error("Z29MatrixDeviceOps::launch_device_inverse3 null device pointer");
    }
    matrix3_inverse_kernel<<<blocks_for(count), kThreadsPerBlock>>>(
        device_matrices_in, device_matrices_out, device_singular, count);
    Status launch =
        CudaError::to_status(cudaGetLastError(), "Z29MatrixDeviceOps::launch_device_inverse3");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "Z29MatrixDeviceOps::launch_device_inverse3 sync");
}

Status Z29MatrixDeviceOps::apply_host_inverse3(std::span<const std::uint8_t> matrices_in,
                                               std::span<std::uint8_t> matrices_out,
                                               std::span<std::uint8_t> singular) {
    if (matrices_in.size() != matrices_out.size()) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse3 matrix size mismatch");
    }
    if (matrices_in.size() % kMatrix3Entries != 0) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse3 matrices not multiple of 9");
    }
    const std::size_t count = matrices_in.size() / kMatrix3Entries;
    if (singular.size() != count) {
        return Status::error("Z29MatrixDeviceOps::apply_host_inverse3 singular size mismatch");
    }
    if (count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> d_in = DeviceBuffer<std::uint8_t>::from_host(matrices_in);
    if (!d_in.ok()) {
        return d_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_out =
        DeviceBuffer<std::uint8_t>::allocate(matrices_out.size());
    if (!d_out.ok()) {
        return d_out.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_sing = DeviceBuffer<std::uint8_t>::allocate(count);
    if (!d_sing.ok()) {
        return d_sing.status();
    }

    Status launched =
        launch_device_inverse3(d_in.value().data(), d_out.value().data(), d_sing.value().data(),
                              count);
    if (!launched.ok()) {
        return launched;
    }
    Status copy_out = d_out.value().copy_to_host(matrices_out);
    if (!copy_out.ok()) {
        return copy_out;
    }
    return d_sing.value().copy_to_host(singular);
}

Status Z29MatrixDeviceOps::launch_device_mul_vec2(const std::uint8_t* device_matrices,
                                                  const std::uint8_t* device_vecs_in,
                                                  std::uint8_t* device_vecs_out,
                                                  std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_matrices == nullptr || device_vecs_in == nullptr || device_vecs_out == nullptr) {
        return Status::error("Z29MatrixDeviceOps::launch_device_mul_vec2 null device pointer");
    }
    matrix2_mul_vec_kernel<<<blocks_for(count), kThreadsPerBlock>>>(device_matrices, device_vecs_in,
                                                                    device_vecs_out, count);
    Status launch =
        CudaError::to_status(cudaGetLastError(), "Z29MatrixDeviceOps::launch_device_mul_vec2");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "Z29MatrixDeviceOps::launch_device_mul_vec2 sync");
}

Status Z29MatrixDeviceOps::apply_host_mul_vec2(std::span<const std::uint8_t> matrices,
                                               std::span<const std::uint8_t> vecs_in,
                                               std::span<std::uint8_t> vecs_out) {
    if (matrices.size() % kMatrix2Entries != 0) {
        return Status::error("Z29MatrixDeviceOps::apply_host_mul_vec2 matrices not multiple of 4");
    }
    const std::size_t count = matrices.size() / kMatrix2Entries;
    if (vecs_in.size() != count * kVec2 || vecs_out.size() != count * kVec2) {
        return Status::error("Z29MatrixDeviceOps::apply_host_mul_vec2 vector size mismatch");
    }
    if (count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> d_m = DeviceBuffer<std::uint8_t>::from_host(matrices);
    if (!d_m.ok()) {
        return d_m.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_in = DeviceBuffer<std::uint8_t>::from_host(vecs_in);
    if (!d_in.ok()) {
        return d_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_out =
        DeviceBuffer<std::uint8_t>::allocate(vecs_out.size());
    if (!d_out.ok()) {
        return d_out.status();
    }

    Status launched =
        launch_device_mul_vec2(d_m.value().data(), d_in.value().data(), d_out.value().data(), count);
    if (!launched.ok()) {
        return launched;
    }
    return d_out.value().copy_to_host(vecs_out);
}

Status Z29MatrixDeviceOps::launch_device_mul_vec3(const std::uint8_t* device_matrices,
                                                  const std::uint8_t* device_vecs_in,
                                                  std::uint8_t* device_vecs_out,
                                                  std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_matrices == nullptr || device_vecs_in == nullptr || device_vecs_out == nullptr) {
        return Status::error("Z29MatrixDeviceOps::launch_device_mul_vec3 null device pointer");
    }
    matrix3_mul_vec_kernel<<<blocks_for(count), kThreadsPerBlock>>>(device_matrices, device_vecs_in,
                                                                    device_vecs_out, count);
    Status launch =
        CudaError::to_status(cudaGetLastError(), "Z29MatrixDeviceOps::launch_device_mul_vec3");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "Z29MatrixDeviceOps::launch_device_mul_vec3 sync");
}

Status Z29MatrixDeviceOps::apply_host_mul_vec3(std::span<const std::uint8_t> matrices,
                                               std::span<const std::uint8_t> vecs_in,
                                               std::span<std::uint8_t> vecs_out) {
    if (matrices.size() % kMatrix3Entries != 0) {
        return Status::error("Z29MatrixDeviceOps::apply_host_mul_vec3 matrices not multiple of 9");
    }
    const std::size_t count = matrices.size() / kMatrix3Entries;
    if (vecs_in.size() != count * kVec3 || vecs_out.size() != count * kVec3) {
        return Status::error("Z29MatrixDeviceOps::apply_host_mul_vec3 vector size mismatch");
    }
    if (count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> d_m = DeviceBuffer<std::uint8_t>::from_host(matrices);
    if (!d_m.ok()) {
        return d_m.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_in = DeviceBuffer<std::uint8_t>::from_host(vecs_in);
    if (!d_in.ok()) {
        return d_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> d_out =
        DeviceBuffer<std::uint8_t>::allocate(vecs_out.size());
    if (!d_out.ok()) {
        return d_out.status();
    }

    Status launched =
        launch_device_mul_vec3(d_m.value().data(), d_in.value().data(), d_out.value().data(), count);
    if (!launched.ok()) {
        return launched;
    }
    return d_out.value().copy_to_host(vecs_out);
}
