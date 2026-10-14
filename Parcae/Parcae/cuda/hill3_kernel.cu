#include "hill3_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "launch_geom.hpp"
#include "z29_matrix3_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__device__ inline void hill3_load_apply(const std::uint8_t* __restrict__ matrix,
                                        std::uint8_t encrypt, std::uint8_t apply[9]) {
    std::uint8_t m[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        m[i] = __ldg(matrix + i);
    }
    if (encrypt != 0u) {
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            apply[i] = m[i];
        }
        return;
    }
    std::uint8_t singular = 0;
    Z29Matrix3Device::try_inverse(m, apply, &singular);
    (void)singular;
}

__global__ void hill3_triple_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                    std::size_t count, const std::uint8_t* __restrict__ matrix,
                                    std::uint8_t encrypt) {
    std::uint8_t apply[9];
    hill3_load_apply(matrix, encrypt, apply);

    const std::size_t ntriples = count / 3u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t p = tid; p < ntriples; p += stride) {
        const std::size_t t = p * 3u;
        std::uint8_t triple[3];
        Z29Matrix3Device::mul_vec(apply, __ldg(in + t), __ldg(in + t + 1u), __ldg(in + t + 2u),
                                  triple);
        out[t] = triple[0];
        out[t + 1u] = triple[1];
        out[t + 2u] = triple[2];
    }
}

Status Hill3Kernel::validate_matrix(std::span<const std::uint8_t> host_matrix) {
    if (host_matrix.size() != 9u) {
        return Status::error("Hill3Kernel: matrix must contain exactly 9 entries");
    }
    for (std::size_t i = 0; i < 9u; ++i) {
        if (host_matrix[i] > 28u) {
            return Status::error("Hill3Kernel: matrix entries must be in 0..28");
        }
    }
    if (Z29Matrix3Device::det(host_matrix.data()) == 0u) {
        return Status::error("Hill3Kernel: matrix must be invertible (det ≢ 0 mod 29)");
    }
    return Status::success();
}

Status Hill3Kernel::launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t count, const std::uint8_t* device_matrix,
                                        CudaDir direction) {
    if (count % 3u != 0u) {
        return Status::error("Hill3Kernel: input length must be a multiple of 3");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr || device_matrix == nullptr) {
        return Status::error("Hill3Kernel::launch_device_async null device pointer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::size_t work = count / 3u;
    const int blocks = LaunchGeom::blocks_for(work);

    hill3_triple_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count,
                                                           device_matrix, encrypt);
    return CudaError::to_status(cudaGetLastError(), "Hill3Kernel::launch_device_async");
}

Status Hill3Kernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                  std::size_t count, const std::uint8_t* device_matrix,
                                  CudaDir direction) {
    Status launched = launch_device_async(device_in, device_out, count, device_matrix, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "Hill3Kernel::launch_device sync");
}

Status Hill3Kernel::apply_host(std::span<const std::uint8_t> host_in,
                               std::span<std::uint8_t> host_out,
                               std::span<const std::uint8_t> host_matrix, CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("Hill3Kernel::apply_host size mismatch");
    }
    Status matrix_ok = validate_matrix(host_matrix);
    if (!matrix_ok.ok()) {
        return matrix_ok;
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_matrix =
        DeviceBuffer<std::uint8_t>::from_host(host_matrix);
    if (!device_matrix.ok()) {
        return device_matrix.status();
    }

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched = launch_device(device.value().data(), device.value().data(),
                                        host_in.size(), device_matrix.value().data(), direction);
        if (!launched.ok()) {
            return launched;
        }
        return device.value().copy_to_host(host_out);
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    if (!device_in.ok()) {
        return device_in.status();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_out =
        DeviceBuffer<std::uint8_t>::allocate(host_out.size());
    if (!device_out.ok()) {
        return device_out.status();
    }

    Status launched = launch_device(device_in.value().data(), device_out.value().data(),
                                    host_in.size(), device_matrix.value().data(), direction);
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
