#include "hill2_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "launch_geom.hpp"
#include "z29_matrix2_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__device__ inline void hill2_load_apply(const std::uint8_t* __restrict__ matrix,
                                        std::uint8_t encrypt, std::uint8_t apply[4]) {
    std::uint8_t m[4];
    m[0] = __ldg(matrix + 0);
    m[1] = __ldg(matrix + 1);
    m[2] = __ldg(matrix + 2);
    m[3] = __ldg(matrix + 3);
    if (encrypt != 0u) {
        apply[0] = m[0];
        apply[1] = m[1];
        apply[2] = m[2];
        apply[3] = m[3];
        return;
    }
    std::uint8_t singular = 0;
    Z29Matrix2Device::try_inverse(m, apply, &singular);
    (void)singular;
}

__global__ void hill2_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                    std::size_t count, const std::uint8_t* __restrict__ matrix,
                                    std::uint8_t encrypt) {
    std::uint8_t apply[4];
    hill2_load_apply(matrix, encrypt, apply);

    const std::size_t n4 = count / 4u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);
    uchar4* out4 = reinterpret_cast<uchar4*>(out);

    for (std::size_t i = tid; i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        std::uint8_t p0[2];
        std::uint8_t p1[2];
        Z29Matrix2Device::mul_vec(apply, v.x, v.y, p0);
        Z29Matrix2Device::mul_vec(apply, v.z, v.w, p1);
        uchar4 w;
        w.x = p0[0];
        w.y = p0[1];
        w.z = p1[0];
        w.w = p1[1];
        out4[i] = w;
    }

    // Even length guaranteed; leftover is 0 or 2 bytes (one Hill pair).
    for (std::size_t t = n4 * 4u + tid * 2u; t + 1u < count; t += stride * 2u) {
        std::uint8_t pair[2];
        Z29Matrix2Device::mul_vec(apply, __ldg(in + t), __ldg(in + t + 1u), pair);
        out[t] = pair[0];
        out[t + 1u] = pair[1];
    }
}

__global__ void hill2_pair_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                  std::size_t count, const std::uint8_t* __restrict__ matrix,
                                  std::uint8_t encrypt) {
    std::uint8_t apply[4];
    hill2_load_apply(matrix, encrypt, apply);

    const std::size_t npairs = count / 2u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t p = tid; p < npairs; p += stride) {
        const std::size_t t = p * 2u;
        std::uint8_t pair[2];
        Z29Matrix2Device::mul_vec(apply, __ldg(in + t), __ldg(in + t + 1u), pair);
        out[t] = pair[0];
        out[t + 1u] = pair[1];
    }
}

Status Hill2Kernel::validate_matrix(std::span<const std::uint8_t> host_matrix) {
    if (host_matrix.size() != 4u) {
        return Status::error("Hill2Kernel: matrix must contain exactly 4 entries");
    }
    for (std::size_t i = 0; i < 4u; ++i) {
        if (host_matrix[i] > 28u) {
            return Status::error("Hill2Kernel: matrix entries must be in 0..28");
        }
    }
    if (Z29Matrix2Device::det(host_matrix.data()) == 0u) {
        return Status::error("Hill2Kernel: matrix must be invertible (det ≢ 0 mod 29)");
    }
    return Status::success();
}

Status Hill2Kernel::launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t count, const std::uint8_t* device_matrix,
                                        CudaDir direction) {
    if (count % 2u != 0u) {
        return Status::error("Hill2Kernel: input length must be even");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr || device_matrix == nullptr) {
        return Status::error("Hill2Kernel::launch_device_async null device pointer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const bool aligned = (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
                         (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
    const std::size_t work = aligned ? (count + 3u) / 4u : count / 2u;
    const int blocks = LaunchGeom::blocks_for(work);

    if (aligned) {
        hill2_uchar4_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count,
                                                               device_matrix, encrypt);
    } else {
        hill2_pair_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count,
                                                             device_matrix, encrypt);
    }
    return CudaError::to_status(cudaGetLastError(), "Hill2Kernel::launch_device_async");
}

Status Hill2Kernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                  std::size_t count, const std::uint8_t* device_matrix,
                                  CudaDir direction) {
    Status launched = launch_device_async(device_in, device_out, count, device_matrix, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "Hill2Kernel::launch_device sync");
}

Status Hill2Kernel::apply_host(std::span<const std::uint8_t> host_in,
                               std::span<std::uint8_t> host_out,
                               std::span<const std::uint8_t> host_matrix, CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("Hill2Kernel::apply_host size mismatch");
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
