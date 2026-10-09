#include "affine_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "launch_geom.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__device__ inline std::uint8_t affine_map(std::uint8_t x, std::uint8_t a, std::uint8_t b,
                                          std::uint8_t inv_a, std::uint8_t encrypt) {
    if (encrypt != 0u) {
        return Z29Device::add(Z29Device::mul(a, x), b);
    }
    return Z29Device::mul(inv_a, Z29Device::sub(x, b));
}

__global__ void affine_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count, std::uint8_t a, std::uint8_t b,
                                     std::uint8_t inv_a, std::uint8_t encrypt) {
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
        uchar4 w;
        w.x = affine_map(v.x, a, b, inv_a, encrypt);
        w.y = affine_map(v.y, a, b, inv_a, encrypt);
        w.z = affine_map(v.z, a, b, inv_a, encrypt);
        w.w = affine_map(v.w, a, b, inv_a, encrypt);
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        out[t] = affine_map(__ldg(in + t), a, b, inv_a, encrypt);
    }
}

__global__ void affine_scalar_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count, std::uint8_t a, std::uint8_t b,
                                     std::uint8_t inv_a, std::uint8_t encrypt) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        out[t] = affine_map(__ldg(in + t), a, b, inv_a, encrypt);
    }
}

Status AffineKernel::validate_params(std::uint8_t a, std::uint8_t b) {
    if (a < 1 || a > 28) {
        return Status::error("AffineKernel: a must be in 1..28");
    }
    if (b > 28) {
        return Status::error("AffineKernel: b must be in 0..28");
    }
    return Status::success();
}

Status AffineKernel::launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out,
                                         std::size_t count, std::uint8_t a, std::uint8_t b,
                                         CudaDir direction) {
    Status params_ok = validate_params(a, b);
    if (!params_ok.ok()) {
        return params_ok;
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("AffineKernel::launch_device_async null device pointer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint8_t inv_a = encrypt != 0u ? static_cast<std::uint8_t>(0) : Z29Device::inv(a);
    const bool aligned = (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
                         (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
    const std::size_t work = aligned ? (count + 3u) / 4u : count;
    const int blocks =
        LaunchGeom::blocks_for(work);

    if (aligned) {
        affine_uchar4_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count, a, b,
                                                            inv_a, encrypt);
    } else {
        affine_scalar_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count, a, b,
                                                            inv_a, encrypt);
    }
    return CudaError::to_status(cudaGetLastError(), "AffineKernel::launch_device_async");
}

Status AffineKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                   std::size_t count, std::uint8_t a, std::uint8_t b,
                                   CudaDir direction) {
    Status launched = launch_device_async(device_in, device_out, count, a, b, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AffineKernel::launch_device sync");
}

Status AffineKernel::apply_host(std::span<const std::uint8_t> host_in,
                                std::span<std::uint8_t> host_out, std::uint8_t a, std::uint8_t b,
                                CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("AffineKernel::apply_host size mismatch");
    }

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched = launch_device(device.value().data(), device.value().data(),
                                        host_in.size(), a, b, direction);
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
                                    host_in.size(), a, b, direction);
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
