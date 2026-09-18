#include "affine_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void affine_kernel(
    const std::uint8_t* in,
    std::uint8_t* out,
    std::size_t count,
    std::uint8_t a,
    std::uint8_t b,
    std::uint8_t encrypt) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t x = in[i];
    if (encrypt != 0u) {
        out[i] = Z29Device::add(Z29Device::mul(a, x), b);
    } else {
        const std::uint8_t inv_a = Z29Device::inv(a);
        out[i] = Z29Device::mul(inv_a, Z29Device::sub(x, b));
    }
}

[[nodiscard]] Status validate_params(std::uint8_t a, std::uint8_t b) {
    if (a < 1 || a > 28) {
        return Status::error("AffineKernel: a must be in 1..28");
    }
    if (b > 28) {
        return Status::error("AffineKernel: b must be in 0..28");
    }
    return Status::success();
}

}  // namespace

Status AffineKernel::launch_device(
    const std::uint8_t* device_in,
    std::uint8_t* device_out,
    std::size_t count,
    std::uint8_t a,
    std::uint8_t b,
    CudaDir direction) {
    Status params_ok = validate_params(a, b);
    if (!params_ok.ok()) {
        return params_ok;
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("AffineKernel::launch_device null device pointer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    affine_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, count, a, b, encrypt);

    Status launch = CudaError::to_status(cudaGetLastError(), "AffineKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AffineKernel::launch_device sync");
}

Status AffineKernel::apply_host(
    std::span<const std::uint8_t> host_in,
    std::span<std::uint8_t> host_out,
    std::uint8_t a,
    std::uint8_t b,
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
        Status launched = launch_device(
            device.value().data(), device.value().data(), host_in.size(), a, b, direction);
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

    Status launched = launch_device(
        device_in.value().data(),
        device_out.value().data(),
        host_in.size(),
        a,
        b,
        direction);
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
