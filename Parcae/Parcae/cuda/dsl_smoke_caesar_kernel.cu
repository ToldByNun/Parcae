#include "dsl_smoke_caesar_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void dsl_smoke_caesar_kernel(
    const std::uint8_t* in,
    std::uint8_t* out,
    std::size_t count,
    std::uint8_t shift,
    std::uint8_t encrypt) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    if (encrypt != 0u) {
        out[i] = Z29Device::add(in[i], shift);
    } else {
        out[i] = Z29Device::sub(in[i], shift);
    }
}

}  // namespace

Status DslSmokeCaesarKernel::launch_device(
    const std::uint8_t* device_in,
    std::uint8_t* device_out,
    std::size_t count,
    std::uint8_t shift,
    CudaDir direction) {
    if (shift > 28u) {
        return Status::error("DslSmokeCaesarKernel: shift must be in 0..28");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("DslSmokeCaesarKernel::launch_device null device pointer");
    }
    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1)
                                      : static_cast<std::uint8_t>(0);
    const int blocks = static_cast<int>(
        (count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
        static_cast<std::size_t>(kThreadsPerBlock));
    dsl_smoke_caesar_kernel<<<blocks, kThreadsPerBlock>>>(
        device_in, device_out, count, shift, encrypt);

    Status launch =
        CudaError::to_status(cudaGetLastError(), "DslSmokeCaesarKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(
        cudaDeviceSynchronize(), "DslSmokeCaesarKernel::launch_device sync");
}

Status DslSmokeCaesarKernel::apply_host(
    std::span<const std::uint8_t> host_in,
    std::span<std::uint8_t> host_out,
    std::uint8_t shift,
    CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("DslSmokeCaesarKernel::apply_host size mismatch");
    }
    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched = launch_device(
            device.value().data(),
            device.value().data(),
            host_in.size(),
            shift,
            direction);
        if (!launched.ok()) {
            return launched;
        }
        return device.value().copy_to_host(host_out);
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_in);
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
        shift,
        direction);
    if (!launched.ok()) {
        return launched;
    }
    return device_out.value().copy_to_host(host_out);
}
