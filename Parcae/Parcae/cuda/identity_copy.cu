#include "identity_copy.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void identity_copy_kernel(
    const std::uint8_t* in,
    std::uint8_t* out,
    std::size_t count) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i < count) {
        out[i] = in[i];
    }
}

}  // namespace

Status IdentityCopy::launch_device(
    const std::uint8_t* device_in,
    std::uint8_t* device_out,
    std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("IdentityCopy::launch_device null device pointer");
    }

    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    identity_copy_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, count);

    Status launch = CudaError::to_status(cudaGetLastError(), "IdentityCopy::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "IdentityCopy::launch_device sync");
}

Status IdentityCopy::apply_host(
    std::span<const std::uint8_t> host_in,
    std::span<std::uint8_t> host_out) {
    if (host_in.size() != host_out.size()) {
        return Status::error("IdentityCopy::apply_host size mismatch");
    }

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched =
            launch_device(device.value().data(), device.value().data(), host_in.size());
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

    Status launched =
        launch_device(device_in.value().data(), device_out.value().data(), host_in.size());
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
