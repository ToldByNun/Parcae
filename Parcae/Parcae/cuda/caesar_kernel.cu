#include "caesar_kernel.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::uint8_t kModulus = 29;

__global__ void caesar_kernel(const std::uint8_t* in, std::uint8_t* out, std::size_t count,
                              std::uint8_t shift, std::uint8_t encrypt) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t x = in[i];
    if (encrypt != 0u) {
        out[i] = static_cast<std::uint8_t>((x + shift) % kModulus);
    } else {
        out[i] = static_cast<std::uint8_t>((x + kModulus - shift) % kModulus);
    }
}

[[nodiscard]] Status validate_shift(std::uint8_t shift) {
    if (shift >= kModulus) {
        return Status::error("CaesarKernel: shift must be in 0..28");
    }
    return Status::success();
}

} // namespace

Status CaesarKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                   std::size_t count, std::uint8_t shift, CudaDir direction) {
    Status shift_ok = validate_shift(shift);
    if (!shift_ok.ok()) {
        return shift_ok;
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("CaesarKernel::launch_device null device pointer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    caesar_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, count, shift, encrypt);

    Status launch = CudaError::to_status(cudaGetLastError(), "CaesarKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "CaesarKernel::launch_device sync");
}

Status CaesarKernel::apply_host(std::span<const std::uint8_t> host_in,
                                std::span<std::uint8_t> host_out, std::uint8_t shift,
                                CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("CaesarKernel::apply_host size mismatch");
    }

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched = launch_device(device.value().data(), device.value().data(),
                                        host_in.size(), shift, direction);
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
                                    host_in.size(), shift, direction);
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
