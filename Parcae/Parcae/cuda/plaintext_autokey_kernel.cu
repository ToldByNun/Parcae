#include "plaintext_autokey_kernel.hpp"

#include "parcae/core/status.hpp"

Status PlaintextAutokeyKernel::launch_device(const std::uint8_t* /*device_in*/,
                                             std::uint8_t* /*device_out*/, std::size_t /*count*/,
                                             const std::uint8_t* /*device_key*/,
                                             std::uint32_t /*key_len*/,
                                             const InterruptDeviceView& /*interrupts*/,
                                             const std::uint32_t* /*device_bitmask_or_skips*/,
                                             CudaDir /*direction*/) {
    return Status::error("PlaintextAutokeyKernel: device twin not wired yet");
}

Status PlaintextAutokeyKernel::launch_device_async(
    const std::uint8_t* /*device_in*/, std::uint8_t* /*device_out*/, std::size_t /*count*/,
    const std::uint8_t* /*device_key*/, std::uint32_t /*key_len*/,
    const InterruptDeviceView& /*interrupts*/, const std::uint32_t* /*device_bitmask_or_skips*/,
    CudaDir /*direction*/) {
    return Status::error("PlaintextAutokeyKernel: device twin not wired yet");
}

Status PlaintextAutokeyKernel::apply_host(std::span<const std::uint8_t> /*host_in*/,
                                          std::span<std::uint8_t> /*host_out*/,
                                          std::span<const std::uint8_t> /*host_key*/,
                                          const InterruptDeviceView& /*interrupts*/,
                                          CudaDir /*direction*/) {
    return Status::error("PlaintextAutokeyKernel: device twin not wired yet");
}
