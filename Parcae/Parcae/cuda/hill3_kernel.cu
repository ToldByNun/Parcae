#include "hill3_kernel.hpp"

#include "parcae/core/status.hpp"

Status Hill3Kernel::launch_device(const std::uint8_t* /*device_in*/, std::uint8_t* /*device_out*/,
                                  std::size_t /*count*/, const std::uint8_t* /*device_matrix*/,
                                  CudaDir /*direction*/) {
    return Status::error("Hill3Kernel: device twin not wired yet");
}

Status Hill3Kernel::launch_device_async(const std::uint8_t* /*device_in*/,
                                        std::uint8_t* /*device_out*/, std::size_t /*count*/,
                                        const std::uint8_t* /*device_matrix*/,
                                        CudaDir /*direction*/) {
    return Status::error("Hill3Kernel: device twin not wired yet");
}

Status Hill3Kernel::apply_host(std::span<const std::uint8_t> /*host_in*/,
                               std::span<std::uint8_t> /*host_out*/,
                               std::span<const std::uint8_t> /*host_matrix*/,
                               CudaDir /*direction*/) {
    return Status::error("Hill3Kernel: device twin not wired yet");
}
