#include "grid_permute_kernel.hpp"

#include "parcae/core/status.hpp"

Status GridPermuteKernel::launch_device(const std::uint8_t* /*device_in*/,
                                        std::uint8_t* /*device_out*/, std::size_t /*count*/,
                                        const std::uint32_t* /*device_order*/,
                                        CudaDir /*direction*/) {
    return Status::error("GridPermuteKernel: device twin not wired yet");
}

Status GridPermuteKernel::launch_device_async(const std::uint8_t* /*device_in*/,
                                              std::uint8_t* /*device_out*/, std::size_t /*count*/,
                                              const std::uint32_t* /*device_order*/,
                                              CudaDir /*direction*/) {
    return Status::error("GridPermuteKernel: device twin not wired yet");
}

Status GridPermuteKernel::apply_host(std::span<const std::uint8_t> /*host_in*/,
                                     std::span<std::uint8_t> /*host_out*/,
                                     std::span<const std::uint32_t> /*host_order*/,
                                     CudaDir /*direction*/) {
    return Status::error("GridPermuteKernel: device twin not wired yet");
}
