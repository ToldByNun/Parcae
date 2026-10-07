#include "cipher_hist_once.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__global__ void cipher_hist_once_local_kernel(const std::uint8_t* __restrict__ in,
                                              std::uint32_t* __restrict__ counts,
                                              std::size_t token_count) {
    __shared__ std::uint32_t stage[HistFast::local_shared_uints];
    HistFast::clear_local(stage);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        HistFast::add_local(stage, v.x);
        HistFast::add_local(stage, v.y);
        HistFast::add_local(stage, v.z);
        HistFast::add_local(stage, v.w);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_local(stage, __ldg(in + t));
    }
    HistFast::flush_local(stage, counts);
}

__global__ void cipher_hist_once_local_scalar_kernel(const std::uint8_t* __restrict__ in,
                                                     std::uint32_t* __restrict__ counts,
                                                     std::size_t token_count) {
    __shared__ std::uint32_t stage[HistFast::local_shared_uints];
    HistFast::clear_local(stage);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;

    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_local(stage, __ldg(in + t));
    }
    HistFast::flush_local(stage, counts);
}

int CipherHistOnce::tiles_for(std::size_t token_count) {
    static_assert(kProductionTileCap == HistFast::production_tile_cap,
                  "CipherHistOnce tile cap must match HistFast");
    return HistFast::tiles_for_capped(token_count, kProductionTileCap);
}

int CipherHistOnce::tiles_for_public(std::size_t token_count) {
    return tiles_for(token_count);
}

Status CipherHistOnce::validate(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                std::size_t token_count) {
    if (token_count > kMaxTokens) {
        return Status::error("CipherHistOnce: token_count exceeds kMaxTokens");
    }
    if (device_counts == nullptr) {
        return Status::error("CipherHistOnce: null device_counts");
    }
    if (token_count != 0u && device_in == nullptr) {
        return Status::error("CipherHistOnce: null device_in");
    }
    return Status::success();
}

Status CipherHistOnce::launch_async(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                    std::size_t token_count, cudaStream_t stream) {
    Status valid = validate(device_in, device_counts, token_count);
    if (!valid.ok()) {
        return valid;
    }

    Status cleared = CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, alphabet * sizeof(std::uint32_t), stream),
        "CipherHistOnce::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    if (token_count == 0u) {
        return Status::success();
    }

    const dim3 grid(1u, static_cast<unsigned>(tiles_for(token_count)));
    const bool aligned =
        (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u;
    if (aligned) {
        cipher_hist_once_local_kernel<<<grid, HistFast::threads, 0, stream>>>(device_in,
                                                                             device_counts,
                                                                             token_count);
    } else {
        cipher_hist_once_local_scalar_kernel<<<grid, HistFast::threads, 0, stream>>>(
            device_in, device_counts, token_count);
    }
    return CudaError::to_status(cudaGetLastError(), "CipherHistOnce::launch_async");
}

Status CipherHistOnce::launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                              std::size_t token_count) {
    Status launched = launch_async(device_in, device_counts, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "CipherHistOnce::launch sync");
}

Status CipherHistOnce::count_host(std::span<const std::uint8_t> host_cipher,
                                  std::span<std::uint32_t> host_counts) {
    if (host_counts.size() != alphabet) {
        return Status::error("CipherHistOnce::count_host: host_counts size must be 29");
    }
    if (host_cipher.size() > kMaxTokens) {
        return Status::error("CipherHistOnce::count_host: token_count exceeds kMaxTokens");
    }

    for (std::size_t i = 0; i < alphabet; ++i) {
        host_counts[i] = 0u;
    }
    if (host_cipher.empty()) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_cipher);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(alphabet);
    if (!device_counts.ok()) {
        return device_counts.status();
    }

    Status launched =
        launch(device_in.value().data(), device_counts.value().data(), host_cipher.size());
    if (!launched.ok()) {
        return launched;
    }
    return device_counts.value().copy_to_host(host_counts);
}
