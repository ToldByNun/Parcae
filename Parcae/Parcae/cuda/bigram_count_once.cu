#include "bigram_count_once.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

/// Block-shared 841-bin stage + fat-tile grid-stride over adjacent pairs.
__global__ void bigram_count_once_kernel(const std::uint8_t* __restrict__ in,
                                         std::uint32_t* __restrict__ counts,
                                         std::size_t pair_count) {
    __shared__ std::uint32_t stage[BigramCountOnce::bins];
    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(BigramCountOnce::bins);
         i += static_cast<int>(blockDim.x)) {
        stage[i] = 0u;
    }
    __syncthreads();

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;

    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < pair_count; t += stride) {
        const std::uint8_t x0 = __ldg(in + t);
        const std::uint8_t x1 = __ldg(in + t + 1u);
        atomicAdd(&stage[static_cast<int>(x0) * HistFast::alphabet + static_cast<int>(x1)], 1u);
    }
    __syncthreads();

    for (int i = static_cast<int>(threadIdx.x); i < static_cast<int>(BigramCountOnce::bins);
         i += static_cast<int>(blockDim.x)) {
        const std::uint32_t v = stage[i];
        if (v != 0u) {
            atomicAdd(&counts[i], v);
        }
    }
}

int BigramCountOnce::tiles_for(std::size_t token_count) {
    const std::size_t pairs = token_count > 0u ? token_count - 1u : 0u;
    return HistFast::tiles_for_capped(pairs, kProductionTileCap);
}

int BigramCountOnce::tiles_for_public(std::size_t token_count) {
    return tiles_for(token_count);
}

Status BigramCountOnce::validate(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                 std::size_t token_count) {
    if (token_count > kMaxTokens) {
        return Status::error("BigramCountOnce: token_count exceeds kMaxTokens");
    }
    if (device_counts == nullptr) {
        return Status::error("BigramCountOnce: null device_counts");
    }
    if (token_count != 0u && device_in == nullptr) {
        return Status::error("BigramCountOnce: null device_in");
    }
    return Status::success();
}

Status BigramCountOnce::launch_async(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                     std::size_t token_count, cudaStream_t stream) {
    Status valid = validate(device_in, device_counts, token_count);
    if (!valid.ok()) {
        return valid;
    }

    Status cleared = CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, bins * sizeof(std::uint32_t), stream),
        "BigramCountOnce::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    if (token_count < 2u) {
        return Status::success();
    }

    const std::size_t pairs = token_count - 1u;
    const dim3 grid(1u, static_cast<unsigned>(tiles_for(token_count)));
    bigram_count_once_kernel<<<grid, HistFast::threads, 0, stream>>>(device_in, device_counts,
                                                                    pairs);
    return CudaError::to_status(cudaGetLastError(), "BigramCountOnce::launch_async");
}

Status BigramCountOnce::launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                               std::size_t token_count) {
    Status launched = launch_async(device_in, device_counts, token_count, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "BigramCountOnce::launch sync");
}

Status BigramCountOnce::count_host(std::span<const std::uint8_t> host_cipher,
                                   std::span<std::uint32_t> host_counts) {
    if (host_counts.size() != bins) {
        return Status::error("BigramCountOnce::count_host: host_counts size must be 841");
    }
    if (host_cipher.size() > kMaxTokens) {
        return Status::error("BigramCountOnce::count_host: token_count exceeds kMaxTokens");
    }

    for (std::size_t i = 0; i < bins; ++i) {
        host_counts[i] = 0u;
    }
    if (host_cipher.size() < 2u) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_cipher);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
        DeviceBuffer<std::uint32_t>::allocate(bins);
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
