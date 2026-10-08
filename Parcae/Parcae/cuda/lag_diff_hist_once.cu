#include "lag_diff_hist_once.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

/// Tail indices `i ∈ [0, T-L)` map to absolute `t = i + L`.
/// `use_sum != 0` → `(in[t] + in[t-L])`; else `(in[t] - in[t-L])`.
__global__ void lag_combine_hist_once_kernel(const std::uint8_t* __restrict__ in,
                                             std::uint32_t* __restrict__ counts,
                                             std::size_t tail_count, std::uint32_t lag,
                                             std::uint8_t use_sum) {
    __shared__ std::uint32_t stage[HistFast::local_shared_uints];
    HistFast::clear_local(stage);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t lag_sz = static_cast<std::size_t>(lag);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < tail_count; i += stride) {
        const std::size_t t = i + lag_sz;
        const std::uint8_t a = __ldg(in + t);
        const std::uint8_t b = __ldg(in + (t - lag_sz));
        const std::uint8_t y =
            use_sum != 0u ? HistFast::enc_caesar(a, b) : HistFast::dec_sub(a, b);
        HistFast::add_local(stage, y);
    }
    HistFast::flush_local(stage, counts);
}

int LagDiffHistOnce::tiles_for(std::size_t token_count, std::uint32_t lag) {
    if (static_cast<std::size_t>(lag) >= token_count) {
        return 1;
    }
    const std::size_t tail = token_count - static_cast<std::size_t>(lag);
    return HistFast::tiles_for_capped(tail, kProductionTileCap);
}

int LagDiffHistOnce::tiles_for_public(std::size_t token_count, std::uint32_t lag) {
    return tiles_for(token_count, lag);
}

Status LagDiffHistOnce::validate(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                 std::size_t token_count, std::uint32_t lag) {
    if (lag == 0u) {
        return Status::error("LagDiffHistOnce: lag must be >= 1");
    }
    if (token_count > kMaxTokens) {
        return Status::error("LagDiffHistOnce: token_count exceeds kMaxTokens");
    }
    if (device_counts == nullptr) {
        return Status::error("LagDiffHistOnce: null device_counts");
    }
    if (token_count != 0u && device_in == nullptr) {
        return Status::error("LagDiffHistOnce: null device_in");
    }
    return Status::success();
}

Status LagDiffHistOnce::launch_combine_async(const std::uint8_t* device_in,
                                             std::uint32_t* device_counts, std::size_t token_count,
                                             std::uint32_t lag, bool use_sum,
                                             cudaStream_t stream) {
    Status valid = validate(device_in, device_counts, token_count, lag);
    if (!valid.ok()) {
        return valid;
    }

    Status cleared = CudaError::to_status(
        cudaMemsetAsync(device_counts, 0, alphabet * sizeof(std::uint32_t), stream),
        "LagDiffHistOnce::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    if (token_count == 0u || static_cast<std::size_t>(lag) >= token_count) {
        return Status::success();
    }

    const std::size_t tail = token_count - static_cast<std::size_t>(lag);
    const dim3 grid(1u, static_cast<unsigned>(tiles_for(token_count, lag)));
    lag_combine_hist_once_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_counts, tail, lag, use_sum ? 1u : 0u);
    return CudaError::to_status(cudaGetLastError(),
                                use_sum ? "LagDiffHistOnce::launch_sum_async"
                                        : "LagDiffHistOnce::launch_async");
}

Status LagDiffHistOnce::launch_async(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                     std::size_t token_count, std::uint32_t lag,
                                     cudaStream_t stream) {
    return launch_combine_async(device_in, device_counts, token_count, lag, /*use_sum=*/false,
                                stream);
}

Status LagDiffHistOnce::launch_sum_async(const std::uint8_t* device_in,
                                         std::uint32_t* device_counts, std::size_t token_count,
                                         std::uint32_t lag, cudaStream_t stream) {
    return launch_combine_async(device_in, device_counts, token_count, lag, /*use_sum=*/true,
                                stream);
}

Status LagDiffHistOnce::launch(const std::uint8_t* device_in, std::uint32_t* device_counts,
                               std::size_t token_count, std::uint32_t lag) {
    Status launched = launch_async(device_in, device_counts, token_count, lag, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "LagDiffHistOnce::launch sync");
}

Status LagDiffHistOnce::launch_sum(const std::uint8_t* device_in, std::uint32_t* device_counts,
                                   std::size_t token_count, std::uint32_t lag) {
    Status launched = launch_sum_async(device_in, device_counts, token_count, lag, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "LagDiffHistOnce::launch_sum sync");
}

Status LagDiffHistOnce::count_host(std::span<const std::uint8_t> host_cipher, std::uint32_t lag,
                                   std::span<std::uint32_t> host_counts) {
    if (host_counts.size() != alphabet) {
        return Status::error("LagDiffHistOnce::count_host: host_counts size must be 29");
    }
    if (lag == 0u) {
        return Status::error("LagDiffHistOnce::count_host: lag must be >= 1");
    }
    if (host_cipher.size() > kMaxTokens) {
        return Status::error("LagDiffHistOnce::count_host: token_count exceeds kMaxTokens");
    }

    for (std::size_t i = 0; i < alphabet; ++i) {
        host_counts[i] = 0u;
    }
    if (host_cipher.empty() || static_cast<std::size_t>(lag) >= host_cipher.size()) {
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
        launch(device_in.value().data(), device_counts.value().data(), host_cipher.size(), lag);
    if (!launched.ok()) {
        return launched;
    }
    return device_counts.value().copy_to_host(host_counts);
}

Status LagDiffHistOnce::count_sum_host(std::span<const std::uint8_t> host_cipher,
                                       std::uint32_t lag, std::span<std::uint32_t> host_counts) {
    if (host_counts.size() != alphabet) {
        return Status::error("LagDiffHistOnce::count_sum_host: host_counts size must be 29");
    }
    if (lag == 0u) {
        return Status::error("LagDiffHistOnce::count_sum_host: lag must be >= 1");
    }
    if (host_cipher.size() > kMaxTokens) {
        return Status::error("LagDiffHistOnce::count_sum_host: token_count exceeds kMaxTokens");
    }

    for (std::size_t i = 0; i < alphabet; ++i) {
        host_counts[i] = 0u;
    }
    if (host_cipher.empty() || static_cast<std::size_t>(lag) >= host_cipher.size()) {
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
        launch_sum(device_in.value().data(), device_counts.value().data(), host_cipher.size(), lag);
    if (!launched.ok()) {
        return launched;
    }
    return device_counts.value().copy_to_host(host_counts);
}
