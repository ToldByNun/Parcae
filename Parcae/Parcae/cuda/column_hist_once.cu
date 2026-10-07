#include "column_hist_once.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

/// One block-column `blockIdx.x = j` walks tokens `t ≡ j (mod L)`.
__global__ void column_hist_once_kernel(const std::uint8_t* __restrict__ in,
                                        std::uint32_t* __restrict__ cols, std::size_t token_count,
                                        std::uint32_t period) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::uint32_t j = static_cast<std::uint32_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride_k = static_cast<std::size_t>(blockDim.x) * tiles;

    for (std::size_t k = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         ; k += stride_k) {
        const std::size_t t = static_cast<std::size_t>(j) + k * static_cast<std::size_t>(period);
        if (t >= token_count) {
            break;
        }
        HistFast::add_private(priv, __ldg(in + t));
    }
    HistFast::flush_private(priv, cols + static_cast<std::size_t>(j) * HistFast::alphabet);
}

int ColumnHistOnce::tiles_for(std::size_t token_count, std::uint32_t period) {
    if (period == 0u) {
        return 1;
    }
    const std::size_t per_col =
        (token_count + static_cast<std::size_t>(period) - 1u) / static_cast<std::size_t>(period);
    return HistFast::tiles_for_capped(per_col, kProductionTileCap);
}

int ColumnHistOnce::tiles_for_public(std::size_t token_count, std::uint32_t period) {
    return tiles_for(token_count, period);
}

Status ColumnHistOnce::validate(const std::uint8_t* device_in, std::uint32_t* device_cols,
                                std::size_t token_count, std::uint32_t period) {
    if (period == 0u || period > kMaxPeriod) {
        return Status::error("ColumnHistOnce: period out of range 1..kMaxPeriod");
    }
    if (token_count > kMaxTokens) {
        return Status::error("ColumnHistOnce: token_count exceeds kMaxTokens");
    }
    if (device_cols == nullptr) {
        return Status::error("ColumnHistOnce: null device_cols");
    }
    if (token_count != 0u && device_in == nullptr) {
        return Status::error("ColumnHistOnce: null device_in");
    }
    return Status::success();
}

Status ColumnHistOnce::launch_async(const std::uint8_t* device_in, std::uint32_t* device_cols,
                                    std::size_t token_count, std::uint32_t period,
                                    cudaStream_t stream) {
    Status valid = validate(device_in, device_cols, token_count, period);
    if (!valid.ok()) {
        return valid;
    }

    const std::size_t col_bytes =
        static_cast<std::size_t>(period) * alphabet * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_cols, 0, col_bytes, stream),
                             "ColumnHistOnce::clear cols");
    if (!cleared.ok()) {
        return cleared;
    }
    if (token_count == 0u) {
        return Status::success();
    }

    const dim3 grid(period, static_cast<unsigned>(tiles_for(token_count, period)));
    column_hist_once_kernel<<<grid, HistFast::threads, 0, stream>>>(device_in, device_cols,
                                                                   token_count, period);
    return CudaError::to_status(cudaGetLastError(), "ColumnHistOnce::launch_async");
}

Status ColumnHistOnce::launch(const std::uint8_t* device_in, std::uint32_t* device_cols,
                              std::size_t token_count, std::uint32_t period) {
    Status launched = launch_async(device_in, device_cols, token_count, period, nullptr);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "ColumnHistOnce::launch sync");
}

Status ColumnHistOnce::count_host(std::span<const std::uint8_t> host_cipher, std::uint32_t period,
                                  std::span<std::uint32_t> host_cols) {
    if (period == 0u || period > kMaxPeriod) {
        return Status::error("ColumnHistOnce::count_host: period out of range");
    }
    if (host_cols.size() != static_cast<std::size_t>(period) * alphabet) {
        return Status::error("ColumnHistOnce::count_host: host_cols size must be L*29");
    }
    if (host_cipher.size() > kMaxTokens) {
        return Status::error("ColumnHistOnce::count_host: token_count exceeds kMaxTokens");
    }

    for (std::size_t i = 0; i < host_cols.size(); ++i) {
        host_cols[i] = 0u;
    }
    if (host_cipher.empty()) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(host_cipher);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint32_t>> device_cols =
        DeviceBuffer<std::uint32_t>::allocate(static_cast<std::size_t>(period) * alphabet);
    if (!device_cols.ok()) {
        return device_cols.status();
    }

    Status launched =
        launch(device_in.value().data(), device_cols.value().data(), host_cipher.size(), period);
    if (!launched.ok()) {
        return launched;
    }
    return device_cols.value().copy_to_host(host_cols);
}
