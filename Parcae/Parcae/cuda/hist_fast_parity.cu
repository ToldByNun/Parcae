#include "hist_fast_parity.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime.h>

#include <span>
#include <vector>

__global__ void hist_identity_warp_kernel(const std::uint8_t* in, std::uint32_t* counts,
                                          std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t base = (tile * static_cast<std::size_t>(HistFast::threads) +
                              static_cast<std::size_t>(threadIdx.x)) *
                             4u;
    if (base + 3u < token_count) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(in + base);
        HistFast::add_private(priv, v.x);
        HistFast::add_private(priv, v.y);
        HistFast::add_private(priv, v.z);
        HistFast::add_private(priv, v.w);
    } else {
        for (std::size_t t = base; t < token_count && t < base + 4u; ++t) {
            HistFast::add_private(priv, in[t]);
        }
    }
    HistFast::flush_private(priv, counts);
}

__global__ void hist_identity_local_kernel(const std::uint8_t* in, std::uint32_t* counts,
                                           std::size_t token_count) {
    __shared__ std::uint32_t stage[HistFast::local_shared_uints];
    HistFast::clear_local(stage);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t base = (tile * static_cast<std::size_t>(HistFast::threads) +
                              static_cast<std::size_t>(threadIdx.x)) *
                             4u;
    if (base + 3u < token_count) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(in + base);
        HistFast::add_local(stage, v.x);
        HistFast::add_local(stage, v.y);
        HistFast::add_local(stage, v.z);
        HistFast::add_local(stage, v.w);
    } else {
        for (std::size_t t = base; t < token_count && t < base + 4u; ++t) {
            HistFast::add_local(stage, in[t]);
        }
    }
    HistFast::flush_local(stage, counts);
}

__global__ void hist_caesar_warp_kernel(const std::uint8_t* in, std::uint8_t shift,
                                        std::uint32_t* counts, std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t base = (tile * static_cast<std::size_t>(HistFast::threads) +
                              static_cast<std::size_t>(threadIdx.x)) *
                             4u;
    if (base + 3u < token_count) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(in + base);
        HistFast::add_private(priv, HistFast::dec_caesar(v.x, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.y, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.z, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.w, shift));
    } else {
        for (std::size_t t = base; t < token_count && t < base + 4u; ++t) {
            HistFast::add_private(priv, HistFast::dec_caesar(in[t], shift));
        }
    }
    HistFast::flush_private(priv, counts);
}

__global__ void hist_caesar_local_kernel(const std::uint8_t* in, std::uint8_t shift,
                                         std::uint32_t* counts, std::size_t token_count) {
    __shared__ std::uint32_t stage[HistFast::local_shared_uints];
    HistFast::clear_local(stage);

    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t base = (tile * static_cast<std::size_t>(HistFast::threads) +
                              static_cast<std::size_t>(threadIdx.x)) *
                             4u;
    if (base + 3u < token_count) {
        const uchar4 v = *reinterpret_cast<const uchar4*>(in + base);
        HistFast::add_local(stage, HistFast::dec_caesar(v.x, shift));
        HistFast::add_local(stage, HistFast::dec_caesar(v.y, shift));
        HistFast::add_local(stage, HistFast::dec_caesar(v.z, shift));
        HistFast::add_local(stage, HistFast::dec_caesar(v.w, shift));
    } else {
        for (std::size_t t = base; t < token_count && t < base + 4u; ++t) {
            HistFast::add_local(stage, HistFast::dec_caesar(in[t], shift));
        }
    }
    HistFast::flush_local(stage, counts);
}

class HistFastParityLaunch {
public:
    [[nodiscard]] static Status run(const std::uint8_t* host_cipher, std::size_t token_count,
                                    bool caesar, std::uint8_t shift,
                                    std::vector<std::uint32_t>& warp_counts,
                                    std::vector<std::uint32_t>& local_counts) {
        if (host_cipher == nullptr && token_count != 0u) {
            return Status::error("HistFastParity: null cipher");
        }
        if (token_count > HistFastParity::kMaxTokens) {
            return Status::error("HistFastParity: token_count exceeds kMaxTokens");
        }

        warp_counts.assign(HistFastParity::alphabet, 0u);
        local_counts.assign(HistFastParity::alphabet, 0u);
        if (token_count == 0u) {
            return Status::success();
        }

        StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(
            std::span<const std::uint8_t>(host_cipher, token_count));
        if (!device_in.ok()) {
            return device_in.status();
        }

        StatusOr<DeviceBuffer<std::uint32_t>> device_warp =
            DeviceBuffer<std::uint32_t>::allocate(HistFastParity::alphabet);
        if (!device_warp.ok()) {
            return device_warp.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_local =
            DeviceBuffer<std::uint32_t>::allocate(HistFastParity::alphabet);
        if (!device_local.ok()) {
            return device_local.status();
        }

        const Status zero_w = CudaError::to_status(
            cudaMemset(device_warp.value().data(), 0,
                       HistFastParity::alphabet * sizeof(std::uint32_t)),
            "HistFastParity zero warp counts");
        if (!zero_w.ok()) {
            return zero_w;
        }
        const Status zero_l = CudaError::to_status(
            cudaMemset(device_local.value().data(), 0,
                       HistFastParity::alphabet * sizeof(std::uint32_t)),
            "HistFastParity zero local counts");
        if (!zero_l.ok()) {
            return zero_l;
        }

        const int tiles = HistFast::tiles_for(token_count);
        const dim3 grid(1, static_cast<unsigned>(tiles));
        if (caesar) {
            hist_caesar_warp_kernel<<<grid, HistFast::threads>>>(
                device_in.value().data(), shift, device_warp.value().data(), token_count);
            hist_caesar_local_kernel<<<grid, HistFast::threads>>>(
                device_in.value().data(), shift, device_local.value().data(), token_count);
        } else {
            hist_identity_warp_kernel<<<grid, HistFast::threads>>>(
                device_in.value().data(), device_warp.value().data(), token_count);
            hist_identity_local_kernel<<<grid, HistFast::threads>>>(
                device_in.value().data(), device_local.value().data(), token_count);
        }
        const Status launch_st =
            CudaError::to_status(cudaGetLastError(), "HistFastParity launch");
        if (!launch_st.ok()) {
            return launch_st;
        }
        const Status sync_st =
            CudaError::to_status(cudaDeviceSynchronize(), "HistFastParity sync");
        if (!sync_st.ok()) {
            return sync_st;
        }

        const Status st_w = device_warp.value().copy_to_host(std::span<std::uint32_t>(warp_counts));
        if (!st_w.ok()) {
            return st_w;
        }
        return device_local.value().copy_to_host(std::span<std::uint32_t>(local_counts));
    }

private:
    HistFastParityLaunch() = delete;
};

Status HistFastParity::compare_identity_hist(const std::uint8_t* host_cipher,
                                             std::size_t token_count,
                                             std::vector<std::uint32_t>& warp_counts,
                                             std::vector<std::uint32_t>& local_counts) {
    return HistFastParityLaunch::run(host_cipher, token_count, false, 0u, warp_counts,
                                     local_counts);
}

Status HistFastParity::compare_caesar_hist(const std::uint8_t* host_cipher, std::size_t token_count,
                                           std::uint8_t shift,
                                           std::vector<std::uint32_t>& warp_counts,
                                           std::vector<std::uint32_t>& local_counts) {
    if (shift >= static_cast<std::uint8_t>(alphabet)) {
        return Status::error("HistFastParity: shift out of range");
    }
    return HistFastParityLaunch::run(host_cipher, token_count, true, shift, warp_counts,
                                     local_counts);
}
