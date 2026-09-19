#include "caesar_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

__global__ void caesar_chi2_histogram_decrypt_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         i < n4;
         i += stride) {
        const uchar4 v = in4[i];
        HistFast::add_private(priv, HistFast::dec_caesar(v.x, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.y, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.z, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.w, shift));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        HistFast::add_private(priv, HistFast::dec_caesar(in[t], shift));
    }
    HistFast::flush_private(
        priv, counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void caesar_chi2_histogram_kernel(
    const std::uint8_t* in,
    const std::uint8_t* shifts,
    const std::uint8_t* directions,
    std::uint32_t* counts,
    std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = shifts[candidate];
    const std::uint8_t encrypt = directions[candidate];
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;

    for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count;
         t += stride) {
        const std::uint8_t x = in[t];
        const std::uint8_t y =
            encrypt != 0u ? HistFast::enc_caesar(x, shift) : HistFast::dec_caesar(x, shift);
        HistFast::add_private(priv, y);
    }
    HistFast::flush_private(
        priv, counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

int CaesarChi2Batch::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status CaesarChi2Batch::validate(
    std::size_t candidate_count,
    std::size_t token_count,
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    bool decrypt_only) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("CaesarChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("CaesarChi2Batch: bad T");
    }
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("CaesarChi2Batch: null device pointer");
    }
    if (!decrypt_only && device_directions == nullptr) {
        return Status::error("CaesarChi2Batch: null directions");
    }
    return Status::success();
}

Status CaesarChi2Batch::launch_impl(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count,
    bool synchronize,
    bool decrypt_only) {
    Status valid = validate(
        candidate_count,
        token_count,
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores,
        decrypt_only);
    if (!valid.ok()) {
        return valid;
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared = CudaError::to_status(
        synchronize ? cudaMemset(device_counts, 0, hist_bytes)
                    : cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
        "CaesarChi2Batch::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(
        static_cast<unsigned>(candidate_count),
        static_cast<unsigned>(tiles_for(token_count)));
    if (decrypt_only) {
        caesar_chi2_histogram_decrypt_kernel<<<grid, HistFast::threads>>>(
            device_in, device_shifts, device_counts, token_count);
    } else {
        caesar_chi2_histogram_kernel<<<grid, HistFast::threads>>>(
            device_in, device_shifts, device_directions, device_counts, token_count);
    }
    Status hist = CudaError::to_status(cudaGetLastError(), "CaesarChi2Batch::histogram");
    if (!hist.ok()) {
        return hist;
    }

    Status fin = Chi2BatchScore::finalize_async(
        device_counts, device_probabilities, device_scores, candidate_count, token_count);
    if (!fin.ok()) {
        return fin;
    }
    if (synchronize) {
        return CudaError::to_status(cudaDeviceSynchronize(), "CaesarChi2Batch::sync");
    }
    return Status::success();
}

Status CaesarChi2Batch::launch(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return launch_impl(
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores,
        candidate_count,
        token_count,
        true,
        false);
}

Status CaesarChi2Batch::launch_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const std::uint8_t* device_directions,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return launch_impl(
        device_in,
        device_shifts,
        device_directions,
        device_probabilities,
        device_counts,
        device_scores,
        candidate_count,
        token_count,
        false,
        false);
}

Status CaesarChi2Batch::launch_decrypt_async(
    const std::uint8_t* device_in,
    const std::uint8_t* device_shifts,
    const double* device_probabilities,
    std::uint32_t* device_counts,
    double* device_scores,
    std::size_t candidate_count,
    std::size_t token_count) {
    return launch_impl(
        device_in,
        device_shifts,
        nullptr,
        device_probabilities,
        device_counts,
        device_scores,
        candidate_count,
        token_count,
        false,
        true);
}
