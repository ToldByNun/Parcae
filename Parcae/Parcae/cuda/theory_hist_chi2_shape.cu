#include "theory_hist_chi2_shape.hpp"

#include "alphabet_chi2_batch.hpp"
#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

/// File-scope — no anonymous namespace (theory hist emit contract).
__global__ void theory_hist_chi2_shape_atbash_kernel(const std::uint8_t* __restrict__ in,
                                                     std::uint32_t* __restrict__ counts,
                                                     std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        HistFast::add_private(priv, HistFast::dec_atbash(v.x));
        HistFast::add_private(priv, HistFast::dec_atbash(v.y));
        HistFast::add_private(priv, HistFast::dec_atbash(v.z));
        HistFast::add_private(priv, HistFast::dec_atbash(v.w));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, HistFast::dec_atbash(__ldg(in + t)));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void theory_hist_chi2_shape_caesar_kernel(const std::uint8_t* __restrict__ in,
                                                     const std::uint8_t* __restrict__ shifts,
                                                     std::uint32_t* __restrict__ counts,
                                                     std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t shift = __ldg(shifts + candidate);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        HistFast::add_private(priv, HistFast::dec_caesar(v.x, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.y, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.z, shift));
        HistFast::add_private(priv, HistFast::dec_caesar(v.w, shift));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, HistFast::dec_caesar(__ldg(in + t), shift));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void theory_hist_chi2_shape_affine_kernel(const std::uint8_t* __restrict__ in,
                                                     const std::uint8_t* __restrict__ affine_a,
                                                     const std::uint8_t* __restrict__ affine_b,
                                                     std::uint32_t* __restrict__ counts,
                                                     std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t lut[HistFast::alphabet];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t inv_a = Z29Device::inv(__ldg(affine_a + candidate));
    const std::uint8_t b = __ldg(affine_b + candidate);
    if (threadIdx.x < HistFast::alphabet) {
        lut[threadIdx.x] =
            Z29Device::mul(inv_a, Z29Device::sub(static_cast<std::uint8_t>(threadIdx.x), b));
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        HistFast::add_private(priv, lut[v.x]);
        HistFast::add_private(priv, lut[v.y]);
        HistFast::add_private(priv, lut[v.z]);
        HistFast::add_private(priv, lut[v.w]);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, lut[__ldg(in + t)]);
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

int TheoryHistChi2Shape::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status TheoryHistChi2Shape::ensure_cipher_hist_scratch(std::uint32_t** out_hist) {
    // Process-lifetime scratch (29 bins). Not safe for overlapping concurrent
    // ShapeInline remaps on this device.
    static std::uint32_t* device_cipher_hist = nullptr;
    if (device_cipher_hist == nullptr) {
        Status allocated = CudaError::to_status(
            cudaMalloc(reinterpret_cast<void**>(&device_cipher_hist),
                       alphabet_size * sizeof(std::uint32_t)),
            "TheoryHistChi2Shape::cipher hist scratch");
        if (!allocated.ok()) {
            device_cipher_hist = nullptr;
            return allocated;
        }
    }
    *out_hist = device_cipher_hist;
    return Status::success();
}

Status TheoryHistChi2Shape::launch_atbash_decode_hist_async(
    const std::uint8_t* device_in, const double* device_probabilities,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_atbash_decode_hist_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2Shape::atbash clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(
                        HistTileCap::tiles_for(HistTileCap::kAtbash, token_count)));
    theory_hist_chi2_shape_atbash_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2Shape::atbash hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2Shape::launch_atbash_async(const std::uint8_t* device_in,
                                                const double* device_probabilities,
                                                std::uint32_t* device_counts,
                                                double* device_scores,
                                                std::size_t candidate_count,
                                                std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_atbash_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_atbash_async(device_in, device_probabilities,
                                                  device_cipher_hist, device_counts, device_scores,
                                                  candidate_count, token_count, stream);
}

Status TheoryHistChi2Shape::launch_caesar_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_caesar_decode_hist_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2Shape::caesar clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_shape_caesar_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_shifts, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2Shape::caesar hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2Shape::launch_caesar_async(const std::uint8_t* device_in,
                                                const std::uint8_t* device_shifts,
                                                const double* device_probabilities,
                                                std::uint32_t* device_counts,
                                                double* device_scores,
                                                std::size_t candidate_count,
                                                std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_caesar_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_caesar_decrypt_async(
        device_in, device_shifts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, stream);
}

Status TheoryHistChi2Shape::launch_affine_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_affine_decode_hist_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2Shape::affine clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_shape_affine_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_a, device_b, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2Shape::affine hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2Shape::launch_affine_async(const std::uint8_t* device_in,
                                                const std::uint8_t* device_a,
                                                const std::uint8_t* device_b,
                                                const double* device_probabilities,
                                                std::uint32_t* device_counts,
                                                double* device_scores,
                                                std::size_t candidate_count,
                                                std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2Shape::launch_affine_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2Shape: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2Shape: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_affine_decrypt_async(
        device_in, device_a, device_b, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, stream);
}
