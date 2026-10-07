#include "alphabet_chi2_batch.hpp"
#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "family_chi2_batch.hpp"
#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

int FamilyChi2Batch::tiles_for_atbash(std::size_t token_count) {
    return HistTileCap::tiles_for(HistTileCap::kAtbash, token_count);
}

int FamilyChi2Batch::tiles_for_totient(std::size_t token_count) {
    return HistTileCap::tiles_for(HistTileCap::kTotient, token_count);
}

Status FamilyChi2Batch::clear_and_grid(std::uint32_t* device_counts, std::size_t candidate_count,
                                       std::size_t token_count, int tile_slot, dim3* grid_out) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }
    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared = CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
                                          "FamilyChi2Batch::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    const int grid_y = tile_slot >= 0 ? HistTileCap::tiles_for(tile_slot, token_count)
                                      : HistFast::tiles_for(token_count);
    *grid_out = dim3(static_cast<unsigned>(candidate_count), static_cast<unsigned>(grid_y));
    return Status::success();
}

__global__ void atbash_chi2_hist_kernel(const std::uint8_t* __restrict__ in,
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

__global__ void atbash_caesar_chi2_hist_kernel(const std::uint8_t* __restrict__ in,
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
        HistFast::add_private(priv, HistFast::enc_caesar(HistFast::dec_atbash(v.x), shift));
        HistFast::add_private(priv, HistFast::enc_caesar(HistFast::dec_atbash(v.y), shift));
        HistFast::add_private(priv, HistFast::enc_caesar(HistFast::dec_atbash(v.z), shift));
        HistFast::add_private(priv, HistFast::enc_caesar(HistFast::dec_atbash(v.w), shift));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(
            priv, HistFast::enc_caesar(HistFast::dec_atbash(__ldg(in + t)), shift));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void affine_chi2_hist_kernel(const std::uint8_t* __restrict__ in,
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

__global__ void vigenere_chi2_hist_kernel(const std::uint8_t* in, const std::uint8_t* key_bytes,
                                          const std::uint32_t* key_begin,
                                          const std::uint32_t* key_len, std::uint32_t* counts,
                                          std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t key_cache[64];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint32_t begin = key_begin[candidate];
    const std::uint32_t len = key_len[candidate];
    const std::uint32_t cached = len < 64u ? len : 64u;
    const bool pow2 = (len != 0u) && ((len & (len - 1u)) == 0u);
    const std::uint32_t mask = len - 1u;
    for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < cached;
         i += static_cast<std::uint32_t>(blockDim.x)) {
        key_cache[i] = key_bytes[begin + i];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    if (pow2 && cached == len) {
        const std::size_t n4 = token_count / 4u;
        const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
        for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             i < n4; i += stride) {
            const uchar4 v = in4[i];
            const std::uint32_t t32 = static_cast<std::uint32_t>(i * 4u);
            HistFast::add_private(priv, HistFast::dec_sub(v.x, key_cache[t32 & mask]));
            HistFast::add_private(priv, HistFast::dec_sub(v.y, key_cache[(t32 + 1u) & mask]));
            HistFast::add_private(priv, HistFast::dec_sub(v.z, key_cache[(t32 + 2u) & mask]));
            HistFast::add_private(priv, HistFast::dec_sub(v.w, key_cache[(t32 + 3u) & mask]));
        }
        for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             t < token_count; t += stride) {
            HistFast::add_private(
                priv, HistFast::dec_sub(in[t], key_cache[static_cast<std::uint32_t>(t) & mask]));
        }
    } else {
        for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             t < token_count; t += stride) {
            const std::uint32_t ki = pow2 ? (static_cast<std::uint32_t>(t) & mask)
                                          : (static_cast<std::uint32_t>(t) % len);
            const std::uint8_t key_symbol = ki < cached ? key_cache[ki] : key_bytes[begin + ki];
            HistFast::add_private(priv, HistFast::dec_sub(in[t], key_symbol));
        }
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void beaufort_chi2_hist_kernel(const std::uint8_t* in, const std::uint8_t* key_bytes,
                                          const std::uint32_t* key_begin,
                                          const std::uint32_t* key_len, std::uint32_t* counts,
                                          std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t key_cache[64];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint32_t begin = key_begin[candidate];
    const std::uint32_t len = key_len[candidate];
    const std::uint32_t cached = len < 64u ? len : 64u;
    const bool pow2 = (len != 0u) && ((len & (len - 1u)) == 0u);
    const std::uint32_t mask = len - 1u;
    for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < cached;
         i += static_cast<std::uint32_t>(blockDim.x)) {
        key_cache[i] = key_bytes[begin + i];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    if (pow2 && cached == len) {
        const std::size_t n4 = token_count / 4u;
        const uchar4* in4 = reinterpret_cast<const uchar4*>(in);
        for (std::size_t i = tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             i < n4; i += stride) {
            const uchar4 v = in4[i];
            const std::uint32_t t32 = static_cast<std::uint32_t>(i * 4u);
            // Beaufort: key - cipher.
            HistFast::add_private(priv, HistFast::dec_sub(key_cache[t32 & mask], v.x));
            HistFast::add_private(priv, HistFast::dec_sub(key_cache[(t32 + 1u) & mask], v.y));
            HistFast::add_private(priv, HistFast::dec_sub(key_cache[(t32 + 2u) & mask], v.z));
            HistFast::add_private(priv, HistFast::dec_sub(key_cache[(t32 + 3u) & mask], v.w));
        }
        for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             t < token_count; t += stride) {
            HistFast::add_private(
                priv, HistFast::dec_sub(key_cache[static_cast<std::uint32_t>(t) & mask], in[t]));
        }
    } else {
        for (std::size_t t = tile * static_cast<std::size_t>(blockDim.x) +
                             static_cast<std::size_t>(threadIdx.x);
             t < token_count; t += stride) {
            const std::uint32_t ki = pow2 ? (static_cast<std::uint32_t>(t) & mask)
                                          : (static_cast<std::uint32_t>(t) % len);
            const std::uint8_t key_symbol = ki < cached ? key_cache[ki] : key_bytes[begin + ki];
            HistFast::add_private(priv, HistFast::dec_sub(key_symbol, in[t]));
        }
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void totient_chi2_hist_kernel(const std::uint8_t* __restrict__ in,
                                         const std::uint8_t* __restrict__ shifts,
                                         const std::uint32_t* __restrict__ shift_begin,
                                         std::uint32_t* __restrict__ counts,
                                         std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t* __restrict__ lane = shifts + __ldg(shift_begin + candidate);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);
    const uchar4* __restrict__ sh4 = reinterpret_cast<const uchar4*>(lane);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        const uchar4 s = __ldg(sh4 + i);
        HistFast::add_private(priv, HistFast::dec_sub(v.x, s.x));
        HistFast::add_private(priv, HistFast::dec_sub(v.y, s.y));
        HistFast::add_private(priv, HistFast::dec_sub(v.z, s.z));
        HistFast::add_private(priv, HistFast::dec_sub(v.w, s.w));
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, HistFast::dec_sub(__ldg(in + t), __ldg(lane + t)));
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

Status FamilyChi2Batch::ensure_cipher_hist_scratch(std::uint32_t** out_hist) {
    // Process-lifetime scratch (29 bins). Not safe for overlapping concurrent
    // Atbash / Atbash∘Caesar remaps on this device.
    static std::uint32_t* device_cipher_hist = nullptr;
    if (device_cipher_hist == nullptr) {
        Status allocated = CudaError::to_status(
            cudaMalloc(reinterpret_cast<void**>(&device_cipher_hist),
                       alphabet_size * sizeof(std::uint32_t)),
            "FamilyChi2Batch::cipher hist scratch");
        if (!allocated.ok()) {
            device_cipher_hist = nullptr;
            return allocated;
        }
    }
    *out_hist = device_cipher_hist;
    return Status::success();
}

Status FamilyChi2Batch::launch_atbash_decode_hist_async(
    const std::uint8_t* device_in, const double* device_probabilities,
    std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash decode null");
    }
    dim3 grid;
    Status prep =
        clear_and_grid(device_counts, candidate_count, token_count, HistTileCap::kAtbash, &grid);
    if (!prep.ok()) {
        return prep;
    }
    atbash_chi2_hist_kernel<<<grid, HistFast::threads>>>(device_in, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status FamilyChi2Batch::launch_atbash_async(const std::uint8_t* device_in,
                                            const double* device_probabilities,
                                            std::uint32_t* device_counts, double* device_scores,
                                            std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_atbash_async(device_in, device_probabilities,
                                                  device_cipher_hist, device_counts, device_scores,
                                                  candidate_count, token_count, nullptr);
}

Status FamilyChi2Batch::launch_atbash_caesar_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash_caesar decode null");
    }
    dim3 grid;
    Status prep =
        clear_and_grid(device_counts, candidate_count, token_count, HistTileCap::kAtbash, &grid);
    if (!prep.ok()) {
        return prep;
    }
    atbash_caesar_chi2_hist_kernel<<<grid, HistFast::threads>>>(device_in, device_shifts,
                                                                device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::atbash_caesar hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status FamilyChi2Batch::launch_atbash_caesar_async(
    const std::uint8_t* device_in, const std::uint8_t* device_shifts,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_shifts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::atbash_caesar null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_atbash_caesar_async(
        device_in, device_shifts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
}

Status FamilyChi2Batch::launch_affine_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::affine decode null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, /*tile_slot=*/-1, &grid);
    if (!prep.ok()) {
        return prep;
    }
    affine_chi2_hist_kernel<<<grid, HistFast::threads>>>(device_in, device_a, device_b,
                                                         device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::affine hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status FamilyChi2Batch::launch_affine_async(const std::uint8_t* device_in,
                                            const std::uint8_t* device_a,
                                            const std::uint8_t* device_b,
                                            const double* device_probabilities,
                                            std::uint32_t* device_counts, double* device_scores,
                                            std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::affine null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_affine_decrypt_async(
        device_in, device_a, device_b, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, nullptr);
}

Status FamilyChi2Batch::launch_vigenere_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::vigenere decode null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, /*tile_slot=*/-1, &grid);
    if (!prep.ok()) {
        return prep;
    }
    vigenere_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::vigenere hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status FamilyChi2Batch::launch_vigenere_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::vigenere null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("FamilyChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("FamilyChi2Batch: bad T");
    }
    return AlphabetChi2Batch::launch_vigenere_decrypt_async(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_probabilities,
        /*device_column_scratch=*/nullptr, device_counts, device_scores, candidate_count,
        token_count, nullptr);
}

Status FamilyChi2Batch::launch_beaufort_async(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::beaufort null");
    }
    dim3 grid;
    Status prep = clear_and_grid(device_counts, candidate_count, token_count, /*tile_slot=*/-1, &grid);
    if (!prep.ok()) {
        return prep;
    }
    beaufort_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::beaufort hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}

Status FamilyChi2Batch::launch_totient_async(const std::uint8_t* device_in,
                                             const std::uint8_t* device_shifts,
                                             const std::uint32_t* device_shift_begin,
                                             const double* device_probabilities,
                                             std::uint32_t* device_counts, double* device_scores,
                                             std::size_t candidate_count, std::size_t token_count) {
    if (device_in == nullptr || device_shifts == nullptr || device_shift_begin == nullptr ||
        device_probabilities == nullptr || device_counts == nullptr || device_scores == nullptr) {
        return Status::error("FamilyChi2Batch::totient null");
    }
    dim3 grid;
    Status prep =
        clear_and_grid(device_counts, candidate_count, token_count, HistTileCap::kTotient, &grid);
    if (!prep.ok()) {
        return prep;
    }
    totient_chi2_hist_kernel<<<grid, HistFast::threads>>>(
        device_in, device_shifts, device_shift_begin, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "FamilyChi2Batch::totient hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count);
}
