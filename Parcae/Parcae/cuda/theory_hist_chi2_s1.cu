#include "theory_hist_chi2_s1.hpp"

#include "alphabet_chi2_batch.hpp"
#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "parcae/search/nvtx_range.hpp"
#include "z29_bytecode_device.hpp"

#include <cuda_runtime_api.h>

#include <cmath>

/// File-scope — no anonymous namespace (theory hist emit contract).

/// Global identity alphabet for S1 bake. `eval_at_trusted` loads cipher via
/// `__ldg(stream + i)` which requires **global** memory (not stack/shared).
__device__ std::uint8_t g_s1_bake_identity[TheoryHistChi2S1::alphabet_size] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

__global__ void theory_hist_chi2_s1_bake_kernel(const std::uint8_t* __restrict__ ops,
                                                const std::uint8_t* __restrict__ imm,
                                                std::uint32_t op_count,
                                                const std::uint8_t* __restrict__ slots,
                                                std::uint16_t slot_count, std::uint16_t cipher_slot,
                                                std::uint16_t index_slot,
                                                std::uint8_t binds_index_i, std::uint16_t max_stack,
                                                std::uint8_t* __restrict__ luts,
                                                std::uint8_t* __restrict__ lane_err,
                                                std::size_t candidate_count) {
    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::uint8_t sym = static_cast<std::uint8_t>(threadIdx.x);
    if (candidate >= candidate_count || sym >= TheoryHistChi2S1::alphabet_size) {
        return;
    }

    std::uint8_t local_slots[TheoryHistChi2S1::kMaxSlots];
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#endif
    for (std::uint16_t s = 0; s < slot_count; ++s) {
        local_slots[s] = slots[candidate * static_cast<std::size_t>(slot_count) + s];
    }

    std::uint8_t stack[TheoryHistChi2S1::kMaxDeviceStack];
    std::uint8_t out_byte = 0;
    std::uint8_t err_flag = 0;
    const bool ok = Z29BytecodeDevice::eval_at_trusted(
        ops, imm, op_count, local_slots, slot_count, cipher_slot, index_slot, binds_index_i,
        g_s1_bake_identity, /*stream_len=*/0, /*i=*/static_cast<std::size_t>(sym), stack,
        max_stack, &out_byte, &err_flag);
    const std::size_t lut_i =
        candidate * static_cast<std::size_t>(TheoryHistChi2S1::alphabet_size) + sym;
    if (!ok || err_flag != Z29BytecodeDevice::kErrOk) {
        lane_err[candidate] = 1u;
        luts[lut_i] = 0u;
        return;
    }
    luts[lut_i] = out_byte;
}

__global__ void theory_hist_chi2_s1_lut_kernel(const std::uint8_t* in, const std::uint8_t* luts,
                                               std::uint32_t* counts, std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    __shared__ std::uint8_t lut[HistFast::alphabet];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::uint8_t* row = luts + candidate * static_cast<std::size_t>(HistFast::alphabet);
    if (threadIdx.x < HistFast::alphabet) {
        lut[threadIdx.x] = row[threadIdx.x];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;
    const std::size_t n4 = token_count / 4u;
    const uchar4* in4 = reinterpret_cast<const uchar4*>(in);

    for (std::size_t i =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         i < n4; i += stride) {
        const uchar4 v = in4[i];
        HistFast::add_private(priv, lut[v.x]);
        HistFast::add_private(priv, lut[v.y]);
        HistFast::add_private(priv, lut[v.z]);
        HistFast::add_private(priv, lut[v.w]);
    }
    for (std::size_t t = n4 * 4u + tile * static_cast<std::size_t>(blockDim.x) +
                         static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        HistFast::add_private(priv, lut[in[t]]);
    }
    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void theory_hist_chi2_s1_patch_inf_kernel(const std::uint8_t* lane_err, double* scores,
                                                     std::size_t candidate_count) {
    const std::size_t c =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (c >= candidate_count) {
        return;
    }
    if (lane_err[c] != 0u) {
        scores[c] = INFINITY;
    }
}

int TheoryHistChi2S1::tiles_for(std::size_t token_count) {
    return HistFast::tiles_for(token_count);
}

Status TheoryHistChi2S1::ensure_cipher_hist_scratch(std::uint32_t** out_hist) {
    // Process-lifetime scratch (29 bins). Not safe for overlapping concurrent
    // S1 remaps on this device.
    static std::uint32_t* device_cipher_hist = nullptr;
    if (device_cipher_hist == nullptr) {
        Status allocated = CudaError::to_status(
            cudaMalloc(reinterpret_cast<void**>(&device_cipher_hist),
                       alphabet_size * sizeof(std::uint32_t)),
            "TheoryHistChi2S1::cipher hist scratch");
        if (!allocated.ok()) {
            device_cipher_hist = nullptr;
            return allocated;
        }
    }
    *out_hist = device_cipher_hist;
    return Status::success();
}

Status TheoryHistChi2S1::launch_bake_async(const std::uint8_t* device_ops,
                                           const std::uint8_t* device_imm, std::uint32_t op_count,
                                           const std::uint8_t* device_slots,
                                           std::uint16_t slot_count, std::uint16_t cipher_slot,
                                           std::uint16_t index_slot, std::uint8_t binds_index_i,
                                           std::uint16_t max_stack, std::uint8_t* device_luts,
                                           std::uint8_t* device_lane_err,
                                           std::size_t candidate_count, cudaStream_t stream) {
    if (device_ops == nullptr || device_imm == nullptr || device_slots == nullptr ||
        device_luts == nullptr || device_lane_err == nullptr) {
        return Status::error("TheoryHistChi2S1::launch_bake_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S1::bake bad C");
    }
    if (op_count == 0u || op_count > kMaxProgramOps) {
        return Status::error("TheoryHistChi2S1::bake bad op_count");
    }
    if (slot_count == 0u || slot_count > kMaxSlots) {
        return Status::error("TheoryHistChi2S1::bake bad slot_count");
    }
    if (cipher_slot >= slot_count) {
        return Status::error("TheoryHistChi2S1::bake bad cipher_slot");
    }
    if (binds_index_i != 0u && index_slot >= slot_count) {
        return Status::error("TheoryHistChi2S1::bake bad index_slot");
    }
    if (max_stack == 0u || max_stack > kMaxDeviceStack) {
        return Status::error("TheoryHistChi2S1::bake bad max_stack");
    }

    NvtxRange nvtx_bake("s1_bake_luts");
    // 32 threads / block: first 29 bake one alphabet symbol each.
    theory_hist_chi2_s1_bake_kernel<<<static_cast<unsigned>(candidate_count), 32, 0, stream>>>(
        device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot, index_slot,
        binds_index_i, max_stack, device_luts, device_lane_err, candidate_count);
    return CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S1::bake");
}

Status TheoryHistChi2S1::launch_lut_decode_hist_async(
    const std::uint8_t* device_in, const std::uint8_t* device_luts,
    const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
    std::size_t candidate_count, std::size_t token_count, cudaStream_t stream) {
    if (device_in == nullptr || device_luts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S1::launch_lut_decode_hist_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S1: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S1: bad T");
    }

    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared =
        CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, stream),
                             "TheoryHistChi2S1::decode clear counts");
    if (!cleared.ok()) {
        return cleared;
    }

    const dim3 grid(static_cast<unsigned>(candidate_count),
                    static_cast<unsigned>(tiles_for(token_count)));
    theory_hist_chi2_s1_lut_kernel<<<grid, HistFast::threads, 0, stream>>>(
        device_in, device_luts, device_counts, token_count);
    Status hist = CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S1::decode hist");
    if (!hist.ok()) {
        return hist;
    }
    return Chi2BatchScore::finalize_async(device_counts, device_probabilities, device_scores,
                                          candidate_count, token_count, stream);
}

Status TheoryHistChi2S1::launch_lut_async(const std::uint8_t* device_in,
                                          const std::uint8_t* device_luts,
                                          const double* device_probabilities,
                                          std::uint32_t* device_counts, double* device_scores,
                                          std::size_t candidate_count, std::size_t token_count,
                                          cudaStream_t stream) {
    if (device_in == nullptr || device_luts == nullptr || device_probabilities == nullptr ||
        device_counts == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S1::launch_lut_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S1: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryHistChi2S1: bad T");
    }

    std::uint32_t* device_cipher_hist = nullptr;
    Status scratch = ensure_cipher_hist_scratch(&device_cipher_hist);
    if (!scratch.ok()) {
        return scratch;
    }
    return AlphabetChi2Batch::launch_lut_decrypt_async(
        device_in, device_luts, device_probabilities, device_cipher_hist, device_counts,
        device_scores, candidate_count, token_count, stream);
}

Status TheoryHistChi2S1::patch_inf_async(const std::uint8_t* device_lane_err,
                                         double* device_scores, std::size_t candidate_count,
                                         cudaStream_t stream) {
    if (device_lane_err == nullptr || device_scores == nullptr) {
        return Status::error("TheoryHistChi2S1::patch_inf_async null");
    }
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryHistChi2S1::patch_inf bad C");
    }
    const int threads = 128;
    const int blocks =
        static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                         static_cast<std::size_t>(threads));
    theory_hist_chi2_s1_patch_inf_kernel<<<blocks, threads, 0, stream>>>(
        device_lane_err, device_scores, candidate_count);
    return CudaError::to_status(cudaGetLastError(), "TheoryHistChi2S1::patch_inf");
}

Status TheoryHistChi2S1::launch_from_slots_async(
    const std::uint8_t* device_in, const std::uint8_t* device_ops, const std::uint8_t* device_imm,
    std::uint32_t op_count, const std::uint8_t* device_slots, std::uint16_t slot_count,
    std::uint16_t cipher_slot, std::uint16_t index_slot, std::uint8_t binds_index_i,
    std::uint16_t max_stack, std::uint8_t* device_luts, const double* device_probabilities,
    std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
    std::size_t candidate_count, std::size_t token_count, cudaStream_t stream) {
    if (device_lane_err == nullptr) {
        return Status::error("TheoryHistChi2S1::launch_from_slots_async null lane_err");
    }
    Status err_cleared =
        CudaError::to_status(cudaMemsetAsync(device_lane_err, 0, candidate_count, stream),
                             "TheoryHistChi2S1::clear lane_err");
    if (!err_cleared.ok()) {
        return err_cleared;
    }
    Status baked =
        launch_bake_async(device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
                          index_slot, binds_index_i, max_stack, device_luts, device_lane_err,
                          candidate_count, stream);
    if (!baked.ok()) {
        return baked;
    }
    Status hist = launch_lut_async(device_in, device_luts, device_probabilities, device_counts,
                                   device_scores, candidate_count, token_count, stream);
    if (!hist.ok()) {
        return hist;
    }
    return patch_inf_async(device_lane_err, device_scores, candidate_count, stream);
}
