#include "theory_chi2_batch.hpp"

#include "chi2_batch_score.hpp"
#include "cuda_error.hpp"
#include "hist_fast.hpp"
#include "parcae/search/nvtx_range.hpp"
#include "z29_bytecode_device.hpp"

#include <cmath>
#include <cuda_runtime_api.h>

__global__ void theory_chi2_hist_kernel(const std::uint8_t* in, const std::uint8_t* ops,
                                        const std::uint8_t* imm, std::uint32_t op_count,
                                        const std::uint8_t* slots, std::uint16_t slot_count,
                                        std::uint16_t cipher_slot, std::uint16_t index_slot,
                                        std::uint8_t binds_index_i, std::uint16_t max_stack,
                                        std::uint32_t* counts, std::uint8_t* lane_err,
                                        std::size_t token_count) {
    __shared__ std::uint32_t priv[HistFast::warps * HistFast::priv_stride];
    HistFast::clear_private(priv);

    const std::size_t candidate = static_cast<std::size_t>(blockIdx.x);
    const std::size_t tile = static_cast<std::size_t>(blockIdx.y);
    const std::size_t tiles = static_cast<std::size_t>(gridDim.y);
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * tiles;

    std::uint8_t local_slots[TheoryChi2Batch::kMaxSlots];
    const std::uint8_t* row = slots + candidate * static_cast<std::size_t>(slot_count);
    for (std::uint16_t s = 0; s < slot_count; ++s) {
        local_slots[s] = row[s];
    }

    std::uint8_t stack[Z29BytecodeDevice::kMaxDeviceStack];

    for (std::size_t t =
             tile * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
         t < token_count; t += stride) {
        std::uint8_t out_byte = 0;
        std::uint8_t err = 0;
        const bool ok = Z29BytecodeDevice::eval_at(
            ops, imm, op_count, local_slots, slot_count, cipher_slot, index_slot, binds_index_i, in,
            token_count, t, stack, max_stack, &out_byte, &err);
        if (!ok) {
            lane_err[candidate] = 1u;
        } else {
            HistFast::add_private(priv, out_byte);
        }
    }

    HistFast::flush_private(priv,
                            counts + candidate * static_cast<std::size_t>(HistFast::alphabet));
}

__global__ void theory_chi2_patch_inf_kernel(const std::uint8_t* lane_err, double* scores,
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

int TheoryChi2Batch::tiles_for(std::size_t token_count) {
    // Scalar token loop: tile by thread coverage (not uchar4 packs).
    const int by_work =
        static_cast<int>((token_count + static_cast<std::size_t>(HistFast::threads) - 1u) /
                         static_cast<std::size_t>(HistFast::threads));
    if (by_work < 1) {
        return 1;
    }
    return by_work < HistFast::max_tiles ? by_work : HistFast::max_tiles;
}

Status TheoryChi2Batch::clear_and_grid(std::uint32_t* device_counts, std::uint8_t* device_lane_err,
                                       std::size_t candidate_count, std::size_t token_count,
                                       dim3* grid_out) {
    if (candidate_count == 0 || candidate_count > kMaxCandidates) {
        return Status::error("TheoryChi2Batch: bad C");
    }
    if (token_count == 0 || token_count > kMaxTokens) {
        return Status::error("TheoryChi2Batch: bad T");
    }
    const std::size_t hist_bytes = candidate_count * alphabet_size * sizeof(std::uint32_t);
    Status cleared = CudaError::to_status(cudaMemsetAsync(device_counts, 0, hist_bytes, 0),
                                          "TheoryChi2Batch::clear counts");
    if (!cleared.ok()) {
        return cleared;
    }
    Status err_cleared =
        CudaError::to_status(cudaMemsetAsync(device_lane_err, 0, candidate_count, 0),
                             "TheoryChi2Batch::clear lane_err");
    if (!err_cleared.ok()) {
        return err_cleared;
    }
    *grid_out =
        dim3(static_cast<unsigned>(candidate_count), static_cast<unsigned>(tiles_for(token_count)));
    return Status::success();
}

Status TheoryChi2Batch::launch_async(
    const std::uint8_t* device_in, const std::uint8_t* device_ops, const std::uint8_t* device_imm,
    std::uint32_t op_count, const std::uint8_t* device_slots, std::uint16_t slot_count,
    std::uint16_t cipher_slot, std::uint16_t index_slot, std::uint8_t binds_index_i,
    std::uint16_t max_stack, const double* device_probabilities, std::uint32_t* device_counts,
    double* device_scores, std::uint8_t* device_lane_err, std::size_t candidate_count,
    std::size_t token_count) {
    if (device_in == nullptr || device_ops == nullptr || device_imm == nullptr ||
        device_slots == nullptr || device_probabilities == nullptr || device_counts == nullptr ||
        device_scores == nullptr || device_lane_err == nullptr) {
        return Status::error("TheoryChi2Batch::launch_async null");
    }
    if (op_count == 0u || op_count > kMaxProgramOps) {
        return Status::error("TheoryChi2Batch: bad op_count");
    }
    if (slot_count == 0u || slot_count > kMaxSlots) {
        return Status::error("TheoryChi2Batch: bad slot_count");
    }
    if (cipher_slot >= slot_count) {
        return Status::error("TheoryChi2Batch: bad cipher_slot");
    }
    if (binds_index_i != 0u && index_slot >= slot_count) {
        return Status::error("TheoryChi2Batch: bad index_slot");
    }
    if (max_stack == 0u || max_stack > kMaxDeviceStack) {
        return Status::error("TheoryChi2Batch: bad max_stack");
    }

    dim3 grid;
    Status prep =
        clear_and_grid(device_counts, device_lane_err, candidate_count, token_count, &grid);
    if (!prep.ok()) {
        return prep;
    }

    {
        NvtxRange nvtx_hist("hist_kernel");
        theory_chi2_hist_kernel<<<grid, HistFast::threads>>>(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, binds_index_i, max_stack, device_counts, device_lane_err, token_count);
        Status hist = CudaError::to_status(cudaGetLastError(), "TheoryChi2Batch::hist");
        if (!hist.ok()) {
            return hist;
        }
    }

    {
        NvtxRange nvtx_finalize("finalize");
        Status finalized = Chi2BatchScore::finalize_async(
            device_counts, device_probabilities, device_scores, candidate_count, token_count);
        if (!finalized.ok()) {
            return finalized;
        }

        const int threads = 128;
        const int blocks =
            static_cast<int>((candidate_count + static_cast<std::size_t>(threads) - 1u) /
                             static_cast<std::size_t>(threads));
        theory_chi2_patch_inf_kernel<<<blocks, threads>>>(device_lane_err, device_scores,
                                                          candidate_count);
        return CudaError::to_status(cudaGetLastError(), "TheoryChi2Batch::patch_inf");
    }
}
