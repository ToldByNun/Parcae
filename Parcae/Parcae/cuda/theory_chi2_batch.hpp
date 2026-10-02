#ifndef THEORY_CHI2_BATCH_HPP
#define THEORY_CHI2_BATCH_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// Fused bytecode decrypt + χ² histogram for `family=theory` search export.
///
/// One HotLoop program (`ops`/`imm`) is shared across candidates; per-candidate
/// param slots are a dense `C * slot_count` row-major matrix. Scores-only D2H
/// path matches `FamilyChi2Batch` (hist on device → `Chi2BatchScore::finalize`).
/// Domain errors set `device_lane_err[c]=1` and patch `device_scores[c]` to +inf
/// after finalize so Top-k never retains broken lanes.
///
/// Kernel stages `ops`/`imm` (+ candidate slots) into block shared memory when
/// `op_count <= kSharedProgramOps`, then evaluates via
/// `Z29BytecodeDevice::eval_at_trusted` (structural checks done in `launch_async`).
///
/// Launch: ~16 tokens/thread. No-index vs index path split. Pass-2 measured
/// uchar4 packs / 32-tok / launch_bounds — all slower on sm_120 (compute-bound).
class TheoryChi2Batch {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::uint16_t kMaxDeviceStack = 64;
    static constexpr std::uint16_t kMaxSlots = 64;
    static constexpr std::uint32_t kMaxProgramOps = 4096;
    /// Ops fitting in block shared mem together with HistFast private bins + slots.
    /// Larger programs fall back to global loads (still trusted eval).
    static constexpr std::uint32_t kSharedProgramOps = 512;

    /// Launch hist + χ² finalize + inf-patch (async on default stream).
    /// `device_lane_err` must hold `candidate_count` bytes (cleared here).
    [[nodiscard]] static Status
    launch_async(const std::uint8_t* device_in, const std::uint8_t* device_ops,
                 const std::uint8_t* device_imm, std::uint32_t op_count,
                 const std::uint8_t* device_slots, std::uint16_t slot_count,
                 std::uint16_t cipher_slot, std::uint16_t index_slot, std::uint8_t binds_index_i,
                 std::uint16_t max_stack, const double* device_probabilities,
                 std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
                 std::size_t candidate_count, std::size_t token_count);

private:
    TheoryChi2Batch() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status clear_and_grid(std::uint32_t* device_counts,
                                               std::uint8_t* device_lane_err,
                                               std::size_t candidate_count, std::size_t token_count,
                                               dim3* grid_out);
};

#endif // THEORY_CHI2_BATCH_HPP
