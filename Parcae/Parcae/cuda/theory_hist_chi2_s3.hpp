#ifndef THEORY_HIST_CHI2_S3_HPP
#define THEORY_HIST_CHI2_S3_HPP

#include "parcae/core/status.hpp"

#include "theory_chi2_batch.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S3 scalar fused χ² hist for bounded general `i`-exprs (`TheoryHistExprLower`).
///
/// In-lib twin: validates S3 caps then launches the trusted bytecode hist path
/// (`TheoryChi2Batch`) — always index-bound programs within ExprLower ceilings.
/// Beyond caps the caller must soft-fall to S0. No C++ namespaces.
class TheoryHistChi2S3 {
public:
    static constexpr std::size_t alphabet_size = TheoryChi2Batch::alphabet_size;
    static constexpr std::size_t kMaxCandidates = TheoryChi2Batch::kMaxCandidates;
    static constexpr std::size_t kMaxTokens = TheoryChi2Batch::kMaxTokens;
    /// Keep in sync with `TheoryHistExprLower` / `TheoryHistChi2Emit::kS3*`.
    static constexpr std::uint32_t kMaxProgramOps = 128;
    static constexpr std::uint16_t kMaxDeviceStack = 16;
    static constexpr std::uint16_t kMaxSlots = 16;

    /// Launch hist + χ² finalize + inf-patch (async). Requires `binds_index_i` and
    /// program within S3 caps; otherwise returns error (export soft-falls S0).
    [[nodiscard]] static Status
    launch_async(const std::uint8_t* device_in, const std::uint8_t* device_ops,
                 const std::uint8_t* device_imm, std::uint32_t op_count,
                 const std::uint8_t* device_slots, std::uint16_t slot_count,
                 std::uint16_t cipher_slot, std::uint16_t index_slot,
                 const double* device_probabilities, std::uint32_t* device_counts,
                 double* device_scores, std::uint8_t* device_lane_err,
                 std::size_t candidate_count, std::size_t token_count, std::uint16_t max_stack,
                 cudaStream_t stream = nullptr) {
        if (device_in == nullptr || device_ops == nullptr || device_imm == nullptr ||
            device_slots == nullptr || device_probabilities == nullptr ||
            device_counts == nullptr || device_scores == nullptr || device_lane_err == nullptr) {
            return Status::error("TheoryHistChi2S3::launch_async null");
        }
        if (op_count == 0 || op_count > kMaxProgramOps) {
            return Status::error("TheoryHistChi2S3: op_count outside S3 caps");
        }
        if (max_stack == 0 || max_stack > kMaxDeviceStack) {
            return Status::error("TheoryHistChi2S3: max_stack outside S3 caps");
        }
        if (slot_count == 0 || slot_count > kMaxSlots) {
            return Status::error("TheoryHistChi2S3: slot_count outside S3 caps");
        }
        if (cipher_slot >= slot_count || index_slot >= slot_count) {
            return Status::error("TheoryHistChi2S3: bad cipher/index slot");
        }
        return TheoryChi2Batch::launch_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, /*binds_index_i=*/1u, max_stack, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count, stream);
    }

private:
    TheoryHistChi2S3() = delete;
};

#endif // THEORY_HIST_CHI2_S3_HPP
