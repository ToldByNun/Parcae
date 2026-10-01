#ifndef THEORY_HIST_CHI2_LAUNCH_HPP
#define THEORY_HIST_CHI2_LAUNCH_HPP

#include "parcae/core/status.hpp"
#include "parcae/dsl/theory_hist_chi2_emit.hpp"

#include "theory_chi2_batch.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

/// Runtime launch façade for theory fused χ² hist.
///
/// Skeleton: always delegates to `TheoryChi2Batch` (S0 bytecode). When
/// `TheoryHistChi2Emit` grows specialized kernels, this class prefers them
/// via `has_specialized` / strategy — `GpuCandidateExport` will call here.
///
/// Caps and ABI match `TheoryChi2Batch`. No C++ namespaces.
class TheoryHistChi2Launch {
public:
    static constexpr std::size_t alphabet_size = TheoryChi2Batch::alphabet_size;
    static constexpr std::size_t kMaxCandidates = TheoryChi2Batch::kMaxCandidates;
    static constexpr std::size_t kMaxTokens = TheoryChi2Batch::kMaxTokens;
    static constexpr std::uint16_t kMaxDeviceStack = TheoryChi2Batch::kMaxDeviceStack;
    static constexpr std::uint16_t kMaxSlots = TheoryChi2Batch::kMaxSlots;
    static constexpr std::uint32_t kMaxProgramOps = TheoryChi2Batch::kMaxProgramOps;

    /// True when a specialized hist twin is loaded for `theory_id` (skeleton: always false).
    [[nodiscard]] static bool has_specialized(std::string_view /*theory_id*/) noexcept {
        return false;
    }

    /// Preferred strategy for an emit bundle (S0 when not specialized).
    [[nodiscard]] static TheoryHistChi2Emit::Strategy
    effective_strategy(const TheoryHistChi2Emit::EmitBundle& bundle) noexcept {
        return bundle.specialized() ? bundle.emitted_strategy()
                                    : TheoryHistChi2Emit::Strategy::S0Bytecode;
    }

    /// Fused hist + χ² finalize + inf-patch (async). Skeleton path = bytecode.
    [[nodiscard]] static Status
    launch_async(const std::uint8_t* device_in, const std::uint8_t* device_ops,
                 const std::uint8_t* device_imm, std::uint32_t op_count,
                 const std::uint8_t* device_slots, std::uint16_t slot_count,
                 std::uint16_t cipher_slot, std::uint16_t index_slot, std::uint8_t binds_index_i,
                 std::uint16_t max_stack, const double* device_probabilities,
                 std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
                 std::size_t candidate_count, std::size_t token_count) {
        return TheoryChi2Batch::launch_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, binds_index_i, max_stack, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count);
    }

    /// Same as `launch_async` — explicit S0 entry for call sites that already chose bytecode.
    [[nodiscard]] static Status launch_bytecode_async(
        const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
        std::size_t candidate_count, std::size_t token_count) {
        return launch_async(device_in, device_ops, device_imm, op_count, device_slots, slot_count,
                            cipher_slot, index_slot, binds_index_i, max_stack, device_probabilities,
                            device_counts, device_scores, device_lane_err, candidate_count,
                            token_count);
    }

private:
    TheoryHistChi2Launch() = delete;
};

#endif // THEORY_HIST_CHI2_LAUNCH_HPP
