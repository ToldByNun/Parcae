#ifndef THEORY_HIST_CHI2_S1_HPP
#define THEORY_HIST_CHI2_S1_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// S1 LUT-29 fused χ² hist for residual FxOnly decrypt (`f(x; params)`).
///
/// Hot export path: device-bake LUTs from bytecode + bound slots into scratch
/// (`launch_bake_async`), then hist (`launch_lut_async`). When the program and
/// bound slots are unchanged, export skips the bake and reuses the resident
/// LUT (`TheoryDeviceScratch::s1_lut_resident`). Fair `T.theory.s1_lut29`
/// bakes once outside `BenchTimer` so the gate is hist+finalize, same as Caesar.
/// Domain errors set `device_lane_err[c]=1`; `patch_inf_async` after finalize
/// patches those scores to +inf. No per-chunk host `eval_at × 29 × C`.
///
/// `device_luts` is row-major `C × 29`: `lut[c*29 + x] = decrypt(x; params_c)`.
/// Hist is uchar4 via shared LUT + fat-tile (`HistFast::production_tile_cap`).
/// Finalize via `Chi2BatchScore`. No C++ namespaces.
class TheoryHistChi2S1 {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;
    static constexpr std::uint16_t kMaxDeviceStack = 64;
    static constexpr std::uint16_t kMaxSlots = 64;
    static constexpr std::uint32_t kMaxProgramOps = 4096;

    /// Fill `device_luts` (C×29) by evaluating the shared HotLoop program at
    /// each alphabet symbol. Mutates a private slot copy per thread; writes
    /// `device_lane_err[c]=1` on domain/program faults (caller clears first).
    [[nodiscard]] static Status
    launch_bake_async(const std::uint8_t* device_ops, const std::uint8_t* device_imm,
                      std::uint32_t op_count, const std::uint8_t* device_slots,
                      std::uint16_t slot_count, std::uint16_t cipher_slot,
                      std::uint16_t index_slot, std::uint8_t binds_index_i,
                      std::uint16_t max_stack, std::uint8_t* device_luts,
                      std::uint8_t* device_lane_err, std::size_t candidate_count,
                      cudaStream_t stream = nullptr);

    [[nodiscard]] static Status
    launch_lut_async(const std::uint8_t* device_in, const std::uint8_t* device_luts,
                     const double* device_probabilities, std::uint32_t* device_counts,
                     double* device_scores, std::size_t candidate_count, std::size_t token_count,
                     cudaStream_t stream = nullptr);

    /// Patch `device_scores[c] = +inf` where `device_lane_err[c] != 0`.
    [[nodiscard]] static Status patch_inf_async(const std::uint8_t* device_lane_err,
                                                double* device_scores, std::size_t candidate_count,
                                                cudaStream_t stream = nullptr);

    /// Bake → hist → χ² finalize → inf-patch (async). Preferred export entry.
    [[nodiscard]] static Status launch_from_slots_async(
        const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, std::uint8_t* device_luts,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::uint8_t* device_lane_err, std::size_t candidate_count, std::size_t token_count,
        cudaStream_t stream = nullptr);

private:
    TheoryHistChi2S1() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);
};

#endif // THEORY_HIST_CHI2_S1_HPP
