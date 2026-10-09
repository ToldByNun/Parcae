#ifndef THEORY_HIST_CHI2_LAUNCH_HPP
#define THEORY_HIST_CHI2_LAUNCH_HPP

#include "parcae/core/status.hpp"
#include "parcae/dsl/theory_hist_chi2_emit.hpp"

#include "theory_chi2_batch.hpp"
#include "theory_hist_chi2_s1.hpp"
#include "theory_hist_chi2_s2.hpp"
#include "theory_hist_chi2_s3.hpp"
#include "theory_hist_chi2_s4.hpp"
#include "theory_hist_chi2_s5.hpp"
#include "theory_hist_chi2_shape.hpp"
#include "theory_hist_module.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string_view>

/// Runtime launch façade for theory fused χ² hist.
///
/// S0: `TheoryChi2Batch` bytecode. ShapeInline Atbash: `TheoryHistChi2Shape`.
/// S1: `TheoryHistChi2S1` LUT-29 twin. S2 linear: `TheoryHistChi2S2` uchar4 twin.
/// S3: `TheoryHistChi2S3` bounded ExprLower scalar twin.
/// S4: `TheoryHistChi2S4` AutokeyRing + hist (vigenere_lag class).
/// Module: `TheoryHistModule` cubin/NVRTC cache (URI+digest); fail → soft S0.
/// `GpuCandidateExport` prefers module then in-lib twins when present;
/// otherwise bytecode.
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

    /// True when a per-URI hist module is cached (`TheoryHistModule`).
    [[nodiscard]] static bool has_specialized(std::string_view theory_id) noexcept {
        return TheoryHistModule::has(theory_id);
    }

    /// Module path (URI cache). Soft error → caller falls back to S0.
    [[nodiscard]] static Status launch_module_async(
        std::string_view theory_uri, const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistModule::launch_async(
            theory_uri, device_in, device_ops, device_imm, op_count, device_slots, slot_count,
            cipher_slot, index_slot, binds_index_i, max_stack, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count, stream);
    }

    /// Preferred strategy for an emit bundle (S0 when not specialized).
    [[nodiscard]] static TheoryHistChi2Emit::Strategy
    effective_strategy(const TheoryHistChi2Emit::EmitBundle& bundle) noexcept {
        return bundle.specialized() ? bundle.emitted_strategy()
                                    : TheoryHistChi2Emit::Strategy::S0Bytecode;
    }

    /// Fused hist + χ² finalize + inf-patch (async). Default path = bytecode.
    [[nodiscard]] static Status
    launch_async(const std::uint8_t* device_in, const std::uint8_t* device_ops,
                 const std::uint8_t* device_imm, std::uint32_t op_count,
                 const std::uint8_t* device_slots, std::uint16_t slot_count,
                 std::uint16_t cipher_slot, std::uint16_t index_slot, std::uint8_t binds_index_i,
                 std::uint16_t max_stack, const double* device_probabilities,
                 std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
                 std::size_t candidate_count, std::size_t token_count,
                 cudaStream_t stream = nullptr) {
        return TheoryChi2Batch::launch_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, binds_index_i, max_stack, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count, stream);
    }

    /// Explicit S0 entry for call sites that already chose bytecode.
    [[nodiscard]] static Status launch_bytecode_async(
        const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr) {
        return launch_async(device_in, device_ops, device_imm, op_count, device_slots, slot_count,
                            cipher_slot, index_slot, binds_index_i, max_stack, device_probabilities,
                            device_counts, device_scores, device_lane_err, candidate_count,
                            token_count, stream);
    }

    /// ShapeInline Atbash — production alphabet mirror remap.
    [[nodiscard]] static Status
    launch_shape_atbash_async(const std::uint8_t* device_in, const double* device_probabilities,
                              std::uint32_t* device_counts, double* device_scores,
                              std::size_t candidate_count, std::size_t token_count,
                              cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_atbash_async(device_in, device_probabilities,
                                                        device_counts, device_scores,
                                                        candidate_count, token_count, stream);
    }

    /// ShapeInline Atbash legacy decode→hist.
    [[nodiscard]] static Status launch_shape_atbash_decode_hist_async(
        const std::uint8_t* device_in, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_atbash_decode_hist_async(
            device_in, device_probabilities, device_counts, device_scores, candidate_count,
            token_count, stream);
    }

    /// ShapeInline Caesar — production alphabet rotate remap. `device_shifts` length C.
    [[nodiscard]] static Status
    launch_shape_caesar_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                              const double* device_probabilities, std::uint32_t* device_counts,
                              double* device_scores, std::size_t candidate_count,
                              std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_caesar_async(
            device_in, device_shifts, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, stream);
    }

    /// ShapeInline Caesar legacy decode→hist.
    [[nodiscard]] static Status launch_shape_caesar_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_shifts,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_caesar_decode_hist_async(
            device_in, device_shifts, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, stream);
    }

    /// ShapeInline Affine decrypt — production remap. Host must reject a==0.
    [[nodiscard]] static Status
    launch_shape_affine_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                              const std::uint8_t* device_b, const double* device_probabilities,
                              std::uint32_t* device_counts, double* device_scores,
                              std::size_t candidate_count, std::size_t token_count,
                              cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_affine_async(
            device_in, device_a, device_b, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, stream);
    }

    /// ShapeInline Affine decrypt legacy decode→hist.
    [[nodiscard]] static Status launch_shape_affine_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_a, const std::uint8_t* device_b,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2Shape::launch_affine_decode_hist_async(
            device_in, device_a, device_b, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, stream);
    }

    /// S1 mono-LUT-29 twin — production CipherHistOnce + bin remap.
    /// `device_luts` row-major `C × 29`.
    [[nodiscard]] static Status
    launch_s1_lut_async(const std::uint8_t* device_in, const std::uint8_t* device_luts,
                        const double* device_probabilities, std::uint32_t* device_counts,
                        double* device_scores, std::size_t candidate_count,
                        std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2S1::launch_lut_async(device_in, device_luts, device_probabilities,
                                                  device_counts, device_scores, candidate_count,
                                                  token_count, stream);
    }

    /// S1 LUT-29 legacy decode→hist.
    [[nodiscard]] static Status launch_s1_lut_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_luts,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr) {
        return TheoryHistChi2S1::launch_lut_decode_hist_async(
            device_in, device_luts, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, stream);
    }

    /// Residual FxOnly: device-bake LUTs from slots into `device_luts`, then hist.
    /// Clears / uses `device_lane_err` for domain → +inf patch (no host eval_at×29×C).
    [[nodiscard]] static Status launch_s1_from_slots_async(
        const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, std::uint8_t* device_luts,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::uint8_t* device_lane_err, std::size_t candidate_count, std::size_t token_count,
        cudaStream_t stream = nullptr) {
        return TheoryHistChi2S1::launch_from_slots_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, binds_index_i, max_stack, device_luts, device_probabilities, device_counts,
            device_scores, device_lane_err, candidate_count, token_count, stream);
    }

    /// S2 linear — production ColumnHistOnce(L=29) + keystream remap.
    [[nodiscard]] static Status launch_s2_linear_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, bool cipher_minus_ks, cudaStream_t stream = nullptr) {
        return TheoryHistChi2S2::launch_linear_async(
            device_in, device_b0, device_b1, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, cipher_minus_ks, stream);
    }

    /// S2 linear legacy decode→hist.
    [[nodiscard]] static Status launch_s2_linear_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::size_t candidate_count,
        std::size_t token_count, bool cipher_minus_ks, cudaStream_t stream = nullptr) {
        return TheoryHistChi2S2::launch_linear_decode_hist_async(
            device_in, device_b0, device_b1, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, cipher_minus_ks, stream);
    }

    /// S3 bounded scalar twin (ExprLower within caps; index-bound bytecode hist).
    [[nodiscard]] static Status launch_s3_scalar_async(
        const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::uint8_t* device_lane_err, std::size_t candidate_count, std::size_t token_count,
        std::uint16_t max_stack, cudaStream_t stream = nullptr) {
        return TheoryHistChi2S3::launch_async(
            device_in, device_ops, device_imm, op_count, device_slots, slot_count, cipher_slot,
            index_slot, device_probabilities, device_counts, device_scores, device_lane_err,
            candidate_count, token_count, max_stack, stream);
    }

    /// S4 AutokeyRing twin (`device_lags` length C) — production remap.
    [[nodiscard]] static Status launch_s4_autokey_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr) {
        return TheoryHistChi2S4::launch_autokey_async(
            device_in, device_lags, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, cipher_minus_ks, stream);
    }

    /// S4 AutokeyRing legacy decode→hist.
    [[nodiscard]] static Status launch_s4_autokey_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_lags,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr) {
        return TheoryHistChi2S4::launch_autokey_decode_hist_async(
            device_in, device_lags, device_probabilities, device_counts, device_scores,
            candidate_count, token_count, cipher_minus_ks, stream);
    }

    /// S5 poly — production ColumnHistOnce(L=29) + keystream remap.
    [[nodiscard]] static Status launch_s5_poly_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const std::uint8_t* device_b2,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr) {
        return TheoryHistChi2S5::launch_poly_async(
            device_in, device_b0, device_b1, device_b2, device_probabilities, device_counts,
            device_scores, candidate_count, token_count, cipher_minus_ks, stream);
    }

    /// S5 poly legacy decode→hist.
    [[nodiscard]] static Status launch_s5_poly_decode_hist_async(
        const std::uint8_t* device_in, const std::uint8_t* device_b0,
        const std::uint8_t* device_b1, const std::uint8_t* device_b2,
        const double* device_probabilities, std::uint32_t* device_counts, double* device_scores,
        std::size_t candidate_count, std::size_t token_count, bool cipher_minus_ks,
        cudaStream_t stream = nullptr) {
        return TheoryHistChi2S5::launch_poly_decode_hist_async(
            device_in, device_b0, device_b1, device_b2, device_probabilities, device_counts,
            device_scores, candidate_count, token_count, cipher_minus_ks, stream);
    }

private:
    TheoryHistChi2Launch() = delete;
};

#endif // THEORY_HIST_CHI2_LAUNCH_HPP
