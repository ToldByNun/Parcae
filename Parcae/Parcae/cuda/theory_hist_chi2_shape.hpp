#ifndef THEORY_HIST_CHI2_SHAPE_HPP
#define THEORY_HIST_CHI2_SHAPE_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

/// ShapeInline fused χ² for algebraic ShapeIds (`docs/architecture/dsl-smart-hist.md`).
///
/// **Production:** alphabet remap after one ciphertext hist (`AlphabetChi2Batch`):
/// - Atbash: `P[b] = H[28 - b]`
/// - Caesar: `P[b] = H[(b + shift) mod 29]`
/// - Affine decrypt: `P[inv(a)·(x−b)] = H[x]` (host must reject `a==0`)
///
/// Legacy decode→hist remains via `launch_*_decode_hist_async`. Finalize via
/// `Chi2BatchScore`. No C++ namespaces.
class TheoryHistChi2Shape {
public:
    static constexpr std::size_t alphabet_size = 29;
    static constexpr std::size_t kMaxCandidates = 16384;
    static constexpr std::size_t kMaxTokens = 1u << 22;

    /// Production Atbash: alphabet mirror remap.
    [[nodiscard]] static Status
    launch_atbash_async(const std::uint8_t* device_in, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count,
                        cudaStream_t stream = nullptr);

    /// Legacy Atbash decode→hist (`HistTileCap::kAtbash`).
    [[nodiscard]] static Status
    launch_atbash_decode_hist_async(const std::uint8_t* device_in,
                                    const double* device_probabilities,
                                    std::uint32_t* device_counts, double* device_scores,
                                    std::size_t candidate_count, std::size_t token_count,
                                    cudaStream_t stream = nullptr);

    /// Production Caesar: alphabet rotate remap. `device_shifts` length C.
    [[nodiscard]] static Status
    launch_caesar_async(const std::uint8_t* device_in, const std::uint8_t* device_shifts,
                        const double* device_probabilities, std::uint32_t* device_counts,
                        double* device_scores, std::size_t candidate_count,
                        std::size_t token_count, cudaStream_t stream = nullptr);

    /// Legacy Caesar decode→hist.
    [[nodiscard]] static Status
    launch_caesar_decode_hist_async(const std::uint8_t* device_in,
                                    const std::uint8_t* device_shifts,
                                    const double* device_probabilities,
                                    std::uint32_t* device_counts, double* device_scores,
                                    std::size_t candidate_count, std::size_t token_count,
                                    cudaStream_t stream = nullptr);

    /// Production Affine decrypt remap. Caller must ensure every `a` is invertible.
    [[nodiscard]] static Status
    launch_affine_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                        const std::uint8_t* device_b, const double* device_probabilities,
                        std::uint32_t* device_counts, double* device_scores,
                        std::size_t candidate_count, std::size_t token_count,
                        cudaStream_t stream = nullptr);

    /// Legacy Affine decrypt decode→hist.
    [[nodiscard]] static Status
    launch_affine_decode_hist_async(const std::uint8_t* device_in, const std::uint8_t* device_a,
                                    const std::uint8_t* device_b,
                                    const double* device_probabilities,
                                    std::uint32_t* device_counts, double* device_scores,
                                    std::size_t candidate_count, std::size_t token_count,
                                    cudaStream_t stream = nullptr);

private:
    TheoryHistChi2Shape() = delete;

    [[nodiscard]] static int tiles_for(std::size_t token_count);

    [[nodiscard]] static Status ensure_cipher_hist_scratch(std::uint32_t** out_hist);
};

#endif // THEORY_HIST_CHI2_SHAPE_HPP
