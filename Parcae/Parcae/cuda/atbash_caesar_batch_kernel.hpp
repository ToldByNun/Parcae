#ifndef ATBASH_CAESAR_BATCH_KERNEL_HPP
#define ATBASH_CAESAR_BATCH_KERNEL_HPP

#include "candidate_batch_buffers.hpp"
#include "params.hpp"

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA batch twin of `gen_atbash_caesar` / Koan-1 compose (Atbash then Caesar +shift).
///
/// Shared ciphertext + per-candidate shifts. Fused device math matches
/// `ComposeTransform::apply_atbash_then_caesar`:
/// - decrypt: `(28 - in + shift) % 29`
/// - encrypt: `28 - ((in + 29 - shift) % 29)`
///
/// Candidate-major `out[c * T + t]`. Typical sweep: `C == 29`, shifts `0..28`.
class AtbashCaesarBatchKernel {
public:
    /// `device_directions[C]`: `0` decrypt / `1` encrypt (outer compose recipe).
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        const std::uint8_t* device_shifts,
        const std::uint8_t* device_directions,
        std::uint8_t* device_out,
        std::size_t candidate_count,
        std::size_t token_count);

    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> shared_in,
        std::span<const std::uint8_t> shifts,
        std::span<const std::uint8_t> directions,
        std::span<std::uint8_t> out);

    /// Run on filled `CandidateBatchBuffers` (Shared + Compose + caesar shifts).
    [[nodiscard]] static Status apply_host(CandidateBatchBuffers& buffers);

private:
    AtbashCaesarBatchKernel() = delete;
};

#endif // ATBASH_CAESAR_BATCH_KERNEL_HPP
