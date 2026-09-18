#ifndef ATBASH_BATCH_KERNEL_HPP
#define ATBASH_BATCH_KERNEL_HPP

#include "candidate_batch_buffers.hpp"
#include "params.hpp"

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA batch twin: shared ciphertext → atbash into each candidate lane (SoA ABI v0).
///
/// Candidate-major `out[c * T + t] = 28 - in[t]`. Typical `C == 1` matches
/// `AtbashCandidateGenerator`. Direction unused (involution). Interrupts unused.
class AtbashBatchKernel {
public:
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t candidate_count,
        std::size_t token_count);

    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> shared_in,
        std::span<std::uint8_t> out,
        std::size_t candidate_count);

    /// Run on filled `CandidateBatchBuffers` (Shared + Atbash). Writes `out_index29`.
    [[nodiscard]] static Status apply_host(CandidateBatchBuffers& buffers);

private:
    AtbashBatchKernel() = delete;
};

#endif // ATBASH_BATCH_KERNEL_HPP
