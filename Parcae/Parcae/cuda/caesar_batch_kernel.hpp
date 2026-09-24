#ifndef CAESAR_BATCH_KERNEL_HPP
#define CAESAR_BATCH_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "candidate_batch_buffers.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA batch twin: shared ciphertext + per-candidate caesar shifts (SoA ABI v0).
///
/// Candidate-major `out[c * T + t]`. Typical sweep: `C == 29`, shifts `0..28`
/// (matches `CaesarCandidateGenerator`). Interrupts unused (same as CPU caesar).
class CaesarBatchKernel {
public:
    /// Device pointers: `device_in[T]`, `device_shifts[C]`, `device_directions[C]`
    /// (`0` decrypt / `1` encrypt), `device_out[C * T]`.
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              const std::uint8_t* device_shifts,
                                              const std::uint8_t* device_directions,
                                              std::uint8_t* device_out, std::size_t candidate_count,
                                              std::size_t token_count);

    /// H2D → kernel → D2H for raw SoA spans.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> shared_in,
                                           std::span<const std::uint8_t> shifts,
                                           std::span<const std::uint8_t> directions,
                                           std::span<std::uint8_t> out);

    /// Run on a filled `CandidateBatchBuffers` (Shared + Caesar). Writes `out_index29`.
    [[nodiscard]] static Status apply_host(CandidateBatchBuffers& buffers);

private:
    CaesarBatchKernel() = delete;
};

#endif // CAESAR_BATCH_KERNEL_HPP
