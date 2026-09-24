#ifndef AFFINE_BATCH_KERNEL_HPP
#define AFFINE_BATCH_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "candidate_batch_buffers.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA batch twin: shared ciphertext + per-candidate affine `(a,b)` (SoA ABI v0).
///
/// Candidate-major `out[c * T + t]`. Typical sweep: `C == 812` nested `a` then `b`
/// (matches `AffineCandidateGenerator` / `fill_affine_ab_nested`).
/// Encrypt: `a·x+b`; decrypt: `inv(a)·(x-b)` via `Z29Device`. Interrupts unused.
class AffineBatchKernel {
public:
    /// Device: `device_in[T]`, `device_a[C]`, `device_b[C]`, `device_directions[C]`
    /// (`0` decrypt / `1` encrypt), `device_out[C * T]`.
    [[nodiscard]] static Status
    launch_device(const std::uint8_t* device_in, const std::uint8_t* device_a,
                  const std::uint8_t* device_b, const std::uint8_t* device_directions,
                  std::uint8_t* device_out, std::size_t candidate_count, std::size_t token_count);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> shared_in,
                                           std::span<const std::uint8_t> affine_a,
                                           std::span<const std::uint8_t> affine_b,
                                           std::span<const std::uint8_t> directions,
                                           std::span<std::uint8_t> out);

    /// Run on filled `CandidateBatchBuffers` (Shared + Affine). Writes `out_index29`.
    [[nodiscard]] static Status apply_host(CandidateBatchBuffers& buffers);

private:
    AffineBatchKernel() = delete;
};

#endif // AFFINE_BATCH_KERNEL_HPP
