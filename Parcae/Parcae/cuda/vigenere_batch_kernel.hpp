#ifndef VIGENERE_BATCH_KERNEL_HPP
#define VIGENERE_BATCH_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "candidate_batch_buffers.hpp"
#include "interrupt_device_view.hpp"
#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// CUDA batch twin of `gen_vigenere_explicit_keys` (caller-supplied key list).
///
/// Shared ciphertext + packed key arena (`key_begin` / `key_len` / `key_bytes`).
/// Optional shared interrupt encoding (same policy for every candidate, matching
/// the CPU generator). Candidate-major `out[c * T + t]`.
class VigenereBatchKernel {
public:
    [[nodiscard]] static Status
    launch_device(const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
                  const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
                  const std::uint8_t* device_directions, std::uint8_t* device_out,
                  std::size_t candidate_count, std::size_t token_count,
                  const InterruptDeviceView& interrupts,
                  const std::uint32_t* device_bitmask_or_skips);

    [[nodiscard]] static Status
    apply_host(std::span<const std::uint8_t> shared_in, std::span<const std::uint8_t> key_bytes,
               std::span<const std::uint32_t> key_begin, std::span<const std::uint32_t> key_len,
               std::span<const std::uint8_t> directions, std::span<std::uint8_t> out,
               const InterruptDeviceView& interrupts);

    /// Run on packed `CandidateBatchBuffers` (Shared + VigenereKey + arena).
    [[nodiscard]] static Status apply_host(CandidateBatchBuffers& buffers,
                                           const InterruptDeviceView& interrupts);

private:
    VigenereBatchKernel() = delete;
};

#endif // VIGENERE_BATCH_KERNEL_HPP
