#ifndef HIST_FAST_PARITY_HPP
#define HIST_FAST_PARITY_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

/// Host helpers that run warp-private vs thread-local `HistFast` identity hists
/// on the same cipher and compare bin counts (golden parity for the local path).
class HistFastParity {
public:
    static constexpr std::size_t alphabet = 29;
    static constexpr std::size_t kMaxTokens = 1u << 20;

    /// Identity histogram: count raw cipher bytes (must already be in `0..28`).
    /// Launches one candidate row with warp-private hist and one with local hist.
    /// On success, `warp_counts` and `local_counts` each have `alphabet` entries
    /// and are bit-identical.
    [[nodiscard]] static Status compare_identity_hist(const std::uint8_t* host_cipher,
                                                      std::size_t token_count,
                                                      std::vector<std::uint32_t>& warp_counts,
                                                      std::vector<std::uint32_t>& local_counts);

    /// Caesar-decrypt hist for a single shift (same decrypt as catalog twin).
    /// Compares warp-private vs local accumulation for one candidate.
    [[nodiscard]] static Status compare_caesar_hist(const std::uint8_t* host_cipher,
                                                    std::size_t token_count, std::uint8_t shift,
                                                    std::vector<std::uint32_t>& warp_counts,
                                                    std::vector<std::uint32_t>& local_counts);

private:
    HistFastParity() = delete;
};

#endif // HIST_FAST_PARITY_HPP
