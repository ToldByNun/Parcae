#ifndef HIST_ALPHABET_MAP_HPP
#define HIST_ALPHABET_MAP_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

/// Host (and future device-shared) alphabet-histogram remaps for Z/29.
///
/// Monoalphabetic decrypt χ² and related scores need only the ciphertext
/// histogram `H` (or lag/bigram aggregates) plus a cheap bin remap — not a
/// full re-decode of the stream per candidate.
///
/// Conventions (decrypt Caesar shift `s`):
///   plaintext bin `b` comes from cipher bin `(b + s) mod 29`,
///   so `P[b] = H[(b + s) mod 29]`.
class HistAlphabetMap {
public:
    static constexpr std::size_t alphabet = Index29::modulus;
    static constexpr std::size_t bigram_bins = alphabet * alphabet;

    using Hist = std::array<std::uint32_t, alphabet>;
    using BigramHist = std::array<std::uint32_t, bigram_bins>;

    /// `P[b] = H[(b + shift) mod 29]` (Caesar decrypt).
    [[nodiscard]] static StatusOr<Hist> rotate_decrypt(const Hist& cipher_hist,
                                                       std::uint8_t shift) {
        Status ok = require_shift(shift);
        if (!ok.ok()) {
            return ok;
        }
        Hist plain{};
        for (std::size_t b = 0; b < alphabet; ++b) {
            const std::size_t src = (b + static_cast<std::size_t>(shift)) % alphabet;
            plain[b] = cipher_hist[src];
        }
        return plain;
    }

    /// `P[b] = H[(b - shift) mod 29]` (Caesar encrypt on bins).
    [[nodiscard]] static StatusOr<Hist> rotate_encrypt(const Hist& cipher_hist,
                                                       std::uint8_t shift) {
        Status ok = require_shift(shift);
        if (!ok.ok()) {
            return ok;
        }
        Hist plain{};
        for (std::size_t b = 0; b < alphabet; ++b) {
            const std::size_t src =
                (b + alphabet - static_cast<std::size_t>(shift)) % alphabet;
            plain[b] = cipher_hist[src];
        }
        return plain;
    }

    /// Atbash: `P[b] = H[28 - b]`.
    [[nodiscard]] static Hist mirror_atbash(const Hist& cipher_hist) noexcept {
        Hist plain{};
        for (std::size_t b = 0; b < alphabet; ++b) {
            plain[b] = cipher_hist[alphabet - 1u - b];
        }
        return plain;
    }

    /// Affine decrypt bins: `y = inv(a)·(x - b)`, then `P[y] += H[x]`.
    [[nodiscard]] static StatusOr<Hist> permute_affine_decrypt(const Hist& cipher_hist,
                                                               std::uint8_t a,
                                                               std::uint8_t b) {
        StatusOr<Index29> a_idx = Index29::try_make(a);
        if (!a_idx.ok()) {
            return a_idx.status();
        }
        StatusOr<Index29> b_idx = Index29::try_make(b);
        if (!b_idx.ok()) {
            return b_idx.status();
        }
        StatusOr<Index29> inv_a = Z29::try_inv(a_idx.value());
        if (!inv_a.ok()) {
            return inv_a.status();
        }

        Hist plain{};
        for (std::size_t x = 0; x < alphabet; ++x) {
            const Index29 y =
                Z29::mul(inv_a.value(),
                         Z29::sub(Index29{static_cast<std::uint8_t>(x)}, b_idx.value()));
            plain[y.value()] += cipher_hist[x];
        }
        return plain;
    }

    /// Catalog Atbash∘Caesar-encrypt on bins: mirror, then encrypt-rotate `shift`.
    [[nodiscard]] static StatusOr<Hist> compose_atbash_caesar_encrypt(const Hist& cipher_hist,
                                                                      std::uint8_t shift) {
        return rotate_encrypt(mirror_atbash(cipher_hist), shift);
    }

    /// Apply a cipher→plain bin map: `P[map[x]] += H[x]` (`map[x]` in 0..28).
    [[nodiscard]] static StatusOr<Hist> apply_bin_map(const Hist& cipher_hist,
                                                      std::span<const std::uint8_t> map) {
        if (map.size() != alphabet) {
            return Status::error("HistAlphabetMap::apply_bin_map: map size must be 29");
        }
        Hist plain{};
        for (std::size_t x = 0; x < alphabet; ++x) {
            if (map[x] >= alphabet) {
                return Status::error("HistAlphabetMap::apply_bin_map: map entry out of range");
            }
            plain[map[x]] += cipher_hist[x];
        }
        return plain;
    }

    /// `P[b] = lag_diff[b] + prefix[b]` (CTAK/S4 merge after once-counts).
    [[nodiscard]] static Hist merge_lag_diff_and_prefix(const Hist& lag_diff,
                                                       const Hist& prefix) noexcept {
        Hist plain{};
        for (std::size_t b = 0; b < alphabet; ++b) {
            plain[b] = lag_diff[b] + prefix[b];
        }
        return plain;
    }

    /// Cipher histogram of a dense Index29 byte stream.
    [[nodiscard]] static StatusOr<Hist> count_stream_hist(std::span<const std::uint8_t> cipher) {
        Hist hist{};
        for (std::size_t i = 0; i < cipher.size(); ++i) {
            if (cipher[i] >= alphabet) {
                return Status::error("HistAlphabetMap::count_stream_hist: symbol out of range");
            }
            ++hist[cipher[i]];
        }
        return hist;
    }

    /// Tail lag-diff hist: `D[b] = |{t ≥ L : (in[t] - in[t-L]) ≡ b}|`.
    /// `L == 0` is an error. If `L >= T`, returns an all-zero hist.
    [[nodiscard]] static StatusOr<Hist> count_lag_diff_hist(std::span<const std::uint8_t> cipher,
                                                            std::uint32_t lag) {
        if (lag == 0u) {
            return Status::error("HistAlphabetMap::count_lag_diff_hist: lag must be >= 1");
        }
        Hist hist{};
        if (static_cast<std::size_t>(lag) >= cipher.size()) {
            return hist;
        }
        for (std::size_t t = static_cast<std::size_t>(lag); t < cipher.size(); ++t) {
            if (cipher[t] >= alphabet || cipher[t - lag] >= alphabet) {
                return Status::error("HistAlphabetMap::count_lag_diff_hist: symbol out of range");
            }
            const std::uint8_t diff = sub_mod(cipher[t], cipher[t - static_cast<std::size_t>(lag)]);
            ++hist[diff];
        }
        return hist;
    }

    /// CTAK primer prefix: `t < L`, `out = in[t] - primer[t]`.
    [[nodiscard]] static StatusOr<Hist>
    count_ctak_prefix_hist(std::span<const std::uint8_t> cipher,
                           std::span<const std::uint8_t> primer) {
        if (primer.empty()) {
            return Status::error("HistAlphabetMap::count_ctak_prefix_hist: empty primer");
        }
        if (primer.size() > cipher.size()) {
            return Status::error(
                "HistAlphabetMap::count_ctak_prefix_hist: primer longer than cipher");
        }
        Hist hist{};
        for (std::size_t t = 0; t < primer.size(); ++t) {
            if (cipher[t] >= alphabet || primer[t] >= alphabet) {
                return Status::error(
                    "HistAlphabetMap::count_ctak_prefix_hist: symbol out of range");
            }
            ++hist[sub_mod(cipher[t], primer[t])];
        }
        return hist;
    }

    /// AutokeyRing prefix (`key = 0` for `t < lag`): `out = in[t]`.
    [[nodiscard]] static StatusOr<Hist> count_ring_prefix_hist(std::span<const std::uint8_t> cipher,
                                                               std::uint32_t lag) {
        if (lag == 0u) {
            return Status::error("HistAlphabetMap::count_ring_prefix_hist: lag must be >= 1");
        }
        Hist hist{};
        const std::size_t n = cipher.size() < static_cast<std::size_t>(lag)
                                  ? cipher.size()
                                  : static_cast<std::size_t>(lag);
        for (std::size_t t = 0; t < n; ++t) {
            if (cipher[t] >= alphabet) {
                return Status::error(
                    "HistAlphabetMap::count_ring_prefix_hist: symbol out of range");
            }
            ++hist[cipher[t]];
        }
        return hist;
    }

    /// Full dense CTAK plaintext hist via lag-diff + primer prefix merge.
    [[nodiscard]] static StatusOr<Hist>
    ctak_plain_hist_from_once(std::span<const std::uint8_t> cipher,
                              std::span<const std::uint8_t> primer) {
        StatusOr<Hist> lag = count_lag_diff_hist(cipher, static_cast<std::uint32_t>(primer.size()));
        if (!lag.ok()) {
            return lag.status();
        }
        StatusOr<Hist> prefix = count_ctak_prefix_hist(cipher, primer);
        if (!prefix.ok()) {
            return prefix.status();
        }
        return merge_lag_diff_and_prefix(lag.value(), prefix.value());
    }

    /// Full AutokeyRing plaintext hist via lag-diff + key-0 prefix merge.
    [[nodiscard]] static StatusOr<Hist>
    ring_plain_hist_from_once(std::span<const std::uint8_t> cipher, std::uint32_t lag) {
        StatusOr<Hist> lag_hist = count_lag_diff_hist(cipher, lag);
        if (!lag_hist.ok()) {
            return lag_hist.status();
        }
        StatusOr<Hist> prefix = count_ring_prefix_hist(cipher, lag);
        if (!prefix.ok()) {
            return prefix.status();
        }
        return merge_lag_diff_and_prefix(lag_hist.value(), prefix.value());
    }

    /// Adjacent-pair counts `B[x0*29 + x1]`. `T < 2` → all zeros.
    [[nodiscard]] static StatusOr<BigramHist>
    count_bigrams(std::span<const std::uint8_t> cipher) {
        BigramHist hist{};
        if (cipher.size() < 2u) {
            return hist;
        }
        for (std::size_t t = 0; t + 1u < cipher.size(); ++t) {
            if (cipher[t] >= alphabet || cipher[t + 1u] >= alphabet) {
                return Status::error("HistAlphabetMap::count_bigrams: symbol out of range");
            }
            const std::size_t idx =
                static_cast<std::size_t>(cipher[t]) * alphabet + static_cast<std::size_t>(cipher[t + 1u]);
            ++hist[idx];
        }
        return hist;
    }

    /// `score = -Σ B[x0][x1] · ll[(x0-s)%29][(x1-s)%29]` with fixed `x0,x1` order 0..28.
    [[nodiscard]] static StatusOr<double>
    bigram_ll_dot_rotated(const BigramHist& cipher_bigrams, std::span<const float> log_bigram_ll,
                          std::uint8_t shift) {
        Status ok = require_shift(shift);
        if (!ok.ok()) {
            return ok;
        }
        if (log_bigram_ll.size() != bigram_bins) {
            return Status::error("HistAlphabetMap::bigram_ll_dot_rotated: ll table size must be 841");
        }

        double score = 0.0;
        for (std::size_t x0 = 0; x0 < alphabet; ++x0) {
            for (std::size_t x1 = 0; x1 < alphabet; ++x1) {
                const std::uint32_t count = cipher_bigrams[x0 * alphabet + x1];
                if (count == 0u) {
                    continue;
                }
                const std::size_t y0 =
                    (x0 + alphabet - static_cast<std::size_t>(shift)) % alphabet;
                const std::size_t y1 =
                    (x1 + alphabet - static_cast<std::size_t>(shift)) % alphabet;
                score += static_cast<double>(count) *
                         static_cast<double>(log_bigram_ll[y0 * alphabet + y1]);
            }
        }
        return -score;
    }

    /// Brute-force Caesar bigram LL over the stream (same LL table, same score sign).
    [[nodiscard]] static StatusOr<double>
    bigram_ll_brute_force(std::span<const std::uint8_t> cipher, std::span<const float> log_bigram_ll,
                          std::uint8_t shift) {
        Status ok = require_shift(shift);
        if (!ok.ok()) {
            return ok;
        }
        if (log_bigram_ll.size() != bigram_bins) {
            return Status::error("HistAlphabetMap::bigram_ll_brute_force: ll table size must be 841");
        }
        if (cipher.size() < 2u) {
            return 0.0;
        }

        double acc = 0.0;
        for (std::size_t t = 0; t + 1u < cipher.size(); ++t) {
            if (cipher[t] >= alphabet || cipher[t + 1u] >= alphabet) {
                return Status::error("HistAlphabetMap::bigram_ll_brute_force: symbol out of range");
            }
            const std::uint8_t y0 = sub_mod(cipher[t], shift);
            const std::uint8_t y1 = sub_mod(cipher[t + 1u], shift);
            acc += static_cast<double>(
                log_bigram_ll[static_cast<std::size_t>(y0) * alphabet + static_cast<std::size_t>(y1)]);
        }
        return -acc;
    }

    /// Pearson χ² from an already-remapped plaintext hist (`N = Σ P`).
    [[nodiscard]] static StatusOr<double> chi2_from_hist(const Hist& plain_hist,
                                                         std::span<const double> probabilities) {
        if (probabilities.size() != alphabet) {
            return Status::error("HistAlphabetMap::chi2_from_hist: probabilities size must be 29");
        }
        std::uint64_t n = 0;
        for (std::size_t i = 0; i < alphabet; ++i) {
            n += plain_hist[i];
        }
        if (n == 0u) {
            return Status::error("HistAlphabetMap::chi2_from_hist: empty histogram");
        }
        const double n_d = static_cast<double>(n);
        double chi2 = 0.0;
        for (std::size_t c = 0; c < alphabet; ++c) {
            const double e = probabilities[c] * n_d;
            if (!(e > 0.0)) {
                return Status::error("HistAlphabetMap::chi2_from_hist: expected frequency is zero");
            }
            const double diff = static_cast<double>(plain_hist[c]) - e;
            chi2 += (diff * diff) / e;
        }
        return chi2;
    }

    [[nodiscard]] static std::uint32_t hist_total(const Hist& hist) noexcept {
        std::uint32_t n = 0;
        for (std::size_t i = 0; i < alphabet; ++i) {
            n += hist[i];
        }
        return n;
    }

private:
    HistAlphabetMap() = delete;

    [[nodiscard]] static Status require_shift(std::uint8_t shift) {
        if (shift >= alphabet) {
            return Status::error("HistAlphabetMap: shift must be in 0..28");
        }
        return Status::success();
    }

    [[nodiscard]] static std::uint8_t sub_mod(std::uint8_t x, std::uint8_t y) noexcept {
        const unsigned s = static_cast<unsigned>(x) + 29u - static_cast<unsigned>(y);
        return static_cast<std::uint8_t>(s >= 29u ? s - 29u : s);
    }
};

#endif // HIST_ALPHABET_MAP_HPP
