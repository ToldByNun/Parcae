#ifndef BENCH_METRIC_HPP
#define BENCH_METRIC_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

/// Throughput metric helpers for Parcae bench / diagnostics.
///
/// **Dual rates** (setup excluded):
/// - `logical_runes_per_sec` (= historical `runes_per_sec`) =
///   `repeats × C × T / seconds`. For alphabet/column/lag/bigram **remap**
///   paths this counts candidate×token pairs scored, not DRAM bytes streamed
///   (cipher is read ~once). Keep for trend / `%peak` vs Remap roofs.
/// - `cipher_bytes_per_sec` = `repeats × T × bytes_per_token / seconds` —
///   physical cipher traffic model (remap mono/column/lag: ~1 B/token; bigram
///   once: ~2 B/token). Compare to GDDR7 BW (896e9), not to Remap roofs.
/// - `keys_per_sec` = `repeats × C / seconds`
///
/// Median-of-3 aggregation matches the CUDA SLO protocol in
/// `docs/architecture/cuda-throughput.md` / `BenchTimer`.
class BenchMetric {
public:
    enum class Kind : std::uint8_t {
        RunesPerSec = 0,
        KeysPerSec,
        CipherBytesPerSec,
    };

    /// One timed window (single sample or median aggregate).
    class Sample {
    public:
        Sample() = default;

        Sample(double wall_seconds, double runes_per_sec, double keys_per_sec) noexcept
            : wall_seconds_(wall_seconds), runes_per_sec_(runes_per_sec),
              keys_per_sec_(keys_per_sec), cipher_bytes_per_sec_(0.0) {}

        Sample(double wall_seconds, double runes_per_sec, double keys_per_sec,
               double cipher_bytes_per_sec) noexcept
            : wall_seconds_(wall_seconds), runes_per_sec_(runes_per_sec),
              keys_per_sec_(keys_per_sec), cipher_bytes_per_sec_(cipher_bytes_per_sec) {}

        [[nodiscard]] double wall_seconds() const noexcept { return wall_seconds_; }

        /// Logical C·T rate (inflated under remap). Alias of historical runes/s.
        [[nodiscard]] double runes_per_sec() const noexcept { return runes_per_sec_; }

        [[nodiscard]] double logical_runes_per_sec() const noexcept { return runes_per_sec_; }

        [[nodiscard]] double keys_per_sec() const noexcept { return keys_per_sec_; }

        /// Physical cipher traffic estimate (0 if not filled by `from_elapsed`).
        [[nodiscard]] double cipher_bytes_per_sec() const noexcept {
            return cipher_bytes_per_sec_;
        }

        /// Build rates from a timed window. Zero / non-positive seconds → rates 0.
        /// `bytes_per_token` defaults to 1 (mono/column/lag once-count).
        [[nodiscard]] static Sample from_elapsed(std::size_t repeats, std::size_t candidates,
                                                 std::size_t tokens, double wall_seconds,
                                                 double bytes_per_token = 1.0) noexcept {
            return Sample{wall_seconds,
                          BenchMetric::logical_runes_per_sec(repeats, candidates, tokens,
                                                            wall_seconds),
                          BenchMetric::keys_per_sec(repeats, candidates, wall_seconds),
                          BenchMetric::cipher_bytes_per_sec(repeats, tokens, wall_seconds,
                                                           bytes_per_token)};
        }

    private:
        double wall_seconds_ = 0.0;
        double runes_per_sec_ = 0.0;
        double keys_per_sec_ = 0.0;
        double cipher_bytes_per_sec_ = 0.0;
    };

    /// Historical name for logical `repeats × C × T / seconds`.
    [[nodiscard]] static double runes_per_sec(std::size_t repeats, std::size_t candidates,
                                              std::size_t tokens, double wall_seconds) noexcept {
        return logical_runes_per_sec(repeats, candidates, tokens, wall_seconds);
    }

    /// Logical scored runes/s (`C·T`). Inflated vs DRAM for remap shapes.
    [[nodiscard]] static double logical_runes_per_sec(std::size_t repeats, std::size_t candidates,
                                                      std::size_t tokens,
                                                      double wall_seconds) noexcept {
        if (wall_seconds <= 0.0) {
            return 0.0;
        }
        const double runes = static_cast<double>(repeats) * static_cast<double>(candidates) *
                             static_cast<double>(tokens);
        return runes / wall_seconds;
    }

    /// Physical cipher bytes/s for once-count traffic (`repeats · T · bytes_per_token`).
    [[nodiscard]] static double cipher_bytes_per_sec(std::size_t repeats, std::size_t tokens,
                                                     double wall_seconds,
                                                     double bytes_per_token = 1.0) noexcept {
        if (wall_seconds <= 0.0 || bytes_per_token <= 0.0) {
            return 0.0;
        }
        const double bytes = static_cast<double>(repeats) * static_cast<double>(tokens) *
                             bytes_per_token;
        return bytes / wall_seconds;
    }

    [[nodiscard]] static double keys_per_sec(std::size_t repeats, std::size_t candidates,
                                             double wall_seconds) noexcept {
        if (wall_seconds <= 0.0) {
            return 0.0;
        }
        const double keys = static_cast<double>(repeats) * static_cast<double>(candidates);
        return keys / wall_seconds;
    }

    /// Median of three samples (sort network). Used by `BenchTimer` timed windows.
    [[nodiscard]] static constexpr double median3(double a, double b, double c) noexcept {
        if (a > b) {
            const double t = a;
            a = b;
            b = t;
        }
        if (b > c) {
            const double t = b;
            b = c;
            c = t;
        }
        if (a > b) {
            const double t = a;
            a = b;
            b = t;
        }
        return b;
    }

    /// Median-of-3 on `Sample::runes_per_sec`, rebuilding wall/keys from the
    /// chosen middle runes/s sample's sibling rates (keeps the matching triple).
    [[nodiscard]] static Sample median3_sample(Sample a, Sample b, Sample c) noexcept {
        double ra = a.runes_per_sec();
        double rb = b.runes_per_sec();
        double rc = c.runes_per_sec();
        Sample sa = a;
        Sample sb = b;
        Sample sc = c;
        if (ra > rb) {
            std::swap(ra, rb);
            std::swap(sa, sb);
        }
        if (rb > rc) {
            std::swap(rb, rc);
            std::swap(sb, sc);
        }
        if (ra > rb) {
            std::swap(ra, rb);
            std::swap(sa, sb);
        }
        return sb;
    }

    [[nodiscard]] static constexpr const char* kind_str(Kind kind) noexcept {
        switch (kind) {
        case Kind::RunesPerSec:
            return "runes/s";
        case Kind::KeysPerSec:
            return "keys/s";
        case Kind::CipherBytesPerSec:
            return "cipher_B/s";
        }
        return "unknown";
    }

private:
    BenchMetric() = delete;
};

#endif // BENCH_METRIC_HPP
