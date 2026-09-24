#ifndef BENCH_METRIC_HPP
#define BENCH_METRIC_HPP

#include <cstddef>
#include <cstdint>
#include <utility>

/// Throughput metric helpers for Parcae bench / diagnostics.
///
/// Canonical rates (setup excluded):
/// - runes/s = `repeats × C × T / seconds`
/// - keys/s  = `repeats × C / seconds`
///
/// Median-of-3 aggregation matches the CUDA SLO protocol in
/// `docs/architecture/cuda-throughput.md` / `BenchTimer`.
class BenchMetric {
public:
    enum class Kind : std::uint8_t {
        RunesPerSec = 0,
        KeysPerSec,
    };

    /// One timed window (single sample or median aggregate).
    class Sample {
    public:
        Sample() = default;

        Sample(double wall_seconds, double runes_per_sec, double keys_per_sec) noexcept
            : wall_seconds_(wall_seconds),
              runes_per_sec_(runes_per_sec),
              keys_per_sec_(keys_per_sec) {}

        [[nodiscard]] double wall_seconds() const noexcept {
            return wall_seconds_;
        }

        [[nodiscard]] double runes_per_sec() const noexcept {
            return runes_per_sec_;
        }

        [[nodiscard]] double keys_per_sec() const noexcept {
            return keys_per_sec_;
        }

        /// Build rates from a timed window. Zero / non-positive seconds → rates 0.
        [[nodiscard]] static Sample from_elapsed(
            std::size_t repeats,
            std::size_t candidates,
            std::size_t tokens,
            double wall_seconds) noexcept {
            return Sample{
                wall_seconds,
                BenchMetric::runes_per_sec(repeats, candidates, tokens, wall_seconds),
                BenchMetric::keys_per_sec(repeats, candidates, wall_seconds)};
        }

    private:
        double wall_seconds_ = 0.0;
        double runes_per_sec_ = 0.0;
        double keys_per_sec_ = 0.0;
    };

    [[nodiscard]] static double runes_per_sec(
        std::size_t repeats,
        std::size_t candidates,
        std::size_t tokens,
        double wall_seconds) noexcept {
        if (wall_seconds <= 0.0) {
            return 0.0;
        }
        const double runes = static_cast<double>(repeats) * static_cast<double>(candidates) *
                             static_cast<double>(tokens);
        return runes / wall_seconds;
    }

    [[nodiscard]] static double keys_per_sec(
        std::size_t repeats, std::size_t candidates, double wall_seconds) noexcept {
        if (wall_seconds <= 0.0) {
            return 0.0;
        }
        const double keys =
            static_cast<double>(repeats) * static_cast<double>(candidates);
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
        }
        return "unknown";
    }

private:
    BenchMetric() = delete;
};

#endif // BENCH_METRIC_HPP
