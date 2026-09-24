#ifndef BENCH_SLO_SUITE_HPP
#define BENCH_SLO_SUITE_HPP

#include "parcae/bench/bench_report.hpp"
#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <cstddef>
#include <string>
#include <utility>

#if defined(PARCAE_HAS_CUDA)
#include "parcae/run/throughput_tiers.hpp"
#endif

/// Orchestrates fused CUDA SLO tiers (T1–T3) into a `BenchReport::Document`.
///
/// Primary path reuses `ThroughputTiers` kernels (CaesarChi2Batch /
/// FamilyChi2Batch / DeepScoreBatch). Optional `--extended` adds F.* and C.*
/// rows (same as historical full `parcae-throughput-tiers` suite).
///
/// Requires a CUDA build + usable device. CPU-only builds return a Status error.
class BenchSloSuite {
public:
    class Options {
    public:
        // No NSDMI: GCC rejects Options{} while BenchSloSuite is incomplete.
        Options() noexcept : extended_(false) {}

        explicit Options(bool extended) noexcept : extended_(extended) {}

        [[nodiscard]] bool extended() const noexcept { return extended_; }

        void set_extended(bool enabled) noexcept { extended_ = enabled; }

    private:
        /// When true, include F.* transform-family and C.* compose rows.
        bool extended_;
    };

    /// Build a measured SLO row (derives keys/s and wall from runes/s).
    /// Host-only helper — usable without CUDA for unit tests.
    [[nodiscard]] static BenchReport::Row
    make_measured_row(std::string name, std::string workload, double runes_per_sec,
                      double target_min, double target_max, bool pass, std::size_t candidates,
                      std::size_t tokens, std::size_t repeats, std::string detail = {}) {
        const double peak = BenchTierSpec::estimated_peak(name);
        double keys_per_sec = 0.0;
        double wall_seconds = 0.0;
        if (runes_per_sec > 0.0 && tokens > 0) {
            keys_per_sec = runes_per_sec / static_cast<double>(tokens);
            wall_seconds = (static_cast<double>(repeats) * static_cast<double>(candidates) *
                            static_cast<double>(tokens)) /
                           runes_per_sec;
        }
        return BenchReport::Row::make(
            std::move(name), std::move(workload), BenchReport::Suite::Slo,
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail, runes_per_sec,
            keys_per_sec, wall_seconds, target_min, target_max, peak, candidates, tokens, repeats,
            std::move(detail));
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static BenchReport::Row
    from_tier_result(const ThroughputTiers::TierResult& tier) {
        return make_measured_row(tier.name, tier.workload, tier.runes_per_sec, tier.target_min,
                                 tier.target_max, tier.pass, tier.candidates, tier.tokens,
                                 tier.repeats);
    }
#endif

    /// Run T1–T3 (and optionally F.*/C.*). Returns `BenchReport::Document`.
    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs) {
        return run(freqs, Options{});
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs,
                                                             const Options& options) {
#if !defined(PARCAE_HAS_CUDA)
        (void)freqs;
        (void)options;
        return Status::error("BenchSloSuite: requires a CUDA build (PARCAE_HAS_CUDA)");
#else
        StatusOr<ThroughputTiers::Report> raw = ThroughputTiers::run(freqs, options.extended());
        if (!raw.ok()) {
            return raw.status();
        }

        BenchReport::Document doc(BenchReport::Suite::Slo);
        for (const ThroughputTiers::TierResult& tier : raw.value().tiers) {
            doc.add_row(from_tier_result(tier));
        }
        doc.recompute_all_pass();
        return doc;
#endif
    }

private:
    BenchSloSuite() = delete;
};

#endif // BENCH_SLO_SUITE_HPP
