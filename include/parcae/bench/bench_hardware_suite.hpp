#ifndef BENCH_HARDWARE_SUITE_HPP
#define BENCH_HARDWARE_SUITE_HPP

#include "parcae/bench/bench_metric.hpp"
#include "parcae/bench/bench_report.hpp"
#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/bench/bench_timer.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "parcae/run/throughput_tiers.hpp"

#include "parcae_cuda.hpp"
#endif

/// CPU vs CUDA side-by-side timing for primary T1–T3 workloads.
///
/// - CPU: Caesar χ² smoke by default (scaled); `--cpu-full` uses full C/T/reps.
///   T2/T3 CPU legs are documented partial proxies (`cpu_partial: caesar_chi2_proxy`).
/// - CUDA: same fused kernels as `BenchSloSuite` / `ThroughputTiers` (T1–T3 only).
/// - Missing CUDA → `RowStatus::Skipped` with detail `skipped_not_built`
///   (`--require-cuda` fails; `--allow-skip` or CPU-capable backends OK).
/// - `gpu/cpu` ratio is informational in `detail` (no absolute hosted CI gate).
class BenchHardwareSuite {
public:
    enum class BackendSelect : std::uint8_t {
        Cpu = 0,
        Cuda,
        Both,
    };

    class Options {
    public:
        Options() = default;

        [[nodiscard]] bool allow_cuda() const noexcept { return allow_cuda_; }

        void set_allow_cuda(bool enabled) noexcept { allow_cuda_ = enabled; }

        [[nodiscard]] bool require_cuda() const noexcept { return require_cuda_; }

        void set_require_cuda(bool enabled) noexcept {
            require_cuda_ = enabled;
            if (enabled) {
                allow_cuda_ = true;
            }
        }

        [[nodiscard]] bool allow_skip() const noexcept { return allow_skip_; }

        void set_allow_skip(bool enabled) noexcept { allow_skip_ = enabled; }

        [[nodiscard]] bool cpu_full() const noexcept { return cpu_full_; }

        void set_cpu_full(bool enabled) noexcept { cpu_full_ = enabled; }

        [[nodiscard]] BackendSelect backend() const noexcept { return backend_; }

        void set_backend(BackendSelect backend) noexcept { backend_ = backend; }

        [[nodiscard]] std::uint32_t seed() const noexcept { return seed_; }

        void set_seed(std::uint32_t seed) noexcept { seed_ = seed; }

    private:
        bool allow_cuda_ = false;
        bool require_cuda_ = false;
        bool allow_skip_ = false;
        bool cpu_full_ = false;
        BackendSelect backend_ = BackendSelect::Both;
        std::uint32_t seed_ = 0x48415244u; // 'HARD'
    };

    [[nodiscard]] static StatusOr<BackendSelect> parse_backend(std::string_view text) {
        if (text == "cpu") {
            return BackendSelect::Cpu;
        }
        if (text == "cuda") {
            return BackendSelect::Cuda;
        }
        if (text == "both" || text.empty()) {
            return BackendSelect::Both;
        }
        return Status::error("BenchHardwareSuite: --backend must be cpu|cuda|both (got '" +
                             std::string(text) + "')");
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const Context& ctx,
                                                             const Options& options = Options{}) {
        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }

        const bool want_cpu =
            options.backend() == BackendSelect::Cpu || options.backend() == BackendSelect::Both;
        const bool want_cuda_leg =
            options.backend() == BackendSelect::Cuda || options.backend() == BackendSelect::Both;

        if (want_cuda_leg && options.backend() == BackendSelect::Cuda && !options.allow_cuda() &&
            !options.require_cuda()) {
            return Status::error(
                "BenchHardwareSuite: CUDA backend requires --allow-cuda or --require-cuda");
        }

        // Both without CUDA opt-in → CPU-only (CI-friendly).
        const bool attempt_cuda = want_cuda_leg && (options.allow_cuda() || options.require_cuda());

        bool cuda_usable = false;
        if (attempt_cuda) {
#if defined(PARCAE_HAS_CUDA)
            cuda_usable = BackendUtil::cuda_built() && ParcaeCuda::available();
#else
            cuda_usable = false;
#endif
            if (!cuda_usable) {
                if (options.require_cuda()) {
                    return Status::error(
                        "BenchHardwareSuite: --require-cuda but CUDA is unavailable "
                        "(not built or no device)");
                }
                if (options.backend() == BackendSelect::Cuda && !options.allow_skip()) {
                    return Status::error(
                        "BenchHardwareSuite: CUDA unavailable (use --allow-skip or "
                        "--backend cpu|both)");
                }
            }
        }

        double cpu_rps[BenchTierSpec::primary_tier_count] = {0.0, 0.0, 0.0};
        bool have_cpu[BenchTierSpec::primary_tier_count] = {false, false, false};

        BenchReport::Document doc(BenchReport::Suite::Hardware);

        if (want_cpu) {
            for (std::size_t i = 0; i < BenchTierSpec::primary_tier_count; ++i) {
                const BenchTierSpec::Tier& tier = BenchTierSpec::tier_at(i);
                StatusOr<BenchReport::Row> cpu_row =
                    measure_cpu_tier(tier, freqs.value(), options, i);
                if (!cpu_row.ok()) {
                    return cpu_row.status();
                }
                cpu_rps[i] = cpu_row.value().runes_per_sec();
                have_cpu[i] = true;
                doc.add_row(std::move(cpu_row.value()));
            }
        }

        if (attempt_cuda) {
            if (!cuda_usable) {
                for (std::size_t i = 0; i < BenchTierSpec::primary_tier_count; ++i) {
                    doc.add_row(make_skipped_cuda_row(BenchTierSpec::tier_at(i)));
                }
            } else {
#if defined(PARCAE_HAS_CUDA)
                StatusOr<ThroughputTiers::Report> raw =
                    ThroughputTiers::run(freqs.value(), /*extended=*/false);
                if (!raw.ok()) {
                    return raw.status();
                }
                for (std::size_t i = 0; i < BenchTierSpec::primary_tier_count; ++i) {
                    const char* id = BenchTierSpec::tier_at(i).id;
                    const ThroughputTiers::TierResult* found = nullptr;
                    for (const ThroughputTiers::TierResult& t : raw.value().tiers) {
                        if (t.name == id) {
                            found = &t;
                            break;
                        }
                    }
                    if (found == nullptr) {
                        return Status::error(std::string("BenchHardwareSuite: missing CUDA tier ") +
                                             id);
                    }
                    doc.add_row(from_cuda_tier(*found, have_cpu[i] ? cpu_rps[i] : 0.0));
                }
#else
                (void)cpu_rps;
                (void)have_cpu;
                for (std::size_t i = 0; i < BenchTierSpec::primary_tier_count; ++i) {
                    doc.add_row(make_skipped_cuda_row(BenchTierSpec::tier_at(i)));
                }
#endif
            }
        }

        doc.recompute_all_pass();
        return doc;
    }

private:
    BenchHardwareSuite() = delete;

    class CpuScale {
    public:
        std::size_t candidates = 0;
        std::size_t tokens = 0;
        std::size_t repeats = 0;
        double scale_factor = 1.0;
        bool partial_proxy = false;
    };

    [[nodiscard]] static CpuScale scale_for(const BenchTierSpec::Tier& tier, bool cpu_full) {
        CpuScale scale;
        scale.partial_proxy = (std::string_view{tier.id} != "T1");
        if (cpu_full) {
            scale.candidates = tier.candidates;
            scale.tokens = tier.tokens;
            scale.repeats = tier.repeats;
            scale.scale_factor = 1.0;
            return scale;
        }
        constexpr std::size_t kSmokeTokens = 4096u;
        constexpr std::size_t kSmokeCapC = 64u;
        scale.candidates = tier.candidates > kSmokeCapC ? kSmokeCapC : tier.candidates;
        scale.tokens = kSmokeTokens;
        scale.repeats = 1u;
        const double full = static_cast<double>(tier.candidates) *
                            static_cast<double>(tier.tokens) * static_cast<double>(tier.repeats);
        const double smoke = static_cast<double>(scale.candidates) *
                             static_cast<double>(scale.tokens) * static_cast<double>(scale.repeats);
        scale.scale_factor = full > 0.0 ? (smoke / full) : 0.0;
        return scale;
    }

    [[nodiscard]] static std::vector<Index29> random_indices(std::size_t n, std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, Index29::modulus - 1);
        std::vector<Index29> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
        }
        return out;
    }

    [[nodiscard]] static BenchReport::Row make_skipped_cuda_row(const BenchTierSpec::Tier& tier) {
        return BenchReport::Row::make(
            tier.id, tier.workload, BenchReport::Suite::Hardware, BenchReport::Backend::Cuda,
            BenchReport::RowStatus::Skipped, 0.0, 0.0, 0.0, tier.slo_min, tier.slo_max,
            tier.estimated_peak, tier.candidates, tier.tokens, tier.repeats, "skipped_not_built");
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static BenchReport::Row from_cuda_tier(const ThroughputTiers::TierResult& tier,
                                                         double cpu_runes_per_sec) {
        std::ostringstream detail;
        if (cpu_runes_per_sec > 0.0 && tier.runes_per_sec > 0.0) {
            const double ratio = tier.runes_per_sec / cpu_runes_per_sec;
            detail << "gpu/cpu=" << std::fixed << std::setprecision(2) << ratio << "x";
        } else {
            detail << "gpu/cpu=n/a";
        }
        const double keys =
            tier.tokens > 0 ? tier.runes_per_sec / static_cast<double>(tier.tokens) : 0.0;
        const double wall =
            (tier.runes_per_sec > 0.0 && tier.tokens > 0)
                ? (static_cast<double>(tier.repeats) * static_cast<double>(tier.candidates) *
                   static_cast<double>(tier.tokens)) /
                      tier.runes_per_sec
                : 0.0;
        // Measured CUDA row passes if timing succeeded (no hosted absolute SLO gate).
        return BenchReport::Row::make(tier.name, tier.workload, BenchReport::Suite::Hardware,
                                      BenchReport::Backend::Cuda, BenchReport::RowStatus::Pass,
                                      tier.runes_per_sec, keys, wall, tier.target_min,
                                      tier.target_max, BenchTierSpec::estimated_peak(tier.name),
                                      tier.candidates, tier.tokens, tier.repeats, detail.str());
    }
#endif

    [[nodiscard]] static StatusOr<BenchReport::Row>
    measure_cpu_tier(const BenchTierSpec::Tier& tier, const ExpectedFrequencyTable& freqs,
                     const Options& options, std::size_t tier_index) {
        const CpuScale scale = scale_for(tier, options.cpu_full());
        const std::uint32_t seed =
            options.seed() ^ static_cast<std::uint32_t>(0x100u * (tier_index + 1u));
        std::vector<Index29> cipher = random_indices(scale.tokens, seed);
        std::vector<Index29> plain(scale.tokens);

        volatile double sink = 0.0;

        StatusOr<BenchMetric::Sample> sample =
            BenchTimer::time_cpu(scale.repeats, scale.candidates, scale.tokens, [&]() -> Status {
                for (std::size_t c = 0; c < scale.candidates; ++c) {
                    const Index29 shift{static_cast<std::uint8_t>(c % Index29::modulus)};
                    Status dec =
                        CaesarTransform::kernel(cipher, plain, shift, TransformDirection::Decrypt);
                    if (!dec.ok()) {
                        return dec;
                    }
                    StatusOr<double> chi2 = Chi2EnglishGp::score(plain, freqs);
                    if (!chi2.ok()) {
                        return chi2.status();
                    }
                    sink = chi2.value();
                }
                return Status::success();
            });
        if (!sample.ok()) {
            return sample.status();
        }
        (void)sink;

        std::ostringstream detail;
        detail << "scale_factor=" << std::scientific << std::setprecision(3) << scale.scale_factor;
        if (scale.partial_proxy) {
            detail << "; cpu_partial: caesar_chi2_proxy";
        }
        if (!options.cpu_full()) {
            detail << "; smoke";
        } else {
            detail << "; cpu_full";
        }

        return BenchReport::Row::make(tier.id, tier.workload, BenchReport::Suite::Hardware,
                                      BenchReport::Backend::Cpu, BenchReport::RowStatus::Pass,
                                      sample.value().runes_per_sec(), sample.value().keys_per_sec(),
                                      sample.value().wall_seconds(), 0.0, 0.0, 0.0,
                                      scale.candidates, scale.tokens, scale.repeats, detail.str());
    }
};

#endif // BENCH_HARDWARE_SUITE_HPP
