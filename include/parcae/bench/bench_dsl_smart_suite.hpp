#ifndef BENCH_DSL_SMART_SUITE_HPP
#define BENCH_DSL_SMART_SUITE_HPP

#include "parcae/bench/bench_metric.hpp"
#include "parcae/bench/bench_report.hpp"
#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/bench/bench_timer.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_hist_chi2_emit.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/search/nvtx_range.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "theory_hist_chi2_launch.hpp"
#endif

/// Fair Kernel-SLO microbench for hand-written HotLoop customs
/// (`docs/architecture/dsl-smart-hist.md` → `profiles/dsl_smart/`).
///
/// Rows pair self-written ShapeInline twins with catalog `F.atbash` /
/// CaesarChi2 / `F.affine` on the **same** C/T. PRIMARY gate is fair
/// `T≥2^20` vs 896B DRAM roof — campaign wall is never PRIMARY.
///
/// CLI: `parcae-bench --suite dsl_smart --allow-cuda`.
class BenchDslSmartSuite {
public:
    class Options {
    public:
        Options() noexcept : tokens_(0), repeats_(0), compare_catalog_(true) {}

        [[nodiscard]] std::size_t tokens() const noexcept { return tokens_; }

        [[nodiscard]] std::size_t repeats() const noexcept { return repeats_; }

        [[nodiscard]] bool compare_catalog() const noexcept { return compare_catalog_; }

        void set_tokens(std::size_t tokens) noexcept { tokens_ = tokens; }

        void set_repeats(std::size_t repeats) noexcept { repeats_ = repeats; }

        void set_compare_catalog(bool enabled) noexcept { compare_catalog_ = enabled; }

    private:
        /// 0 → fair `BenchTierSpec::t1.tokens` (2^20).
        std::size_t tokens_;
        /// 0 → per-tier default repeats.
        std::size_t repeats_;
        bool compare_catalog_;
    };

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs) {
        return run(freqs, Options{});
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs,
                                                             const Options& options) {
#if !defined(PARCAE_HAS_CUDA)
        (void)freqs;
        (void)options;
        return Status::error("BenchDslSmartSuite: requires a CUDA build (PARCAE_HAS_CUDA)");
#else
        BenchReport::Document doc(BenchReport::Suite::DslSmart);

        const std::size_t fair_T =
            options.tokens() == 0 ? BenchTierSpec::fair_gate_tokens() : options.tokens();

        // --- Atbash custom vs F.atbash ---
        {
            const BenchTierSpec::Tier& tier = BenchTierSpec::dsl_smart_atbash;
            const std::size_t reps =
                options.repeats() == 0 ? tier.repeats : options.repeats();
            StatusOr<BenchReport::Row> custom = run_custom_atbash(
                freqs, tier.candidates, fair_T, reps, tier.id, tier.workload, &tier);
            if (!custom.ok()) {
                return custom.status();
            }
            doc.add_row(std::move(custom.value()));

            if (options.compare_catalog()) {
                const BenchTierSpec::Tier& cmp = BenchTierSpec::dsl_smart_compare_atbash;
                StatusOr<BenchReport::Row> cat = run_catalog_atbash(
                    freqs, cmp.candidates, fair_T, reps, cmp.id, cmp.workload, &cmp);
                if (!cat.ok()) {
                    return cat.status();
                }
                doc.add_row(std::move(cat.value()));
            }
        }

        // --- Caesar custom vs CaesarChi2Batch ---
        {
            const BenchTierSpec::Tier& tier = BenchTierSpec::dsl_smart_caesar;
            const std::size_t reps =
                options.repeats() == 0 ? tier.repeats : options.repeats();
            StatusOr<BenchReport::Row> custom = run_custom_caesar(
                freqs, tier.candidates, fair_T, reps, tier.id, tier.workload, &tier);
            if (!custom.ok()) {
                return custom.status();
            }
            doc.add_row(std::move(custom.value()));

            if (options.compare_catalog()) {
                const BenchTierSpec::Tier& cmp = BenchTierSpec::dsl_smart_compare_caesar;
                StatusOr<BenchReport::Row> cat = run_catalog_caesar(
                    freqs, cmp.candidates, fair_T, reps, cmp.id, cmp.workload, &cmp);
                if (!cat.ok()) {
                    return cat.status();
                }
                doc.add_row(std::move(cat.value()));
            }
        }

        // --- Affine decrypt custom vs F.affine ---
        {
            const BenchTierSpec::Tier& tier = BenchTierSpec::dsl_smart_affine;
            const std::size_t reps =
                options.repeats() == 0 ? tier.repeats : options.repeats();
            // Catch2 short-T may shrink C via options: when T is underfill and
            // candidates not overridden, keep Spec C; for tiny smoke use Spec C
            // only when fair — else allow smaller via tokens-only path (still Spec C).
            StatusOr<BenchReport::Row> custom = run_custom_affine(
                freqs, tier.candidates, fair_T, reps, tier.id, tier.workload, &tier);
            if (!custom.ok()) {
                return custom.status();
            }
            doc.add_row(std::move(custom.value()));

            if (options.compare_catalog()) {
                const BenchTierSpec::Tier& cmp = BenchTierSpec::dsl_smart_compare_affine;
                StatusOr<BenchReport::Row> cat = run_catalog_affine(
                    freqs, cmp.candidates, fair_T, reps, cmp.id, cmp.workload, &cmp);
                if (!cat.ok()) {
                    return cat.status();
                }
                doc.add_row(std::move(cat.value()));
            }
        }

        doc.recompute_all_pass();
        return doc;
#endif
    }

private:
    BenchDslSmartSuite() = delete;

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static std::string
    annotate_detail(std::string detail, double rps, double peak, std::size_t tokens) {
        if (!BenchTierSpec::is_fair_gate_tokens(tokens)) {
            if (!detail.empty()) {
                detail += ";";
            }
            detail += "underfill_not_slo_gate";
            return detail;
        }
        if (peak > 0.0) {
            if (!detail.empty()) {
                detail += ";";
            }
            detail += "pct_peak=" + std::to_string(BenchTierSpec::percent_peak(rps, peak));
            if (BenchTierSpec::checkpoint_50B_applicable(peak)) {
                detail += BenchTierSpec::checkpoint_50B_hit(rps, peak) ? ";checkpoint_50B=hit"
                                                                        : ";checkpoint_50B=miss";
            }
        }
        return detail;
    }

    [[nodiscard]] static bool gate_pass(double rps, std::size_t tokens, double slo_min,
                                        double peak) noexcept {
        if (rps <= 0.0) {
            return false;
        }
        if (!BenchTierSpec::is_fair_gate_tokens(tokens) || peak <= 0.0) {
            return true;
        }
        return BenchTierSpec::pass_tier(rps, slo_min, peak);
    }

    [[nodiscard]] static BenchReport::Row
    row_from_sample(std::string name, std::string workload, const BenchMetric::Sample& sample,
                    std::size_t C, std::size_t T, std::size_t reps,
                    const BenchTierSpec::Tier* tier, std::string detail_prefix) {
        const double peak = tier != nullptr ? tier->estimated_peak : 0.0;
        const double slo_min = tier != nullptr ? tier->slo_min : 0.0;
        const bool pass = gate_pass(sample.runes_per_sec(), T, slo_min, peak);
        std::string detail =
            annotate_detail(std::move(detail_prefix), sample.runes_per_sec(), peak, T);
        return BenchReport::Row::make(
            std::move(name), std::move(workload), BenchReport::Suite::DslSmart,
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            sample.runes_per_sec(), sample.keys_per_sec(), sample.wall_seconds(), slo_min, 0.0,
            peak, C, T, reps, std::move(detail));
    }

    [[nodiscard]] static std::vector<std::uint8_t> random_stream(std::size_t n,
                                                                 std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 28);
        std::vector<std::uint8_t> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<std::uint8_t>(dist(rng));
        }
        return out;
    }

    [[nodiscard]] static StatusOr<TheoryIr> make_atbash_arith_theory() {
        const Z29Expr::Ptr x = Z29Expr::var("x");
        StatusOr<Z29Expr::Ptr> twenty_eight = Z29Expr::constant(28);
        if (!twenty_eight.ok()) {
            return twenty_eight.status();
        }
        return TheoryIr::make(
            "dsl_smart_atbash_arith", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault, {},
            Z29Expr::sub(twenty_eight.value(), x), Z29Expr::sub(twenty_eight.value(), x),
            std::string("Hand-written Atbash arith (name-irrelevant)."));
    }

    [[nodiscard]] static StatusOr<TheoryIr> make_caesar_theory() {
        StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
        if (!shift_p.ok()) {
            return shift_p.status();
        }
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr shift = Z29Expr::var("shift");
        return TheoryIr::make("dsl_smart_caesar", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
                              TheoryIr::InterruptMode::ElementwiseDefault, {shift_p.value()},
                              Z29Expr::add(x, shift), Z29Expr::sub(x, shift),
                              std::string("Hand-written Caesar."));
    }

    [[nodiscard]] static StatusOr<TheoryIr> make_affine_decrypt_theory() {
        StatusOr<ParamIr> a = ParamIr::make("a", 1, 28);
        if (!a.ok()) {
            return a.status();
        }
        StatusOr<ParamIr> b = ParamIr::make("b", 0, 28);
        if (!b.ok()) {
            return b.status();
        }
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr av = Z29Expr::var("a");
        const Z29Expr::Ptr bv = Z29Expr::var("b");
        return TheoryIr::make(
            "dsl_smart_affine", TheoryIr::Family::Elementwise, TheoryIr::Tier::A,
            TheoryIr::InterruptMode::ElementwiseDefault, {a.value(), b.value()},
            Z29Expr::add(Z29Expr::mul(av, x), bv),
            Z29Expr::mul(Z29Expr::inv(av), Z29Expr::sub(x, bv)),
            std::string("Hand-written Affine decrypt."));
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_custom_atbash(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                      std::size_t reps, std::string name, std::string workload,
                      const BenchTierSpec::Tier* tier) {
        StatusOr<TheoryIr> theory = make_atbash_arith_theory();
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<TheoryHistChi2Emit::EmitBundle> emit =
            TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
        if (!emit.ok() || !emit.value().has_shape_atbash_kernel()) {
            return Status::error("BenchDslSmartSuite: Atbash custom did not emit ShapeInline twin");
        }

        const auto host_in = random_stream(T, 0xD51Au);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_custom_atbash");
            return TheoryHistChi2Launch::launch_shape_atbash_async(
                device_in.value().data(), device_probs.value().data(),
                device_counts.value().data(), device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "ShapeInline_atbash");
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_catalog_atbash(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                       std::size_t reps, std::string name, std::string workload,
                       const BenchTierSpec::Tier* tier) {
        const auto host_in = random_stream(T, 0xD51Au);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * FamilyChi2Batch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_compare_Fatbash");
            return FamilyChi2Batch::launch_atbash_async(
                device_in.value().data(), device_probs.value().data(),
                device_counts.value().data(), device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "catalog_F.atbash");
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_custom_caesar(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                      std::size_t reps, std::string name, std::string workload,
                      const BenchTierSpec::Tier* tier) {
        StatusOr<TheoryIr> theory = make_caesar_theory();
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<TheoryHistChi2Emit::EmitBundle> emit =
            TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
        if (!emit.ok() || !emit.value().has_shape_caesar_kernel()) {
            return Status::error("BenchDslSmartSuite: Caesar custom did not emit ShapeInline twin");
        }

        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c % 29);
        }
        const auto host_in = random_stream(T, 0xD51Cu);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_custom_caesar");
            return TheoryHistChi2Launch::launch_shape_caesar_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "ShapeInline_caesar");
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_catalog_caesar(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                       std::size_t reps, std::string name, std::string workload,
                       const BenchTierSpec::Tier* tier) {
        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c % 29);
        }
        const auto host_in = random_stream(T, 0xD51Cu);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * CaesarChi2Batch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_compare_caesar");
            return CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "catalog_CaesarChi2");
    }

    static void fill_affine_grid(std::size_t C, std::vector<std::uint8_t>& a,
                                 std::vector<std::uint8_t>& b) {
        a.assign(C, 0);
        b.assign(C, 0);
        std::size_t c = 0;
        for (std::uint8_t ai = 1; ai <= 28 && c < C; ++ai) {
            for (std::uint8_t bi = 0; bi < 29 && c < C; ++bi) {
                a[c] = ai;
                b[c] = bi;
                ++c;
            }
        }
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_custom_affine(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                      std::size_t reps, std::string name, std::string workload,
                      const BenchTierSpec::Tier* tier) {
        StatusOr<TheoryIr> theory = make_affine_decrypt_theory();
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<TheoryHistChi2Emit::EmitBundle> emit =
            TheoryHistChi2Emit::emit_decrypt_hist(theory.value());
        if (!emit.ok() || !emit.value().has_shape_affine_kernel()) {
            return Status::error("BenchDslSmartSuite: Affine custom did not emit ShapeInline twin");
        }

        std::vector<std::uint8_t> host_a;
        std::vector<std::uint8_t> host_b;
        fill_affine_grid(C, host_a, host_b);

        const auto host_in = random_stream(T, 0xD51Fu);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_a =
            DeviceBuffer<std::uint8_t>::from_host(host_a);
        if (!device_a.ok()) {
            return device_a.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_b =
            DeviceBuffer<std::uint8_t>::from_host(host_b);
        if (!device_b.ok()) {
            return device_b.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * TheoryHistChi2Launch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_custom_affine");
            return TheoryHistChi2Launch::launch_shape_affine_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "ShapeInline_affine");
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_catalog_affine(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                       std::size_t reps, std::string name, std::string workload,
                       const BenchTierSpec::Tier* tier) {
        std::vector<std::uint8_t> host_a;
        std::vector<std::uint8_t> host_b;
        fill_affine_grid(C, host_a, host_b);

        const auto host_in = random_stream(T, 0xD51Fu);
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_a =
            DeviceBuffer<std::uint8_t>::from_host(host_a);
        if (!device_a.ok()) {
            return device_a.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_b =
            DeviceBuffer<std::uint8_t>::from_host(host_b);
        if (!device_b.ok()) {
            return device_b.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * FamilyChi2Batch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            NvtxRange nvtx("dsl_smart_compare_Faffine");
            return FamilyChi2Batch::launch_affine_async(
                device_in.value().data(), device_a.value().data(), device_b.value().data(),
                device_probs.value().data(), device_counts.value().data(),
                device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               tier, "catalog_F.affine");
    }
#endif
};

#endif // BENCH_DSL_SMART_SUITE_HPP
