#ifndef BENCH_THEORY_SUITE_HPP
#define BENCH_THEORY_SUITE_HPP

#include "parcae/bench/bench_metric.hpp"
#include "parcae/bench/bench_report.hpp"
#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/bench/bench_timer.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/param_ir.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/dsl/z29_expr.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "theory_chi2_batch.hpp"
#endif

/// Theory fused-χ² cudaEvent microbench (`parcae-bench --suite theory`).
///
/// Times **kernel only** via `BenchTimer` (4 warmups + median-of-3 cudaEvent;
/// H2D / host prepare excluded). Side-by-side Caesar twin uses the same
/// `(C,T,reps)` and cipher. Peaks are uncalibrated until `BenchTierSpec`
/// theory rows land — rows pass when measurement succeeds (`peak_uncalibrated`
/// in detail). Acceptance gate remains ≥90% `estimated_peak` once calibrated
/// (see `docs/architecture/cuda-profile-theory.md`).
class BenchTheorySuite {
public:
    class Options {
    public:
        Options() noexcept
            : tokens_(0), candidates_(0), repeats_(0), campaign_grid_(false),
              compare_catalog_(true) {}

        [[nodiscard]] std::size_t tokens() const noexcept { return tokens_; }

        [[nodiscard]] std::size_t candidates() const noexcept { return candidates_; }

        [[nodiscard]] std::size_t repeats() const noexcept { return repeats_; }

        [[nodiscard]] bool campaign_grid() const noexcept { return campaign_grid_; }

        [[nodiscard]] bool compare_catalog() const noexcept { return compare_catalog_; }

        void set_tokens(std::size_t tokens) noexcept { tokens_ = tokens; }

        void set_candidates(std::size_t candidates) noexcept { candidates_ = candidates; }

        void set_repeats(std::size_t repeats) noexcept { repeats_ = repeats; }

        void set_campaign_grid(bool enabled) noexcept { campaign_grid_ = enabled; }

        void set_compare_catalog(bool enabled) noexcept { compare_catalog_ = enabled; }

    private:
        /// 0 → default fair `BenchTierSpec::t1.tokens` (or campaign T when set).
        std::size_t tokens_;
        /// 0 → default per workload (29 / 9 / 16384).
        std::size_t candidates_;
        /// 0 → default per workload.
        std::size_t repeats_;
        bool campaign_grid_;
        bool compare_catalog_;
    };

    /// Host-only helper for unit tests (no CUDA).
    [[nodiscard]] static BenchReport::Row
    make_measured_row(std::string name, std::string workload, double runes_per_sec,
                      double keys_per_sec, double wall_seconds, bool pass, std::size_t candidates,
                      std::size_t tokens, std::size_t repeats, std::string detail = {}) {
        return BenchReport::Row::make(
            std::move(name), std::move(workload), BenchReport::Suite::Theory,
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail, runes_per_sec,
            keys_per_sec, wall_seconds, /*target_min=*/0.0, /*target_max=*/0.0,
            /*estimated_peak=*/0.0, candidates, tokens, repeats, std::move(detail));
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs) {
        return run(freqs, Options{});
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(const ExpectedFrequencyTable& freqs,
                                                             const Options& options) {
#if !defined(PARCAE_HAS_CUDA)
        (void)freqs;
        (void)options;
        return Status::error("BenchTheorySuite: requires a CUDA build (PARCAE_HAS_CUDA)");
#else
        BenchReport::Document doc(BenchReport::Suite::Theory);

        const std::size_t fair_T =
            options.tokens() == 0 ? BenchTierSpec::t1.tokens : options.tokens();
        const std::size_t fair_C =
            options.candidates() == 0 ? BenchTierSpec::t1.candidates : options.candidates();
        const std::size_t fair_reps = options.repeats() == 0 ? 8u : options.repeats();

        StatusOr<BenchReport::Row> caesar_bc =
            run_caesar_bytecode(freqs, fair_C, fair_T, fair_reps, "T.theory.caesar_bytecode",
                                "TheoryChi2Batch Caesar bytecode", "peak_uncalibrated");
        if (!caesar_bc.ok()) {
            return caesar_bc.status();
        }
        doc.add_row(std::move(caesar_bc.value()));

        if (options.compare_catalog()) {
            StatusOr<BenchReport::Row> caesar_twin =
                run_caesar_catalog(freqs, fair_C, fair_T, fair_reps, "T.theory.compare_caesar",
                                   "CaesarChi2Batch twin (same C/T)", "catalog_compare");
            if (!caesar_twin.ok()) {
                return caesar_twin.status();
            }
            doc.add_row(std::move(caesar_twin.value()));
        }

        const std::size_t prog_C = options.candidates() == 0 ? 9u : options.candidates();
        const std::size_t prog_reps = options.repeats() == 0 ? 4u : options.repeats();
        StatusOr<BenchReport::Row> progressive =
            run_progressive(freqs, prog_C, fair_T, prog_reps, "T.theory.progressive",
                            "TheoryChi2Batch keyed stream (b0+b1*i)", "peak_uncalibrated");
        if (!progressive.ok()) {
            return progressive.status();
        }
        doc.add_row(std::move(progressive.value()));

        if (options.campaign_grid()) {
            constexpr std::size_t camp_C = 16384;
            constexpr std::size_t camp_T = 262;
            const std::size_t camp_reps = options.repeats() == 0 ? 32u : options.repeats();
            StatusOr<BenchReport::Row> campaign = run_caesar_bytecode(
                freqs, camp_C, camp_T, camp_reps, "T.theory.caesar_campaign",
                "TheoryChi2Batch Caesar @ campaign-like T", "underfill_not_slo_gate");
            if (!campaign.ok()) {
                return campaign.status();
            }
            doc.add_row(std::move(campaign.value()));
        }

        doc.recompute_all_pass();
        return doc;
#endif
    }

private:
    BenchTheorySuite() = delete;

#if defined(PARCAE_HAS_CUDA)
    class PackedProgram {
    public:
        std::vector<std::uint8_t> ops;
        std::vector<std::uint8_t> imm;
        std::vector<std::uint8_t> slots;
        std::uint16_t slot_count = 0;
        std::uint16_t cipher_slot = 0;
        std::uint16_t index_slot = 0;
        std::uint8_t binds_index_i = 0;
        std::uint16_t max_stack = 0;
        std::size_t C = 0;
    };

    class Scratch {
    public:
        DeviceBuffer<std::uint8_t> in;
        DeviceBuffer<double> probs;
        DeviceBuffer<std::uint32_t> counts;
        DeviceBuffer<double> scores;
        DeviceBuffer<std::uint8_t> lane_err;
        DeviceBuffer<std::uint8_t> ops;
        DeviceBuffer<std::uint8_t> imm;
        DeviceBuffer<std::uint8_t> slots;
        std::size_t C = 0;
        std::size_t T = 0;
        std::uint32_t op_count = 0;
        std::uint16_t slot_count = 0;
        std::uint16_t cipher_slot = 0;
        std::uint16_t index_slot = 0;
        std::uint8_t binds_index_i = 0;
        std::uint16_t max_stack = 0;
    };

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

    [[nodiscard]] static StatusOr<TheoryIr> make_caesar_theory() {
        StatusOr<ParamIr> shift_p = ParamIr::make("shift", 0, 28);
        if (!shift_p.ok()) {
            return shift_p.status();
        }
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr shift = Z29Expr::var("shift");
        return TheoryIr::make("bench_theory_caesar", TheoryIr::Family::Elementwise,
                              TheoryIr::Tier::A, TheoryIr::InterruptMode::ElementwiseDefault,
                              {shift_p.value()}, Z29Expr::add(x, shift), Z29Expr::sub(x, shift));
    }

    [[nodiscard]] static StatusOr<TheoryIr> make_progressive_theory() {
        StatusOr<ParamIr> b0 = ParamIr::make("b0", 0, 28);
        if (!b0.ok()) {
            return b0.status();
        }
        StatusOr<ParamIr> b1 = ParamIr::make("b1", 0, 28);
        if (!b1.ok()) {
            return b1.status();
        }
        const Z29Expr::Ptr x = Z29Expr::var("x");
        const Z29Expr::Ptr i = Z29Expr::var("i");
        const Z29Expr::Ptr s =
            Z29Expr::add(Z29Expr::var("b0"), Z29Expr::mul(Z29Expr::var("b1"), i));
        return TheoryIr::make("bench_theory_progressive", TheoryIr::Family::KeyedStream,
                              TheoryIr::Tier::B, TheoryIr::InterruptMode::NoneByDesign,
                              {b0.value(), b1.value()}, Z29Expr::add(x, s), Z29Expr::sub(x, s),
                              std::string("BenchTheorySuite progressive."));
    }

    [[nodiscard]] static StatusOr<PackedProgram>
    pack_grid(const Z29Bytecode::Program& prog, const TheoryIr& theory,
              const std::vector<nlohmann::json>& params_list) {
        PackedProgram out;
        out.C = params_list.size();
        if (out.C == 0 || out.C > TheoryChi2Batch::kMaxCandidates) {
            return Status::error("BenchTheorySuite: bad candidate count");
        }
        out.slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
        out.cipher_slot = prog.cipher_slot;
        out.index_slot = prog.index_slot;
        out.binds_index_i = prog.binds_index_i ? 1u : 0u;
        out.max_stack = prog.max_stack == 0 ? 8 : prog.max_stack;
        if (out.max_stack > TheoryChi2Batch::kMaxDeviceStack) {
            return Status::error("BenchTheorySuite: max_stack exceeds device cap");
        }
        if (out.slot_count > TheoryChi2Batch::kMaxSlots) {
            return Status::error("BenchTheorySuite: slot_count exceeds device cap");
        }
        if (prog.ops.size() > TheoryChi2Batch::kMaxProgramOps) {
            return Status::error("BenchTheorySuite: program too large");
        }

        out.ops.reserve(prog.ops.size());
        for (Z29Bytecode::Op op : prog.ops) {
            out.ops.push_back(Z29Bytecode::op_as_u8(op));
        }
        out.imm = prog.imm;
        out.slots.assign(out.C * out.slot_count, 0);
        for (std::size_t c = 0; c < out.C; ++c) {
            StatusOr<std::vector<Index29>> bound =
                Z29Bytecode::bind_theory_slots(prog, theory, params_list[c]);
            if (!bound.ok()) {
                return bound.status();
            }
            if (bound.value().size() != out.slot_count) {
                return Status::error("BenchTheorySuite: slot bind size mismatch");
            }
            for (std::uint16_t s = 0; s < out.slot_count; ++s) {
                out.slots[c * out.slot_count + s] = bound.value()[s].value();
            }
        }
        return out;
    }

    [[nodiscard]] static StatusOr<Scratch> make_scratch(std::span<const std::uint8_t> host_in,
                                                        const ExpectedFrequencyTable& freqs,
                                                        const PackedProgram& packed) {
        if (host_in.empty() || host_in.size() > TheoryChi2Batch::kMaxTokens) {
            return Status::error("BenchTheorySuite: bad token count");
        }
        Scratch s;
        s.C = packed.C;
        s.T = host_in.size();
        s.op_count = static_cast<std::uint32_t>(packed.ops.size());
        s.slot_count = packed.slot_count;
        s.cipher_slot = packed.cipher_slot;
        s.index_slot = packed.index_slot;
        s.binds_index_i = packed.binds_index_i;
        s.max_stack = packed.max_stack;

        StatusOr<DeviceBuffer<std::uint8_t>> in = DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!in.ok()) {
            return in.status();
        }
        s.in = std::move(in.value());

        StatusOr<DeviceBuffer<double>> probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!probs.ok()) {
            return probs.status();
        }
        s.probs = std::move(probs.value());

        StatusOr<DeviceBuffer<std::uint32_t>> counts =
            DeviceBuffer<std::uint32_t>::allocate(s.C * TheoryChi2Batch::alphabet_size);
        if (!counts.ok()) {
            return counts.status();
        }
        s.counts = std::move(counts.value());

        StatusOr<DeviceBuffer<double>> scores = DeviceBuffer<double>::allocate(s.C);
        if (!scores.ok()) {
            return scores.status();
        }
        s.scores = std::move(scores.value());

        StatusOr<DeviceBuffer<std::uint8_t>> lane_err = DeviceBuffer<std::uint8_t>::allocate(s.C);
        if (!lane_err.ok()) {
            return lane_err.status();
        }
        s.lane_err = std::move(lane_err.value());

        StatusOr<DeviceBuffer<std::uint8_t>> ops = DeviceBuffer<std::uint8_t>::from_host(packed.ops);
        if (!ops.ok()) {
            return ops.status();
        }
        s.ops = std::move(ops.value());

        StatusOr<DeviceBuffer<std::uint8_t>> imm = DeviceBuffer<std::uint8_t>::from_host(packed.imm);
        if (!imm.ok()) {
            return imm.status();
        }
        s.imm = std::move(imm.value());

        StatusOr<DeviceBuffer<std::uint8_t>> slots =
            DeviceBuffer<std::uint8_t>::from_host(packed.slots);
        if (!slots.ok()) {
            return slots.status();
        }
        s.slots = std::move(slots.value());
        return s;
    }

    [[nodiscard]] static BenchReport::Row
    row_from_sample(std::string name, std::string workload, const BenchMetric::Sample& sample,
                    std::size_t C, std::size_t T, std::size_t reps, std::string detail) {
        const bool pass = sample.runes_per_sec() > 0.0;
        return make_measured_row(std::move(name), std::move(workload), sample.runes_per_sec(),
                                 sample.keys_per_sec(), sample.wall_seconds(), pass, C, T, reps,
                                 std::move(detail));
    }

    [[nodiscard]] static std::vector<nlohmann::json> caesar_params(std::size_t C) {
        std::vector<nlohmann::json> params;
        params.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            params.push_back(nlohmann::json{{"shift", static_cast<int>(c % 29)}});
        }
        return params;
    }

    [[nodiscard]] static std::vector<nlohmann::json> progressive_params(std::size_t C) {
        std::vector<nlohmann::json> params;
        params.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            params.push_back(nlohmann::json{{"b0", static_cast<int>(c % 29)},
                                            {"b1", static_cast<int>((c / 29) % 29)}});
        }
        return params;
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_caesar_bytecode(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                        std::size_t reps, std::string name, std::string workload,
                        std::string detail) {
        StatusOr<TheoryIr> theory = make_caesar_theory();
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory.value(), TransformDirection::Decrypt);
        if (!prog.ok()) {
            return prog.status();
        }
        StatusOr<PackedProgram> packed = pack_grid(prog.value(), theory.value(), caesar_params(C));
        if (!packed.ok()) {
            return packed.status();
        }
        const auto host_in = random_stream(T, 0x71EFu);
        StatusOr<Scratch> scratch = make_scratch(host_in, freqs, packed.value());
        if (!scratch.ok()) {
            return scratch.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return TheoryChi2Batch::launch_async(
                scratch.value().in.data(), scratch.value().ops.data(), scratch.value().imm.data(),
                scratch.value().op_count, scratch.value().slots.data(), scratch.value().slot_count,
                scratch.value().cipher_slot, scratch.value().index_slot,
                scratch.value().binds_index_i, scratch.value().max_stack,
                scratch.value().probs.data(), scratch.value().counts.data(),
                scratch.value().scores.data(), scratch.value().lane_err.data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               std::move(detail));
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_progressive(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                    std::size_t reps, std::string name, std::string workload, std::string detail) {
        StatusOr<TheoryIr> theory = make_progressive_theory();
        if (!theory.ok()) {
            return theory.status();
        }
        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory.value(), TransformDirection::Decrypt);
        if (!prog.ok()) {
            return prog.status();
        }
        StatusOr<PackedProgram> packed =
            pack_grid(prog.value(), theory.value(), progressive_params(C));
        if (!packed.ok()) {
            return packed.status();
        }
        const auto host_in = random_stream(T, 0xA11Au);
        StatusOr<Scratch> scratch = make_scratch(host_in, freqs, packed.value());
        if (!scratch.ok()) {
            return scratch.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return TheoryChi2Batch::launch_async(
                scratch.value().in.data(), scratch.value().ops.data(), scratch.value().imm.data(),
                scratch.value().op_count, scratch.value().slots.data(), scratch.value().slot_count,
                scratch.value().cipher_slot, scratch.value().index_slot,
                scratch.value().binds_index_i, scratch.value().max_stack,
                scratch.value().probs.data(), scratch.value().counts.data(),
                scratch.value().scores.data(), scratch.value().lane_err.data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               std::move(detail));
    }

    [[nodiscard]] static StatusOr<BenchReport::Row>
    run_caesar_catalog(const ExpectedFrequencyTable& freqs, std::size_t C, std::size_t T,
                       std::size_t reps, std::string name, std::string workload,
                       std::string detail) {
        if (C > CaesarChi2Batch::kMaxCandidates) {
            return Status::error("BenchTheorySuite: C exceeds CaesarChi2Batch cap");
        }
        const auto host_in = random_stream(T, 0x71EFu);
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
            DeviceBuffer<std::uint32_t>::allocate(C * CaesarChi2Batch::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c % 29);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }

        StatusOr<BenchMetric::Sample> sample = BenchTimer::time_cuda(reps, C, T, [&]() {
            return CaesarChi2Batch::launch_decrypt_async(
                device_in.value().data(), device_shifts.value().data(), device_probs.value().data(),
                device_counts.value().data(), device_scores.value().data(), C, T);
        });
        if (!sample.ok()) {
            return sample.status();
        }
        return row_from_sample(std::move(name), std::move(workload), sample.value(), C, T, reps,
                               std::move(detail));
    }
#endif
};

#endif // BENCH_THEORY_SUITE_HPP
