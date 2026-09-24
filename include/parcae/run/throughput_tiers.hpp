#ifndef THROUGHPUT_TIERS_HPP
#define THROUGHPUT_TIERS_HPP

#if !defined(PARCAE_HAS_CUDA)
#error "throughput_tiers.hpp requires PARCAE_HAS_CUDA"
#endif

#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "params.hpp"
#include "atbash_kernel.hpp"

#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

/// Gated CUDA throughput tiers matching the search-stack SLO targets.
class ThroughputTiers {
public:
    struct TierResult {
        std::string name;
        std::string workload;
        double runes_per_sec = 0.0;
        double target_min = 0.0;
        double target_max = 0.0;  // 0 = no upper bound
        bool pass = false;
        std::size_t candidates = 0;
        std::size_t tokens = 0;
        std::size_t repeats = 0;
    };

    struct Report {
        std::vector<TierResult> tiers;
        bool all_pass = false;
    };

    /// Run SLO tiers. When `extended` is true (default), also includes F.* and
    /// C.* rows (historical `parcae-throughput-tiers` behavior). When false,
    /// only primary T1–T3 (`BenchSloSuite` default).
    [[nodiscard]] static StatusOr<Report> run(
        const ExpectedFrequencyTable& freqs, bool extended = true) {
        Report report;
        StatusOr<TierResult> t1 = tier1_simple_sub(freqs);
        if (!t1.ok()) {
            return t1.status();
        }
        report.tiers.push_back(std::move(t1.value()));

        StatusOr<TierResult> t2 = tier2_filtered_multikey(freqs);
        if (!t2.ok()) {
            return t2.status();
        }
        report.tiers.push_back(std::move(t2.value()));

        StatusOr<TierResult> t3 = tier3_ngram_dict();
        if (!t3.ok()) {
            return t3.status();
        }
        report.tiers.push_back(std::move(t3.value()));

        if (extended) {
            StatusOr<std::vector<TierResult>> families = family_suite(freqs);
            if (!families.ok()) {
                return families.status();
            }
            for (TierResult& f : families.value()) {
                report.tiers.push_back(std::move(f));
            }

            StatusOr<std::vector<TierResult>> compose = compose_suite(freqs);
            if (!compose.ok()) {
                return compose.status();
            }
            for (TierResult& c : compose.value()) {
                report.tiers.push_back(std::move(c));
            }
        }

        report.all_pass = true;
        for (const TierResult& t : report.tiers) {
            if (!t.pass) {
                report.all_pass = false;
                break;
            }
        }
        return report;
    }

    [[nodiscard]] static std::string format_rps(double rps) {
        std::ostringstream out;
        out << std::fixed;
        if (rps >= 1.0e9) {
            out << std::setprecision(2) << (rps / 1.0e9) << "B";
        } else if (rps >= 1.0e6) {
            out << std::setprecision(2) << (rps / 1.0e6) << "M";
        } else {
            out << std::setprecision(2) << rps;
        }
        return out.str();
    }

    [[nodiscard]] static std::string format(const Report& report) {
        std::ostringstream out;
        out << "PARCAE — THROUGHPUT TIERS (CUDA fused)\n";
        out << "Metric: repeats x C x T / median-of-3 cudaEvent (setup excluded)\n";
        out << "Peaks: practical ceilings on RTX 5070 Ti (%peak must stay at or under 100)\n\n";

        auto emit_section = [&](std::string_view title, auto&& name_pred) {
            bool any = false;
            for (const TierResult& t : report.tiers) {
                if (name_pred(t.name)) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                return;
            }
            out << title << '\n';
            out << "Tier              Workload                      runes/s     target      "
                   "est.peak   %peak  result\n";
            out << "------------------------------------------------------------------------"
                   "------------------------\n";
            for (const TierResult& t : report.tiers) {
                if (!name_pred(t.name)) {
                    continue;
                }
                const double peak = estimated_peak(t.name);
                const double pct = peak > 0.0 ? (100.0 * t.runes_per_sec / peak) : 0.0;
                std::ostringstream target;
                if (t.target_max > 0.0) {
                    target << format_rps(t.target_min) << "-" << format_rps(t.target_max);
                } else {
                    target << ">=" << format_rps(t.target_min);
                }
                out << std::left << std::setw(17) << t.name << " " << std::setw(27) << t.workload
                    << " " << std::right << std::setw(10) << format_rps(t.runes_per_sec)
                    << "  " << std::left << std::setw(11) << target.str()
                    << "  " << std::right << std::setw(8) << format_rps(peak)
                    << "  " << std::setw(5) << std::fixed << std::setprecision(0) << pct << "%"
                    << "  " << (t.pass ? "PASS" : "FAIL") << '\n';
                out << "      C=" << t.candidates << " T=" << t.tokens << " reps=" << t.repeats
                    << '\n';
            }
            out << "------------------------------------------------------------------------"
                   "------------------------\n\n";
        };

        emit_section("SLO TIERS", [](const std::string& n) {
            return n == "T1" || n == "T2" || n == "T3";
        });
        emit_section("TRANSFORM FAMILIES", [](const std::string& n) {
            return n.size() >= 2 && n[0] == 'F' && n[1] == '.';
        });
        emit_section("COMPOSE", [](const std::string& n) {
            return n.size() >= 2 && n[0] == 'C' && n[1] == '.';
        });

        out << (report.all_pass ? "ALL TIERS PASS\n" : "TIERS FAILED\n");
        return out.str();
    }

    /// Practical **ceilings** on RTX 5070 Ti — delegated to `BenchTierSpec`
    /// (docs/architecture/cuda-throughput.md). Recalibrate there, never clamp %.
    [[nodiscard]] static double estimated_peak(const std::string& tier) {
        return BenchTierSpec::estimated_peak(tier);
    }

    /// Pass: SLO floor and ≥90% of the practical ceiling. Delegates to
    /// `BenchTierSpec::pass_tier`. First idle-GPU run can miss; re-run warm.
    [[nodiscard]] static bool pass_tier(double rps, double slo_min, double peak) {
        return BenchTierSpec::pass_tier(rps, slo_min, peak);
    }

private:
    ThroughputTiers() = delete;

    struct Scratch {
        DeviceBuffer<std::uint8_t> in;
        DeviceBuffer<double> probs;
        DeviceBuffer<std::uint32_t> counts;
        DeviceBuffer<double> scores;
        std::size_t C = 0;
        std::size_t T = 0;
    };

    [[nodiscard]] static std::vector<std::uint8_t> random_stream(
        std::size_t n, std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 28);
        std::vector<std::uint8_t> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<std::uint8_t>(dist(rng));
        }
        return out;
    }

    [[nodiscard]] static StatusOr<Scratch> make_scratch(
        std::span<const std::uint8_t> host_in,
        const ExpectedFrequencyTable& freqs,
        std::size_t C) {
        Scratch s;
        s.C = C;
        s.T = host_in.size();
        StatusOr<DeviceBuffer<std::uint8_t>> in = DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!in.ok()) {
            return in.status();
        }
        s.in = std::move(in.value());

        StatusOr<DeviceBuffer<double>> dp = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!dp.ok()) {
            return dp.status();
        }
        s.probs = std::move(dp.value());

        StatusOr<DeviceBuffer<std::uint32_t>> counts =
            DeviceBuffer<std::uint32_t>::allocate(C * 29);
        if (!counts.ok()) {
            return counts.status();
        }
        s.counts = std::move(counts.value());

        StatusOr<DeviceBuffer<double>> scores = DeviceBuffer<double>::allocate(C);
        if (!scores.ok()) {
            return scores.status();
        }
        s.scores = std::move(scores.value());
        return s;
    }

    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<double> timed_rps_once(
        Scratch& scratch, std::size_t repeats, LaunchFn&& launch) {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        Status ev0 = CudaError::to_status(cudaEventCreate(&start), "event create start");
        if (!ev0.ok()) {
            return ev0;
        }
        Status ev1 = CudaError::to_status(cudaEventCreate(&stop), "event create stop");
        if (!ev1.ok()) {
            cudaEventDestroy(start);
            return ev1;
        }

        Status rec0 = CudaError::to_status(cudaEventRecord(start, 0), "event record start");
        if (!rec0.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return rec0;
        }
        for (std::size_t r = 0; r < repeats; ++r) {
            Status launched = launch();
            if (!launched.ok()) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return launched;
            }
        }
        Status rec1 = CudaError::to_status(cudaEventRecord(stop, 0), "event record stop");
        if (!rec1.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return rec1;
        }
        Status synced = CudaError::to_status(cudaEventSynchronize(stop), "event sync");
        if (!synced.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return synced;
        }
        float ms = 0.0f;
        Status elapsed =
            CudaError::to_status(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (!elapsed.ok()) {
            return elapsed;
        }
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        const double runes = static_cast<double>(repeats) * static_cast<double>(scratch.C) *
                             static_cast<double>(scratch.T);
        return seconds > 0.0 ? (runes / seconds) : 0.0;
    }

    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<double> timed_rps(
        Scratch& scratch, std::size_t repeats, LaunchFn&& launch) {
        // Extra warmups so clocks / caches settle before the timed window.
        for (int w = 0; w < 4; ++w) {
            Status warm = launch();
            if (!warm.ok()) {
                return warm;
            }
        }
        Status warm_sync = CudaError::to_status(cudaDeviceSynchronize(), "warmup sync");
        if (!warm_sync.ok()) {
            return warm_sync;
        }

        // Median of 3 samples — GPU boost clocks otherwise swing %peak over 100.
        double samples[3]{};
        for (int i = 0; i < 3; ++i) {
            StatusOr<double> one = timed_rps_once(scratch, repeats, launch);
            if (!one.ok()) {
                return one.status();
            }
            samples[i] = one.value();
        }
        if (samples[0] > samples[1]) {
            std::swap(samples[0], samples[1]);
        }
        if (samples[1] > samples[2]) {
            std::swap(samples[1], samples[2]);
        }
        if (samples[0] > samples[1]) {
            std::swap(samples[0], samples[1]);
        }
        return samples[1];
    }

    [[nodiscard]] static bool in_band(double value, double lo, double hi) {
        if (value < lo) {
            return false;
        }
        if (hi > 0.0 && value > hi) {
            // Above band still counts as pass for throughput SLOs (faster is fine).
            return true;
        }
        return true;
    }

    /// Tier 1: simple Caesar fused χ² — config from `BenchTierSpec::t1`.
    [[nodiscard]] static StatusOr<TierResult> tier1_simple_sub(
        const ExpectedFrequencyTable& freqs) {
        constexpr std::size_t C = BenchTierSpec::t1.candidates;
        constexpr std::size_t T = BenchTierSpec::t1.tokens;
        constexpr std::size_t reps = BenchTierSpec::t1.repeats;
        static_assert(C == Index29::modulus);
        const auto host_in = random_stream(T, 0x71EFu);
        StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }

        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!d_shifts.ok()) {
            return d_shifts.status();
        }

        StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
            return CaesarChi2Batch::launch_decrypt_async(
                scratch.value().in.data(),
                d_shifts.value().data(),
                scratch.value().probs.data(),
                scratch.value().counts.data(),
                scratch.value().scores.data(),
                scratch.value().C,
                scratch.value().T);
        });
        if (!rps.ok()) {
            return rps.status();
        }

        TierResult out;
        out.name = BenchTierSpec::t1.id;
        out.workload = BenchTierSpec::t1.workload;
        out.runes_per_sec = rps.value();
        out.target_min = BenchTierSpec::t1.slo_min;
        out.target_max = BenchTierSpec::t1.slo_max;
        out.pass = pass_tier(out.runes_per_sec, out.target_min, estimated_peak(out.name));
        out.candidates = scratch.value().C;
        out.tokens = scratch.value().T;
        out.repeats = reps;
        return out;
    }

    /// Tier 2: multi-key Vigenère + autokey + dynamic-shift — `BenchTierSpec::t2`.
    [[nodiscard]] static StatusOr<TierResult> tier2_filtered_multikey(
        const ExpectedFrequencyTable& freqs) {
        constexpr std::size_t C = BenchTierSpec::t2.candidates;
        constexpr std::size_t key_len = 8;
        constexpr std::size_t T = BenchTierSpec::t2.tokens;
        constexpr std::size_t reps = BenchTierSpec::t2.repeats;
        const auto host_in = random_stream(T, 0xA11Au);

        // Split wall across three filtered families; report min (bottleneck).
        double worst = 1.0e300;
        std::string label;

        // --- multi-key Vigenère ---
        {
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            std::vector<std::uint8_t> keys(C * key_len);
            std::vector<std::uint32_t> begin(C);
            std::vector<std::uint32_t> len(C, static_cast<std::uint32_t>(key_len));
            for (std::size_t c = 0; c < C; ++c) {
                begin[c] = static_cast<std::uint32_t>(c * key_len);
                for (std::size_t j = 0; j < key_len; ++j) {
                    keys[c * key_len + j] =
                        static_cast<std::uint8_t>((c * 3 + j * 7 + 1) % 29);
                }
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_keys =
                DeviceBuffer<std::uint8_t>::from_host(keys);
            if (!d_keys.ok()) {
                return d_keys.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_begin =
                DeviceBuffer<std::uint32_t>::from_host(begin);
            if (!d_begin.ok()) {
                return d_begin.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_len =
                DeviceBuffer<std::uint32_t>::from_host(len);
            if (!d_len.ok()) {
                return d_len.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return FamilyChi2Batch::launch_vigenere_async(
                    scratch.value().in.data(),
                    d_keys.value().data(),
                    d_begin.value().data(),
                    d_len.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            if (rps.value() < worst) {
                worst = rps.value();
                label = "multi-key Vigenere chi2";
            }
        }

        // --- autokey ---
        {
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            std::vector<std::uint8_t> keys(C * key_len);
            std::vector<std::uint32_t> begin(C);
            std::vector<std::uint32_t> len(C, static_cast<std::uint32_t>(key_len));
            for (std::size_t c = 0; c < C; ++c) {
                begin[c] = static_cast<std::uint32_t>(c * key_len);
                for (std::size_t j = 0; j < key_len; ++j) {
                    keys[c * key_len + j] =
                        static_cast<std::uint8_t>((c + j * 5 + 2) % 29);
                }
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_keys =
                DeviceBuffer<std::uint8_t>::from_host(keys);
            if (!d_keys.ok()) {
                return d_keys.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_begin =
                DeviceBuffer<std::uint32_t>::from_host(begin);
            if (!d_begin.ok()) {
                return d_begin.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_len =
                DeviceBuffer<std::uint32_t>::from_host(len);
            if (!d_len.ok()) {
                return d_len.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return DeepScoreBatch::launch_autokey_chi2_async(
                    scratch.value().in.data(),
                    d_keys.value().data(),
                    d_begin.value().data(),
                    d_len.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            if (rps.value() < worst) {
                worst = rps.value();
                label = "ciphertext-autokey chi2";
            }
        }

        // --- dynamic shift ---
        {
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            std::vector<std::uint8_t> base(C);
            std::vector<std::uint8_t> step(C);
            for (std::size_t c = 0; c < C; ++c) {
                base[c] = static_cast<std::uint8_t>(c % 29);
                step[c] = static_cast<std::uint8_t>((c % 28) + 1);
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_base =
                DeviceBuffer<std::uint8_t>::from_host(base);
            if (!d_base.ok()) {
                return d_base.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_step =
                DeviceBuffer<std::uint8_t>::from_host(step);
            if (!d_step.ok()) {
                return d_step.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return DeepScoreBatch::launch_dynamic_shift_chi2_async(
                    scratch.value().in.data(),
                    d_base.value().data(),
                    d_step.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            if (rps.value() < worst) {
                worst = rps.value();
                label = "dynamic-shift chi2";
            }
        }

        TierResult out;
        out.name = BenchTierSpec::t2.id;
        out.workload = std::string("Filtered multi-key/autokey/dyn (") + label + " worst)";
        out.runes_per_sec = worst;
        out.target_min = BenchTierSpec::t2.slo_min;
        out.target_max = BenchTierSpec::t2.slo_max;
        out.pass = pass_tier(out.runes_per_sec, out.target_min, estimated_peak(out.name));
        out.candidates = C;
        out.tokens = T;
        out.repeats = reps;
        return out;
    }

    /// Tier 3: deep bigram + dictionary validation — `BenchTierSpec::t3`.
    [[nodiscard]] static StatusOr<TierResult> tier3_ngram_dict() {
        constexpr std::size_t C = BenchTierSpec::t3.candidates;
        constexpr std::size_t T = BenchTierSpec::t3.tokens;
        constexpr std::size_t reps = BenchTierSpec::t3.repeats;
        constexpr std::size_t dict_n = 64;
        const auto host_in = random_stream(T, 0xD1C7u);

        // Synthetic GP-ish bigram LL table.
        std::vector<float> bigram(29 * 29);
        for (std::size_t i = 0; i < bigram.size(); ++i) {
            bigram[i] = -3.5f + 0.01f * static_cast<float>(i % 29);
        }
        StatusOr<DeviceBuffer<float>> d_bigram = DeviceBuffer<float>::from_host(bigram);
        if (!d_bigram.ok()) {
            return d_bigram.status();
        }

        std::vector<std::uint8_t> dict_words(dict_n * DeepScoreBatch::kDictWordLen, 0);
        std::vector<std::uint8_t> dict_lens(dict_n, 0);
        for (std::size_t w = 0; w < dict_n; ++w) {
            const std::uint8_t len = static_cast<std::uint8_t>(3 + (w % 4));
            dict_lens[w] = len;
            for (std::uint8_t i = 0; i < len; ++i) {
                dict_words[w * DeepScoreBatch::kDictWordLen + i] =
                    static_cast<std::uint8_t>((w * 5 + i * 3) % 29);
            }
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_words =
            DeviceBuffer<std::uint8_t>::from_host(dict_words);
        if (!d_words.ok()) {
            return d_words.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_lens =
            DeviceBuffer<std::uint8_t>::from_host(dict_lens);
        if (!d_lens.ok()) {
            return d_lens.status();
        }

        StatusOr<DeviceBuffer<std::uint8_t>> d_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!d_in.ok()) {
            return d_in.status();
        }
        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c % 29);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!d_shifts.ok()) {
            return d_shifts.status();
        }
        StatusOr<DeviceBuffer<double>> d_scores = DeviceBuffer<double>::allocate(C);
        if (!d_scores.ok()) {
            return d_scores.status();
        }

        Scratch scratch;
        scratch.in = std::move(d_in.value());
        scratch.scores = std::move(d_scores.value());
        scratch.C = C;
        scratch.T = T;

        StatusOr<double> rps = timed_rps(scratch, reps, [&]() {
            return DeepScoreBatch::launch_caesar_ngram_dict_async(
                scratch.in.data(),
                d_shifts.value().data(),
                d_bigram.value().data(),
                d_words.value().data(),
                d_lens.value().data(),
                scratch.scores.data(),
                C,
                T,
                dict_n);
        });
        if (!rps.ok()) {
            return rps.status();
        }

        TierResult out;
        out.name = BenchTierSpec::t3.id;
        out.workload = BenchTierSpec::t3.workload;
        out.runes_per_sec = rps.value();
        out.target_min = BenchTierSpec::t3.slo_min;
        out.target_max = BenchTierSpec::t3.slo_max;
        out.pass = pass_tier(out.runes_per_sec, out.target_min, estimated_peak(out.name));
        out.candidates = C;
        out.tokens = T;
        out.repeats = reps;
        return out;
    }

    [[nodiscard]] static StatusOr<TierResult> make_family_result(
        std::string name,
        std::string workload,
        double rps,
        double slo_min,
        std::size_t C,
        std::size_t T,
        std::size_t reps) {
        TierResult out;
        out.name = std::move(name);
        out.workload = std::move(workload);
        out.runes_per_sec = rps;
        out.target_min = slo_min;
        out.target_max = 0.0;
        out.pass = pass_tier(rps, slo_min, estimated_peak(out.name));
        out.candidates = C;
        out.tokens = T;
        out.repeats = reps;
        return out;
    }

    /// Per-family fused χ² gates (search-stack transform catalog).
    [[nodiscard]] static StatusOr<std::vector<TierResult>> family_suite(
        const ExpectedFrequencyTable& freqs) {
        std::vector<TierResult> out;
        out.reserve(5);

        // --- atbash (involutory; pad C for occupancy) ---
        {
            constexpr std::size_t C = 512;
            constexpr std::size_t T = 1u << 20;
            constexpr std::size_t reps = 64;
            const auto host_in = random_stream(T, 0xA7BAu);
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return FamilyChi2Batch::launch_atbash_async(
                    scratch.value().in.data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            StatusOr<TierResult> row = make_family_result(
                "F.atbash",
                "Atbash fused chi2",
                rps.value(),
                BenchTierSpec::slo_floor("F.atbash"),
                C,
                T,
                reps);
            if (!row.ok()) {
                return row.status();
            }
            out.push_back(std::move(row.value()));
        }

        // --- affine a=1..28, b=0..28 ---
        {
            constexpr std::size_t C = 28u * 29u;
            constexpr std::size_t T = 1u << 18;
            constexpr std::size_t reps = 16;
            const auto host_in = random_stream(T, 0xA7BCu);
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            std::vector<std::uint8_t> a(C);
            std::vector<std::uint8_t> b(C);
            std::size_t c = 0;
            for (std::uint8_t ai = 1; ai <= 28; ++ai) {
                for (std::uint8_t bi = 0; bi < 29; ++bi) {
                    a[c] = ai;
                    b[c] = bi;
                    ++c;
                }
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_a = DeviceBuffer<std::uint8_t>::from_host(a);
            if (!d_a.ok()) {
                return d_a.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_b = DeviceBuffer<std::uint8_t>::from_host(b);
            if (!d_b.ok()) {
                return d_b.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return FamilyChi2Batch::launch_affine_async(
                    scratch.value().in.data(),
                    d_a.value().data(),
                    d_b.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            StatusOr<TierResult> row = make_family_result(
                "F.affine",
                "Affine fused chi2 (812)",
                rps.value(),
                BenchTierSpec::slo_floor("F.affine"),
                C,
                T,
                reps);
            if (!row.ok()) {
                return row.status();
            }
            out.push_back(std::move(row.value()));
        }

        // --- vigenere + beaufort (pow2 key len 8) ---
        {
            constexpr std::size_t C = 4096;
            constexpr std::size_t key_len = 8;
            constexpr std::size_t T = 1u << 18;
            constexpr std::size_t reps = 16;
            const auto host_in = random_stream(T, 0xA7BDu);
            std::vector<std::uint8_t> keys(C * key_len);
            std::vector<std::uint32_t> begin(C);
            std::vector<std::uint32_t> len(C, static_cast<std::uint32_t>(key_len));
            for (std::size_t c = 0; c < C; ++c) {
                begin[c] = static_cast<std::uint32_t>(c * key_len);
                for (std::size_t j = 0; j < key_len; ++j) {
                    keys[c * key_len + j] =
                        static_cast<std::uint8_t>((c * 3 + j * 7 + 1) % 29);
                }
            }
            StatusOr<DeviceBuffer<std::uint8_t>> d_keys =
                DeviceBuffer<std::uint8_t>::from_host(keys);
            if (!d_keys.ok()) {
                return d_keys.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_begin =
                DeviceBuffer<std::uint32_t>::from_host(begin);
            if (!d_begin.ok()) {
                return d_begin.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_len =
                DeviceBuffer<std::uint32_t>::from_host(len);
            if (!d_len.ok()) {
                return d_len.status();
            }

            {
                StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
                if (!scratch.ok()) {
                    return scratch.status();
                }
                StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                    return FamilyChi2Batch::launch_vigenere_async(
                        scratch.value().in.data(),
                        d_keys.value().data(),
                        d_begin.value().data(),
                        d_len.value().data(),
                        scratch.value().probs.data(),
                        scratch.value().counts.data(),
                        scratch.value().scores.data(),
                        C,
                        T);
                });
                if (!rps.ok()) {
                    return rps.status();
                }
                StatusOr<TierResult> row = make_family_result(
                    "F.vigenere",
                    "Vigenere fused chi2 (key=8)",
                    rps.value(),
                    BenchTierSpec::slo_floor("F.vigenere"),
                    C,
                    T,
                    reps);
                if (!row.ok()) {
                    return row.status();
                }
                out.push_back(std::move(row.value()));
            }
            {
                StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
                if (!scratch.ok()) {
                    return scratch.status();
                }
                StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                    return FamilyChi2Batch::launch_beaufort_async(
                        scratch.value().in.data(),
                        d_keys.value().data(),
                        d_begin.value().data(),
                        d_len.value().data(),
                        scratch.value().probs.data(),
                        scratch.value().counts.data(),
                        scratch.value().scores.data(),
                        C,
                        T);
                });
                if (!rps.ok()) {
                    return rps.status();
                }
                StatusOr<TierResult> row = make_family_result(
                    "F.beaufort",
                    "Beaufort fused chi2 (key=8)",
                    rps.value(),
                    BenchTierSpec::slo_floor("F.beaufort"),
                    C,
                    T,
                    reps);
                if (!row.ok()) {
                    return row.status();
                }
                out.push_back(std::move(row.value()));
            }
        }

        // --- totient prime−1 stream (shared shifts, C starts at 0) ---
        {
            constexpr std::size_t C = 512;
            constexpr std::size_t T = 1u << 18;
            constexpr std::size_t reps = 16;
            const auto host_in = random_stream(T, 0xA7BEu);
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            std::vector<std::uint8_t> shifts(T);
            for (std::size_t t = 0; t < T; ++t) {
                shifts[t] = static_cast<std::uint8_t>((t * 3 + 5) % 29);
            }
            std::vector<std::uint32_t> begin(C, 0);
            StatusOr<DeviceBuffer<std::uint8_t>> d_shifts =
                DeviceBuffer<std::uint8_t>::from_host(shifts);
            if (!d_shifts.ok()) {
                return d_shifts.status();
            }
            StatusOr<DeviceBuffer<std::uint32_t>> d_begin =
                DeviceBuffer<std::uint32_t>::from_host(begin);
            if (!d_begin.ok()) {
                return d_begin.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return FamilyChi2Batch::launch_totient_async(
                    scratch.value().in.data(),
                    d_shifts.value().data(),
                    d_begin.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            StatusOr<TierResult> row = make_family_result(
                "F.totient",
                "Totient stream fused chi2",
                rps.value(),
                BenchTierSpec::slo_floor("F.totient"),
                C,
                T,
                reps);
            if (!row.ok()) {
                return row.status();
            }
            out.push_back(std::move(row.value()));
        }

        return out;
    }

    /// Compose recipes (catalog `compose` / Koan-1 atbash→caesar+shift).
    [[nodiscard]] static StatusOr<std::vector<TierResult>> compose_suite(
        const ExpectedFrequencyTable& freqs) {
        std::vector<TierResult> out;
        out.reserve(2);

        constexpr std::size_t C = Index29::modulus;
        constexpr std::size_t T = 1u << 20;
        constexpr std::size_t reps = 64;
        const auto host_in = random_stream(T, 0xC0A1u);

        std::vector<std::uint8_t> shifts(C);
        std::vector<std::uint8_t> dirs(C, 1u);  // caesar stage encrypt = +shift
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!d_shifts.ok()) {
            return d_shifts.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> d_dirs =
            DeviceBuffer<std::uint8_t>::from_host(dirs);
        if (!d_dirs.ok()) {
            return d_dirs.status();
        }

        // --- fused Koan-1 χ² ---
        {
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                return FamilyChi2Batch::launch_atbash_caesar_async(
                    scratch.value().in.data(),
                    d_shifts.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            StatusOr<TierResult> row = make_family_result(
                "C.koan1_fused",
                "Atbash→Caesar+shift fused",
                rps.value(),
                BenchTierSpec::slo_floor("C.koan1_fused"),
                C,
                T,
                reps);
            if (!row.ok()) {
                return row.status();
            }
            out.push_back(std::move(row.value()));
        }

        // --- staged: atbash kernel then caesar χ² (encrypt dirs) ---
        {
            StatusOr<Scratch> scratch = make_scratch(host_in, freqs, C);
            if (!scratch.ok()) {
                return scratch.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> mid =
                DeviceBuffer<std::uint8_t>::allocate(T);
            if (!mid.ok()) {
                return mid.status();
            }
            StatusOr<double> rps = timed_rps(scratch.value(), reps, [&]() {
                Status atb = AtbashKernel::launch_device_async(
                    scratch.value().in.data(), mid.value().data(), T);
                if (!atb.ok()) {
                    return atb;
                }
                return CaesarChi2Batch::launch_async(
                    mid.value().data(),
                    d_shifts.value().data(),
                    d_dirs.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    T);
            });
            if (!rps.ok()) {
                return rps.status();
            }
            StatusOr<TierResult> row = make_family_result(
                "C.koan1_stages",
                "Atbash kern + Caesar chi2",
                rps.value(),
                BenchTierSpec::slo_floor("C.koan1_stages"),
                C,
                T,
                reps);
            if (!row.ok()) {
                return row.status();
            }
            out.push_back(std::move(row.value()));
        }

        return out;
    }
};

#endif  // THROUGHPUT_TIERS_HPP
