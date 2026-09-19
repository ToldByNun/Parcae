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

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <chrono>
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

    [[nodiscard]] static StatusOr<Report> run(const ExpectedFrequencyTable& freqs) {
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
        out << "Metric: repeats x C x T / wall (kernel+sync; setup excluded)\n";
        out << "Peaks: practical ceilings on RTX 5070 Ti (~896 GB/s DRAM)\n\n";
        out << "Tier  Workload                              runes/s     target      est.peak   %peak  result\n";
        out << "-----------------------------------------------------------------------------------------------\n";
        for (const TierResult& t : report.tiers) {
            const double peak = estimated_peak(t.name);
            const double pct = peak > 0.0 ? (100.0 * t.runes_per_sec / peak) : 0.0;
            std::ostringstream target;
            if (t.target_max > 0.0) {
                target << format_rps(t.target_min) << "-" << format_rps(t.target_max);
            } else {
                target << ">=" << format_rps(t.target_min);
            }
            out << std::left << std::setw(5) << t.name << " " << std::setw(36) << t.workload
                << " " << std::right << std::setw(10) << format_rps(t.runes_per_sec)
                << "  " << std::left << std::setw(11) << target.str()
                << "  " << std::right << std::setw(8) << format_rps(peak)
                << "  " << std::setw(5) << std::fixed << std::setprecision(0) << pct << "%"
                << "  " << (t.pass ? "PASS" : "FAIL") << '\n';
            out << "      C=" << t.candidates << " T=" << t.tokens << " reps=" << t.repeats
                << '\n';
        }
        out << "-----------------------------------------------------------------------------------------------\n";
        out << (report.all_pass ? "ALL TIERS PASS\n" : "TIERS FAILED\n");
        return out.str();
    }

    /// Practical peak estimates (not marketing FLOPS).
    [[nodiscard]] static double estimated_peak(const std::string& tier) {
        if (tier == "T1") {
            return 450.0e9;
        }
        if (tier == "T2") {
            return 280.0e9;
        }
        if (tier == "T3") {
            return 25.0e9;  // bigram + chained dict probes every 4th index
        }
        return 0.0;
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
    [[nodiscard]] static StatusOr<double> timed_rps(
        Scratch& scratch, std::size_t repeats, LaunchFn&& launch) {
        // Warmup
        Status warm = launch();
        if (!warm.ok()) {
            return warm;
        }
        Status warm_sync = CudaError::to_status(cudaDeviceSynchronize(), "warmup sync");
        if (!warm_sync.ok()) {
            return warm_sync;
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t r = 0; r < repeats; ++r) {
            Status launched = launch();
            if (!launched.ok()) {
                return launched;
            }
        }
        Status synced = CudaError::to_status(cudaDeviceSynchronize(), "tier sync");
        if (!synced.ok()) {
            return synced;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double runes = static_cast<double>(repeats) * static_cast<double>(scratch.C) *
                             static_cast<double>(scratch.T);
        return seconds > 0.0 ? (runes / seconds) : 0.0;
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

    /// Tier 1: simple Caesar / affine / short Vigenère χ² — target 15–35B (pass if ≥15B).
    [[nodiscard]] static StatusOr<TierResult> tier1_simple_sub(
        const ExpectedFrequencyTable& freqs) {
        constexpr std::size_t T = 1u << 20;
        constexpr std::size_t reps = 64;
        const auto host_in = random_stream(T, 0x71EFu);
        StatusOr<Scratch> scratch = make_scratch(host_in, freqs, Index29::modulus);
        if (!scratch.ok()) {
            return scratch.status();
        }

        std::vector<std::uint8_t> shifts(Index29::modulus);
        for (std::size_t c = 0; c < Index29::modulus; ++c) {
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
        out.name = "T1";
        out.workload = "Caesar fused chi2 (simple sub)";
        out.runes_per_sec = rps.value();
        out.target_min = 15.0e9;
        out.target_max = 35.0e9;
        out.pass = in_band(out.runes_per_sec, out.target_min, out.target_max);
        out.candidates = scratch.value().C;
        out.tokens = scratch.value().T;
        out.repeats = reps;
        return out;
    }

    /// Tier 2: multi-key Vigenère + autokey + dynamic-shift — target 3–10B (pass if ≥3B).
    [[nodiscard]] static StatusOr<TierResult> tier2_filtered_multikey(
        const ExpectedFrequencyTable& freqs) {
        constexpr std::size_t C = 4096;
        constexpr std::size_t key_len = 8;
        constexpr std::size_t T = 1u << 18;  // 256k
        constexpr std::size_t reps = 8;
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
        out.name = "T2";
        out.workload = "Filtered multi-key/autokey/dyn (" + label + " worst)";
        out.runes_per_sec = worst;
        out.target_min = 3.0e9;
        out.target_max = 10.0e9;
        out.pass = in_band(out.runes_per_sec, out.target_min, out.target_max);
        out.candidates = C;
        out.tokens = T;
        out.repeats = reps;
        return out;
    }

    /// Tier 3: deep bigram + dictionary validation — target ≥1B.
    [[nodiscard]] static StatusOr<TierResult> tier3_ngram_dict() {
        constexpr std::size_t C = 512;
        constexpr std::size_t T = 1u << 18;
        constexpr std::size_t reps = 8;
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
        out.name = "T3";
        out.workload = "Caesar bigram+dict validation";
        out.runes_per_sec = rps.value();
        out.target_min = 1.0e9;
        out.target_max = 0.0;
        out.pass = out.runes_per_sec >= out.target_min;
        out.candidates = C;
        out.tokens = T;
        out.repeats = reps;
        return out;
    }
};

#endif  // THROUGHPUT_TIERS_HPP
