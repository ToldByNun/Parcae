#ifndef SEARCH_RUN_CUDA_HPP
#define SEARCH_RUN_CUDA_HPP

#if !defined(PARCAE_HAS_CUDA)
#error "search_run_cuda.hpp requires PARCAE_HAS_CUDA"
#endif

#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "params.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/run/search_run_metrics.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>

/// CUDA family sweeps for `SearchRun` (fused decrypt+χ², scores-only D2H).
class SearchRunCuda {
public:
    struct SweepResult {
        double tok_per_sec = 0.0;
        double score_mean = 0.0;
        double score_std = 0.0;
        std::vector<SearchRunStep> steps;
        std::string transform_id;
        std::string parameters_label;
    };

    [[nodiscard]] static StatusOr<SweepResult> run(
        std::string_view family,
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t throughput_repeats) {
        if (family == "caesar") {
            return run_caesar(cipher, freqs, throughput_repeats);
        }
        if (family == "atbash") {
            return run_atbash(cipher, freqs, throughput_repeats);
        }
        if (family == "atbash_caesar") {
            return run_atbash_caesar(cipher, freqs, throughput_repeats);
        }
        if (family == "affine") {
            return run_affine(cipher, freqs, throughput_repeats);
        }
        if (family == "vigenere") {
            return run_vigenere(cipher, freqs, throughput_repeats);
        }
        return Status::error(
            "SearchRun CUDA: unknown family (caesar|atbash|atbash_caesar|affine|vigenere)");
    }

private:
    SearchRunCuda() = delete;

    [[nodiscard]] static std::string hash_params(const nlohmann::json& params) {
        return Sha256::hex_digest(params.dump());
    }

    [[nodiscard]] static double mean_of(std::span<const double> values) {
        if (values.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (double v : values) {
            sum += v;
        }
        return sum / static_cast<double>(values.size());
    }

    [[nodiscard]] static double stddev_of(std::span<const double> values, double mean) {
        if (values.size() < 2) {
            return 0.0;
        }
        double acc = 0.0;
        for (double v : values) {
            const double d = v - mean;
            acc += d * d;
        }
        return std::sqrt(acc / static_cast<double>(values.size()));
    }

    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(std::span<const Index29> cipher) {
        std::vector<std::uint8_t> out(cipher.size());
        for (std::size_t i = 0; i < cipher.size(); ++i) {
            out[i] = cipher[i].value();
        }
        return out;
    }

    struct DeviceScoreScratch {
        DeviceBuffer<std::uint8_t> in;
        DeviceBuffer<double> probs;
        DeviceBuffer<unsigned long long> counts;
        DeviceBuffer<double> scores;
        std::size_t C = 0;
        std::size_t T = 0;
    };

    [[nodiscard]] static StatusOr<DeviceScoreScratch> make_scratch(
        std::span<const std::uint8_t> host_in,
        const ExpectedFrequencyTable& freqs,
        std::size_t C) {
        DeviceScoreScratch s;
        s.C = C;
        s.T = host_in.size();
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
        StatusOr<DeviceBuffer<unsigned long long>> counts =
            DeviceBuffer<unsigned long long>::allocate(C * 29);
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
    [[nodiscard]] static StatusOr<SweepResult> timed_loop(
        DeviceScoreScratch& scratch,
        std::size_t repeats,
        LaunchFn&& launch,
        std::string transform_id,
        std::string parameters_label,
        const std::vector<nlohmann::json>& step_params) {
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < repeats; ++rep) {
            Status launched = launch();
            if (!launched.ok()) {
                return launched;
            }
        }
        Status synced =
            CudaError::to_status(cudaDeviceSynchronize(), "SearchRunCuda::sync");
        if (!synced.ok()) {
            return synced;
        }
        std::vector<double> scores(scratch.C, 0.0);
        Status copied = scratch.scores.copy_to_host(scores);
        if (!copied.ok()) {
            return copied;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double runes =
            static_cast<double>(repeats) * static_cast<double>(scratch.C) *
            static_cast<double>(scratch.T);

        SweepResult result;
        result.tok_per_sec = seconds > 0.0 ? (runes / seconds) : 0.0;
        result.score_mean = mean_of(scores);
        result.score_std = stddev_of(scores, result.score_mean);
        result.transform_id = transform_id;
        result.parameters_label = std::move(parameters_label);
        result.steps.reserve(scratch.C);
        for (std::size_t c = 0; c < scratch.C; ++c) {
            result.steps.emplace_back(
                c,
                transform_id,
                hash_params(step_params[c]),
                scores[c]);
        }
        return result;
    }

    [[nodiscard]] static StatusOr<SweepResult> run_caesar(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t repeats) {
        const std::size_t C = Index29::modulus;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScoreScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        std::vector<std::uint8_t> shifts(C);
        std::vector<std::uint8_t> dirs(C, static_cast<std::uint8_t>(CudaDir::Decrypt));
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_dirs =
            DeviceBuffer<std::uint8_t>::from_host(dirs);
        if (!device_dirs.ok()) {
            return device_dirs.status();
        }

        std::vector<nlohmann::json> params;
        params.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            params.push_back({{"shift", static_cast<int>(c)}});
        }

        return timed_loop(
            scratch.value(),
            repeats,
            [&]() {
                return CaesarChi2Batch::launch_async(
                    scratch.value().in.data(),
                    device_shifts.value().data(),
                    device_dirs.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "caesar",
            "shift 0-28",
            params);
    }

    [[nodiscard]] static StatusOr<SweepResult> run_atbash(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t repeats) {
        constexpr std::size_t C = 1;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScoreScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        std::vector<nlohmann::json> params{nlohmann::json::object()};
        return timed_loop(
            scratch.value(),
            repeats,
            [&]() {
                return FamilyChi2Batch::launch_atbash_async(
                    scratch.value().in.data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "atbash",
            "involutory",
            params);
    }

    [[nodiscard]] static StatusOr<SweepResult> run_atbash_caesar(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t repeats) {
        const std::size_t C = Index29::modulus;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScoreScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        std::vector<nlohmann::json> params;
        params.reserve(C);
        for (std::size_t c = 0; c < C; ++c) {
            params.push_back({{"shift", static_cast<int>(c)}});
        }
        return timed_loop(
            scratch.value(),
            repeats,
            [&]() {
                return FamilyChi2Batch::launch_atbash_caesar_async(
                    scratch.value().in.data(),
                    device_shifts.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "atbash_caesar",
            "atbash then caesar shift 0-28",
            params);
    }

    [[nodiscard]] static StatusOr<SweepResult> run_affine(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t repeats) {
        constexpr std::size_t C = 28u * 29u;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScoreScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        std::vector<std::uint8_t> a(C);
        std::vector<std::uint8_t> b(C);
        std::vector<nlohmann::json> params;
        params.reserve(C);
        std::size_t c = 0;
        for (std::uint8_t ai = 1; ai <= 28; ++ai) {
            for (std::uint8_t bi = 0; bi < 29; ++bi) {
                a[c] = ai;
                b[c] = bi;
                params.push_back({{"a", static_cast<int>(ai)}, {"b", static_cast<int>(bi)}});
                ++c;
            }
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_a = DeviceBuffer<std::uint8_t>::from_host(a);
        if (!device_a.ok()) {
            return device_a.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_b = DeviceBuffer<std::uint8_t>::from_host(b);
        if (!device_b.ok()) {
            return device_b.status();
        }
        return timed_loop(
            scratch.value(),
            repeats,
            [&]() {
                return FamilyChi2Batch::launch_affine_async(
                    scratch.value().in.data(),
                    device_a.value().data(),
                    device_b.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "affine",
            "a=1..28, b=0..28 (812)",
            params);
    }

    [[nodiscard]] static StatusOr<SweepResult> run_vigenere(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t repeats) {
        constexpr std::size_t C = 20;  // key lengths 1..20
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScoreScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }

        // Pack keys: length L key is [1,2,...,L] (mod 29) for L=1..20.
        std::size_t arena = 0;
        for (std::size_t L = 1; L <= C; ++L) {
            arena += L;
        }
        std::vector<std::uint8_t> key_bytes;
        key_bytes.reserve(arena);
        std::vector<std::uint32_t> key_begin(C);
        std::vector<std::uint32_t> key_len(C);
        std::vector<nlohmann::json> params;
        params.reserve(C);
        std::uint32_t cursor = 0;
        for (std::size_t L = 1; L <= C; ++L) {
            key_begin[L - 1] = cursor;
            key_len[L - 1] = static_cast<std::uint32_t>(L);
            params.push_back({{"key_length", static_cast<int>(L)}});
            for (std::size_t j = 0; j < L; ++j) {
                key_bytes.push_back(static_cast<std::uint8_t>((j + 1) % 29));
            }
            cursor += static_cast<std::uint32_t>(L);
        }

        StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
            DeviceBuffer<std::uint8_t>::from_host(key_bytes);
        if (!device_keys.ok()) {
            return device_keys.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
            DeviceBuffer<std::uint32_t>::from_host(key_begin);
        if (!device_begin.ok()) {
            return device_begin.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_len =
            DeviceBuffer<std::uint32_t>::from_host(key_len);
        if (!device_len.ok()) {
            return device_len.status();
        }

        return timed_loop(
            scratch.value(),
            repeats,
            [&]() {
                return FamilyChi2Batch::launch_vigenere_async(
                    scratch.value().in.data(),
                    device_keys.value().data(),
                    device_begin.value().data(),
                    device_len.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "vigenere_key",
            "key length 1-20",
            params);
    }
};

#endif  // SEARCH_RUN_CUDA_HPP
