#ifndef SEARCH_RUN_HPP
#define SEARCH_RUN_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/sha256.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/run/search_run_metrics.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_batch_kernel.hpp"
#include "chi2_english_gp_score.hpp"
#include "device_buffer.hpp"
#include "params.hpp"
#endif

/// End-to-end search-run tracker: throughput + sweep scores + locked-fixture eval.
///
/// Timing window covers transform+score on device (when CUDA). Setup/init and
/// the one-time H2D of cipher/params are excluded; only score histograms (29
/// bins/lane) cross PCIe in the timed path — not the full C*T plaintext.
class SearchRun {
public:
    class Options {
    public:
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu;
        std::string family = "caesar";           // caesar (v0)
        std::string score_id = "chi2_english_gp_v0";
        std::uint32_t seed = 0;                  // 0 → random_device
        std::size_t stream_length = 4096;        // synthetic cipher length for sweep
        std::size_t throughput_repeats = 8;      // timed loops after setup
        bool compare_cpu_cuda = true;            // when backend is cuda
    };

    [[nodiscard]] static StatusOr<SearchRunMetrics> run(
        const parcae::tool::Context& ctx,
        Options options = {}) {
        Status usable = parcae::tool::BackendUtil::ensure_usable(options.backend);
        if (!usable.ok()) {
            return usable;
        }
        if (options.family != "caesar") {
            return Status::error("SearchRun: only family=caesar is supported in v0");
        }
        if (options.stream_length == 0 || options.stream_length > 4096) {
            return Status::error("SearchRun: stream_length must be in 1..4096");
        }
        if (options.throughput_repeats == 0) {
            return Status::error("SearchRun: throughput_repeats must be >= 1");
        }

        std::uint32_t seed = options.seed;
        if (seed == 0) {
            seed = std::random_device{}();
        }

        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }

        // Synthetic plaintext → encrypt with known shift, then sweep decrypt.
        constexpr std::uint8_t kTrueShift = 11;
        const std::vector<Index29> plain = lcg_indices(options.stream_length, seed);
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain,
            nlohmann::json{{"shift", static_cast<int>(kTrueShift)}},
            TransformDirection::Encrypt);
        if (!cipher.ok()) {
            return cipher.status();
        }

        SearchRunMetrics metrics;
        metrics.set_seed(seed);
        metrics.set_transform_id("caesar");
        metrics.set_parameters_label("shift 0-28");
        metrics.set_score_id(options.score_id);
        metrics.set_backend(std::string(parcae::tool::BackendUtil::to_string(options.backend)));

        StatusOr<SweepResult> sweep = run_caesar_sweep(
            cipher.value(), options, freqs.value());
        if (!sweep.ok()) {
            return sweep.status();
        }
        metrics.set_tok_per_sec(sweep.value().tok_per_sec);
        metrics.set_score_mean(sweep.value().score_mean);
        metrics.set_score_std(sweep.value().score_std);
        metrics.set_steps(std::move(sweep.value().steps));

        StatusOr<EvalResult> eval = run_fixture_eval(ctx, options.score_id, freqs.value(), seed);
        if (!eval.ok()) {
            return eval.status();
        }
        metrics.set_eval(eval.value().passed, eval.value().total);

        if (options.compare_cpu_cuda && options.backend == parcae::tool::Backend::Cuda) {
            StatusOr<bool> parity = compare_cpu_cuda_caesar(
                cipher.value(), freqs.value(), options.score_id);
            if (!parity.ok()) {
                return parity.status();
            }
            metrics.set_cpu_cuda_pass(parity.value());
        }

        return metrics;
    }

private:
    SearchRun() = delete;

    struct SweepResult {
        double tok_per_sec = 0.0;
        double score_mean = 0.0;
        double score_std = 0.0;
        std::vector<SearchRunStep> steps;
    };

    struct EvalResult {
        std::size_t passed = 0;
        std::size_t total = 0;
    };

    [[nodiscard]] static std::vector<Index29> lcg_indices(std::size_t n, std::uint64_t seed) {
        std::vector<Index29> out;
        out.reserve(n);
        std::uint64_t state = seed;
        for (std::size_t i = 0; i < n; ++i) {
            state = state * 6364136223846793005ULL + 1ULL;
            out.push_back(Index29{static_cast<std::uint8_t>((state >> 33) % 29)});
        }
        return out;
    }

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

    [[nodiscard]] static StatusOr<SweepResult> run_caesar_sweep(
        std::span<const Index29> cipher,
        const Options& options,
        const ExpectedFrequencyTable& freqs) {
        ScoreRequest request;
        request.expected_frequencies = &freqs;

        if (options.backend == parcae::tool::Backend::Cuda) {
#if defined(PARCAE_HAS_CUDA)
            return run_caesar_sweep_cuda(cipher, options, request);
#else
            return Status::error("SearchRun: CUDA not built");
#endif
        }
        return run_caesar_sweep_cpu(cipher, options, request);
    }

    [[nodiscard]] static StatusOr<SweepResult> run_caesar_sweep_cpu(
        std::span<const Index29> cipher,
        const Options& options,
        const ScoreRequest& request) {
        // Timed: transform + score for all shifts (E2E; setup excluded).
        std::vector<double> scores(Index29::modulus, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < options.throughput_repeats; ++rep) {
            for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
                StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
                    cipher,
                    nlohmann::json{{"shift", static_cast<int>(shift)}},
                    TransformDirection::Decrypt);
                if (!out.ok()) {
                    return out.status();
                }
                StatusOr<double> scored = ScoreRegistry::score(
                    options.score_id, out.value(), "v0", nlohmann::json::object(), request);
                if (!scored.ok()) {
                    return scored.status();
                }
                scores[shift] = scored.value();
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double e2e_seconds = std::chrono::duration<double>(t1 - t0).count();

        const double runes =
            static_cast<double>(options.throughput_repeats) *
            static_cast<double>(Index29::modulus) *
            static_cast<double>(cipher.size());
        SweepResult result;
        result.tok_per_sec = e2e_seconds > 0.0 ? (runes / e2e_seconds) : 0.0;
        result.score_mean = mean_of(scores);
        result.score_std = stddev_of(scores, result.score_mean);
        result.steps.reserve(Index29::modulus);
        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            const nlohmann::json params = {{"shift", static_cast<int>(shift)}};
            result.steps.emplace_back(
                static_cast<std::size_t>(shift),
                "caesar",
                hash_params(params),
                scores[shift]);
        }
        return result;
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static StatusOr<SweepResult> run_caesar_sweep_cuda(
        std::span<const Index29> cipher,
        const Options& options,
        const ScoreRequest& request) {
        if (options.score_id != "chi2_english_gp_v0") {
            return Status::error(
                "SearchRun CUDA resident path requires score_id=chi2_english_gp_v0");
        }
        if (request.expected_frequencies == nullptr) {
            return Status::error("SearchRun CUDA requires expected_frequencies");
        }

        const std::size_t C = Index29::modulus;
        const std::size_t T = cipher.size();
        std::vector<std::uint8_t> host_in(T);
        for (std::size_t i = 0; i < T; ++i) {
            host_in[i] = cipher[i].value();
        }
        std::vector<std::uint8_t> host_shifts(C);
        std::vector<std::uint8_t> host_dirs(C, static_cast<std::uint8_t>(CudaDir::Decrypt));
        for (std::size_t c = 0; c < C; ++c) {
            host_shifts[c] = static_cast<std::uint8_t>(c);
        }

        // Setup OUTSIDE the timed window: one H2D of inputs + device allocs.
        StatusOr<DeviceBuffer<std::uint8_t>> device_in =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device_in.ok()) {
            return device_in.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(host_shifts);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_dirs =
            DeviceBuffer<std::uint8_t>::from_host(host_dirs);
        if (!device_dirs.ok()) {
            return device_dirs.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_out =
            DeviceBuffer<std::uint8_t>::allocate(C * T);
        if (!device_out.ok()) {
            return device_out.status();
        }
        StatusOr<DeviceBuffer<unsigned long long>> device_counts =
            DeviceBuffer<unsigned long long>::allocate(Chi2EnglishGpScore::alphabet_size);
        if (!device_counts.ok()) {
            return device_counts.status();
        }

        const std::span<const double> probs(
            request.expected_frequencies->probabilities().data(),
            request.expected_frequencies->probabilities().size());

        // Timed: transform stays on device; score only D2Hs 29-bin histograms
        // (not C*T plaintext). Cipher/params already resident.
        std::vector<double> scores(C, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < options.throughput_repeats; ++rep) {
            Status launched = CaesarBatchKernel::launch_device(
                device_in.value().data(),
                device_shifts.value().data(),
                device_dirs.value().data(),
                device_out.value().data(),
                C,
                T);
            if (!launched.ok()) {
                return launched;
            }
            for (std::size_t c = 0; c < C; ++c) {
                const std::uint8_t* lane = device_out.value().data() + (c * T);
                StatusOr<double> scored = Chi2EnglishGpScore::score_device(
                    lane, T, probs, device_counts.value().data());
                if (!scored.ok()) {
                    return scored.status();
                }
                scores[c] = scored.value();
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double runes =
            static_cast<double>(options.throughput_repeats) *
            static_cast<double>(C) *
            static_cast<double>(T);

        SweepResult result;
        result.tok_per_sec = seconds > 0.0 ? (runes / seconds) : 0.0;
        result.score_mean = mean_of(scores);
        result.score_std = stddev_of(scores, result.score_mean);
        result.steps.reserve(C);
        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            const nlohmann::json params = {{"shift", static_cast<int>(shift)}};
            result.steps.emplace_back(
                static_cast<std::size_t>(shift),
                "caesar",
                hash_params(params),
                scores[shift]);
        }
        return result;
    }

    [[nodiscard]] static StatusOr<bool> compare_cpu_cuda_caesar(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::string_view score_id) {
        ScoreRequest request;
        request.expected_frequencies = &freqs;

        Options cpu_opts;
        cpu_opts.backend = parcae::tool::Backend::Cpu;
        cpu_opts.score_id = std::string(score_id);
        cpu_opts.throughput_repeats = 1;
        cpu_opts.stream_length = cipher.size();

        Options cuda_opts = cpu_opts;
        cuda_opts.backend = parcae::tool::Backend::Cuda;

        StatusOr<SweepResult> cpu = run_caesar_sweep_cpu(cipher, cpu_opts, request);
        if (!cpu.ok()) {
            return cpu.status();
        }
        StatusOr<SweepResult> cuda = run_caesar_sweep_cuda(cipher, cuda_opts, request);
        if (!cuda.ok()) {
            return cuda.status();
        }
        if (cpu.value().steps.size() != cuda.value().steps.size()) {
            return false;
        }
        for (std::size_t i = 0; i < cpu.value().steps.size(); ++i) {
            if (cpu.value().steps[i].score() != cuda.value().steps[i].score()) {
                return false;
            }
            if (cpu.value().steps[i].param_hash() != cuda.value().steps[i].param_hash()) {
                return false;
            }
        }
        return true;
    }
#else
    [[nodiscard]] static StatusOr<bool> compare_cpu_cuda_caesar(
        std::span<const Index29>,
        const ExpectedFrequencyTable&,
        std::string_view) {
        return Status::error("SearchRun: CUDA not built");
    }
#endif

    [[nodiscard]] static StatusOr<EvalResult> run_fixture_eval(
        const parcae::tool::Context& ctx,
        std::string_view score_id,
        const ExpectedFrequencyTable& freqs,
        std::uint32_t seed) {
        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            return profile.status();
        }
        const LatinCodec codec(profile.value());
        const PlaintextNormalizer normalizer(codec);

        const std::filesystem::path solved = ctx.data_root() / "fixtures" / "solved";
        if (!std::filesystem::is_directory(solved)) {
            return Status::error("SearchRun: fixtures/solved missing");
        }

        ScoreRequest request;
        request.expected_frequencies = &freqs;

        EvalResult eval;
        for (const std::filesystem::directory_entry& entry :
             std::filesystem::directory_iterator(solved)) {
            if (!entry.is_directory()) {
                continue;
            }
            StatusOr<Fixture> fixture = FixtureLoader::load_directory(entry.path().string());
            if (!fixture.ok()) {
                continue;
            }
            if (fixture.value().verification_status() != "locked") {
                continue;
            }

            StatusOr<std::string> normalized = normalizer.normalize(fixture.value().plaintext());
            if (!normalized.ok()) {
                return normalized.status();
            }
            StatusOr<std::vector<Index29>> plain = codec.delatinize(normalized.value());
            if (!plain.ok()) {
                return plain.status();
            }
            if (plain.value().empty()) {
                continue;
            }

            const std::vector<Index29> noise =
                lcg_indices(plain.value().size(), seed ^ (0x9E3779B9u + eval.total));

            StatusOr<double> plain_score = ScoreRegistry::score(
                score_id, plain.value(), "v0", nlohmann::json::object(), request);
            if (!plain_score.ok()) {
                return plain_score.status();
            }
            StatusOr<double> noise_score = ScoreRegistry::score(
                score_id, noise, "v0", nlohmann::json::object(), request);
            if (!noise_score.ok()) {
                return noise_score.status();
            }

            StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
            if (!order.ok()) {
                return order.status();
            }

            ++eval.total;
            const bool better =
                order.value() == ScoreOrder::Asc
                    ? (plain_score.value() < noise_score.value())
                    : (plain_score.value() > noise_score.value());
            if (better) {
                ++eval.passed;
            }
        }

        if (eval.total == 0) {
            return Status::error("SearchRun: no locked fixtures found for eval");
        }
        return eval;
    }
};

#endif // SEARCH_RUN_HPP
