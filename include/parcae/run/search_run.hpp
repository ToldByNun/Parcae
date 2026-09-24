#ifndef SEARCH_RUN_HPP
#define SEARCH_RUN_HPP

#include "parcae/core/index29.hpp"
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
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "parcae/run/search_run_cuda.hpp"
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
        Backend backend = Backend::Cpu;
        std::string family = "caesar"; // caesar (v0)
        std::string score_id = "chi2_english_gp_v0";
        std::uint32_t seed = 0;             // 0 → random_device
        std::size_t stream_length = 4096;   // synthetic cipher length for sweep
        std::size_t throughput_repeats = 8; // timed loops after setup
        bool compare_cpu_cuda = true;       // when backend is cuda
    };

    [[nodiscard]] static StatusOr<SearchRunMetrics> run(const Context& ctx) {
        return run(ctx, Options{});
    }

    [[nodiscard]] static StatusOr<SearchRunMetrics> run(const Context& ctx, Options options) {
        Status usable = BackendUtil::ensure_usable(options.backend);
        if (!usable.ok()) {
            return usable;
        }
        static const std::string_view kFamilies[] = {"caesar", "atbash", "atbash_caesar", "affine",
                                                     "vigenere"};
        bool family_ok = false;
        for (std::string_view f : kFamilies) {
            if (options.family == f) {
                family_ok = true;
                break;
            }
        }
        if (!family_ok) {
            return Status::error(
                "SearchRun: family must be caesar|atbash|atbash_caesar|affine|vigenere");
        }
        if (options.backend == Backend::Cpu && options.family != "caesar") {
            return Status::error(
                "SearchRun: CPU backend currently supports family=caesar; use --backend cuda");
        }
        if (options.stream_length == 0 || options.stream_length > (1u << 22)) {
            return Status::error("SearchRun: stream_length must be in 1..4194304");
        }
        if (options.throughput_repeats == 0) {
            return Status::error("SearchRun: throughput_repeats must be >= 1");
        }
        if (options.score_id != "chi2_english_gp_v0") {
            return Status::error("SearchRun: score_id must be chi2_english_gp_v0");
        }

        std::uint32_t seed = options.seed;
        if (seed == 0) {
            seed = std::random_device{}();
        }

        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }

        // Synthetic plaintext → encrypt with known Caesar shift (shared input stream).
        constexpr std::uint8_t kTrueShift = 11;
        const std::vector<Index29> plain = lcg_indices(options.stream_length, seed);
        StatusOr<std::vector<Index29>> cipher =
            CaesarTransform{}.apply(plain, nlohmann::json{{"shift", static_cast<int>(kTrueShift)}},
                                    TransformDirection::Encrypt);
        if (!cipher.ok()) {
            return cipher.status();
        }

        SearchRunMetrics metrics;
        metrics.set_seed(seed);
        metrics.set_score_id(options.score_id);
        metrics.set_backend(std::string(BackendUtil::to_string(options.backend)));

        StatusOr<SweepResult> sweep = run_family_sweep(cipher.value(), options, freqs.value());
        if (!sweep.ok()) {
            return sweep.status();
        }
        metrics.set_transform_id(sweep.value().transform_id);
        metrics.set_parameters_label(sweep.value().parameters_label);
        metrics.set_tok_per_sec(sweep.value().tok_per_sec);
        metrics.set_score_mean(sweep.value().score_mean);
        metrics.set_score_std(sweep.value().score_std);
        metrics.set_steps(std::move(sweep.value().steps));

        StatusOr<EvalResult> eval = run_fixture_eval(ctx, options.score_id, freqs.value(), seed);
        if (!eval.ok()) {
            return eval.status();
        }
        metrics.set_eval(eval.value().passed, eval.value().total);

        if (options.compare_cpu_cuda && options.backend == Backend::Cuda) {
            const std::size_t parity_n = std::min<std::size_t>(cipher.value().size(), 256);
            StatusOr<bool> parity = compare_cpu_cuda_family(
                options.family, std::span<const Index29>(cipher.value().data(), parity_n),
                freqs.value());
            if (!parity.ok()) {
                return parity.status();
            }
            metrics.set_cpu_cuda_pass(parity.value());
        }

        return metrics;
    }

    struct EvalResult {
        std::size_t passed = 0;
        std::size_t total = 0;
    };

    [[nodiscard]] static StatusOr<EvalResult> run_fixture_eval(const Context& ctx,
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

            StatusOr<double> plain_score = ScoreRegistry::score(score_id, plain.value(), "v0",
                                                                nlohmann::json::object(), request);
            if (!plain_score.ok()) {
                return plain_score.status();
            }
            StatusOr<double> noise_score =
                ScoreRegistry::score(score_id, noise, "v0", nlohmann::json::object(), request);
            if (!noise_score.ok()) {
                return noise_score.status();
            }

            StatusOr<ScoreOrder> order = ScoreRegistry::order_of(score_id);
            if (!order.ok()) {
                return order.status();
            }

            ++eval.total;
            const bool better = order.value() == ScoreOrder::Asc
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

private:
    SearchRun() = delete;

    struct SweepResult {
        double tok_per_sec = 0.0;
        double score_mean = 0.0;
        double score_std = 0.0;
        std::string transform_id;
        std::string parameters_label;
        std::vector<SearchRunStep> steps;
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

    [[nodiscard]] static StatusOr<SweepResult>
    run_family_sweep(std::span<const Index29> cipher, const Options& options,
                     const ExpectedFrequencyTable& freqs) {
        if (options.backend == Backend::Cuda) {
#if defined(PARCAE_HAS_CUDA)
            StatusOr<SearchRunCuda::SweepResult> cuda =
                SearchRunCuda::run(options.family, cipher, freqs, options.throughput_repeats);
            if (!cuda.ok()) {
                return cuda.status();
            }
            SweepResult result;
            result.tok_per_sec = cuda.value().tok_per_sec;
            result.score_mean = cuda.value().score_mean;
            result.score_std = cuda.value().score_std;
            result.transform_id = std::move(cuda.value().transform_id);
            result.parameters_label = std::move(cuda.value().parameters_label);
            result.steps = std::move(cuda.value().steps);
            return result;
#else
            return Status::error("SearchRun: CUDA not built");
#endif
        }
        return run_caesar_sweep_cpu(cipher, options, freqs);
    }

    [[nodiscard]] static StatusOr<SweepResult>
    run_caesar_sweep_cpu(std::span<const Index29> cipher, const Options& options,
                         const ExpectedFrequencyTable& freqs) {
        ScoreRequest request;
        request.expected_frequencies = &freqs;

        std::vector<double> scores(Index29::modulus, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < options.throughput_repeats; ++rep) {
            for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
                StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
                    cipher, nlohmann::json{{"shift", static_cast<int>(shift)}},
                    TransformDirection::Decrypt);
                if (!out.ok()) {
                    return out.status();
                }
                StatusOr<double> scored = ScoreRegistry::score(options.score_id, out.value(), "v0",
                                                               nlohmann::json::object(), request);
                if (!scored.ok()) {
                    return scored.status();
                }
                scores[shift] = scored.value();
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double e2e_seconds = std::chrono::duration<double>(t1 - t0).count();

        const double runes = static_cast<double>(options.throughput_repeats) *
                             static_cast<double>(Index29::modulus) *
                             static_cast<double>(cipher.size());
        SweepResult result;
        result.tok_per_sec = e2e_seconds > 0.0 ? (runes / e2e_seconds) : 0.0;
        result.score_mean = mean_of(scores);
        result.score_std = stddev_of(scores, result.score_mean);
        result.transform_id = "caesar";
        result.parameters_label = "shift 0-28";
        result.steps.reserve(Index29::modulus);
        for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
            const nlohmann::json params = {{"shift", static_cast<int>(shift)}};
            result.steps.emplace_back(static_cast<std::size_t>(shift), "caesar", params,
                                      scores[shift]);
        }
        return result;
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static StatusOr<std::vector<double>>
    cpu_family_scores(std::string_view family, std::span<const Index29> cipher,
                      const ExpectedFrequencyTable& freqs) {
        ScoreRequest request;
        request.expected_frequencies = &freqs;
        std::vector<double> scores;

        if (family == "caesar") {
            scores.resize(Index29::modulus);
            for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
                StatusOr<std::vector<Index29>> out = CaesarTransform{}.apply(
                    cipher, nlohmann::json{{"shift", static_cast<int>(shift)}},
                    TransformDirection::Decrypt);
                if (!out.ok()) {
                    return out.status();
                }
                StatusOr<double> scored = ScoreRegistry::score(
                    "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
                if (!scored.ok()) {
                    return scored.status();
                }
                scores[shift] = scored.value();
            }
            return scores;
        }
        if (family == "atbash") {
            StatusOr<std::vector<Index29>> out = AtbashTransform{}.apply(
                cipher, nlohmann::json::object(), TransformDirection::Decrypt);
            if (!out.ok()) {
                return out.status();
            }
            StatusOr<double> scored = ScoreRegistry::score("chi2_english_gp_v0", out.value(), "v0",
                                                           nlohmann::json::object(), request);
            if (!scored.ok()) {
                return scored.status();
            }
            return std::vector<double>{scored.value()};
        }
        if (family == "atbash_caesar") {
            scores.resize(Index29::modulus);
            for (std::uint8_t shift = 0; shift < Index29::modulus; ++shift) {
                StatusOr<std::vector<Index29>> out = ComposeTransform::apply_atbash_then_caesar(
                    cipher, shift, TransformDirection::Decrypt);
                if (!out.ok()) {
                    return out.status();
                }
                StatusOr<double> scored = ScoreRegistry::score(
                    "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
                if (!scored.ok()) {
                    return scored.status();
                }
                scores[shift] = scored.value();
            }
            return scores;
        }
        if (family == "affine") {
            scores.reserve(812);
            for (std::uint8_t a = 1; a <= 28; ++a) {
                for (std::uint8_t b = 0; b < 29; ++b) {
                    StatusOr<std::vector<Index29>> out = AffineTransform{}.apply(
                        cipher,
                        nlohmann::json{{"a", static_cast<int>(a)}, {"b", static_cast<int>(b)}},
                        TransformDirection::Decrypt);
                    if (!out.ok()) {
                        return out.status();
                    }
                    StatusOr<double> scored = ScoreRegistry::score(
                        "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
                    if (!scored.ok()) {
                        return scored.status();
                    }
                    scores.push_back(scored.value());
                }
            }
            return scores;
        }
        if (family == "vigenere") {
            scores.reserve(20);
            for (int L = 1; L <= 20; ++L) {
                nlohmann::json params;
                params["key_indices"] = nlohmann::json::array();
                for (int j = 0; j < L; ++j) {
                    params["key_indices"].push_back((j + 1) % 29);
                }
                StatusOr<std::vector<Index29>> out =
                    VigenereKeyTransform{}.apply(cipher, params, TransformDirection::Decrypt);
                if (!out.ok()) {
                    return out.status();
                }
                StatusOr<double> scored = ScoreRegistry::score(
                    "chi2_english_gp_v0", out.value(), "v0", nlohmann::json::object(), request);
                if (!scored.ok()) {
                    return scored.status();
                }
                scores.push_back(scored.value());
            }
            return scores;
        }
        return Status::error("SearchRun: unknown family for CPU reference");
    }

    [[nodiscard]] static StatusOr<bool>
    compare_cpu_cuda_family(std::string_view family, std::span<const Index29> cipher,
                            const ExpectedFrequencyTable& freqs) {
        StatusOr<std::vector<double>> cpu = cpu_family_scores(family, cipher, freqs);
        if (!cpu.ok()) {
            return cpu.status();
        }
        StatusOr<SearchRunCuda::SweepResult> cuda =
            SearchRunCuda::run(family, cipher, freqs, /*throughput_repeats=*/1);
        if (!cuda.ok()) {
            return cuda.status();
        }
        if (cpu.value().size() != cuda.value().steps.size()) {
            return false;
        }
        for (std::size_t i = 0; i < cpu.value().size(); ++i) {
            if (cpu.value()[i] != cuda.value().steps[i].score()) {
                return false;
            }
        }
        return true;
    }
#else
    [[nodiscard]] static StatusOr<bool> compare_cpu_cuda_family(std::string_view,
                                                                std::span<const Index29>,
                                                                const ExpectedFrequencyTable&) {
        return Status::error("SearchRun: CUDA not built");
    }
#endif
};

#endif // SEARCH_RUN_HPP
