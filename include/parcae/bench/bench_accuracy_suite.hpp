#ifndef BENCH_ACCURACY_SUITE_HPP
#define BENCH_ACCURACY_SUITE_HPP

#include "parcae/bench/bench_report.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/run/search_run.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/search/cpu_candidate_export.hpp"
#include "parcae/search/workspace_cipher.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "deep_score_batch.hpp"
#include "device_buffer.hpp"
#include "params.hpp"
#include "parcae_cuda.hpp"
#endif

/// Deterministic statistical validation suite for `parcae-bench --suite accuracy`.
///
/// CPU (always):
/// - `A.fixture_eval` — locked plaintexts beat LCG noise under χ²
/// - `A.chi2_sanity` — welcome plaintext χ² < length-matched uniform
/// - `A.oracle_rank` — a-warning ciphertext ranks true atbash at top-1
///
/// CUDA (when built and `Options::allow_cuda()`):
/// - `A.fused_parity` — CaesarChi2Batch vs CPU ScoreRegistry (small stream)
/// - `A.planted_caesar` — χ² argmin recovers planted shift
/// - `A.planted_bigram` — bigram-LL argmin recovers planted shift
class BenchAccuracySuite {
public:
    class Options {
    public:
        Options() = default;

        [[nodiscard]] bool allow_cuda() const noexcept {
            return allow_cuda_;
        }

        void set_allow_cuda(bool enabled) noexcept {
            allow_cuda_ = enabled;
        }

        [[nodiscard]] std::uint32_t seed() const noexcept {
            return seed_;
        }

        void set_seed(std::uint32_t seed) noexcept {
            seed_ = seed;
        }

    private:
        bool allow_cuda_ = false;
        std::uint32_t seed_ = 2109016688u;
    };

    [[nodiscard]] static BenchReport::Row make_check_row(
        std::string name,
        std::string workload,
        BenchReport::Backend backend,
        BenchReport::RowStatus status,
        std::string detail = {}) {
        return BenchReport::Row::make(
            std::move(name),
            std::move(workload),
            BenchReport::Suite::Accuracy,
            backend,
            status,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            std::move(detail));
    }

    [[nodiscard]] static StatusOr<BenchReport::Document> run(
        const parcae::tool::Context& ctx, const Options& options = Options{}) {
        StatusOr<ExpectedFrequencyTable> freqs = ctx.load_english_gp_expected();
        if (!freqs.ok()) {
            return freqs.status();
        }

        BenchReport::Document doc(BenchReport::Suite::Accuracy);

        StatusOr<BenchReport::Row> eval = check_fixture_eval(ctx, freqs.value(), options.seed());
        if (!eval.ok()) {
            return eval.status();
        }
        doc.add_row(std::move(eval.value()));

        StatusOr<BenchReport::Row> chi2 = check_chi2_sanity(ctx, freqs.value());
        if (!chi2.ok()) {
            return chi2.status();
        }
        doc.add_row(std::move(chi2.value()));

        StatusOr<BenchReport::Row> oracle = check_oracle_rank(ctx);
        if (!oracle.ok()) {
            return oracle.status();
        }
        doc.add_row(std::move(oracle.value()));

        if (options.allow_cuda()) {
#if defined(PARCAE_HAS_CUDA)
            if (!parcae::tool::BackendUtil::cuda_built() || !ParcaeCuda::available()) {
                doc.add_row(make_check_row(
                    "A.fused_parity",
                    "CaesarChi2Batch vs CPU chi2",
                    BenchReport::Backend::Cuda,
                    BenchReport::RowStatus::Skipped,
                    "CUDA unavailable"));
                doc.add_row(make_check_row(
                    "A.planted_caesar",
                    "planted Caesar chi2 argmin",
                    BenchReport::Backend::Cuda,
                    BenchReport::RowStatus::Skipped,
                    "CUDA unavailable"));
                doc.add_row(make_check_row(
                    "A.planted_bigram",
                    "planted Caesar bigram-LL argmin",
                    BenchReport::Backend::Cuda,
                    BenchReport::RowStatus::Skipped,
                    "CUDA unavailable"));
            } else {
                StatusOr<BenchReport::Row> fused = check_fused_parity(freqs.value());
                if (!fused.ok()) {
                    return fused.status();
                }
                doc.add_row(std::move(fused.value()));

                StatusOr<BenchReport::Row> planted = check_planted_caesar(freqs.value());
                if (!planted.ok()) {
                    return planted.status();
                }
                doc.add_row(std::move(planted.value()));

                StatusOr<BenchReport::Row> bigram = check_planted_bigram();
                if (!bigram.ok()) {
                    return bigram.status();
                }
                doc.add_row(std::move(bigram.value()));
            }
#else
            doc.add_row(make_check_row(
                "A.fused_parity",
                "CaesarChi2Batch vs CPU chi2",
                BenchReport::Backend::Cuda,
                BenchReport::RowStatus::Skipped,
                "CUDA not built"));
            doc.add_row(make_check_row(
                "A.planted_caesar",
                "planted Caesar chi2 argmin",
                BenchReport::Backend::Cuda,
                BenchReport::RowStatus::Skipped,
                "CUDA not built"));
            doc.add_row(make_check_row(
                "A.planted_bigram",
                "planted Caesar bigram-LL argmin",
                BenchReport::Backend::Cuda,
                BenchReport::RowStatus::Skipped,
                "CUDA not built"));
#endif
        }

        doc.recompute_all_pass();
        return doc;
    }

private:
    BenchAccuracySuite() = delete;

    [[nodiscard]] static StatusOr<std::vector<Index29>> plaintext_indices_of(
        const parcae::tool::Context& ctx, std::string_view fixture_id) {
        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            return profile.status();
        }
        const LatinCodec codec(profile.value());
        const PlaintextNormalizer normalizer(codec);

        StatusOr<std::filesystem::path> dir = ctx.resolve_fixture_dir(fixture_id);
        if (!dir.ok()) {
            return dir.status();
        }
        StatusOr<Fixture> fixture = FixtureLoader::load_directory(dir.value().string());
        if (!fixture.ok()) {
            return fixture.status();
        }
        StatusOr<std::string> normalized = normalizer.normalize(fixture.value().plaintext());
        if (!normalized.ok()) {
            return normalized.status();
        }
        return codec.delatinize(normalized.value());
    }

    [[nodiscard]] static StatusOr<BenchReport::Row> check_fixture_eval(
        const parcae::tool::Context& ctx,
        const ExpectedFrequencyTable& freqs,
        std::uint32_t seed) {
        StatusOr<SearchRun::EvalResult> eval =
            SearchRun::run_fixture_eval(ctx, "chi2_english_gp_v0", freqs, seed);
        if (!eval.ok()) {
            return eval.status();
        }
        const bool pass = eval.value().passed == eval.value().total && eval.value().total > 0;
        std::ostringstream detail;
        detail << "passed=" << eval.value().passed << "/" << eval.value().total
               << " seed=" << seed;
        return make_check_row(
            "A.fixture_eval",
            "locked plaintext vs LCG noise (chi2)",
            BenchReport::Backend::Cpu,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }

    [[nodiscard]] static StatusOr<BenchReport::Row> check_chi2_sanity(
        const parcae::tool::Context& ctx, const ExpectedFrequencyTable& freqs) {
        StatusOr<std::vector<Index29>> welcome = plaintext_indices_of(ctx, "welcome");
        if (!welcome.ok()) {
            return welcome.status();
        }
        if (welcome.value().empty()) {
            return Status::error("BenchAccuracySuite: welcome plaintext empty");
        }

        std::vector<Index29> uniform;
        uniform.reserve(welcome.value().size());
        for (std::size_t i = 0; i < welcome.value().size(); ++i) {
            uniform.push_back(Index29{static_cast<std::uint8_t>(i % Index29::modulus)});
        }

        StatusOr<double> chi_plain = Chi2EnglishGp::score(welcome.value(), freqs);
        if (!chi_plain.ok()) {
            return chi_plain.status();
        }
        StatusOr<double> chi_uniform = Chi2EnglishGp::score(uniform, freqs);
        if (!chi_uniform.ok()) {
            return chi_uniform.status();
        }

        const bool pass = chi_plain.value() < chi_uniform.value();
        std::ostringstream detail;
        detail << "chi_plain=" << chi_plain.value() << " chi_uniform=" << chi_uniform.value();
        return make_check_row(
            "A.chi2_sanity",
            "welcome plaintext chi2 < uniform",
            BenchReport::Backend::Cpu,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }

    [[nodiscard]] static StatusOr<BenchReport::Row> check_oracle_rank(
        const parcae::tool::Context& ctx) {
        StatusOr<WorkspaceCipher> cipher =
            WorkspaceCipher::from_fixture(ctx.data_root(), "bench-acc", "a-warning");
        if (!cipher.ok()) {
            return cipher.status();
        }

        StatusOr<CpuCandidateExport::Result> ranked = CpuCandidateExport::run(
            cipher.value().indices(),
            "atbash",
            "chi2_english_gp_v0",
            /*k=*/1,
            ctx,
            TransformDirection::Decrypt);
        if (!ranked.ok()) {
            return ranked.status();
        }
        if (ranked.value().size() == 0) {
            return Status::error("BenchAccuracySuite: oracle rank empty");
        }

        const std::string top_id =
            ranked.value().rows().front().candidate().transform_id().str();
        const bool pass = (top_id == "atbash");
        std::ostringstream detail;
        detail << "top_transform=" << top_id
               << " score=" << ranked.value().rows().front().score();
        return make_check_row(
            "A.oracle_rank",
            "a-warning ciphertext ranks atbash top-1",
            BenchReport::Backend::Cpu,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static StatusOr<BenchReport::Row> check_fused_parity(
        const ExpectedFrequencyTable& freqs) {
        std::vector<Index29> plain;
        plain.reserve(128);
        for (std::uint8_t i = 0; i < 128; ++i) {
            plain.push_back(Index29{static_cast<std::uint8_t>(i % Index29::modulus)});
        }
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", 7}}, TransformDirection::Encrypt);
        if (!cipher.ok()) {
            return cipher.status();
        }

        constexpr std::size_t C = Index29::modulus;
        const std::size_t T = cipher.value().size();
        std::vector<std::uint8_t> host_in(T);
        for (std::size_t i = 0; i < T; ++i) {
            host_in[i] = cipher.value()[i].value();
        }
        std::vector<std::uint8_t> shifts(C);
        std::vector<std::uint8_t> dirs(C, static_cast<std::uint8_t>(CudaDir::Decrypt));
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }

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
        StatusOr<DeviceBuffer<std::uint8_t>> device_dirs =
            DeviceBuffer<std::uint8_t>::from_host(dirs);
        if (!device_dirs.ok()) {
            return device_dirs.status();
        }
        StatusOr<DeviceBuffer<double>> device_probs = DeviceBuffer<double>::from_host(
            std::span<const double>(freqs.probabilities().data(), 29));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * 29);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        Status launched = CaesarChi2Batch::launch(
            device_in.value().data(),
            device_shifts.value().data(),
            device_dirs.value().data(),
            device_probs.value().data(),
            device_counts.value().data(),
            device_scores.value().data(),
            C,
            T);
        if (!launched.ok()) {
            return launched;
        }

        std::vector<double> gpu(C);
        Status copied = device_scores.value().copy_to_host(gpu);
        if (!copied.ok()) {
            return copied;
        }

        ScoreRequest request;
        request.expected_frequencies = &freqs;
        std::size_t mismatches = 0;
        for (std::size_t shift = 0; shift < C; ++shift) {
            StatusOr<std::vector<Index29>> dec = CaesarTransform{}.apply(
                cipher.value(),
                nlohmann::json{{"shift", static_cast<int>(shift)}},
                TransformDirection::Decrypt);
            if (!dec.ok()) {
                return dec.status();
            }
            StatusOr<double> cpu = ScoreRegistry::score(
                "chi2_english_gp_v0", dec.value(), "v0", nlohmann::json::object(), request);
            if (!cpu.ok()) {
                return cpu.status();
            }
            if (gpu[shift] != cpu.value()) {
                ++mismatches;
            }
        }

        const bool pass = mismatches == 0;
        std::ostringstream detail;
        detail << "mismatches=" << mismatches << "/" << C;
        return make_check_row(
            "A.fused_parity",
            "CaesarChi2Batch vs CPU chi2",
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }

    [[nodiscard]] static StatusOr<BenchReport::Row> check_planted_caesar(
        const ExpectedFrequencyTable& freqs) {
        constexpr std::uint8_t kPlanted = 11;
        std::vector<Index29> plain;
        plain.reserve(256);
        for (std::size_t i = 0; i < 256; ++i) {
            // Mild English-GP-ish bias toward low indices.
            plain.push_back(Index29{static_cast<std::uint8_t>((i * 3 + 1) % 17)});
        }
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", static_cast<int>(kPlanted)}}, TransformDirection::Encrypt);
        if (!cipher.ok()) {
            return cipher.status();
        }

        constexpr std::size_t C = Index29::modulus;
        const std::size_t T = cipher.value().size();
        std::vector<std::uint8_t> host_in(T);
        for (std::size_t i = 0; i < T; ++i) {
            host_in[i] = cipher.value()[i].value();
        }
        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }

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
            std::span<const double>(freqs.probabilities().data(), 29));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * 29);
        if (!device_counts.ok()) {
            return device_counts.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        Status launched = CaesarChi2Batch::launch_decrypt_async(
            device_in.value().data(),
            device_shifts.value().data(),
            device_probs.value().data(),
            device_counts.value().data(),
            device_scores.value().data(),
            C,
            T);
        if (!launched.ok()) {
            return launched;
        }
        Status synced = CudaError::to_status(cudaDeviceSynchronize(), "planted caesar sync");
        if (!synced.ok()) {
            return synced;
        }

        std::vector<double> gpu(C);
        Status copied = device_scores.value().copy_to_host(gpu);
        if (!copied.ok()) {
            return copied;
        }

        std::size_t best = 0;
        for (std::size_t c = 1; c < C; ++c) {
            if (gpu[c] < gpu[best]) {
                best = c;
            }
        }
        const bool pass = best == kPlanted;
        std::ostringstream detail;
        detail << "planted=" << static_cast<int>(kPlanted) << " argmin=" << best;
        return make_check_row(
            "A.planted_caesar",
            "planted Caesar chi2 argmin",
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }

    [[nodiscard]] static StatusOr<BenchReport::Row> check_planted_bigram() {
        constexpr std::uint8_t kPlanted = 5;
        constexpr std::uint8_t kA = 3;
        constexpr std::uint8_t kB = 7;
        std::vector<Index29> plain;
        plain.reserve(200);
        for (std::size_t i = 0; i < 200; ++i) {
            plain.push_back(Index29{(i % 2 == 0) ? kA : kB});
        }
        StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
            plain, nlohmann::json{{"shift", static_cast<int>(kPlanted)}}, TransformDirection::Encrypt);
        if (!cipher.ok()) {
            return cipher.status();
        }

        // Prefer A↔B transitions; everything else heavily penalized.
        std::vector<float> bigram(29 * 29, -20.0f);
        bigram[static_cast<std::size_t>(kA) * 29 + kB] = 0.0f;
        bigram[static_cast<std::size_t>(kB) * 29 + kA] = 0.0f;

        constexpr std::size_t C = Index29::modulus;
        const std::size_t T = cipher.value().size();
        std::vector<std::uint8_t> host_in(T);
        for (std::size_t i = 0; i < T; ++i) {
            host_in[i] = cipher.value()[i].value();
        }
        std::vector<std::uint8_t> shifts(C);
        for (std::size_t c = 0; c < C; ++c) {
            shifts[c] = static_cast<std::uint8_t>(c);
        }

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
        StatusOr<DeviceBuffer<float>> device_bigram = DeviceBuffer<float>::from_host(bigram);
        if (!device_bigram.ok()) {
            return device_bigram.status();
        }
        StatusOr<DeviceBuffer<double>> device_scores = DeviceBuffer<double>::allocate(C);
        if (!device_scores.ok()) {
            return device_scores.status();
        }

        Status launched = DeepScoreBatch::launch_caesar_bigram_ll_async(
            device_in.value().data(),
            device_shifts.value().data(),
            device_bigram.value().data(),
            device_scores.value().data(),
            C,
            T);
        if (!launched.ok()) {
            return launched;
        }
        Status synced = CudaError::to_status(cudaDeviceSynchronize(), "planted bigram sync");
        if (!synced.ok()) {
            return synced;
        }

        std::vector<double> gpu(C);
        Status copied = device_scores.value().copy_to_host(gpu);
        if (!copied.ok()) {
            return copied;
        }

        std::size_t best = 0;
        for (std::size_t c = 1; c < C; ++c) {
            if (gpu[c] < gpu[best]) {
                best = c;
            }
        }
        const bool pass = best == kPlanted;
        std::ostringstream detail;
        detail << "planted=" << static_cast<int>(kPlanted) << " argmin=" << best;
        return make_check_row(
            "A.planted_bigram",
            "planted Caesar bigram-LL argmin",
            BenchReport::Backend::Cuda,
            pass ? BenchReport::RowStatus::Pass : BenchReport::RowStatus::Fail,
            detail.str());
    }
#endif
};

#endif // BENCH_ACCURACY_SUITE_HPP
