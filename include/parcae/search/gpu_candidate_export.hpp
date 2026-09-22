#ifndef GPU_CANDIDATE_EXPORT_HPP
#define GPU_CANDIDATE_EXPORT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "parcae_cuda.hpp"

#include <cuda_runtime_api.h>
#endif

/// Fused GPU (or host-scored) export → top-k `TransformCandidate`s.
///
/// Scores-only fused χ² → D2H scores → host `BatchOrdering` → apply transform
/// **only** for retained lanes (search-loop.md). Caesar uses `CaesarChi2Batch`;
/// atbash / atbash_caesar / affine / vigenere use `FamilyChi2Batch`.
/// Vigenère is **explicit keys or a bounded synthetic grid only** — never an
/// unbounded dictionary search.
class GpuCandidateExport {
public:
    static constexpr std::string_view score_id = "chi2_english_gp_v0";
    static constexpr std::string_view score_version = "v0";
    /// Default max key length for `vigenere_bounded` (matches SearchRun CUDA sweep).
    static constexpr std::size_t default_vigenere_max_key_length = 20;

    /// One best-first row after export (envelope + plaintext indices + score).
    class Row {
    public:
        Row(TransformCandidate candidate, double score, std::size_t rank, std::size_t source_index)
            : candidate_(std::move(candidate)),
              score_(score),
              rank_(rank),
              source_index_(source_index) {}

        [[nodiscard]] const TransformCandidate& candidate() const noexcept {
            return candidate_;
        }

        [[nodiscard]] double score() const noexcept {
            return score_;
        }

        [[nodiscard]] std::size_t rank() const noexcept {
            return rank_;
        }

        /// Lane index in the family grid.
        [[nodiscard]] std::size_t source_index() const noexcept {
            return source_index_;
        }

        [[nodiscard]] nlohmann::json to_wire(parcae::tool::Backend backend) const {
            nlohmann::json base = candidate_.to_json();
            base["rank"] = rank_;
            base["score"] = nlohmann::json{
                {"score_id", std::string(score_id)},
                {"score_version", std::string(score_version)},
                {"value", score_},
                {"backend", std::string(parcae::tool::BackendUtil::to_string(backend))},
            };
            return base;
        }

    private:
        TransformCandidate candidate_;
        double score_ = 0.0;
        std::size_t rank_ = 0;
        std::size_t source_index_ = 0;
    };

    class Result {
    public:
        Result() = default;

        explicit Result(std::vector<Row> rows, parcae::tool::Backend backend)
            : rows_(std::move(rows)), backend_(backend) {}

        [[nodiscard]] const std::vector<Row>& rows() const noexcept {
            return rows_;
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return rows_.size();
        }

        [[nodiscard]] parcae::tool::Backend backend() const noexcept {
            return backend_;
        }

        [[nodiscard]] std::vector<nlohmann::json> to_wire_lines() const {
            std::vector<nlohmann::json> out;
            out.reserve(rows_.size());
            for (const Row& row : rows_) {
                out.push_back(row.to_wire(backend_));
            }
            return out;
        }

    private:
        std::vector<Row> rows_;
        parcae::tool::Backend backend_ = parcae::tool::Backend::Cuda;
    };

    // --- Caesar ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> caesar_from_host_scores(
        std::span<const Index29> cipher,
        std::span<const double> scores_by_shift,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores_by_shift.size() != Index29::modulus) {
            return Status::error("GpuCandidateExport: caesar scores must have length 29");
        }
        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores_by_shift,
            k,
            [](std::size_t shift) {
                return CaesarCandidateGenerator::make_candidate_id(
                    static_cast<std::uint8_t>(shift));
            });
        if (!hits.ok()) {
            return hits.status();
        }

        const CaesarTransform transform;
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            const std::uint8_t shift = static_cast<std::uint8_t>(hit.source_index());
            const nlohmann::json params = {{"shift", static_cast<int>(shift)}};
            StatusOr<std::vector<Index29>> plain =
                transform.apply(cipher, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(
                hit.candidate_id(),
                TransformId::caesar(),
                direction,
                params,
                std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result> caesar(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::caesar fused path supports decrypt only "
                "(use caesar_from_host_scores for encrypt)");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        return Status::error(
            "GpuCandidateExport::caesar requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        StatusOr<std::vector<double>> scores = fused_caesar_scores(cipher, freqs);
        if (!scores.ok()) {
            return scores.status();
        }
        return caesar_from_host_scores(
            cipher, scores.value(), k, TransformDirection::Decrypt, parcae::tool::Backend::Cuda);
#endif
    }

    // --- Atbash ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> atbash_from_host_scores(
        std::span<const Index29> cipher,
        std::span<const double> scores,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores.size() != AtbashCandidateGenerator::candidate_count) {
            return Status::error("GpuCandidateExport: atbash scores must have length 1");
        }
        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores, k, [](std::size_t) { return AtbashCandidateGenerator::make_candidate_id(); });
        if (!hits.ok()) {
            return hits.status();
        }

        const AtbashTransform transform;
        const nlohmann::json params = nlohmann::json::object();
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            StatusOr<std::vector<Index29>> plain =
                transform.apply(cipher, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(
                hit.candidate_id(),
                TransformId::atbash(),
                direction,
                params,
                std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result> atbash(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::atbash fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        return Status::error(
            "GpuCandidateExport::atbash requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        StatusOr<std::vector<double>> scores = fused_atbash_scores(cipher, freqs);
        if (!scores.ok()) {
            return scores.status();
        }
        return atbash_from_host_scores(
            cipher, scores.value(), k, TransformDirection::Decrypt, parcae::tool::Backend::Cuda);
#endif
    }

    // --- Atbash ∘ Caesar -------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> atbash_caesar_from_host_scores(
        std::span<const Index29> cipher,
        std::span<const double> scores_by_shift,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores_by_shift.size() != AtbashCaesarCandidateGenerator::candidate_count) {
            return Status::error(
                "GpuCandidateExport: atbash_caesar scores must have length 29");
        }
        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores_by_shift,
            k,
            [](std::size_t shift) {
                return AtbashCaesarCandidateGenerator::make_candidate_id(
                    static_cast<std::uint8_t>(shift));
            });
        if (!hits.ok()) {
            return hits.status();
        }

        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            const std::uint8_t shift = static_cast<std::uint8_t>(hit.source_index());
            const nlohmann::json params = ComposeTransform::atbash_then_caesar_params(shift);
            StatusOr<std::vector<Index29>> plain =
                ComposeTransform::apply_atbash_then_caesar(cipher, shift, direction);
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(
                hit.candidate_id(),
                TransformId::compose(),
                direction,
                params,
                std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result> atbash_caesar(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::atbash_caesar fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        return Status::error(
            "GpuCandidateExport::atbash_caesar requires CUDA "
            "(build with PARCAE_BUILD_CUDA=ON)");
#else
        StatusOr<std::vector<double>> scores = fused_atbash_caesar_scores(cipher, freqs);
        if (!scores.ok()) {
            return scores.status();
        }
        return atbash_caesar_from_host_scores(
            cipher, scores.value(), k, TransformDirection::Decrypt, parcae::tool::Backend::Cuda);
#endif
    }

    // --- Affine ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> affine_from_host_scores(
        std::span<const Index29> cipher,
        std::span<const double> scores,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores.size() != AffineCandidateGenerator::candidate_count) {
            return Status::error(
                "GpuCandidateExport: affine scores must have length 812");
        }
        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores,
            k,
            [](std::size_t index) {
                const std::uint8_t a =
                    static_cast<std::uint8_t>(index / Index29::modulus + 1);
                const std::uint8_t b = static_cast<std::uint8_t>(index % Index29::modulus);
                return AffineCandidateGenerator::make_candidate_id(a, b);
            });
        if (!hits.ok()) {
            return hits.status();
        }

        const AffineTransform transform;
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            const std::size_t index = hit.source_index();
            const std::uint8_t a = static_cast<std::uint8_t>(index / Index29::modulus + 1);
            const std::uint8_t b = static_cast<std::uint8_t>(index % Index29::modulus);
            const nlohmann::json params = {
                {"a", static_cast<int>(a)},
                {"b", static_cast<int>(b)},
            };
            StatusOr<std::vector<Index29>> plain =
                transform.apply(cipher, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(
                hit.candidate_id(),
                TransformId::affine(),
                direction,
                params,
                std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result> affine(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::affine fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        return Status::error(
            "GpuCandidateExport::affine requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        StatusOr<std::vector<double>> scores = fused_affine_scores(cipher, freqs);
        if (!scores.ok()) {
            return scores.status();
        }
        return affine_from_host_scores(
            cipher, scores.value(), k, TransformDirection::Decrypt, parcae::tool::Backend::Cuda);
#endif
    }

    // --- Vigenère (explicit keys / bounded grid) -------------------------------

    /// Synthetic bounded grid: for L in `1..max_key_length`, key = `[1,2,…,L] mod 29`.
    /// Same construction as `SearchRunCuda::run_vigenere` — not a dictionary.
    [[nodiscard]] static StatusOr<std::vector<std::vector<Index29>>> default_bounded_key_grid(
        std::size_t max_key_length = default_vigenere_max_key_length) {
        if (max_key_length == 0) {
            return Status::error("GpuCandidateExport: max_key_length must be >= 1");
        }
#if defined(PARCAE_HAS_CUDA)
        if (max_key_length > FamilyChi2Batch::kMaxCandidates) {
            return Status::error(
                "GpuCandidateExport: max_key_length exceeds FamilyChi2Batch::kMaxCandidates");
        }
#else
        if (max_key_length > 16384) {
            return Status::error("GpuCandidateExport: max_key_length exceeds 16384");
        }
#endif
        std::vector<std::vector<Index29>> keys;
        keys.reserve(max_key_length);
        for (std::size_t L = 1; L <= max_key_length; ++L) {
            std::vector<Index29> key;
            key.reserve(L);
            for (std::size_t j = 0; j < L; ++j) {
                key.push_back(Index29{static_cast<std::uint8_t>((j + 1) % Index29::modulus)});
            }
            keys.push_back(std::move(key));
        }
        return keys;
    }

    [[nodiscard]] static StatusOr<Result> vigenere_from_host_scores(
        std::span<const Index29> cipher,
        const std::vector<std::vector<Index29>>& keys,
        std::span<const double> scores,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        Status keys_ok = require_vigenere_keys(keys);
        if (!keys_ok.ok()) {
            return keys_ok;
        }
        if (scores.size() != keys.size()) {
            return Status::error(
                "GpuCandidateExport: vigenere scores length must equal keys length");
        }

        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores,
            k,
            [&](std::size_t index) {
                return VigenereExplicitKeyCandidateGenerator::make_candidate_id(
                    keys[index], index);
            });
        if (!hits.ok()) {
            return hits.status();
        }

        const VigenereKeyTransform transform;
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            const std::size_t index = hit.source_index();
            nlohmann::json params{{"key_indices", nlohmann::json::array()}};
            for (const Index29 idx : keys[index]) {
                params["key_indices"].push_back(static_cast<int>(idx.value()));
            }
            StatusOr<std::vector<Index29>> plain =
                transform.apply(cipher, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(
                hit.candidate_id(),
                TransformId::vigenere_key(),
                direction,
                std::move(params),
                std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        return Result{std::move(rows), backend};
    }

    /// Fused Vigenère decrypt χ² over an **explicit** caller-supplied key list.
    [[nodiscard]] static StatusOr<Result> vigenere(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        const std::vector<std::vector<Index29>>& keys,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::vigenere fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)keys;
        (void)k;
        return Status::error(
            "GpuCandidateExport::vigenere requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        StatusOr<std::vector<double>> scores = fused_vigenere_scores(cipher, freqs, keys);
        if (!scores.ok()) {
            return scores.status();
        }
        return vigenere_from_host_scores(
            cipher,
            keys,
            scores.value(),
            k,
            TransformDirection::Decrypt,
            parcae::tool::Backend::Cuda);
#endif
    }

    /// Bounded synthetic key grid (`default_bounded_key_grid`) then fused export.
    [[nodiscard]] static StatusOr<Result> vigenere_bounded(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        std::size_t k,
        std::size_t max_key_length = default_vigenere_max_key_length,
        TransformDirection direction = TransformDirection::Decrypt) {
        StatusOr<std::vector<std::vector<Index29>>> keys =
            default_bounded_key_grid(max_key_length);
        if (!keys.ok()) {
            return keys.status();
        }
        return vigenere(cipher, freqs, keys.value(), k, direction);
    }

private:
    GpuCandidateExport() = delete;

    [[nodiscard]] static Status require_vigenere_keys(
        const std::vector<std::vector<Index29>>& keys) {
        if (keys.empty()) {
            return Status::error(
                "GpuCandidateExport: vigenere requires a non-empty explicit key list");
        }
#if defined(PARCAE_HAS_CUDA)
        if (keys.size() > FamilyChi2Batch::kMaxCandidates) {
            return Status::error(
                "GpuCandidateExport: vigenere key count exceeds FamilyChi2Batch::kMaxCandidates");
        }
#else
        if (keys.size() > 16384) {
            return Status::error("GpuCandidateExport: vigenere key count exceeds 16384");
        }
#endif
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].empty()) {
                return Status::error(
                    "GpuCandidateExport: vigenere key_indices must be non-empty");
            }
            for (const Index29 idx : keys[i]) {
                if (idx.value() >= Index29::modulus) {
                    return Status::error(
                        "GpuCandidateExport: vigenere key_indices entry out of range");
                }
            }
        }
        return Status::success();
    }

    [[nodiscard]] static Status require_cipher_k(
        std::span<const Index29> cipher,
        std::size_t k) {
        if (cipher.empty()) {
            return Status::error("GpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("GpuCandidateExport: k must be >= 1");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::vector<BatchHit>> select_top_k(
        std::span<const double> scores,
        std::size_t k,
        const std::function<std::string(std::size_t)>& id_for_index) {
        std::vector<BatchHit> hits;
        hits.reserve(scores.size());
        for (std::size_t i = 0; i < scores.size(); ++i) {
            hits.emplace_back(id_for_index(i), scores[i], i);
        }
        std::sort(hits.begin(), hits.end(), BatchOrdering::BestFirst{ScoreOrder::Asc});
        if (hits.size() > k) {
            hits.erase(hits.begin() + static_cast<std::ptrdiff_t>(k), hits.end());
        }
        return hits;
    }

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static Status require_cuda_freqs(const ExpectedFrequencyTable& freqs) {
        if (!ParcaeCuda::available()) {
            return Status::error("GpuCandidateExport: no CUDA device available");
        }
        if (freqs.probabilities().size() != Index29::modulus) {
            return Status::error(
                "GpuCandidateExport: expected frequency table must have 29 bins");
        }
        return Status::success();
    }

    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(std::span<const Index29> cipher) {
        std::vector<std::uint8_t> out(cipher.size());
        for (std::size_t i = 0; i < cipher.size(); ++i) {
            out[i] = cipher[i].value();
        }
        return out;
    }

    struct DeviceScratch {
        DeviceBuffer<std::uint8_t> in;
        DeviceBuffer<double> probs;
        DeviceBuffer<std::uint32_t> counts;
        DeviceBuffer<double> scores;
        std::size_t C = 0;
        std::size_t T = 0;
    };

    [[nodiscard]] static StatusOr<DeviceScratch> make_scratch(
        std::span<const std::uint8_t> host_in,
        const ExpectedFrequencyTable& freqs,
        std::size_t C) {
        DeviceScratch s;
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
        StatusOr<DeviceBuffer<std::uint32_t>> counts =
            DeviceBuffer<std::uint32_t>::allocate(C * Index29::modulus);
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
    [[nodiscard]] static StatusOr<std::vector<double>> launch_sync_copy(
        DeviceScratch& scratch,
        LaunchFn&& launch,
        const char* sync_label) {
        Status launched = launch();
        if (!launched.ok()) {
            return launched;
        }
        Status synced = CudaError::to_status(cudaDeviceSynchronize(), sync_label);
        if (!synced.ok()) {
            return synced;
        }
        std::vector<double> scores(scratch.C, 0.0);
        Status copied = scratch.scores.copy_to_host(scores);
        if (!copied.ok()) {
            return copied;
        }
        return scores;
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_caesar_scores(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        constexpr std::size_t C = Index29::modulus;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
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
        return launch_sync_copy(
            scratch.value(),
            [&]() {
                return CaesarChi2Batch::launch_decrypt_async(
                    scratch.value().in.data(),
                    device_shifts.value().data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "GpuCandidateExport::caesar sync");
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_atbash_scores(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        constexpr std::size_t C = AtbashCandidateGenerator::candidate_count;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        return launch_sync_copy(
            scratch.value(),
            [&]() {
                return FamilyChi2Batch::launch_atbash_async(
                    scratch.value().in.data(),
                    scratch.value().probs.data(),
                    scratch.value().counts.data(),
                    scratch.value().scores.data(),
                    C,
                    scratch.value().T);
            },
            "GpuCandidateExport::atbash sync");
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_atbash_caesar_scores(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        constexpr std::size_t C = AtbashCaesarCandidateGenerator::candidate_count;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
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
        return launch_sync_copy(
            scratch.value(),
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
            "GpuCandidateExport::atbash_caesar sync");
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_affine_scores(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        constexpr std::size_t C = AffineCandidateGenerator::candidate_count;
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }
        std::vector<std::uint8_t> a(C);
        std::vector<std::uint8_t> b(C);
        std::size_t c = 0;
        for (std::uint8_t ai = 1; ai < Index29::modulus; ++ai) {
            for (std::uint8_t bi = 0; bi < Index29::modulus; ++bi) {
                a[c] = ai;
                b[c] = bi;
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
        return launch_sync_copy(
            scratch.value(),
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
            "GpuCandidateExport::affine sync");
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_vigenere_scores(
        std::span<const Index29> cipher,
        const ExpectedFrequencyTable& freqs,
        const std::vector<std::vector<Index29>>& keys) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        Status keys_ok = require_vigenere_keys(keys);
        if (!keys_ok.ok()) {
            return keys_ok;
        }

        const std::size_t C = keys.size();
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }

        std::size_t arena = 0;
        for (const auto& key : keys) {
            arena += key.size();
        }
        std::vector<std::uint8_t> key_bytes;
        key_bytes.reserve(arena);
        std::vector<std::uint32_t> key_begin(C);
        std::vector<std::uint32_t> key_len(C);
        std::uint32_t cursor = 0;
        for (std::size_t i = 0; i < C; ++i) {
            key_begin[i] = cursor;
            key_len[i] = static_cast<std::uint32_t>(keys[i].size());
            for (const Index29 idx : keys[i]) {
                key_bytes.push_back(idx.value());
            }
            cursor += key_len[i];
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

        return launch_sync_copy(
            scratch.value(),
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
            "GpuCandidateExport::vigenere sync");
    }
#endif
};

#endif // GPU_CANDIDATE_EXPORT_HPP
