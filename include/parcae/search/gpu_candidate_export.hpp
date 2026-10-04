#ifndef GPU_CANDIDATE_EXPORT_HPP
#define GPU_CANDIDATE_EXPORT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/batch/batch_runner.hpp"
#include "parcae/cli/console_progress_sink.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/generate/affine_candidate_generator.hpp"
#include "parcae/generate/atbash_caesar_candidate_generator.hpp"
#include "parcae/generate/atbash_candidate_generator.hpp"
#include "parcae/generate/beaufort_explicit_key_candidate_generator.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/compose_recipe_candidate_generator.hpp"
#include "parcae/generate/theory_explicit_params_candidate_generator.hpp"
#include "parcae/generate/totient_offset_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/generate/vigenere_explicit_key_candidate_generator.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/math/totient_keystream.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/search/nvtx_range.hpp"
#include "parcae/search/theory_export_cache.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/totient_prime_stream_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "compose_driver.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "family_chi2_batch.hpp"
#include "interrupt_device_view.hpp"
#include "params_json.hpp"
#include "parcae_cuda.hpp"
#include "cuda_stream_pair.hpp"
#include "theory_chi2_batch.hpp"
#include "theory_device_scratch.hpp"
#include "theory_hist_chi2_launch.hpp"

#include <cuda_runtime_api.h>
#include <cstring>

class TheoryExportPipeline;
#endif

/// Fused GPU (or host-scored) export → top-k `TransformCandidate`s.
///
/// Scores-only fused χ² → D2H scores → host `BatchOrdering` → apply transform
/// **only** for retained lanes (search-loop.md). Caesar uses `CaesarChi2Batch`;
/// atbash / atbash_caesar / affine / vigenere use `FamilyChi2Batch`.
/// Theory: prefers `TheoryHistChi2Emit` ShapeInline→S1→S2 specialized twins when present,
/// soft-fallback S0 bytecode via `TheoryHistChi2Launch`; `export_backend=cuda`.
/// Family `compose`: Atbash∘Caesar grid reuses fused export; arbitrary recipes
/// score via `ComposeDriver` apply + host χ² (top-k only).
/// Vigenère is **explicit keys or a bounded synthetic grid only** — never an
/// unbounded dictionary search.
///
/// Optional `BatchRunner::Progress` emits **coarse** stage ticks only:
/// `fuse` (device launch+sync), `d2h` (scores copy), `materialize` (top-k
/// plaintext apply). Grid size is `candidates_total` for fuse/d2h; materialize
/// uses retained hit count. Observe-only — results are identical with or
/// without a sink. Host-score paths emit `materialize` only (no fuse/d2h).
///
/// When NVTX is available (`NvtxRange`), theory fused export also pushes
/// timeline ranges: `prepare_theory`, `bind_slots`, `h2d`, `hist_kernel`,
/// `finalize`, `d2h`, `materialize` (plus scheduler `ingest`).
///
/// Theory path: pass a long-lived `TheoryExportCache*` to amortize bytecode
/// compile and (CUDA) device `ops`/`imm` uploads across chunks. Pass a
/// long-lived `TheoryDeviceScratch*` to reuse cipher/probs/param device
/// buffers across chunks (`nullptr` → one-shot local scratch). Pass a
/// `CudaStreamPair*` for copy/compute streams (`nullptr` → create_or_legacy).
/// Prefer `theory_scores_only` when the caller only needs χ² scores (no top-k
/// materialize).
class GpuCandidateExport {
public:
    static constexpr std::string_view score_id = "chi2_english_gp_v0";
    static constexpr std::string_view score_version = "v0";
    /// Default max key length for `vigenere_bounded` / `beaufort_bounded`.
    static constexpr std::size_t default_vigenere_max_key_length = 20;
    /// Default contiguous `prime_start_index` count for opt-in `totient` family.
    static constexpr std::size_t default_totient_start_count = 32;

#if defined(PARCAE_HAS_CUDA)
    /// Handle for staged H2D / in-flight hist (pipelined ping-pong export).
    class TheoryLaunchTicket {
    public:
        enum class Phase : std::uint8_t { Idle, Staged, InFlight };

        TheoryLaunchTicket() = default;

        [[nodiscard]] Phase phase() const noexcept { return phase_; }

        [[nodiscard]] bool staged() const noexcept { return phase_ == Phase::Staged; }

        [[nodiscard]] bool in_flight() const noexcept { return phase_ == Phase::InFlight; }

        [[nodiscard]] std::size_t candidate_count() const noexcept { return candidate_count_; }

    private:
        friend class GpuCandidateExport;
        friend class TheoryExportPipeline;

        enum class Kind : std::uint8_t { S0, S1, S2, ShapeAtbash, ShapeCaesar, ShapeAffine };

        Phase phase_ = Phase::Idle;
        Kind kind_ = Kind::S0;
        std::size_t candidate_count_ = 0;
        std::size_t token_count_ = 0;
        std::uint32_t op_count_ = 0;
        std::uint16_t slot_count_ = 0;
        std::uint16_t cipher_slot_ = 0;
        std::uint16_t index_slot_ = 0;
        std::uint16_t max_stack_ = 0;
        std::uint8_t binds_index_i_ = 0;
        bool cipher_minus_ks_ = true;
    };
#endif

    /// One best-first row after export (envelope + plaintext indices + score).
    class Row {
    public:
        Row(TransformCandidate candidate, double score, std::size_t rank, std::size_t source_index)
            : candidate_(std::move(candidate)), score_(score), rank_(rank),
              source_index_(source_index) {}

        [[nodiscard]] const TransformCandidate& candidate() const noexcept { return candidate_; }

        [[nodiscard]] double score() const noexcept { return score_; }

        [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

        /// Lane index in the family grid.
        [[nodiscard]] std::size_t source_index() const noexcept { return source_index_; }

        [[nodiscard]] nlohmann::json to_wire(Backend backend) const {
            nlohmann::json base = candidate_.to_json();
            base["rank"] = rank_;
            base["score"] = nlohmann::json{
                {"score_id", std::string(score_id)},
                {"score_version", std::string(score_version)},
                {"value", score_},
                {"backend", std::string(BackendUtil::to_string(backend))},
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

        explicit Result(std::vector<Row> rows, Backend backend)
            : rows_(std::move(rows)), backend_(backend) {}

        [[nodiscard]] const std::vector<Row>& rows() const noexcept { return rows_; }

        [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }

        [[nodiscard]] Backend backend() const noexcept { return backend_; }

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
        Backend backend_ = Backend::Cuda;
    };

    // --- Caesar ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> caesar_from_host_scores(
        std::span<const Index29> cipher, std::span<const double> scores_by_shift, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores_by_shift.size() != Index29::modulus) {
            return Status::error("GpuCandidateExport: caesar scores must have length 29");
        }
        StatusOr<std::vector<BatchHit>> hits =
            select_top_k(scores_by_shift, k, [](std::size_t shift) {
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::caesar(), direction,
                                         params, std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores_by_shift.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    caesar(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs, std::size_t k,
           TransformDirection direction = TransformDirection::Decrypt,
           BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::caesar fused path supports decrypt only "
                                 "(use caesar_from_host_scores for encrypt)");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::caesar requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_caesar_scores(cipher, freqs, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return caesar_from_host_scores(cipher, scores.value(), k, TransformDirection::Decrypt,
                                       Backend::Cuda, progress);
#endif
    }

    // --- Atbash ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> atbash_from_host_scores(
        std::span<const Index29> cipher, std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::atbash(), direction,
                                         params, std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    atbash(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs, std::size_t k,
           TransformDirection direction = TransformDirection::Decrypt,
           BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::atbash fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::atbash requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_atbash_scores(cipher, freqs, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return atbash_from_host_scores(cipher, scores.value(), k, TransformDirection::Decrypt,
                                       Backend::Cuda, progress);
#endif
    }

    // --- Atbash ∘ Caesar -------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> atbash_caesar_from_host_scores(
        std::span<const Index29> cipher, std::span<const double> scores_by_shift, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores_by_shift.size() != AtbashCaesarCandidateGenerator::candidate_count) {
            return Status::error("GpuCandidateExport: atbash_caesar scores must have length 29");
        }
        StatusOr<std::vector<BatchHit>> hits =
            select_top_k(scores_by_shift, k, [](std::size_t shift) {
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::compose(), direction,
                                         params, std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores_by_shift.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    atbash_caesar(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                  std::size_t k, TransformDirection direction = TransformDirection::Decrypt,
                  BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::atbash_caesar fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        (void)progress;
        return Status::error("GpuCandidateExport::atbash_caesar requires CUDA "
                             "(build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_atbash_caesar_scores(cipher, freqs, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return atbash_caesar_from_host_scores(cipher, scores.value(), k,
                                              TransformDirection::Decrypt, Backend::Cuda, progress);
#endif
    }

    // --- Compose recipes (AtbashCaesar reuse + ComposeDriver) --------------------

    /// Score explicit compose recipes (host χ² after apply). Prefer
    /// `compose_from_param_grid` so Atbash∘Caesar grids hit the fused path.
    [[nodiscard]] static StatusOr<Result>
    compose_recipes(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                    const std::vector<nlohmann::json>& recipes, std::size_t k,
                    TransformDirection direction = TransformDirection::Decrypt,
                    BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (recipes.empty()) {
            return Status::error("GpuCandidateExport::compose_recipes requires recipes");
        }
        if (ComposeRecipeCandidateGenerator::is_full_atbash_caesar_grid(recipes) &&
            direction == TransformDirection::Decrypt) {
            return atbash_caesar(cipher, freqs, k, direction, progress);
        }
        return compose_recipes_driver(cipher, freqs, recipes, k, direction, progress);
    }

    /// Resolve `param_grid` like `ComposeRecipeCandidateGenerator`, then export.
    [[nodiscard]] static StatusOr<Result>
    compose_from_param_grid(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                            const nlohmann::json& param_grid, std::size_t k,
                            TransformDirection direction = TransformDirection::Decrypt,
                            BatchRunner::Progress progress = BatchRunner::Progress{}) {
        StatusOr<std::vector<nlohmann::json>> recipes =
            ComposeRecipeCandidateGenerator::recipes_from_param_grid(
                param_grid.is_null() ? nlohmann::json::object() : param_grid);
        if (!recipes.ok()) {
            return recipes.status();
        }
        return compose_recipes(cipher, freqs, recipes.value(), k, direction, progress);
    }

    // --- Affine ----------------------------------------------------------------

    [[nodiscard]] static StatusOr<Result> affine_from_host_scores(
        std::span<const Index29> cipher, std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (scores.size() != AffineCandidateGenerator::candidate_count) {
            return Status::error("GpuCandidateExport: affine scores must have length 812");
        }
        StatusOr<std::vector<BatchHit>> hits = select_top_k(scores, k, [](std::size_t index) {
            const std::uint8_t a = static_cast<std::uint8_t>(index / Index29::modulus + 1);
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::affine(), direction,
                                         params, std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    affine(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs, std::size_t k,
           TransformDirection direction = TransformDirection::Decrypt,
           BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::affine fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::affine requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_affine_scores(cipher, freqs, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return affine_from_host_scores(cipher, scores.value(), k, TransformDirection::Decrypt,
                                       Backend::Cuda, progress);
#endif
    }

    // --- Vigenère (explicit keys / bounded grid) -------------------------------

    /// Synthetic bounded grid: for L in `1..max_key_length`, key = `[1,2,…,L] mod 29`.
    /// Same construction as `SearchRunCuda::run_vigenere` — not a dictionary.
    [[nodiscard]] static StatusOr<std::vector<std::vector<Index29>>>
    default_bounded_key_grid(std::size_t max_key_length = default_vigenere_max_key_length) {
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
        std::span<const Index29> cipher, const std::vector<std::vector<Index29>>& keys,
        std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
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

        StatusOr<std::vector<BatchHit>> hits = select_top_k(scores, k, [&](std::size_t index) {
            return VigenereExplicitKeyCandidateGenerator::make_candidate_id(keys[index], index);
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::vigenere_key(), direction,
                                         std::move(params), std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    /// Fused Vigenère decrypt χ² over an **explicit** caller-supplied key list.
    [[nodiscard]] static StatusOr<Result>
    vigenere(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
             const std::vector<std::vector<Index29>>& keys, std::size_t k,
             TransformDirection direction = TransformDirection::Decrypt,
             BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::vigenere fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)keys;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::vigenere requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_vigenere_scores(cipher, freqs, keys, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return vigenere_from_host_scores(cipher, keys, scores.value(), k,
                                         TransformDirection::Decrypt, Backend::Cuda, progress);
#endif
    }

    /// Bounded synthetic key grid (`default_bounded_key_grid`) then fused export.
    [[nodiscard]] static StatusOr<Result>
    vigenere_bounded(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                     std::size_t k, std::size_t max_key_length = default_vigenere_max_key_length,
                     TransformDirection direction = TransformDirection::Decrypt,
                     BatchRunner::Progress progress = BatchRunner::Progress{}) {
        StatusOr<std::vector<std::vector<Index29>>> keys = default_bounded_key_grid(max_key_length);
        if (!keys.ok()) {
            return keys.status();
        }
        return vigenere(cipher, freqs, keys.value(), k, direction, progress);
    }

    // --- Beaufort (opt-in; same key grids as Vigenère) --------------------------

    [[nodiscard]] static StatusOr<Result> beaufort_from_host_scores(
        std::span<const Index29> cipher, const std::vector<std::vector<Index29>>& keys,
        std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
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
                "GpuCandidateExport: beaufort scores length must equal keys length");
        }

        StatusOr<std::vector<BatchHit>> hits = select_top_k(scores, k, [&](std::size_t index) {
            return BeaufortExplicitKeyCandidateGenerator::make_candidate_id(keys[index], index);
        });
        if (!hits.ok()) {
            return hits.status();
        }

        const BeaufortKeyTransform transform;
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
            TransformCandidate candidate(hit.candidate_id(), TransformId::beaufort_key(), direction,
                                         std::move(params), std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    beaufort(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
             const std::vector<std::vector<Index29>>& keys, std::size_t k,
             TransformDirection direction = TransformDirection::Decrypt,
             BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::beaufort fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)keys;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::beaufort requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores = fused_beaufort_scores(cipher, freqs, keys, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return beaufort_from_host_scores(cipher, keys, scores.value(), k,
                                         TransformDirection::Decrypt, Backend::Cuda, progress);
#endif
    }

    [[nodiscard]] static StatusOr<Result>
    beaufort_bounded(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                     std::size_t k, std::size_t max_key_length = default_vigenere_max_key_length,
                     TransformDirection direction = TransformDirection::Decrypt,
                     BatchRunner::Progress progress = BatchRunner::Progress{}) {
        StatusOr<std::vector<std::vector<Index29>>> keys = default_bounded_key_grid(max_key_length);
        if (!keys.ok()) {
            return keys.status();
        }
        return beaufort(cipher, freqs, keys.value(), k, direction, progress);
    }

    // --- Totient (opt-in; bounded prime_start_index list) -----------------------

    [[nodiscard]] static StatusOr<std::vector<std::size_t>>
    default_totient_starts(std::size_t count = default_totient_start_count) {
        if (count == 0) {
            return Status::error("GpuCandidateExport: totient start count must be >= 1");
        }
#if defined(PARCAE_HAS_CUDA)
        if (count > FamilyChi2Batch::kMaxCandidates) {
            return Status::error(
                "GpuCandidateExport: totient start count exceeds FamilyChi2Batch::kMaxCandidates");
        }
#else
        if (count > 16384) {
            return Status::error("GpuCandidateExport: totient start count exceeds 16384");
        }
#endif
        std::vector<std::size_t> starts(count);
        for (std::size_t i = 0; i < count; ++i) {
            starts[i] = i;
        }
        return starts;
    }

    [[nodiscard]] static StatusOr<Result> totient_from_host_scores(
        std::span<const Index29> cipher, const std::vector<std::size_t>& prime_start_indices,
        std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        if (prime_start_indices.empty()) {
            return Status::error(
                "GpuCandidateExport: totient requires a non-empty prime_start_indices list");
        }
        if (scores.size() != prime_start_indices.size()) {
            return Status::error(
                "GpuCandidateExport: totient scores length must equal starts length");
        }

        StatusOr<std::vector<BatchHit>> hits = select_top_k(scores, k, [&](std::size_t index) {
            return TotientOffsetCandidateGenerator::make_candidate_id(prime_start_indices[index]);
        });
        if (!hits.ok()) {
            return hits.status();
        }

        const TotientPrimeStreamTransform transform;
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            const std::size_t index = hit.source_index();
            const nlohmann::json params{
                {"prime_start_index", static_cast<std::uint64_t>(prime_start_indices[index])},
                {"shift_mode", "prime_minus_one_mod_29"},
            };
            StatusOr<std::vector<Index29>> plain =
                transform.apply(cipher, params, direction, InterruptPolicy::none());
            if (!plain.ok()) {
                return plain.status();
            }
            TransformCandidate candidate(hit.candidate_id(), TransformId::totient_prime_stream(),
                                         direction, params, std::move(plain.value()));
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<Result>
    totient(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
            const std::vector<std::size_t>& prime_start_indices, std::size_t k,
            TransformDirection direction = TransformDirection::Decrypt,
            BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error("GpuCandidateExport::totient fused path supports decrypt only");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)prime_start_indices;
        (void)k;
        (void)progress;
        return Status::error(
            "GpuCandidateExport::totient requires CUDA (build with PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        StatusOr<std::vector<double>> scores =
            fused_totient_scores(cipher, freqs, prime_start_indices, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return totient_from_host_scores(cipher, prime_start_indices, scores.value(), k,
                                        TransformDirection::Decrypt, Backend::Cuda, progress);
#endif
    }

    [[nodiscard]] static StatusOr<Result>
    totient_bounded(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                    std::size_t k, std::size_t start_count = default_totient_start_count,
                    TransformDirection direction = TransformDirection::Decrypt,
                    BatchRunner::Progress progress = BatchRunner::Progress{}) {
        StatusOr<std::vector<std::size_t>> starts = default_totient_starts(start_count);
        if (!starts.ok()) {
            return starts.status();
        }
        return totient(cipher, freqs, starts.value(), k, direction, progress);
    }

    // --- Theory (explicit params_list + theory_uri) -----------------------------

    /// Top-k from precomputed χ² scores; materialize plaintext via host bytecode.
    /// `scores.size()` MUST equal `params_list.size()`. Interrupt must be empty
    /// (fused search contract); non-empty → error.
    [[nodiscard]] static StatusOr<Result> theory_from_host_scores(
        std::span<const Index29> cipher, const std::filesystem::path& theories_root,
        std::string_view theory_uri_text, const std::vector<nlohmann::json>& params_list,
        std::span<const double> scores, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt, Backend backend = Backend::Cpu,
        BatchRunner::Progress progress = BatchRunner::Progress{},
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        TheoryExportCache* cache = nullptr) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }
        Status grid = require_theory_params_list(params_list);
        if (!grid.ok()) {
            return grid;
        }
        if (scores.size() != params_list.size()) {
            return Status::error(
                "GpuCandidateExport: theory scores length must match params_list size");
        }
        if (!interrupt.skip_indices().empty()) {
            return Status::error(
                "GpuCandidateExport::theory fused path requires empty InterruptPolicy");
        }

        TheoryExportCache local_cache;
        TheoryExportCache& active = cache != nullptr ? *cache : local_cache;
        StatusOr<const TheoryExportCache::Entry*> prepared =
            prepare_theory(active, theories_root, theory_uri_text, direction);
        if (!prepared.ok()) {
            return prepared.status();
        }
        const TheoryExportCache::Entry& entry = *prepared.value();

        StatusOr<std::vector<BatchHit>> hits = select_top_k(
            scores, k, [&](std::size_t index) {
                return TheoryExplicitParamsCandidateGenerator::make_candidate_id(entry.uri_str(),
                                                                                 index);
            });
        if (!hits.ok()) {
            return hits.status();
        }

        const TransformId tid = TransformId::unchecked(entry.uri_str());
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        std::vector<Index29> plain(cipher.size());
        {
            NvtxRange nvtx_materialize("materialize");
            for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
                const BatchHit& hit = hits.value()[rank];
                const std::size_t index = hit.source_index();
                const nlohmann::json& params = params_list[index];
                Status applied = Z29Bytecode::apply_into_theory(
                    entry.program(), entry.theory(), params, cipher, plain, interrupt);
                if (!applied.ok()) {
                    return Status::error("GpuCandidateExport::theory materialize failed for index " +
                                         std::to_string(index) + ": " + applied.message());
                }
                TransformCandidate candidate(hit.candidate_id(), tid, direction, params, plain);
                rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
            }
        }
        emit_materialize(progress, rows.size(), scores.size());
        return Result{std::move(rows), backend};
    }

    /// Fused CUDA χ² over `params_list` for one `theory_uri`, then top-k materialize.
    /// Decrypt + empty interrupt only. Requires `PARCAE_BUILD_CUDA`.
    /// Pass `cache` to reuse bytecode + device program buffers across chunks.
    /// Pass `scratch` to reuse cipher/probs/param device buffers across chunks.
    [[nodiscard]] static StatusOr<Result> theory_explicit_params(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list, std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        BatchRunner::Progress progress = BatchRunner::Progress{},
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        TheoryExportCache* cache = nullptr
#if defined(PARCAE_HAS_CUDA)
        ,
        TheoryDeviceScratch* scratch = nullptr, CudaStreamPair* streams = nullptr
#endif
    ) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::theory_explicit_params fused path supports decrypt only");
        }
        if (!interrupt.skip_indices().empty()) {
            return Status::error(
                "GpuCandidateExport::theory_explicit_params requires empty InterruptPolicy");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)theories_root;
        (void)theory_uri_text;
        (void)params_list;
        (void)k;
        (void)progress;
        (void)cache;
        return Status::error(
            "GpuCandidateExport::theory_explicit_params requires CUDA (PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        TheoryExportCache local_cache;
        TheoryExportCache& active = cache != nullptr ? *cache : local_cache;
        TheoryDeviceScratch local_scratch;
        TheoryDeviceScratch& active_scratch = scratch != nullptr ? *scratch : local_scratch;
        CudaStreamPair local_streams;
        CudaStreamPair* active_streams = streams;
        if (active_streams == nullptr) {
            local_streams = CudaStreamPair::create_or_legacy();
            active_streams = &local_streams;
        }
        StatusOr<std::vector<double>> scores =
            fused_theory_scores(cipher, freqs, theories_root, theory_uri_text, params_list, active,
                                active_scratch, *active_streams, progress);
        if (!scores.ok()) {
            return scores.status();
        }
        return theory_from_host_scores(cipher, theories_root, theory_uri_text, params_list,
                                       scores.value(), k, TransformDirection::Decrypt,
                                       Backend::Cuda, progress, interrupt, &active);
#endif
    }

    /// Scores-only fused CUDA χ² (no top-k materialize). Same contracts as
    /// `theory_explicit_params` (decrypt, empty interrupt, CUDA required).
    [[nodiscard]] static StatusOr<std::vector<double>> theory_scores_only(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list,
        TransformDirection direction = TransformDirection::Decrypt,
        BatchRunner::Progress progress = BatchRunner::Progress{},
        const InterruptPolicy& interrupt = InterruptPolicy::none(),
        TheoryExportCache* cache = nullptr
#if defined(PARCAE_HAS_CUDA)
        ,
        TheoryDeviceScratch* scratch = nullptr, CudaStreamPair* streams = nullptr
#endif
    ) {
        if (direction != TransformDirection::Decrypt) {
            return Status::error(
                "GpuCandidateExport::theory_scores_only fused path supports decrypt only");
        }
        if (!interrupt.skip_indices().empty()) {
            return Status::error(
                "GpuCandidateExport::theory_scores_only requires empty InterruptPolicy");
        }
#if !defined(PARCAE_HAS_CUDA)
        (void)cipher;
        (void)freqs;
        (void)theories_root;
        (void)theory_uri_text;
        (void)params_list;
        (void)progress;
        (void)cache;
        return Status::error(
            "GpuCandidateExport::theory_scores_only requires CUDA (PARCAE_BUILD_CUDA=ON)");
#else
        prepare_progress(progress, cipher);
        TheoryExportCache local_cache;
        TheoryExportCache& active = cache != nullptr ? *cache : local_cache;
        TheoryDeviceScratch local_scratch;
        TheoryDeviceScratch& active_scratch = scratch != nullptr ? *scratch : local_scratch;
        CudaStreamPair local_streams;
        CudaStreamPair* active_streams = streams;
        if (active_streams == nullptr) {
            local_streams = CudaStreamPair::create_or_legacy();
            active_streams = &local_streams;
        }
        return fused_theory_scores(cipher, freqs, theories_root, theory_uri_text, params_list,
                                   active, active_scratch, *active_streams, progress);
#endif
    }

#if defined(PARCAE_HAS_CUDA)
    /// Bind + H2D into the write param slab + `record_h2d_done` (no hist yet).
    [[nodiscard]] static StatusOr<TheoryLaunchTicket> theory_stage_scores_only(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list, TheoryExportCache& cache,
        TheoryDeviceScratch& scratch, CudaStreamPair& streams,
        BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        TheoryLaunchTicket ticket;
        StatusOr<std::vector<double>> staged =
            fused_theory_scores(cipher, freqs, theories_root, theory_uri_text, params_list, cache,
                                scratch, streams, progress, TheoryFuseMode::StageH2D, &ticket);
        if (!staged.ok()) {
            return staged.status();
        }
        return ticket;
    }

    /// `wait_h2d` + hist launch + `record_compute_done` for a staged ticket.
    [[nodiscard]] static Status theory_launch_staged(TheoryExportCache& cache,
                                                     TheoryDeviceScratch& scratch,
                                                     CudaStreamPair& streams,
                                                     TheoryLaunchTicket& ticket) {
        if (!ticket.staged()) {
            return Status::error("GpuCandidateExport::theory_launch_staged: ticket not staged");
        }
        Status wait_h2d = streams.wait_h2d_on_compute();
        if (!wait_h2d.ok()) {
            return wait_h2d;
        }
        Status launched = Status::success();
        switch (ticket.kind_) {
        case TheoryLaunchTicket::Kind::S0:
            launched = TheoryHistChi2Launch::launch_bytecode_async(
                scratch.cipher(), cache.device_ops().data(), cache.device_imm().data(),
                ticket.op_count_, scratch.slots(), ticket.slot_count_, ticket.cipher_slot_,
                ticket.index_slot_, ticket.binds_index_i_, ticket.max_stack_, scratch.probs(),
                scratch.counts(), scratch.scores(), scratch.lane_err(), ticket.candidate_count_,
                ticket.token_count_, streams.compute());
            break;
        case TheoryLaunchTicket::Kind::S1:
            launched = TheoryHistChi2Launch::launch_s1_from_slots_async(
                scratch.cipher(), cache.device_ops().data(), cache.device_imm().data(),
                ticket.op_count_, scratch.slots(), ticket.slot_count_, ticket.cipher_slot_,
                ticket.index_slot_, ticket.binds_index_i_, ticket.max_stack_, scratch.luts(),
                scratch.probs(), scratch.counts(), scratch.scores(), scratch.lane_err(),
                ticket.candidate_count_, ticket.token_count_, streams.compute());
            break;
        case TheoryLaunchTicket::Kind::S2:
            launched = TheoryHistChi2Launch::launch_s2_linear_async(
                scratch.cipher(), scratch.b0(), scratch.b1(), scratch.probs(), scratch.counts(),
                scratch.scores(), ticket.candidate_count_, ticket.token_count_,
                ticket.cipher_minus_ks_, streams.compute());
            break;
        case TheoryLaunchTicket::Kind::ShapeAtbash:
            launched = TheoryHistChi2Launch::launch_shape_atbash_async(
                scratch.cipher(), scratch.probs(), scratch.counts(), scratch.scores(),
                ticket.candidate_count_, ticket.token_count_, streams.compute());
            break;
        case TheoryLaunchTicket::Kind::ShapeCaesar:
            launched = TheoryHistChi2Launch::launch_shape_caesar_async(
                scratch.cipher(), scratch.b0(), scratch.probs(), scratch.counts(),
                scratch.scores(), ticket.candidate_count_, ticket.token_count_, streams.compute());
            break;
        case TheoryLaunchTicket::Kind::ShapeAffine:
            launched = TheoryHistChi2Launch::launch_shape_affine_async(
                scratch.cipher(), scratch.b0(), scratch.b1(), scratch.probs(), scratch.counts(),
                scratch.scores(), ticket.candidate_count_, ticket.token_count_, streams.compute());
            break;
        }
        if (!launched.ok()) {
            return launched;
        }
        Status recorded = streams.record_compute_done();
        if (!recorded.ok()) {
            return recorded;
        }
        ticket.phase_ = TheoryLaunchTicket::Phase::InFlight;
        return Status::success();
    }

    /// Collect scores for an in-flight ticket.
    [[nodiscard]] static StatusOr<std::vector<double>>
    theory_collect_scores_only(TheoryDeviceScratch& scratch, CudaStreamPair& streams,
                               TheoryLaunchTicket& ticket,
                               BatchRunner::Progress progress = BatchRunner::Progress{}) {
        if (!ticket.in_flight()) {
            return Status::error(
                "GpuCandidateExport::theory_collect_scores_only: ticket not in flight");
        }
        StatusOr<std::vector<double>> scores =
            collect_theory_scores(scratch, streams, ticket.candidate_count_, progress);
        ticket = TheoryLaunchTicket{};
        return scores;
    }
#endif

private:
    GpuCandidateExport() = delete;

#if defined(PARCAE_HAS_CUDA)
    enum class TheoryFuseMode : std::uint8_t { Full, StageH2D };
#endif

    [[nodiscard]] static Status
    require_theory_params_list(const std::vector<nlohmann::json>& params_list) {
        if (params_list.empty()) {
            return Status::error("GpuCandidateExport: theory params_list must be non-empty");
        }
#if defined(PARCAE_HAS_CUDA)
        if (params_list.size() > TheoryChi2Batch::kMaxCandidates) {
            return Status::error(
                "GpuCandidateExport: theory params_list exceeds TheoryChi2Batch::kMaxCandidates");
        }
#else
        if (params_list.size() > 16384) {
            return Status::error("GpuCandidateExport: theory params_list exceeds 16384");
        }
#endif
        for (std::size_t i = 0; i < params_list.size(); ++i) {
            if (!params_list[i].is_object()) {
                return Status::error(
                    "GpuCandidateExport: theory params_list entries must be objects");
            }
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<const TheoryExportCache::Entry*>
    prepare_theory(TheoryExportCache& cache, const std::filesystem::path& theories_root,
                   std::string_view theory_uri_text, TransformDirection direction) {
        NvtxRange nvtx_prepare("prepare_theory");
        return cache.ensure(theories_root, theory_uri_text, direction);
    }

    static void emit_stage(BatchRunner::Progress& progress, std::string_view stage,
                           std::size_t done, std::size_t total) {
        if (progress.sink == nullptr) {
            return;
        }
        ConsoleProgressSnapshot snap;
        snap.set_stage(std::string(stage));
        snap.set_candidates_done(done);
        snap.set_candidates_total(total);
        snap.set_rune_count(progress.rune_count);
        progress.sink->on_stage(stage, snap);
    }

    static void prepare_progress(BatchRunner::Progress& progress, std::span<const Index29> cipher) {
        if (progress.rune_count == 0) {
            progress.rune_count = cipher.size();
        }
    }

    static void emit_materialize(BatchRunner::Progress& progress, std::size_t retained,
                                 std::size_t grid_size) {
        emit_stage(progress, "materialize", retained, grid_size);
    }

    [[nodiscard]] static Status
    require_vigenere_keys(const std::vector<std::vector<Index29>>& keys) {
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
                return Status::error("GpuCandidateExport: vigenere key_indices must be non-empty");
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

    /// Apply each recipe (ComposeDriver on CUDA, ComposeTransform otherwise), χ², top-k.
    [[nodiscard]] static StatusOr<Result>
    compose_recipes_driver(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                           const std::vector<nlohmann::json>& recipes, std::size_t k,
                           TransformDirection direction,
                           BatchRunner::Progress progress = BatchRunner::Progress{}) {
        prepare_progress(progress, cipher);
        Status common = require_cipher_k(cipher, k);
        if (!common.ok()) {
            return common;
        }


        std::vector<double> scores;
        std::vector<std::vector<Index29>> plains;
        scores.reserve(recipes.size());
        plains.reserve(recipes.size());

        for (std::size_t i = 0; i < recipes.size(); ++i) {
            Status params_ok = ComposeRecipeCandidateGenerator::require_compose_params(recipes[i]);
            if (!params_ok.ok()) {
                return Status::error("GpuCandidateExport::compose recipes[" + std::to_string(i) +
                                     "]: " + params_ok.message());
            }
            StatusOr<std::vector<Index29>> plain =
                apply_compose_recipe(cipher, recipes[i], direction);
            if (!plain.ok()) {
                return Status::error("GpuCandidateExport::compose apply failed for index " +
                                     std::to_string(i) + ": " + plain.status().message());
            }
            StatusOr<double> score = Chi2EnglishGp::score(plain.value(), freqs);
            if (!score.ok()) {
                return score.status();
            }
            scores.push_back(score.value());
            plains.push_back(std::move(plain.value()));
        }

        StatusOr<std::vector<BatchHit>> hits = select_top_k(scores, k, [&](std::size_t index) {
            return ComposeRecipeCandidateGenerator::make_candidate_id(index, recipes[index]);
        });
        if (!hits.ok()) {
            return hits.status();
        }

        const Backend backend =
#if defined(PARCAE_HAS_CUDA)
            ParcaeCuda::available() ? Backend::Cuda : Backend::Cpu;
#else
            Backend::Cpu;
#endif
        std::vector<Row> rows;
        rows.reserve(hits.value().size());
        for (std::size_t rank = 0; rank < hits.value().size(); ++rank) {
            const BatchHit& hit = hits.value()[rank];
            TransformCandidate candidate(hit.candidate_id(), TransformId::compose(), direction,
                                         recipes[hit.source_index()], plains[hit.source_index()]);
            rows.emplace_back(std::move(candidate), hit.score(), rank, hit.source_index());
        }
        emit_materialize(progress, rows.size(), recipes.size());
        return Result{std::move(rows), backend};
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_compose_recipe(std::span<const Index29> cipher, const nlohmann::json& params,
                         TransformDirection direction) {
#if defined(PARCAE_HAS_CUDA)
        if (ParcaeCuda::available()) {
            StatusOr<ComposeParamsHost> recipe = CudaParamsJson::compose_from_json(params);
            if (!recipe.ok()) {
                return recipe.status();
            }
            StatusOr<InterruptDeviceView> view =
                InterruptDeviceView::from_policy(InterruptPolicy::none(), cipher.size());
            if (!view.ok()) {
                return view.status();
            }
            const auto host_in = to_bytes(cipher);
            std::vector<std::uint8_t> host_out(host_in.size(), 0);
            const CudaDir cuda_dir =
                direction == TransformDirection::Encrypt ? CudaDir::Encrypt : CudaDir::Decrypt;
            Status applied = ComposeDriver::apply_host(host_in, host_out, recipe.value(),
                                                       view.value(), cuda_dir);
            if (!applied.ok()) {
                return applied;
            }
            std::vector<Index29> out;
            out.reserve(host_out.size());
            for (const std::uint8_t b : host_out) {
                out.push_back(Index29{b});
            }
            return out;
        }
#endif
        return ComposeTransform{}.apply(cipher, params, direction);
    }

    [[nodiscard]] static Status require_cipher_k(std::span<const Index29> cipher, std::size_t k) {
        if (cipher.empty()) {
            return Status::error("GpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("GpuCandidateExport: k must be >= 1");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::vector<BatchHit>>
    select_top_k(std::span<const double> scores, std::size_t k,
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
            return Status::error("GpuCandidateExport: expected frequency table must have 29 bins");
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

    [[nodiscard]] static StatusOr<DeviceScratch> make_scratch(std::span<const std::uint8_t> host_in,
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
    [[nodiscard]] static StatusOr<std::vector<double>>
    launch_sync_copy(DeviceScratch& scratch, LaunchFn&& launch, const char* sync_label,
                     BatchRunner::Progress progress = BatchRunner::Progress{}) {
        Status launched = launch();
        if (!launched.ok()) {
            return launched;
        }
        Status synced = CudaError::to_status(cudaDeviceSynchronize(), sync_label);
        if (!synced.ok()) {
            return synced;
        }
        emit_stage(progress, "fuse", scratch.C, scratch.C);
        std::vector<double> scores(scratch.C, 0.0);
        {
            NvtxRange nvtx_d2h("d2h");
            Status copied = scratch.scores.copy_to_host(scores);
            if (!copied.ok()) {
                return copied;
            }
        }
        emit_stage(progress, "d2h", scratch.C, scratch.C);
        return scores;
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_caesar_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                        BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                    scratch.value().in.data(), device_shifts.value().data(),
                    scratch.value().probs.data(), scratch.value().counts.data(),
                    scratch.value().scores.data(), C, scratch.value().T);
            },
            "GpuCandidateExport::caesar sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_atbash_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                        BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                    scratch.value().in.data(), scratch.value().probs.data(),
                    scratch.value().counts.data(), scratch.value().scores.data(), C,
                    scratch.value().T);
            },
            "GpuCandidateExport::atbash sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_atbash_caesar_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                               BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                    scratch.value().in.data(), device_shifts.value().data(),
                    scratch.value().probs.data(), scratch.value().counts.data(),
                    scratch.value().scores.data(), C, scratch.value().T);
            },
            "GpuCandidateExport::atbash_caesar sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_affine_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                        BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                    scratch.value().in.data(), device_a.value().data(), device_b.value().data(),
                    scratch.value().probs.data(), scratch.value().counts.data(),
                    scratch.value().scores.data(), C, scratch.value().T);
            },
            "GpuCandidateExport::affine sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_vigenere_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                          const std::vector<std::vector<Index29>>& keys,
                          BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                    scratch.value().in.data(), device_keys.value().data(),
                    device_begin.value().data(), device_len.value().data(),
                    scratch.value().probs.data(), scratch.value().counts.data(),
                    scratch.value().scores.data(), C, scratch.value().T);
            },
            "GpuCandidateExport::vigenere sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_beaufort_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                          const std::vector<std::vector<Index29>>& keys,
                          BatchRunner::Progress progress = BatchRunner::Progress{}) {
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
                return FamilyChi2Batch::launch_beaufort_async(
                    scratch.value().in.data(), device_keys.value().data(),
                    device_begin.value().data(), device_len.value().data(),
                    scratch.value().probs.data(), scratch.value().counts.data(),
                    scratch.value().scores.data(), C, scratch.value().T);
            },
            "GpuCandidateExport::beaufort sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>>
    fused_totient_scores(std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
                         const std::vector<std::size_t>& prime_start_indices,
                         BatchRunner::Progress progress = BatchRunner::Progress{}) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        if (prime_start_indices.empty()) {
            return Status::error(
                "GpuCandidateExport: totient requires a non-empty prime_start_indices list");
        }
        if (prime_start_indices.size() > FamilyChi2Batch::kMaxCandidates) {
            return Status::error(
                "GpuCandidateExport: totient start count exceeds FamilyChi2Batch::kMaxCandidates");
        }

        const std::size_t C = prime_start_indices.size();
        const auto host_in = to_bytes(cipher);
        StatusOr<DeviceScratch> scratch = make_scratch(host_in, freqs, C);
        if (!scratch.ok()) {
            return scratch.status();
        }

        std::size_t max_start = 0;
        for (const std::size_t s : prime_start_indices) {
            if (s > max_start) {
                max_start = s;
            }
        }
        StatusOr<std::vector<Index29>> shifts =
            TotientKeystream::shifts(max_start + scratch.value().T, 0);
        if (!shifts.ok()) {
            return shifts.status();
        }
        std::vector<std::uint8_t> shift_bytes;
        shift_bytes.reserve(shifts.value().size());
        for (const Index29 idx : shifts.value()) {
            shift_bytes.push_back(idx.value());
        }
        std::vector<std::uint32_t> shift_begin(C);
        for (std::size_t i = 0; i < C; ++i) {
            shift_begin[i] = static_cast<std::uint32_t>(prime_start_indices[i]);
        }

        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shift_bytes);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
            DeviceBuffer<std::uint32_t>::from_host(shift_begin);
        if (!device_begin.ok()) {
            return device_begin.status();
        }

        return launch_sync_copy(
            scratch.value(),
            [&]() {
                return FamilyChi2Batch::launch_totient_async(
                    scratch.value().in.data(), device_shifts.value().data(),
                    device_begin.value().data(), scratch.value().probs.data(),
                    scratch.value().counts.data(), scratch.value().scores.data(), C,
                    scratch.value().T);
            },
            "GpuCandidateExport::totient sync", progress);
    }

    [[nodiscard]] static std::optional<std::uint16_t>
    find_slot_index(const Z29Bytecode::Program& prog, std::string_view name) {
        for (std::uint16_t i = 0; i < static_cast<std::uint16_t>(prog.slot_names.size()); ++i) {
            if (prog.slot_names[static_cast<std::size_t>(i)] == name) {
                return i;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] static std::uint64_t fingerprint_bytes(std::span<const std::uint8_t> bytes) {
        std::uint64_t h = 14695981039346656037ull;
        for (const std::uint8_t b : bytes) {
            h ^= static_cast<std::uint64_t>(b);
            h *= 1099511628211ull;
        }
        h ^= static_cast<std::uint64_t>(bytes.size());
        h *= 1099511628211ull;
        return h == 0 ? 1 : h;
    }

    [[nodiscard]] static std::uint64_t fingerprint_doubles(std::span<const double> values) {
        std::uint64_t h = 14695981039346656037ull;
        for (const double v : values) {
            std::uint64_t bits = 0;
            static_assert(sizeof(double) == sizeof(std::uint64_t));
            std::memcpy(&bits, &v, sizeof(bits));
            h ^= bits;
            h *= 1099511628211ull;
        }
        h ^= static_cast<std::uint64_t>(values.size());
        h *= 1099511628211ull;
        return h == 0 ? 1 : h;
    }

    /// Grow-only reserve + skippable cipher/probs H2D into persistent scratch.
    [[nodiscard]] static Status ensure_theory_resident(TheoryDeviceScratch& scratch,
                                                       std::span<const std::uint8_t> host_in,
                                                       const ExpectedFrequencyTable& freqs,
                                                       std::size_t C, cudaStream_t copy_stream) {
        Status cap = scratch.ensure_capacity(C, host_in.size());
        if (!cap.ok()) {
            return cap;
        }
        StatusOr<bool> cipher_up =
            scratch.upload_cipher_async(host_in, fingerprint_bytes(host_in), copy_stream);
        if (!cipher_up.ok()) {
            return cipher_up.status();
        }
        const std::span<const double> probs(freqs.probabilities().data(),
                                            freqs.probabilities().size());
        StatusOr<bool> probs_up =
            scratch.upload_probs_async(probs, fingerprint_doubles(probs), copy_stream);
        if (!probs_up.ok()) {
            return probs_up.status();
        }
        return Status::success();
    }

    /// After H2D on copy: record → wait on compute → launch → record compute done.
    template <typename LaunchFn>
    [[nodiscard]] static Status enqueue_theory_launch(CudaStreamPair& streams, LaunchFn&& launch) {
        Status h2d_ev = streams.record_h2d_done();
        if (!h2d_ev.ok()) {
            return h2d_ev;
        }
        Status wait_h2d = streams.wait_h2d_on_compute();
        if (!wait_h2d.ok()) {
            return wait_h2d;
        }
        Status launched = launch(streams.compute());
        if (!launched.ok()) {
            return launched;
        }
        return streams.record_compute_done();
    }

    /// Wait for compute, D2H scores on copy stream, synchronize copy.
    [[nodiscard]] static StatusOr<std::vector<double>>
    collect_theory_scores(TheoryDeviceScratch& scratch, CudaStreamPair& streams, std::size_t C,
                          BatchRunner::Progress progress = BatchRunner::Progress{}) {
        Status wait_compute = streams.wait_compute_on_copy();
        if (!wait_compute.ok()) {
            return wait_compute;
        }
        emit_stage(progress, "fuse", C, C);
        std::vector<double> scores(C, 0.0);
        {
            NvtxRange nvtx_d2h("d2h");
            Status copied = scratch.download_scores_async(scores, streams.copy());
            if (!copied.ok()) {
                return copied;
            }
            Status d2h_sync = streams.synchronize_copy();
            if (!d2h_sync.ok()) {
                return d2h_sync;
            }
        }
        emit_stage(progress, "d2h", C, C);
        return scores;
    }

    /// Full sync path: enqueue hist then collect scores.
    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<std::vector<double>>
    launch_theory_stream_copy(TheoryDeviceScratch& scratch, CudaStreamPair& streams, std::size_t C,
                              LaunchFn&& launch, const char* /*sync_label*/,
                              BatchRunner::Progress progress = BatchRunner::Progress{}) {
        Status enqueued = enqueue_theory_launch(streams, std::forward<LaunchFn>(launch));
        if (!enqueued.ok()) {
            return enqueued;
        }
        return collect_theory_scores(scratch, streams, C, progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_s0(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const TheoryExportCache::Entry& entry, const std::vector<nlohmann::json>& params_list,
        TheoryExportCache& cache, TheoryDeviceScratch& scratch, CudaStreamPair& streams,
        BatchRunner::Progress progress, TheoryFuseMode mode, TheoryLaunchTicket* ticket_out) {
        const Z29Bytecode::Program& prog = entry.program();
        const std::size_t C = params_list.size();
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
        const std::uint16_t max_stack = prog.max_stack == 0 ? 8 : prog.max_stack;
        if (max_stack > TheoryChi2Batch::kMaxDeviceStack) {
            return Status::error("GpuCandidateExport::theory max_stack exceeds device cap");
        }

        const std::vector<std::uint8_t>& ops = entry.ops_u8();

        std::vector<std::uint8_t> slots(C * slot_count, 0);
        {
            NvtxRange nvtx_bind("bind_slots");
            for (std::size_t c = 0; c < C; ++c) {
                StatusOr<std::vector<Index29>> bound =
                    Z29Bytecode::bind_theory_slots(prog, entry.theory(), params_list[c]);
                if (!bound.ok()) {
                    return bound.status();
                }
                if (bound.value().size() != slot_count) {
                    return Status::error("GpuCandidateExport::theory slot bind size mismatch");
                }
                for (std::uint16_t s = 0; s < slot_count; ++s) {
                    slots[c * slot_count + s] = bound.value()[s].value();
                }
            }
        }

        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            Status prog_up = cache.ensure_device_program();
            if (!prog_up.ok()) {
                return prog_up;
            }
            Status slots_up =
                scratch.upload_slots_async(slots, C, slot_count, streams.copy());
            if (!slots_up.ok()) {
                return slots_up;
            }
            scratch.commit_param_slab();
        }

        cache.note_hist_launch(TheoryHistChi2Emit::Strategy::S0Bytecode);
        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::S0;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
                ticket_out->op_count_ = static_cast<std::uint32_t>(ops.size());
                ticket_out->slot_count_ = slot_count;
                ticket_out->cipher_slot_ = prog.cipher_slot;
                ticket_out->index_slot_ = prog.index_slot;
                ticket_out->max_stack_ = max_stack;
                ticket_out->binds_index_i_ = prog.binds_index_i ? 1u : 0u;
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_bytecode_async(
                scratch.cipher(), cache.device_ops().data(), cache.device_imm().data(),
                static_cast<std::uint32_t>(ops.size()), scratch.slots(), slot_count,
                prog.cipher_slot, prog.index_slot, prog.binds_index_i ? 1u : 0u, max_stack,
                scratch.probs(), scratch.counts(), scratch.scores(), scratch.lane_err(), C,
                host_in.size(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory S0 sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_shape_atbash(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::vector<nlohmann::json>& params_list, TheoryDeviceScratch& scratch,
        CudaStreamPair& streams, BatchRunner::Progress progress, TheoryFuseMode mode,
        TheoryLaunchTicket* ticket_out) {
        NvtxRange nvtx_shape("specialized_shape_atbash");
        const std::size_t C = params_list.size();
        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            scratch.commit_param_slab();
        }

        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::ShapeAtbash;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_shape_atbash_async(
                scratch.cipher(), scratch.probs(), scratch.counts(), scratch.scores(), C,
                host_in.size(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory ShapeAtbash sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_shape_caesar(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const TheoryExportCache::Entry& entry, const std::vector<nlohmann::json>& params_list,
        TheoryDeviceScratch& scratch, CudaStreamPair& streams, BatchRunner::Progress progress,
        TheoryFuseMode mode, TheoryLaunchTicket* ticket_out) {
        NvtxRange nvtx_shape("specialized_shape_caesar");
        if (!entry.hist_plan().shape().has_value()) {
            return Status::error("GpuCandidateExport::theory ShapeCaesar missing shape plan");
        }
        const TheoryHistChi2Emit::ShapePlan& plan = *entry.hist_plan().shape();
        const Z29Bytecode::Program& prog = entry.program();
        const std::size_t C = params_list.size();
        std::vector<std::uint8_t> host_shifts(C, 0);
        std::vector<std::uint8_t> host_dummy(C, 0);
        {
            NvtxRange nvtx_bind("bind_slots_shape_caesar");
            if (plan.has_const_shift()) {
                std::fill(host_shifts.begin(), host_shifts.end(), plan.const_shift());
            } else {
                const std::optional<std::uint16_t> ishift =
                    find_slot_index(prog, plan.shift_name());
                if (!ishift.has_value()) {
                    return Status::error(
                        "GpuCandidateExport::theory ShapeCaesar shift slot not found");
                }
                const std::uint16_t slot_count =
                    static_cast<std::uint16_t>(prog.slot_names.size());
                for (std::size_t c = 0; c < C; ++c) {
                    StatusOr<std::vector<Index29>> bound =
                        Z29Bytecode::bind_theory_slots(prog, entry.theory(), params_list[c]);
                    if (!bound.ok()) {
                        return bound.status();
                    }
                    if (bound.value().size() != slot_count) {
                        return Status::error(
                            "GpuCandidateExport::theory ShapeCaesar slot bind size mismatch");
                    }
                    host_shifts[c] = bound.value()[*ishift].value();
                }
            }
        }

        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            Status up = scratch.upload_b0_b1_async(host_shifts, host_dummy, C, streams.copy());
            if (!up.ok()) {
                return up;
            }
            scratch.commit_param_slab();
        }

        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::ShapeCaesar;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_shape_caesar_async(
                scratch.cipher(), scratch.b0(), scratch.probs(), scratch.counts(),
                scratch.scores(), C, host_in.size(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory ShapeCaesar sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_shape_affine(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const TheoryExportCache::Entry& entry, const std::vector<nlohmann::json>& params_list,
        TheoryDeviceScratch& scratch, CudaStreamPair& streams, BatchRunner::Progress progress,
        TheoryFuseMode mode, TheoryLaunchTicket* ticket_out) {
        NvtxRange nvtx_shape("specialized_shape_affine");
        if (!entry.hist_plan().shape().has_value()) {
            return Status::error("GpuCandidateExport::theory ShapeAffine missing shape plan");
        }
        const TheoryHistChi2Emit::ShapePlan& plan = *entry.hist_plan().shape();
        const Z29Bytecode::Program& prog = entry.program();
        const std::size_t C = params_list.size();
        std::vector<std::uint8_t> host_a(C, 0);
        std::vector<std::uint8_t> host_b(C, 0);
        {
            NvtxRange nvtx_bind("bind_slots_shape_affine");
            const std::optional<std::uint16_t> ia =
                plan.a_name().empty() ? std::nullopt : find_slot_index(prog, plan.a_name());
            const std::optional<std::uint16_t> ib =
                plan.b_name().empty() ? std::nullopt : find_slot_index(prog, plan.b_name());
            if (!plan.has_const_a() && !ia.has_value()) {
                return Status::error("GpuCandidateExport::theory ShapeAffine a slot not found");
            }
            if (!plan.has_const_b() && !plan.b_name().empty() && !ib.has_value()) {
                return Status::error("GpuCandidateExport::theory ShapeAffine b slot not found");
            }
            const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
            for (std::size_t c = 0; c < C; ++c) {
                StatusOr<std::vector<Index29>> bound =
                    Z29Bytecode::bind_theory_slots(prog, entry.theory(), params_list[c]);
                if (!bound.ok()) {
                    return bound.status();
                }
                if (bound.value().size() != slot_count) {
                    return Status::error(
                        "GpuCandidateExport::theory ShapeAffine slot bind size mismatch");
                }
                const std::uint8_t a =
                    plan.has_const_a() ? plan.const_a() : bound.value()[*ia].value();
                const std::uint8_t b =
                    plan.has_const_b()
                        ? plan.const_b()
                        : (ib.has_value() ? bound.value()[*ib].value() : static_cast<std::uint8_t>(0));
                // inv(a) domain: a==0 is undefined in ℤ₂₉ → soft-fallback S0.
                if (a == 0 || !Z29::try_inv(Index29{a}).ok()) {
                    return Status::error(
                        "GpuCandidateExport::theory ShapeAffine inv(a) domain (a==0)");
                }
                host_a[c] = a;
                host_b[c] = b;
            }
        }

        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            Status up = scratch.upload_b0_b1_async(host_a, host_b, C, streams.copy());
            if (!up.ok()) {
                return up;
            }
            scratch.commit_param_slab();
        }

        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::ShapeAffine;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_shape_affine_async(
                scratch.cipher(), scratch.b0(), scratch.b1(), scratch.probs(), scratch.counts(),
                scratch.scores(), C, host_in.size(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory ShapeAffine sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_s1(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const TheoryExportCache::Entry& entry, const std::vector<nlohmann::json>& params_list,
        TheoryExportCache& cache, TheoryDeviceScratch& scratch, CudaStreamPair& streams,
        BatchRunner::Progress progress, TheoryFuseMode mode, TheoryLaunchTicket* ticket_out) {
        NvtxRange nvtx_s1("specialized_s1");
        const Z29Bytecode::Program& prog = entry.program();
        const std::size_t C = params_list.size();
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
        const std::uint16_t max_stack = prog.max_stack == 0 ? 8 : prog.max_stack;
        if (max_stack > TheoryChi2Batch::kMaxDeviceStack) {
            return Status::error("GpuCandidateExport::theory S1 max_stack exceeds device cap");
        }
        // FxOnly residual: stream index must not participate in the LUT bake.
        if (prog.binds_index_i) {
            return Status::error("GpuCandidateExport::theory S1 rejects binds_index_i");
        }

        const std::vector<std::uint8_t>& ops = entry.ops_u8();

        std::vector<std::uint8_t> slots(C * slot_count, 0);
        {
            NvtxRange nvtx_bind("bind_slots_s1");
            for (std::size_t c = 0; c < C; ++c) {
                StatusOr<std::vector<Index29>> bound =
                    Z29Bytecode::bind_theory_slots(prog, entry.theory(), params_list[c]);
                if (!bound.ok()) {
                    return bound.status();
                }
                if (bound.value().size() != slot_count) {
                    return Status::error("GpuCandidateExport::theory S1 slot bind size mismatch");
                }
                for (std::uint16_t s = 0; s < slot_count; ++s) {
                    slots[c * slot_count + s] = bound.value()[s].value();
                }
            }
        }

        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            Status prog_up = cache.ensure_device_program();
            if (!prog_up.ok()) {
                return prog_up;
            }
            Status slots_up =
                scratch.upload_slots_async(slots, C, slot_count, streams.copy());
            if (!slots_up.ok()) {
                return slots_up;
            }
            // LUT table filled on-device after H2D (into launch slab post-commit).
            scratch.commit_param_slab();
        }

        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::S1;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
                ticket_out->op_count_ = static_cast<std::uint32_t>(ops.size());
                ticket_out->slot_count_ = slot_count;
                ticket_out->cipher_slot_ = prog.cipher_slot;
                ticket_out->index_slot_ = prog.index_slot;
                ticket_out->max_stack_ = max_stack;
                ticket_out->binds_index_i_ = 0u;
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_s1_from_slots_async(
                scratch.cipher(), cache.device_ops().data(), cache.device_imm().data(),
                static_cast<std::uint32_t>(ops.size()), scratch.slots(), slot_count,
                prog.cipher_slot, prog.index_slot, /*binds_index_i=*/0u, max_stack, scratch.luts(),
                scratch.probs(), scratch.counts(), scratch.scores(), scratch.lane_err(), C,
                host_in.size(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory S1 sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores_s2(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const TheoryExportCache::Entry& entry, const std::vector<nlohmann::json>& params_list,
        TheoryDeviceScratch& scratch, CudaStreamPair& streams, BatchRunner::Progress progress,
        TheoryFuseMode mode, TheoryLaunchTicket* ticket_out) {
        NvtxRange nvtx_s2("specialized_s2");
        if (!entry.hist_plan().s2_linear().has_value()) {
            return Status::error("GpuCandidateExport::theory S2 missing linear plan");
        }
        const TheoryHistChi2Emit::S2LinearPlan& plan = *entry.hist_plan().s2_linear();
        if (!plan.coeffs_ok()) {
            return Status::error("GpuCandidateExport::theory S2 incomplete linear coeffs");
        }
        const Z29Bytecode::Program& prog = entry.program();
        const std::optional<std::uint16_t> ib0 =
            plan.has_const_b0() ? std::nullopt : find_slot_index(prog, plan.b0_name());
        const std::optional<std::uint16_t> ib1 =
            plan.has_const_b1() ? std::nullopt : find_slot_index(prog, plan.b1_name());
        if (!plan.has_const_b0() && !ib0.has_value()) {
            return Status::error("GpuCandidateExport::theory S2 b0 slot not found");
        }
        if (!plan.has_const_b1() && !ib1.has_value()) {
            return Status::error("GpuCandidateExport::theory S2 b1 slot not found");
        }

        const std::size_t C = params_list.size();
        const std::uint16_t slot_count = static_cast<std::uint16_t>(prog.slot_names.size());
        std::vector<std::uint8_t> host_b0(C, 0);
        std::vector<std::uint8_t> host_b1(C, 0);
        {
            NvtxRange nvtx_bind("bind_slots_s2");
            for (std::size_t c = 0; c < C; ++c) {
                if (plan.has_const_b0() && plan.has_const_b1()) {
                    host_b0[c] = plan.const_b0();
                    host_b1[c] = plan.const_b1();
                    continue;
                }
                StatusOr<std::vector<Index29>> bound =
                    Z29Bytecode::bind_theory_slots(prog, entry.theory(), params_list[c]);
                if (!bound.ok()) {
                    return bound.status();
                }
                if (bound.value().size() != slot_count) {
                    return Status::error("GpuCandidateExport::theory S2 slot bind size mismatch");
                }
                host_b0[c] = plan.has_const_b0() ? plan.const_b0() : bound.value()[*ib0].value();
                host_b1[c] = plan.has_const_b1() ? plan.const_b1() : bound.value()[*ib1].value();
            }
        }

        const auto host_in = to_bytes(cipher);
        {
            NvtxRange nvtx_h2d("h2d");
            Status resident =
                ensure_theory_resident(scratch, host_in, freqs, C, streams.copy());
            if (!resident.ok()) {
                return resident;
            }
            Status coeffs_up =
                scratch.upload_b0_b1_async(host_b0, host_b1, C, streams.copy());
            if (!coeffs_up.ok()) {
                return coeffs_up;
            }
            scratch.commit_param_slab();
        }

        if (mode == TheoryFuseMode::StageH2D) {
            Status h2d_ev = streams.record_h2d_done();
            if (!h2d_ev.ok()) {
                return h2d_ev;
            }
            if (ticket_out != nullptr) {
                ticket_out->phase_ = TheoryLaunchTicket::Phase::Staged;
                ticket_out->kind_ = TheoryLaunchTicket::Kind::S2;
                ticket_out->candidate_count_ = C;
                ticket_out->token_count_ = host_in.size();
                ticket_out->cipher_minus_ks_ = plan.cipher_minus_ks();
            }
            return std::vector<double>{};
        }
        auto launch = [&](cudaStream_t compute) {
            return TheoryHistChi2Launch::launch_s2_linear_async(
                scratch.cipher(), scratch.b0(), scratch.b1(), scratch.probs(), scratch.counts(),
                scratch.scores(), C, host_in.size(), plan.cipher_minus_ks(), compute);
        };
        return launch_theory_stream_copy(scratch, streams, C, launch,
                                         "GpuCandidateExport::theory S2 sync", progress);
    }

    [[nodiscard]] static StatusOr<std::vector<double>> fused_theory_scores(
        std::span<const Index29> cipher, const ExpectedFrequencyTable& freqs,
        const std::filesystem::path& theories_root, std::string_view theory_uri_text,
        const std::vector<nlohmann::json>& params_list, TheoryExportCache& cache,
        TheoryDeviceScratch& scratch, CudaStreamPair& streams,
        BatchRunner::Progress progress = BatchRunner::Progress{},
        TheoryFuseMode mode = TheoryFuseMode::Full, TheoryLaunchTicket* ticket_out = nullptr) {
        Status ok = require_cuda_freqs(freqs);
        if (!ok.ok()) {
            return ok;
        }
        Status common = require_cipher_k(cipher, 1);
        if (!common.ok()) {
            return common;
        }
        Status grid = require_theory_params_list(params_list);
        if (!grid.ok()) {
            return grid;
        }

        StatusOr<const TheoryExportCache::Entry*> prepared =
            prepare_theory(cache, theories_root, theory_uri_text, TransformDirection::Decrypt);
        if (!prepared.ok()) {
            return prepared.status();
        }
        const TheoryExportCache::Entry& entry = *prepared.value();
        const TheoryExportCache::HistPlan& hist = entry.hist_plan();

        // Prefer ShapeInline twins (Atbash/Caesar/Affine) → S1 soft → S1 → S2 → S0.
        if (hist.specialized() && hist.has_shape_atbash_kernel()) {
            StatusOr<std::vector<double>> shape =
                fused_theory_scores_shape_atbash(cipher, freqs, params_list, scratch, streams,
                                                 progress, mode, ticket_out);
            if (shape.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::ShapeInline);
                return shape;
            }
            // Soft fallback S0.
        }
        if (hist.specialized() && hist.has_shape_caesar_kernel()) {
            StatusOr<std::vector<double>> shape =
                fused_theory_scores_shape_caesar(cipher, freqs, entry, params_list, scratch,
                                                 streams, progress, mode, ticket_out);
            if (shape.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::ShapeInline);
                return shape;
            }
            // Soft fallback S0.
        }
        if (hist.specialized() && hist.has_shape_affine_kernel()) {
            StatusOr<std::vector<double>> shape =
                fused_theory_scores_shape_affine(cipher, freqs, entry, params_list, scratch,
                                                 streams, progress, mode, ticket_out);
            if (shape.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::ShapeInline);
                return shape;
            }
            // Soft fallback S0 (incl. inv(a) domain).
        }
        if (hist.specialized() &&
            hist.emitted_strategy() == TheoryHistChi2Emit::Strategy::ShapeInline &&
            hist.has_s1_soft_path()) {
            StatusOr<std::vector<double>> soft =
                fused_theory_scores_s1(cipher, freqs, entry, params_list, cache, scratch, streams,
                                       progress, mode, ticket_out);
            if (soft.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::ShapeInline);
                return soft;
            }
            // Soft fallback S0 (bind / program issues).
        }
        if (hist.specialized() &&
            hist.emitted_strategy() == TheoryHistChi2Emit::Strategy::S1Lut29 && hist.s1_lut()) {
            StatusOr<std::vector<double>> s1 =
                fused_theory_scores_s1(cipher, freqs, entry, params_list, cache, scratch, streams,
                                       progress, mode, ticket_out);
            if (s1.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::S1Lut29);
                return s1;
            }
            // Soft fallback S0 (bind / program issues).
        }
        if (hist.specialized() &&
            hist.emitted_strategy() == TheoryHistChi2Emit::Strategy::S2Uchar4Inline &&
            hist.s2_linear()) {
            StatusOr<std::vector<double>> s2 =
                fused_theory_scores_s2(cipher, freqs, entry, params_list, scratch, streams,
                                       progress, mode, ticket_out);
            if (s2.ok()) {
                cache.note_hist_launch(TheoryHistChi2Emit::Strategy::S2Uchar4Inline);
                return s2;
            }
        }

        return fused_theory_scores_s0(cipher, freqs, entry, params_list, cache, scratch, streams,
                                      progress, mode, ticket_out);
    }
#endif
};
#endif // GPU_CANDIDATE_EXPORT_HPP
