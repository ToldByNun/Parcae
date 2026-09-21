#ifndef GPU_CANDIDATE_EXPORT_HPP
#define GPU_CANDIDATE_EXPORT_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/batch/batch_ordering.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/generate/caesar_candidate_generator.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/score/expected_frequency_table.hpp"
#include "parcae/score/score_order.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "parcae_cuda.hpp"

#include <cuda_runtime_api.h>
#endif

/// Fused GPU (or host-scored) export → top-k `TransformCandidate`s.
///
/// Caesar v0: scores-only fused χ² (`CaesarChi2Batch`) → D2H scores → host
/// `BatchOrdering` → apply transform **only** for retained lanes (search-loop.md).
class GpuCandidateExport {
public:
    static constexpr std::string_view score_id = "chi2_english_gp_v0";
    static constexpr std::string_view score_version = "v0";

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

        /// Lane index in the family grid (Caesar: shift 0..28).
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

    /// Materialize Caesar top-k from a host score vector indexed by shift.
    /// `scores_by_shift.size()` MUST be 29. Used by the CUDA path after D2H and
    /// by CPU tests that inject oracle scores (no device required).
    [[nodiscard]] static StatusOr<Result> caesar_from_host_scores(
        std::span<const Index29> cipher,
        std::span<const double> scores_by_shift,
        std::size_t k,
        TransformDirection direction = TransformDirection::Decrypt,
        parcae::tool::Backend backend = parcae::tool::Backend::Cpu) {
        if (cipher.empty()) {
            return Status::error("GpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("GpuCandidateExport: k must be >= 1");
        }
        if (scores_by_shift.size() != Index29::modulus) {
            return Status::error("GpuCandidateExport: scores_by_shift must have length 29");
        }
        if (direction != TransformDirection::Decrypt && direction != TransformDirection::Encrypt) {
            return Status::error("GpuCandidateExport: invalid direction");
        }

        std::vector<BatchHit> hits;
        hits.reserve(Index29::modulus);
        for (std::size_t shift = 0; shift < Index29::modulus; ++shift) {
            hits.emplace_back(
                CaesarCandidateGenerator::make_candidate_id(static_cast<std::uint8_t>(shift)),
                scores_by_shift[shift],
                shift);
        }
        std::sort(hits.begin(), hits.end(), BatchOrdering::BestFirst{ScoreOrder::Asc});
        if (hits.size() > k) {
            hits.erase(hits.begin() + static_cast<std::ptrdiff_t>(k), hits.end());
        }

        const CaesarTransform transform;
        std::vector<Row> rows;
        rows.reserve(hits.size());
        for (std::size_t rank = 0; rank < hits.size(); ++rank) {
            const BatchHit& hit = hits[rank];
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

    /// Fused Caesar decrypt χ² on device → host top-k materialization.
    /// Requires `PARCAE_HAS_CUDA` and a usable CUDA device at runtime.
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
        if (!ParcaeCuda::available()) {
            return Status::error("GpuCandidateExport::caesar: no CUDA device available");
        }
        if (cipher.empty()) {
            return Status::error("GpuCandidateExport: ciphertext must be non-empty");
        }
        if (k == 0) {
            return Status::error("GpuCandidateExport: k must be >= 1");
        }
        if (freqs.probabilities().size() != Index29::modulus) {
            return Status::error("GpuCandidateExport: expected frequency table must have 29 bins");
        }

        constexpr std::size_t C = Index29::modulus;
        const std::size_t T = cipher.size();
        std::vector<std::uint8_t> host_in(T);
        for (std::size_t i = 0; i < T; ++i) {
            host_in[i] = cipher[i].value();
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
            std::span<const double>(freqs.probabilities().data(), freqs.probabilities().size()));
        if (!device_probs.ok()) {
            return device_probs.status();
        }
        StatusOr<DeviceBuffer<std::uint32_t>> device_counts =
            DeviceBuffer<std::uint32_t>::allocate(C * Index29::modulus);
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
        Status synced =
            CudaError::to_status(cudaDeviceSynchronize(), "GpuCandidateExport::caesar sync");
        if (!synced.ok()) {
            return synced;
        }

        std::vector<double> scores(C, 0.0);
        Status copied = device_scores.value().copy_to_host(scores);
        if (!copied.ok()) {
            return copied;
        }

        return caesar_from_host_scores(
            cipher, scores, k, TransformDirection::Decrypt, parcae::tool::Backend::Cuda);
#endif
    }

private:
    GpuCandidateExport() = delete;
};

#endif // GPU_CANDIDATE_EXPORT_HPP
