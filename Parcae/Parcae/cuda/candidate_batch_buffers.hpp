#ifndef CANDIDATE_BATCH_BUFFERS_HPP
#define CANDIDATE_BATCH_BUFFERS_HPP

#include "params.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

/// Host-side Candidate Batch ABI v0 SoA staging (`docs/architecture/cuda-abi.md`).
///
/// Candidate-major flat layout: element `(c, t)` lives at `c * T + t`.
/// v0 prefers **shared** ciphertext tokens + per-candidate param lanes (caesar 29,
/// affine 812); optional per-candidate token replication is also supported.
///
/// Built on the host before H2D. No CUDA runtime dependency.
class CandidateBatchBuffers {
public:
    /// Max candidates for a single v0 batch (covers affine 812 + headroom for
    /// explicit key lists). Larger sweeps must be chunked by the caller.
    static constexpr std::size_t kMaxC = 8192;

    /// Max tokens / consumable length per stream (fixture-scale; matches
    /// `InterruptDeviceView::kBitmaskMaxT`).
    static constexpr std::size_t kMaxT = 4096;

    enum class TokenLayout : std::uint8_t {
        Shared = 0,        ///< `token_index29[T]` + params SoA
        PerCandidate = 1,  ///< `token_index29[C * T]`
    };

    /// Options for `allocate`. Param lanes sized to `C` are always reserved for
    /// the declared family; unused lanes stay zero-filled.
    class AllocateOptions {
    public:
        TokenLayout token_layout = TokenLayout::Shared;
        CudaFamilyId family = CudaFamilyId::Caesar;
        bool with_consume_mask = false;
        bool with_scores = true;
        /// Key-byte arena capacity (keyed families). Bytes are packed by the caller.
        std::size_t key_arena_capacity = 0;
    };

    [[nodiscard]] static StatusOr<CandidateBatchBuffers> allocate(
        std::size_t candidate_count,
        std::size_t token_count,
        AllocateOptions options = {}) {
        if (candidate_count == 0) {
            return Status::error("CandidateBatchBuffers: C must be >= 1");
        }
        if (candidate_count > kMaxC) {
            return Status::error("CandidateBatchBuffers: C exceeds kMaxC");
        }
        if (token_count > kMaxT) {
            return Status::error("CandidateBatchBuffers: T exceeds kMaxT");
        }

        CandidateBatchBuffers buffers;
        buffers.candidate_count_ = candidate_count;
        buffers.token_count_ = token_count;
        buffers.token_layout_ = options.token_layout;
        buffers.family_ = options.family;

        const std::size_t flat = candidate_count * token_count;

        if (options.token_layout == TokenLayout::Shared) {
            buffers.token_index29_.assign(token_count, 0);
        } else {
            buffers.token_index29_.assign(flat, 0);
        }

        buffers.out_index29_.assign(flat, 0);
        buffers.directions_.assign(candidate_count, static_cast<std::uint8_t>(CudaDir::Decrypt));

        reserve_param_lanes(buffers, options.family, candidate_count);

        if (options.with_consume_mask) {
            buffers.consume_mask_.assign(flat, 1);  // 1 = consumable
        }
        if (options.with_scores) {
            buffers.scores_.assign(candidate_count, 0.0);
        }
        if (options.key_arena_capacity > 0) {
            buffers.key_bytes_.assign(options.key_arena_capacity, 0);
            buffers.key_begin_.assign(candidate_count, 0);
            buffers.key_len_.assign(candidate_count, 0);
        }

        return buffers;
    }

    [[nodiscard]] std::size_t candidate_count() const noexcept {
        return candidate_count_;
    }

    [[nodiscard]] std::size_t token_count() const noexcept {
        return token_count_;
    }

    [[nodiscard]] TokenLayout token_layout() const noexcept {
        return token_layout_;
    }

    [[nodiscard]] CudaFamilyId family() const noexcept {
        return family_;
    }

    /// Flat index for candidate-major `(c, t)`.
    [[nodiscard]] std::size_t flat_index(std::size_t candidate, std::size_t token) const {
        return candidate * token_count_ + token;
    }

    [[nodiscard]] std::span<std::uint8_t> token_index29() noexcept {
        return token_index29_;
    }

    [[nodiscard]] std::span<const std::uint8_t> token_index29() const noexcept {
        return token_index29_;
    }

    [[nodiscard]] std::span<std::uint8_t> out_index29() noexcept {
        return out_index29_;
    }

    [[nodiscard]] std::span<const std::uint8_t> out_index29() const noexcept {
        return out_index29_;
    }

    [[nodiscard]] std::span<std::uint8_t> consume_mask() noexcept {
        return consume_mask_;
    }

    [[nodiscard]] std::span<const std::uint8_t> consume_mask() const noexcept {
        return consume_mask_;
    }

    [[nodiscard]] std::span<double> scores() noexcept {
        return scores_;
    }

    [[nodiscard]] std::span<const double> scores() const noexcept {
        return scores_;
    }

    [[nodiscard]] std::span<std::uint8_t> directions() noexcept {
        return directions_;
    }

    [[nodiscard]] std::span<const std::uint8_t> directions() const noexcept {
        return directions_;
    }

    [[nodiscard]] std::span<std::uint8_t> caesar_shifts() noexcept {
        return caesar_shifts_;
    }

    [[nodiscard]] std::span<const std::uint8_t> caesar_shifts() const noexcept {
        return caesar_shifts_;
    }

    [[nodiscard]] std::span<std::uint8_t> affine_a() noexcept {
        return affine_a_;
    }

    [[nodiscard]] std::span<const std::uint8_t> affine_a() const noexcept {
        return affine_a_;
    }

    [[nodiscard]] std::span<std::uint8_t> affine_b() noexcept {
        return affine_b_;
    }

    [[nodiscard]] std::span<const std::uint8_t> affine_b() const noexcept {
        return affine_b_;
    }

    [[nodiscard]] std::span<std::uint32_t> key_begin() noexcept {
        return key_begin_;
    }

    [[nodiscard]] std::span<const std::uint32_t> key_begin() const noexcept {
        return key_begin_;
    }

    [[nodiscard]] std::span<std::uint32_t> key_len() noexcept {
        return key_len_;
    }

    [[nodiscard]] std::span<const std::uint32_t> key_len() const noexcept {
        return key_len_;
    }

    [[nodiscard]] std::span<std::uint8_t> key_bytes() noexcept {
        return key_bytes_;
    }

    [[nodiscard]] std::span<const std::uint8_t> key_bytes() const noexcept {
        return key_bytes_;
    }

    /// Pack shared ciphertext tokens (`TokenLayout::Shared` only).
    [[nodiscard]] Status set_shared_tokens(std::span<const Index29> tokens) {
        if (token_layout_ != TokenLayout::Shared) {
            return Status::error("CandidateBatchBuffers::set_shared_tokens requires Shared layout");
        }
        if (tokens.size() != token_count_) {
            return Status::error("CandidateBatchBuffers::set_shared_tokens size mismatch");
        }
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_index29_[i] = tokens[i].value();
        }
        return Status::success();
    }

    /// Pack one candidate's tokens (`TokenLayout::PerCandidate` only).
    [[nodiscard]] Status set_candidate_tokens(
        std::size_t candidate,
        std::span<const Index29> tokens) {
        if (token_layout_ != TokenLayout::PerCandidate) {
            return Status::error(
                "CandidateBatchBuffers::set_candidate_tokens requires PerCandidate layout");
        }
        if (candidate >= candidate_count_) {
            return Status::error("CandidateBatchBuffers::set_candidate_tokens candidate out of range");
        }
        if (tokens.size() != token_count_) {
            return Status::error("CandidateBatchBuffers::set_candidate_tokens size mismatch");
        }
        const std::size_t base = flat_index(candidate, 0);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_index29_[base + i] = tokens[i].value();
        }
        return Status::success();
    }

    /// Read one candidate's `out_index29` lane into `Index29` (host-side D2H helper).
    [[nodiscard]] Status copy_out_candidate(std::size_t candidate, std::span<Index29> dest) const {
        if (candidate >= candidate_count_) {
            return Status::error("CandidateBatchBuffers::copy_out_candidate candidate out of range");
        }
        if (dest.size() != token_count_) {
            return Status::error("CandidateBatchBuffers::copy_out_candidate size mismatch");
        }
        const std::size_t base = flat_index(candidate, 0);
        for (std::size_t i = 0; i < token_count_; ++i) {
            dest[i] = Index29{out_index29_[base + i]};
        }
        return Status::success();
    }

    /// Fill caesar shift lanes `0 .. C-1` (typical `C == 29` decrypt sweep).
    [[nodiscard]] Status fill_caesar_shifts_iota() {
        if (family_ != CudaFamilyId::Caesar) {
            return Status::error("CandidateBatchBuffers::fill_caesar_shifts_iota requires Caesar");
        }
        if (caesar_shifts_.size() != candidate_count_) {
            return Status::error("CandidateBatchBuffers: caesar_shifts lane missing");
        }
        for (std::size_t c = 0; c < candidate_count_; ++c) {
            if (c >= Index29::modulus) {
                return Status::error("CandidateBatchBuffers: caesar shift lane exceeds modulus");
            }
            caesar_shifts_[c] = static_cast<std::uint8_t>(c);
        }
        return Status::success();
    }

    /// Fill affine `(a,b)` lanes in nested `a` then `b` order (`gen_affine`: 28×29).
    [[nodiscard]] Status fill_affine_ab_nested() {
        if (family_ != CudaFamilyId::Affine) {
            return Status::error("CandidateBatchBuffers::fill_affine_ab_nested requires Affine");
        }
        constexpr std::size_t expected = 28u * 29u;
        if (candidate_count_ != expected) {
            return Status::error("CandidateBatchBuffers: affine nested fill requires C == 812");
        }
        std::size_t c = 0;
        for (std::uint8_t a = 1; a <= 28; ++a) {
            for (std::uint8_t b = 0; b < Index29::modulus; ++b) {
                affine_a_[c] = a;
                affine_b_[c] = b;
                ++c;
            }
        }
        return Status::success();
    }

private:
    CandidateBatchBuffers() = default;

    static void reserve_param_lanes(
        CandidateBatchBuffers& buffers,
        CudaFamilyId family,
        std::size_t candidate_count) {
        switch (family) {
        case CudaFamilyId::Caesar:
            buffers.caesar_shifts_.assign(candidate_count, 0);
            break;
        case CudaFamilyId::Affine:
            buffers.affine_a_.assign(candidate_count, 1);
            buffers.affine_b_.assign(candidate_count, 0);
            break;
        case CudaFamilyId::VigenereKey:
        case CudaFamilyId::BeaufortKey:
        case CudaFamilyId::TotientPrimeStream:
        case CudaFamilyId::Identity:
        case CudaFamilyId::Atbash:
        case CudaFamilyId::Compose:
            break;
        }
    }

    std::size_t candidate_count_ = 0;
    std::size_t token_count_ = 0;
    TokenLayout token_layout_ = TokenLayout::Shared;
    CudaFamilyId family_ = CudaFamilyId::Caesar;

    std::vector<std::uint8_t> token_index29_;
    std::vector<std::uint8_t> out_index29_;
    std::vector<std::uint8_t> consume_mask_;
    std::vector<double> scores_;
    std::vector<std::uint8_t> directions_;

    std::vector<std::uint8_t> caesar_shifts_;
    std::vector<std::uint8_t> affine_a_;
    std::vector<std::uint8_t> affine_b_;
    std::vector<std::uint32_t> key_begin_;
    std::vector<std::uint32_t> key_len_;
    std::vector<std::uint8_t> key_bytes_;
};

#endif // CANDIDATE_BATCH_BUFFERS_HPP
