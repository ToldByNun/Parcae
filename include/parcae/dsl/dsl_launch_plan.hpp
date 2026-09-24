#ifndef DSL_LAUNCH_PLAN_HPP
#define DSL_LAUNCH_PLAN_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"
#include "parcae/dsl/theory_ir.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

/// CUDA launch geometry for DSL-emitted twins (docs/architecture/python-transpiler.md).
/// Mirrors hand twins: 1D elementwise `ceil(n/256)×256` (`CaesarKernel`) and
/// HistFast χ² `dim3(C, tiles_for(T))` with `threads=256` (`FamilyChi2Batch`).
/// Host-only arithmetic — no CUDA runtime dependency.
class DslLaunchPlan {
public:
    /// Same as `HistFast::threads` / twin `kThreadsPerBlock`.
    static constexpr int threads_per_block = 256;
    /// Same as `HistFast::max_tiles`.
    static constexpr int hist_max_tiles = 1024;
    /// Same as `HistFast::alphabet`.
    static constexpr int hist_alphabet = 29;

    enum class Kind : std::uint8_t {
        /// Stream / elementwise kernel: `<<<blocks_1d(n), 256>>>`.
        Elementwise1D = 0,
        /// Batch materialize over `C·T` flat indices (same 1D formula on `C*T`).
        BatchFlat1D,
        /// Fused hist χ²: `dim3(C, tiles_for(T))`, block 256.
        HistChi2_2D,
    };

    class Plan {
    public:
        Plan(Kind kind, int grid_x, int grid_y, int threads, std::size_t work_items,
             std::size_t candidates, std::size_t tokens)
            : kind_(kind), grid_x_(grid_x), grid_y_(grid_y), threads_(threads),
              work_items_(work_items), candidates_(candidates), tokens_(tokens) {}

        [[nodiscard]] Kind kind() const noexcept { return kind_; }

        [[nodiscard]] int grid_x() const noexcept { return grid_x_; }

        [[nodiscard]] int grid_y() const noexcept { return grid_y_; }

        [[nodiscard]] int threads() const noexcept { return threads_; }

        /// Elements covered (stream length, or `C*T`, or hist token span).
        [[nodiscard]] std::size_t work_items() const noexcept { return work_items_; }

        [[nodiscard]] std::size_t candidates() const noexcept { return candidates_; }

        [[nodiscard]] std::size_t tokens() const noexcept { return tokens_; }

        [[nodiscard]] bool is_empty_launch() const noexcept {
            return grid_x_ == 0 || (kind_ == Kind::HistChi2_2D && grid_y_ == 0);
        }

    private:
        Kind kind_ = Kind::Elementwise1D;
        int grid_x_ = 0;
        int grid_y_ = 1;
        int threads_ = threads_per_block;
        std::size_t work_items_ = 0;
        std::size_t candidates_ = 1;
        std::size_t tokens_ = 0;
    };

    /// Bit-identical to `HistFast::tiles_for` / `CaesarChi2Batch::tiles_for`.
    [[nodiscard]] static int tiles_for(std::size_t token_count) noexcept {
        const std::size_t packs = (token_count + 3u) / 4u;
        const int by_work =
            static_cast<int>((packs + static_cast<std::size_t>(threads_per_block) - 1u) /
                             static_cast<std::size_t>(threads_per_block));
        if (by_work < 1) {
            return 1;
        }
        return by_work < hist_max_tiles ? by_work : hist_max_tiles;
    }

    /// `ceil(count / threads_per_block)`; `count == 0` → `0` (twins skip launch).
    [[nodiscard]] static int blocks_1d(std::size_t count) noexcept {
        if (count == 0) {
            return 0;
        }
        return static_cast<int>((count + static_cast<std::size_t>(threads_per_block) - 1u) /
                                static_cast<std::size_t>(threads_per_block));
    }

    [[nodiscard]] static StatusOr<Plan> elementwise_1d(std::size_t count) {
        return Plan{Kind::Elementwise1D, blocks_1d(count), 1, threads_per_block, count, 1, count};
    }

    [[nodiscard]] static StatusOr<Plan> batch_flat_1d(std::size_t candidates, std::size_t tokens) {
        if (candidates == 0) {
            return fail("batch_flat_1d: candidates must be > 0");
        }
        if (tokens == 0) {
            return fail("batch_flat_1d: tokens must be > 0");
        }
        const std::size_t work = candidates * tokens;
        return Plan{Kind::BatchFlat1D, blocks_1d(work), 1, threads_per_block, work,
                    candidates,        tokens};
    }

    [[nodiscard]] static StatusOr<Plan> hist_chi2_2d(std::size_t candidates, std::size_t tokens) {
        if (candidates == 0) {
            return fail("hist_chi2_2d: candidates must be > 0");
        }
        if (tokens == 0) {
            return fail("hist_chi2_2d: tokens must be > 0");
        }
        return Plan{Kind::HistChi2_2D,
                    static_cast<int>(candidates),
                    tiles_for(tokens),
                    threads_per_block,
                    tokens,
                    candidates,
                    tokens};
    }

    /// Default plan for a theory family (v0: elementwise / keyed_* → 1D stream).
    [[nodiscard]] static StatusOr<Plan> for_theory(const TheoryIr& theory, std::size_t stream_len,
                                                   std::size_t candidates = 1) {
        Status st = theory.validate();
        if (!st.ok()) {
            return st;
        }
        switch (theory.family()) {
        case TheoryIr::Family::Elementwise:
        case TheoryIr::Family::KeyedStream:
        case TheoryIr::Family::KeyedPermutation:
            if (candidates == 1) {
                return elementwise_1d(stream_len);
            }
            return batch_flat_1d(candidates, stream_len);
        }
        return fail("for_theory: unknown family for theory '" + theory.name() + "'");
    }

    /// Prefer hist 2D when compiling a fused χ² sweep façade (explicit opt-in).
    [[nodiscard]] static StatusOr<Plan> for_fused_chi2(std::size_t candidates, std::size_t tokens) {
        return hist_chi2_2d(candidates, tokens);
    }

    [[nodiscard]] static const char* kind_str(Kind k) noexcept {
        switch (k) {
        case Kind::Elementwise1D:
            return "elementwise_1d";
        case Kind::BatchFlat1D:
            return "batch_flat_1d";
        case Kind::HistChi2_2D:
            return "hist_chi2_2d";
        }
        return "unknown";
    }

private:
    DslLaunchPlan() = delete;

    [[nodiscard]] static Status fail(std::string message) {
        return DslDiag::make(DslRuleId::E032_primitive_body, std::move(message)).to_status();
    }
};

#endif // DSL_LAUNCH_PLAN_HPP
