#ifndef BATCH_ORDERING_HPP
#define BATCH_ORDERING_HPP

#include "parcae/batch/batch_hit.hpp"
#include "parcae/score/score_order.hpp"

#include <cstddef>
#include <string>

/// Deterministic total order for batch / top-k reduction (parity contract).
///
/// **Ordering policy (locked for CPU↔CUDA / multi-thread parity):**
/// 1. Better score first — `ScoreOrder::Desc` ⇒ larger IEEE-754 value wins;
///    `ScoreOrder::Asc` ⇒ smaller wins. Equality uses exact binary64 `==`.
/// 2. On equal score: lexicographically smaller `candidate_id` wins.
/// 3. On equal id (should be rare): smaller `source_index` wins.
///
/// Parallel / `std::execution` paths MAY compute scores in any order, but MUST
/// materialize a per-index score array then apply this same total order in a
/// single-threaded reduction. They MUST NOT use lock-free heaps or
/// first-to-finish insertion that depends on scheduling.
class BatchOrdering {
public:
    /// True if `lhs` should rank strictly better (earlier) than `rhs`.
    [[nodiscard]] static bool better(const BatchHit& lhs, const BatchHit& rhs,
                                     ScoreOrder order) noexcept {
        if (lhs.score() != rhs.score()) {
            if (order == ScoreOrder::Desc) {
                return lhs.score() > rhs.score();
            }
            return lhs.score() < rhs.score();
        }
        if (lhs.candidate_id() != rhs.candidate_id()) {
            return lhs.candidate_id() < rhs.candidate_id();
        }
        return lhs.source_index() < rhs.source_index();
    }

    /// Comparator for `std::sort` into best-first order.
    class BestFirst {
    public:
        explicit BestFirst(ScoreOrder order) noexcept : order_(order) {}

        [[nodiscard]] bool operator()(const BatchHit& a, const BatchHit& b) const noexcept {
            return BatchOrdering::better(a, b, order_);
        }

    private:
        ScoreOrder order_;
    };
};

#endif // BATCH_ORDERING_HPP
