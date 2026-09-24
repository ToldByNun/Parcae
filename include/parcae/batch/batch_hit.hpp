#ifndef BATCH_HIT_HPP
#define BATCH_HIT_HPP

#include <cstddef>
#include <string>
#include <utility>

/// One ranked row from a bounded batch / top-k run.
class BatchHit {
public:
    BatchHit(std::string candidate_id, double score, std::size_t source_index)
        : candidate_id_(std::move(candidate_id)), score_(score), source_index_(source_index) {}

    [[nodiscard]] const std::string& candidate_id() const noexcept { return candidate_id_; }

    [[nodiscard]] double score() const noexcept { return score_; }

    /// Index into the input candidate span (stable tertiary tie-break).
    [[nodiscard]] std::size_t source_index() const noexcept { return source_index_; }

private:
    std::string candidate_id_;
    double score_;
    std::size_t source_index_;
};

#endif // BATCH_HIT_HPP
