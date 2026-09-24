#ifndef BATCH_RESULT_HPP
#define BATCH_RESULT_HPP

#include "parcae/batch/batch_hit.hpp"

#include <cstddef>
#include <utility>
#include <vector>

/// Outcome of `BatchRunner::run` — `top()` is best-first under the score order.
class BatchResult {
public:
    BatchResult(std::vector<BatchHit> top, std::size_t scored_count, std::string score_id,
                std::string score_version)
        : top_(std::move(top)), scored_count_(scored_count), score_id_(std::move(score_id)),
          score_version_(std::move(score_version)) {}

    [[nodiscard]] const std::vector<BatchHit>& top() const noexcept { return top_; }

    [[nodiscard]] std::size_t scored_count() const noexcept { return scored_count_; }

    [[nodiscard]] const std::string& score_id() const noexcept { return score_id_; }

    [[nodiscard]] const std::string& score_version() const noexcept { return score_version_; }

private:
    std::vector<BatchHit> top_;
    std::size_t scored_count_ = 0;
    std::string score_id_;
    std::string score_version_;
};

#endif // BATCH_RESULT_HPP
