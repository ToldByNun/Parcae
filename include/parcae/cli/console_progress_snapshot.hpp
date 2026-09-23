#ifndef CONSOLE_PROGRESS_SNAPSHOT_HPP
#define CONSOLE_PROGRESS_SNAPSHOT_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

/// One observe-only progress tick for human CLI dashboards.
///
/// Digests / ranking MUST NOT depend on this type. Fields are informational;
/// unknown totals use `std::nullopt` (ETA / percent may be unavailable).
class ConsoleProgressSnapshot {
public:
    ConsoleProgressSnapshot() = default;

    void set_tool(std::string value) {
        tool_ = std::move(value);
    }

    void set_workspace_id(std::string value) {
        workspace_id_ = std::move(value);
    }

    void set_family(std::string value) {
        family_ = std::move(value);
    }

    void set_backend(std::string value) {
        backend_ = std::move(value);
    }

    void set_score_id(std::string value) {
        score_id_ = std::move(value);
    }

    void set_stage(std::string value) {
        stage_ = std::move(value);
    }

    void set_candidates_done(std::size_t value) noexcept {
        candidates_done_ = value;
    }

    void set_candidates_total(std::optional<std::size_t> value) noexcept {
        candidates_total_ = value;
    }

    void set_rune_count(std::size_t value) noexcept {
        rune_count_ = value;
    }

    void set_elapsed_seconds(double value) noexcept {
        elapsed_seconds_ = value;
    }

    void set_runes_per_sec(double value) noexcept {
        runes_per_sec_ = value;
    }

    void set_candidates_per_sec(double value) noexcept {
        candidates_per_sec_ = value;
    }

    void set_best_score(std::optional<double> value) noexcept {
        best_score_ = value;
    }

    void set_best_label(std::string value) {
        best_label_ = std::move(value);
    }

    void set_iteration(std::size_t index, std::size_t total) noexcept {
        iteration_index_ = index;
        iteration_total_ = total;
    }

    [[nodiscard]] const std::string& tool() const noexcept {
        return tool_;
    }

    [[nodiscard]] const std::string& workspace_id() const noexcept {
        return workspace_id_;
    }

    [[nodiscard]] const std::string& family() const noexcept {
        return family_;
    }

    [[nodiscard]] const std::string& backend() const noexcept {
        return backend_;
    }

    [[nodiscard]] const std::string& score_id() const noexcept {
        return score_id_;
    }

    [[nodiscard]] const std::string& stage() const noexcept {
        return stage_;
    }

    [[nodiscard]] std::size_t candidates_done() const noexcept {
        return candidates_done_;
    }

    [[nodiscard]] const std::optional<std::size_t>& candidates_total() const noexcept {
        return candidates_total_;
    }

    [[nodiscard]] std::size_t rune_count() const noexcept {
        return rune_count_;
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        return elapsed_seconds_;
    }

    [[nodiscard]] double runes_per_sec() const noexcept {
        return runes_per_sec_;
    }

    [[nodiscard]] double candidates_per_sec() const noexcept {
        return candidates_per_sec_;
    }

    [[nodiscard]] const std::optional<double>& best_score() const noexcept {
        return best_score_;
    }

    [[nodiscard]] const std::string& best_label() const noexcept {
        return best_label_;
    }

    [[nodiscard]] std::size_t iteration_index() const noexcept {
        return iteration_index_;
    }

    [[nodiscard]] std::size_t iteration_total() const noexcept {
        return iteration_total_;
    }

    /// Fraction in `[0, 1]` when total is known and non-zero; otherwise nullopt.
    [[nodiscard]] std::optional<double> fraction_done() const noexcept {
        if (!candidates_total_.has_value() || candidates_total_.value() == 0) {
            return std::nullopt;
        }
        const double total = static_cast<double>(candidates_total_.value());
        const double done = static_cast<double>(candidates_done_);
        double frac = done / total;
        if (frac < 0.0) {
            frac = 0.0;
        }
        if (frac > 1.0) {
            frac = 1.0;
        }
        return frac;
    }

    /// Seconds remaining when total + positive candidates/s are known.
    [[nodiscard]] std::optional<double> eta_seconds() const noexcept {
        if (!candidates_total_.has_value()) {
            return std::nullopt;
        }
        if (candidates_per_sec_ <= 0.0) {
            return std::nullopt;
        }
        const std::size_t total = candidates_total_.value();
        if (candidates_done_ >= total) {
            return 0.0;
        }
        const double remaining = static_cast<double>(total - candidates_done_);
        return remaining / candidates_per_sec_;
    }

    /// Recompute throughput fields from done counts and elapsed.
    /// `runes_per_sec ≈ (candidates_done * rune_count) / elapsed`.
    void refresh_rates() noexcept {
        if (elapsed_seconds_ <= 0.0) {
            runes_per_sec_ = 0.0;
            candidates_per_sec_ = 0.0;
            return;
        }
        candidates_per_sec_ =
            static_cast<double>(candidates_done_) / elapsed_seconds_;
        runes_per_sec_ = candidates_per_sec_ * static_cast<double>(rune_count_);
    }

private:
    std::string tool_;
    std::string workspace_id_;
    std::string family_;
    std::string backend_;
    std::string score_id_;
    std::string stage_;
    std::size_t candidates_done_ = 0;
    std::optional<std::size_t> candidates_total_;
    std::size_t rune_count_ = 0;
    double elapsed_seconds_ = 0.0;
    double runes_per_sec_ = 0.0;
    double candidates_per_sec_ = 0.0;
    std::optional<double> best_score_;
    std::string best_label_;
    std::size_t iteration_index_ = 0;
    std::size_t iteration_total_ = 0;
};

#endif // CONSOLE_PROGRESS_SNAPSHOT_HPP
