#ifndef HYPOTHESIS_STATUS_HPP
#define HYPOTHESIS_STATUS_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <string>
#include <string_view>

/// Lifecycle of a HypothesisRecord (`parcae.hypothesis.v0`).
enum class HypothesisStatus {
    Draft,
    Proposed,
    Scored,
    Rejected,
    Promoted,
};

class HypothesisStatusUtil {
public:
    [[nodiscard]] static constexpr std::string_view to_string(HypothesisStatus status) noexcept {
        switch (status) {
        case HypothesisStatus::Draft:
            return "draft";
        case HypothesisStatus::Proposed:
            return "proposed";
        case HypothesisStatus::Scored:
            return "scored";
        case HypothesisStatus::Rejected:
            return "rejected";
        case HypothesisStatus::Promoted:
            return "promoted";
        }
        return "draft";
    }

    [[nodiscard]] static StatusOr<HypothesisStatus> from_string(std::string_view text) {
        if (text == "draft") {
            return HypothesisStatus::Draft;
        }
        if (text == "proposed") {
            return HypothesisStatus::Proposed;
        }
        if (text == "scored") {
            return HypothesisStatus::Scored;
        }
        if (text == "rejected") {
            return HypothesisStatus::Rejected;
        }
        if (text == "promoted") {
            return HypothesisStatus::Promoted;
        }
        return Status::error("Unknown hypothesis status: " + std::string(text));
    }

    /// v0 transition table from `hypothesis-workspace.md` (same→same allowed).
    [[nodiscard]] static bool can_transition(HypothesisStatus from, HypothesisStatus to) noexcept {
        if (from == to) {
            return true;
        }
        switch (from) {
        case HypothesisStatus::Draft:
            return to == HypothesisStatus::Proposed;
        case HypothesisStatus::Proposed:
            return to == HypothesisStatus::Scored || to == HypothesisStatus::Rejected ||
                   to == HypothesisStatus::Promoted;
        case HypothesisStatus::Scored:
            return to == HypothesisStatus::Rejected || to == HypothesisStatus::Promoted;
        case HypothesisStatus::Rejected:
        case HypothesisStatus::Promoted:
            return false;
        }
        return false;
    }

    [[nodiscard]] static Status require_transition(HypothesisStatus from, HypothesisStatus to) {
        if (can_transition(from, to)) {
            return Status::success();
        }
        return Status::error(
            "Illegal hypothesis status transition: " + std::string(to_string(from)) + " → " +
            std::string(to_string(to)));
    }

private:
    HypothesisStatusUtil() = delete;
};

#endif  // HYPOTHESIS_STATUS_HPP
