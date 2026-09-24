#ifndef SEARCH_RUN_CONSOLE_HPP
#define SEARCH_RUN_CONSOLE_HPP

#include "parcae/cli/console_dashboard.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"
#include "parcae/run/search_run_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/// Human console report for `SearchRunMetrics`.
///
/// Shares the `ConsoleDashboard` panel / throughput vocabulary with
/// `parcae-search-cycle` (same `PARCAE  tool …` header and runes/s labels).
/// Sweep score bars remain search-run-specific detail below the panel.
class SearchRunConsole {
public:
    [[nodiscard]] static std::string format(const SearchRunMetrics& metrics) {
        const ConsoleProgressSnapshot snap = to_snapshot(metrics);
        std::ostringstream out;
        out << ConsoleDashboard::format_panel(snap, 28) << '\n';
        out << "params=" << metrics.parameters_label() << "  seed=" << metrics.seed() << '\n';
        out << '\n';
        out << score_bars(metrics);
        out << "Fixture Eval    " << metrics.eval_passed() << " / " << metrics.eval_total() << '\n';
        if (metrics.cpu_cuda_pass().has_value()) {
            out << "CPU <-> CUDA      " << (metrics.cpu_cuda_pass().value() ? "PASS" : "FAIL")
                << '\n';
        }
        out << ConsoleDashboard::format_line(snap) << "  [done]\n";
        return out.str();
    }

private:
    SearchRunConsole() = delete;

    [[nodiscard]] static ConsoleProgressSnapshot to_snapshot(const SearchRunMetrics& metrics) {
        ConsoleProgressSnapshot snap;
        snap.set_tool("search-run");
        snap.set_family(metrics.transform_id());
        snap.set_backend(metrics.backend());
        snap.set_score_id(metrics.score_id());
        snap.set_stage("done");
        const std::size_t n = metrics.steps().size();
        snap.set_candidates_done(n);
        snap.set_candidates_total(n);
        snap.set_runes_per_sec(metrics.tok_per_sec());
        if (n > 0) {
            // chi2-style scores: lower is better — surface the minimum as best.
            std::size_t best_i = 0;
            for (std::size_t i = 1; i < n; ++i) {
                if (metrics.steps()[i].score() < metrics.steps()[best_i].score()) {
                    best_i = i;
                }
            }
            snap.set_best_score(metrics.steps()[best_i].score());
            snap.set_best_label(step_label(metrics.steps()[best_i]));
        }
        return snap;
    }

    [[nodiscard]] static std::string step_label(const SearchRunStep& step) {
        const nlohmann::json& p = step.params();
        if (p.contains("shift") && p.at("shift").is_number_integer()) {
            return "shift=" + std::to_string(p.at("shift").get<int>());
        }
        if (p.contains("a") && p.contains("b")) {
            std::ostringstream out;
            out << "a=" << p.at("a") << ",b=" << p.at("b");
            return out.str();
        }
        if (!step.transform_id().empty()) {
            return step.transform_id();
        }
        return {};
    }

    /// ASCII bar chart over the sweep axis (high score = taller bar). Pure 7-bit.
    [[nodiscard]] static std::string score_bars(const SearchRunMetrics& metrics) {
        const std::vector<SearchRunStep>& steps = metrics.steps();
        if (steps.empty()) {
            return "Score           (no sweep)\n";
        }

        double lo = steps.front().score();
        double hi = lo;
        for (const SearchRunStep& step : steps) {
            lo = std::min(lo, step.score());
            hi = std::max(hi, step.score());
        }
        const double span = hi - lo;
        constexpr int bar_rows = 4;
        constexpr std::size_t max_cols = 29;

        const std::size_t cols = std::min(steps.size(), max_cols);
        std::vector<int> heights(cols, 1);
        for (std::size_t i = 0; i < cols; ++i) {
            const std::size_t src = cols == 1 ? 0 : (i * (steps.size() - 1)) / (cols - 1);
            int h = 1;
            if (span > 0.0) {
                h = 1 + static_cast<int>(
                            std::lround(((steps[src].score() - lo) / span) * (bar_rows - 1)));
            }
            heights[i] = std::clamp(h, 1, bar_rows);
        }

        std::ostringstream out;
        for (int row = bar_rows; row >= 1; --row) {
            if (row == bar_rows) {
                out << "Score           ";
            } else {
                out << "                ";
            }
            for (std::size_t i = 0; i < cols; ++i) {
                out << (heights[i] >= row ? '#' : ' ');
            }
            out << '\n';
        }
        out << "                ";
        for (std::size_t i = 0; i < cols; ++i) {
            out << '-';
        }
        out << "  sweep\n";
        out << std::setprecision(4) << std::fixed;
        out << "                mean=" << metrics.score_mean() << "  std=" << metrics.score_std()
            << '\n';
        return out.str();
    }
};

#endif // SEARCH_RUN_CONSOLE_HPP
