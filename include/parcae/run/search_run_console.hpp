#ifndef SEARCH_RUN_CONSOLE_HPP
#define SEARCH_RUN_CONSOLE_HPP

#include "parcae/run/search_run_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/// Human console report for `SearchRunMetrics` (AI-training-dashboard analogue).
class SearchRunConsole {
public:
    [[nodiscard]] static std::string format(const SearchRunMetrics& metrics) {
        std::ostringstream out;
        if (metrics.backend() == "cuda") {
            out << "PARCAE - CUDA SEARCH RUN\n\n";
        } else {
            out << "PARCAE - SEARCH RUN\n\n";
        }
        out << "Transform:      " << display_transform(metrics.transform_id()) << '\n';
        out << "Parameters:     " << metrics.parameters_label() << '\n';
        out << "Seed:           " << metrics.seed() << '\n';
        out << '\n';

        out << "Throughput      " << format_throughput(metrics.tok_per_sec()) << '\n';
        out << score_bars(metrics);
        out << "Fixture Eval    " << metrics.eval_passed() << " / " << metrics.eval_total()
            << '\n';

        if (metrics.cpu_cuda_pass().has_value()) {
            out << '\n';
            out << "CPU <-> CUDA      "
                << (metrics.cpu_cuda_pass().value() ? "PASS" : "FAIL") << '\n';
        }
        return out.str();
    }

private:
    SearchRunConsole() = delete;

    [[nodiscard]] static std::string display_transform(const std::string& id) {
        if (id == "caesar") {
            return "Caesar";
        }
        if (id == "atbash_caesar") {
            return "Atbash+Caesar";
        }
        if (id == "vigenere_key") {
            return "Vigenere";
        }
        if (id == "affine") {
            return "Affine";
        }
        if (id == "atbash") {
            return "Atbash";
        }
        if (id == "compose") {
            return "Compose";
        }
        return id;
    }

    [[nodiscard]] static std::string format_throughput(double tok_per_sec) {
        std::ostringstream out;
        if (tok_per_sec >= 1.0e6) {
            out << std::setprecision(2) << std::fixed << (tok_per_sec / 1.0e6) << "M runes/s";
        } else if (tok_per_sec >= 1.0e3) {
            out << std::setprecision(2) << std::fixed << (tok_per_sec / 1.0e3) << "k runes/s";
        } else {
            out << std::setprecision(2) << std::fixed << tok_per_sec << " runes/s";
        }
        return out.str();
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
            const std::size_t src =
                cols == 1 ? 0 : (i * (steps.size() - 1)) / (cols - 1);
            int h = 1;
            if (span > 0.0) {
                h = 1 + static_cast<int>(std::lround(
                    ((steps[src].score() - lo) / span) * (bar_rows - 1)));
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
        out << "                mean=" << metrics.score_mean()
            << "  std=" << metrics.score_std() << '\n';
        return out.str();
    }
};

#endif // SEARCH_RUN_CONSOLE_HPP
