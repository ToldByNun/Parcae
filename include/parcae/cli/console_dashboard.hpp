#ifndef CONSOLE_DASHBOARD_HPP
#define CONSOLE_DASHBOARD_HPP

#include "parcae/cli/console_ansi.hpp"
#include "parcae/cli/console_progress_clock.hpp"
#include "parcae/cli/console_progress_mode.hpp"
#include "parcae/cli/console_progress_sink.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Human stderr dashboard: live panel (TTY) or throttled append-only lines.
///
/// Construct with a **resolved** mode (`Panel` / `Lines` / `Off`). Passing
/// `Auto` is treated as `Lines` (callers should resolve via
/// `ConsoleProgressMode::resolve` first). If Panel is requested but VT enable
/// fails, the dashboard falls back to Lines automatically.
///
/// Output defaults to `stderr`. Tests inject an `std::ostream` and a clock
/// override for deterministic goldens.
class ConsoleDashboard : public ConsoleProgressSink {
public:
    class Options {
    public:
        ConsoleProgressMode mode{ConsoleProgressMode::Kind::Lines};
        /// Minimum seconds between progress paints (stage changes bypass).
        double throttle_seconds = 0.15;
        std::size_t bar_width = 28;
        /// When null, writes to `std::cerr`.
        std::ostream* out = nullptr;
    };

    explicit ConsoleDashboard(Options options)
        : mode_(normalize_mode(options.mode)),
          throttle_seconds_(options.throttle_seconds < 0.0 ? 0.0 : options.throttle_seconds),
          bar_width_(options.bar_width == 0 ? 1 : options.bar_width),
          out_(options.out == nullptr ? &std::cerr : options.out) {}

    [[nodiscard]] ConsoleProgressMode mode() const noexcept {
        return mode_;
    }

    [[nodiscard]] ConsoleProgressClock& clock() noexcept {
        return clock_;
    }

    [[nodiscard]] const ConsoleProgressClock& clock() const noexcept {
        return clock_;
    }

    /// How many panel body lines `format_panel_lines` emits (fixed layout).
    [[nodiscard]] static constexpr std::size_t panel_line_count() noexcept {
        return 6;
    }

    void on_progress(const ConsoleProgressSnapshot& snapshot) override {
        if (mode_.is_off()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        maybe_init_unlocked();
        if (!should_paint_progress_unlocked(snapshot)) {
            return;
        }
        paint_unlocked(snapshot);
    }

    void on_stage(
        std::string_view stage,
        const ConsoleProgressSnapshot& snapshot) override {
        if (mode_.is_off()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        maybe_init_unlocked();
        ConsoleProgressSnapshot copy = snapshot;
        copy.set_stage(std::string(stage));
        last_stage_ = copy.stage();
        paint_unlocked(copy);
    }

    /// Final static line + restore cursor. Leaves the cursor below any panel.
    void finish(const ConsoleProgressSnapshot& snapshot) {
        if (mode_.is_off()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        maybe_init_unlocked();
        if (mode_.is_panel() && panel_rows_painted_ > 0) {
            (*out_) << ConsoleAnsi::show_cursor();
            panel_rows_painted_ = 0;
        }
        (*out_) << format_line(snapshot) << "  [done]\n";
        out_->flush();
        last_paint_elapsed_ = clock_.elapsed_seconds();
    }

    /// Single append-only progress line (no ANSI). Trailing newline omitted.
    [[nodiscard]] static std::string format_line(const ConsoleProgressSnapshot& snap) {
        std::ostringstream out;
        out << '[' << (snap.tool().empty() ? "parcae" : snap.tool()) << "] ";
        out << (snap.stage().empty() ? "?" : snap.stage()) << ' ';
        if (snap.candidates_total().has_value()) {
            out << snap.candidates_done() << '/' << snap.candidates_total().value();
            if (snap.fraction_done().has_value()) {
                out << " (" << format_fixed(100.0 * snap.fraction_done().value(), 1) << "%)";
            }
        } else {
            out << snap.candidates_done() << "/?";
        }
        out << ' ' << format_fixed(snap.candidates_per_sec(), 2) << "c/s";
        out << ' ' << format_throughput(snap.runes_per_sec());
        if (snap.eta_seconds().has_value()) {
            out << " eta=" << format_fixed(snap.eta_seconds().value(), 1) << 's';
        } else {
            out << " eta=?";
        }
        if (snap.best_score().has_value()) {
            out << " best=" << format_fixed(snap.best_score().value(), 4);
            if (!snap.best_label().empty()) {
                out << ' ' << snap.best_label();
            }
        }
        if (snap.iteration_total() > 0) {
            out << " iter=" << snap.iteration_index() << '/' << snap.iteration_total();
        }
        out << " t=" << format_fixed(snap.elapsed_seconds(), 2) << 's';
        return out.str();
    }

    /// Fixed 6-line panel body (no ANSI, no trailing newline after last line).
    [[nodiscard]] static std::vector<std::string> format_panel_lines(
        const ConsoleProgressSnapshot& snap,
        std::size_t bar_width) {
        if (bar_width == 0) {
            bar_width = 1;
        }
        std::vector<std::string> lines;
        lines.reserve(panel_line_count());

        {
            std::ostringstream row;
            row << "PARCAE  " << (snap.tool().empty() ? "parcae" : snap.tool());
            if (!snap.workspace_id().empty()) {
                row << "  ws=" << snap.workspace_id();
            }
            if (!snap.family().empty()) {
                row << "  family=" << snap.family();
            }
            if (!snap.backend().empty()) {
                row << "  backend=" << snap.backend();
            }
            if (!snap.score_id().empty()) {
                row << "  score=" << snap.score_id();
            }
            lines.push_back(row.str());
        }

        {
            std::ostringstream row;
            row << "stage=" << (snap.stage().empty() ? "?" : snap.stage());
            if (snap.iteration_total() > 0) {
                row << "  iter=" << snap.iteration_index() << '/' << snap.iteration_total();
            }
            lines.push_back(row.str());
        }

        {
            std::ostringstream row;
            const double frac =
                snap.fraction_done().has_value() ? snap.fraction_done().value() : 0.0;
            row << '[' << ConsoleAnsi::ascii_bar(frac, bar_width) << "] ";
            if (snap.candidates_total().has_value()) {
                row << format_fixed(100.0 * frac, 1) << "%  " << snap.candidates_done()
                    << '/' << snap.candidates_total().value();
            } else {
                row << "?%  " << snap.candidates_done() << "/?";
            }
            lines.push_back(row.str());
        }

        {
            std::ostringstream row;
            row << "runes=" << snap.rune_count() << "  "
                << format_throughput(snap.runes_per_sec()) << "  "
                << format_fixed(snap.candidates_per_sec(), 2) << " cand/s";
            if (snap.eta_seconds().has_value()) {
                row << "  eta=" << format_fixed(snap.eta_seconds().value(), 1) << 's';
            } else {
                row << "  eta=?";
            }
            lines.push_back(row.str());
        }

        {
            std::ostringstream row;
            row << "best=";
            if (snap.best_score().has_value()) {
                row << format_fixed(snap.best_score().value(), 4);
                if (!snap.best_label().empty()) {
                    row << "  " << snap.best_label();
                }
            } else {
                row << "-";
            }
            lines.push_back(row.str());
        }

        {
            std::ostringstream row;
            row << "elapsed=" << format_fixed(snap.elapsed_seconds(), 2) << 's';
            lines.push_back(row.str());
        }

        return lines;
    }

    [[nodiscard]] static std::string format_panel(
        const ConsoleProgressSnapshot& snap,
        std::size_t bar_width) {
        const std::vector<std::string> lines = format_panel_lines(snap, bar_width);
        std::ostringstream out;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) {
                out << '\n';
            }
            out << lines[i];
        }
        return out.str();
    }

private:
    [[nodiscard]] static ConsoleProgressMode normalize_mode(ConsoleProgressMode mode) {
        if (mode.is_auto()) {
            return ConsoleProgressMode{ConsoleProgressMode::Kind::Lines};
        }
        return mode;
    }

    [[nodiscard]] static std::string format_fixed(double value, int precision) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(precision) << value;
        return out.str();
    }

    [[nodiscard]] static std::string format_throughput(double runes_per_sec) {
        std::ostringstream out;
        if (runes_per_sec >= 1.0e6) {
            out << format_fixed(runes_per_sec / 1.0e6, 2) << "M runes/s";
        } else if (runes_per_sec >= 1.0e3) {
            out << format_fixed(runes_per_sec / 1.0e3, 2) << "k runes/s";
        } else {
            out << format_fixed(runes_per_sec, 2) << " runes/s";
        }
        return out.str();
    }

    void maybe_init_unlocked() {
        if (initialized_) {
            return;
        }
        initialized_ = true;
        if (mode_.is_panel()) {
            if (!ConsoleAnsi::enable_virtual_terminal_stderr()) {
                mode_ = ConsoleProgressMode{ConsoleProgressMode::Kind::Lines};
            } else {
                (*out_) << ConsoleAnsi::hide_cursor();
            }
        }
    }

    [[nodiscard]] bool should_paint_progress_unlocked(
        const ConsoleProgressSnapshot& snapshot) const {
        if (!have_painted_) {
            return true;
        }
        if (snapshot.stage() != last_stage_) {
            return true;
        }
        const double now = clock_.elapsed_seconds();
        if (now - last_paint_elapsed_ + 1e-12 >= throttle_seconds_) {
            return true;
        }
        return false;
    }

    void paint_unlocked(const ConsoleProgressSnapshot& snapshot) {
        last_stage_ = snapshot.stage();
        if (mode_.is_panel()) {
            paint_panel_unlocked(snapshot);
        } else {
            (*out_) << format_line(snapshot) << '\n';
        }
        out_->flush();
        have_painted_ = true;
        last_paint_elapsed_ = clock_.elapsed_seconds();
    }

    void paint_panel_unlocked(const ConsoleProgressSnapshot& snapshot) {
        const std::vector<std::string> lines =
            format_panel_lines(snapshot, bar_width_);
        if (panel_rows_painted_ > 0) {
            (*out_) << ConsoleAnsi::cursor_up(panel_rows_painted_);
        }
        for (const std::string& line : lines) {
            (*out_) << line << ConsoleAnsi::clear_to_eol() << '\n';
        }
        panel_rows_painted_ = lines.size();
    }

    ConsoleProgressMode mode_;
    double throttle_seconds_ = 0.15;
    std::size_t bar_width_ = 28;
    std::ostream* out_ = nullptr;
    ConsoleProgressClock clock_{};
    std::mutex mutex_;
    bool initialized_ = false;
    bool have_painted_ = false;
    std::size_t panel_rows_painted_ = 0;
    double last_paint_elapsed_ = 0.0;
    std::string last_stage_;
};

#endif // CONSOLE_DASHBOARD_HPP
