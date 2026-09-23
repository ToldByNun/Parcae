#ifndef CONSOLE_PROGRESS_CLOCK_HPP
#define CONSOLE_PROGRESS_CLOCK_HPP

#include <chrono>
#include <optional>

/// Steady-clock wrapper for CLI progress elapsed time.
///
/// Production code uses wall steady time from `restart()`. Tests MAY inject a
/// fixed elapsed via `set_elapsed_override` so rates / formatting stay deterministic.
class ConsoleProgressClock {
public:
    ConsoleProgressClock() {
        restart();
    }

    void restart() noexcept {
        start_ = std::chrono::steady_clock::now();
        override_seconds_.reset();
    }

    /// When set, `elapsed_seconds()` returns this value instead of the wall clock.
    void set_elapsed_override(std::optional<double> seconds) noexcept {
        override_seconds_ = seconds;
    }

    [[nodiscard]] const std::optional<double>& elapsed_override() const noexcept {
        return override_seconds_;
    }

    [[nodiscard]] double elapsed_seconds() const noexcept {
        if (override_seconds_.has_value()) {
            return override_seconds_.value();
        }
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> delta = now - start_;
        return delta.count();
    }

private:
    std::chrono::steady_clock::time_point start_{};
    std::optional<double> override_seconds_;
};

#endif // CONSOLE_PROGRESS_CLOCK_HPP
