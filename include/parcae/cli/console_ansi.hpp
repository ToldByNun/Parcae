#ifndef CONSOLE_ANSI_HPP
#define CONSOLE_ANSI_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

/// Pure ASCII / ANSI helpers for human CLI progress panels.
///
/// Escape builders are side-effect free (unit-testable). TTY / VT probes talk to
/// the real stderr handle unless an override is set for tests.
///
/// Classes only — no namespaces. Best-effort VT enable on Windows: failure is
/// not an error (callers fall back to line mode).
class ConsoleAnsi {
public:
    ConsoleAnsi() = delete;

    /// CSI CUP-relative: move cursor up `lines` rows. `lines == 0` → empty.
    [[nodiscard]] static std::string cursor_up(std::size_t lines) {
        if (lines == 0) {
            return {};
        }
        return std::string("\x1b[") + std::to_string(lines) + "A";
    }

    /// CSI EL 0: clear from cursor to end of line.
    [[nodiscard]] static std::string clear_to_eol() { return std::string("\x1b[K"); }

    [[nodiscard]] static std::string carriage_return() { return std::string("\r"); }

    /// Hide / show cursor (DEC private modes). Panel painters MAY use these.
    [[nodiscard]] static std::string hide_cursor() { return std::string("\x1b[?25l"); }

    [[nodiscard]] static std::string show_cursor() { return std::string("\x1b[?25h"); }

    /// 7-bit progress bar: `#` filled, `-` empty. `fraction` clamped to `[0, 1]`.
    [[nodiscard]] static std::string ascii_bar(double fraction, std::size_t width) {
        if (width == 0) {
            return {};
        }
        double frac = fraction;
        if (!(frac > 0.0)) {
            frac = 0.0;
        }
        if (frac > 1.0) {
            frac = 1.0;
        }
        const std::size_t filled =
            static_cast<std::size_t>(std::lround(frac * static_cast<double>(width)));
        const std::size_t n = std::min(filled, width);
        return std::string(n, '#') + std::string(width - n, '-');
    }

    /// Whether stderr is an interactive terminal (or the test override).
    [[nodiscard]] static bool stderr_is_tty() {
        if (tty_override().has_value()) {
            return tty_override().value();
        }
#if defined(_WIN32)
        return _isatty(_fileno(stderr)) != 0;
#else
        return ::isatty(::fileno(stderr)) != 0;
#endif
    }

    /// Best-effort enable ANSI VT processing on Windows stderr.
    /// On non-Windows returns `stderr_is_tty()` (assume ANSI on TTYs).
    /// Returns false when the console rejects VT — callers MUST fall back.
    [[nodiscard]] static bool enable_virtual_terminal_stderr() {
        if (vt_override().has_value()) {
            return vt_override().value();
        }
#if defined(_WIN32)
        const HANDLE handle = GetStdHandle(STD_ERROR_HANDLE);
        if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
            return false;
        }
        DWORD mode = 0;
        if (GetConsoleMode(handle, &mode) == 0) {
            return false;
        }
        mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (SetConsoleMode(handle, mode) == 0) {
            return false;
        }
        return true;
#else
        return stderr_is_tty();
#endif
    }

    /// Test hook: force `stderr_is_tty()` result. `nullopt` clears override.
    static void set_stderr_tty_override(std::optional<bool> value) noexcept {
        tty_override() = value;
    }

    /// Test hook: force `enable_virtual_terminal_stderr()` result.
    static void set_vt_enable_override(std::optional<bool> value) noexcept {
        vt_override() = value;
    }

    static void clear_overrides() noexcept {
        tty_override().reset();
        vt_override().reset();
    }

private:
    [[nodiscard]] static std::optional<bool>& tty_override() noexcept {
        static std::optional<bool> value;
        return value;
    }

    [[nodiscard]] static std::optional<bool>& vt_override() noexcept {
        static std::optional<bool> value;
        return value;
    }
};

#endif // CONSOLE_ANSI_HPP
