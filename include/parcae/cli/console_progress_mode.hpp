#ifndef CONSOLE_PROGRESS_MODE_HPP
#define CONSOLE_PROGRESS_MODE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cctype>
#include <string>
#include <string_view>

/// How human-facing CLI progress should render.
///
/// Resolution (see `resolve`):
/// - `Off` stays off.
/// - `Panel` / `Lines` stay forced.
/// - `Auto` becomes `Panel` when stderr is a TTY, otherwise `Lines`.
class ConsoleProgressMode {
public:
    enum class Kind {
        Auto,
        Panel,
        Lines,
        Off,
    };

    ConsoleProgressMode() = default;

    explicit ConsoleProgressMode(Kind kind) noexcept : kind_(kind) {}

    [[nodiscard]] Kind kind() const noexcept { return kind_; }

    [[nodiscard]] bool is_off() const noexcept { return kind_ == Kind::Off; }

    [[nodiscard]] bool is_auto() const noexcept { return kind_ == Kind::Auto; }

    [[nodiscard]] bool is_panel() const noexcept { return kind_ == Kind::Panel; }

    [[nodiscard]] bool is_lines() const noexcept { return kind_ == Kind::Lines; }

    [[nodiscard]] std::string_view to_string() const noexcept {
        switch (kind_) {
        case Kind::Auto:
            return "auto";
        case Kind::Panel:
            return "panel";
        case Kind::Lines:
            return "lines";
        case Kind::Off:
            return "off";
        }
        return "auto";
    }

    /// Parse `auto|panel|lines|off` (case-insensitive). Empty → error.
    [[nodiscard]] static StatusOr<ConsoleProgressMode> parse(std::string_view text) {
        std::string lowered;
        lowered.reserve(text.size());
        for (char c : text) {
            lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (lowered == "auto") {
            return ConsoleProgressMode{Kind::Auto};
        }
        if (lowered == "panel") {
            return ConsoleProgressMode{Kind::Panel};
        }
        if (lowered == "lines") {
            return ConsoleProgressMode{Kind::Lines};
        }
        if (lowered == "off") {
            return ConsoleProgressMode{Kind::Off};
        }
        return Status::error("ConsoleProgressMode: expected auto|panel|lines|off, got '" +
                             std::string(text) + "'");
    }

    /// Map CLI flags to a requested mode before TTY resolution.
    ///
    /// Precedence: `quiet` → Off; else `plain_progress` → Lines; else `progress_flag`
    /// (default `"auto"` when empty).
    [[nodiscard]] static StatusOr<ConsoleProgressMode> from_flags(bool quiet, bool plain_progress,
                                                                  std::string_view progress_flag) {
        if (quiet) {
            return ConsoleProgressMode{Kind::Off};
        }
        if (plain_progress) {
            return ConsoleProgressMode{Kind::Lines};
        }
        if (progress_flag.empty()) {
            return ConsoleProgressMode{Kind::Auto};
        }
        return parse(progress_flag);
    }

    /// Resolve `Auto` against whether stderr is interactive.
    /// Forced Panel/Lines/Off are unchanged.
    [[nodiscard]] ConsoleProgressMode resolve(bool stderr_is_tty) const noexcept {
        if (kind_ != Kind::Auto) {
            return *this;
        }
        if (stderr_is_tty) {
            return ConsoleProgressMode{Kind::Panel};
        }
        return ConsoleProgressMode{Kind::Lines};
    }

    [[nodiscard]] bool operator==(const ConsoleProgressMode& other) const noexcept {
        return kind_ == other.kind_;
    }

    [[nodiscard]] bool operator!=(const ConsoleProgressMode& other) const noexcept {
        return kind_ != other.kind_;
    }

private:
    Kind kind_ = Kind::Auto;
};

#endif // CONSOLE_PROGRESS_MODE_HPP
