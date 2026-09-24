#ifndef THEORY_URI_HPP
#define THEORY_URI_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

/// `parcae://theories/<name>@<version>` (docs/spec/theory-artifact.md).
class TheoryUri {
public:
    static constexpr std::string_view scheme = "parcae";
    static constexpr std::string_view theories_prefix = "theories/";

    [[nodiscard]] static StatusOr<std::string> validate_name(std::string_view name) {
        if (name.empty() || name.size() > 64) {
            return Status::error("theory name length must be 1..64");
        }
        const unsigned char first = static_cast<unsigned char>(name[0]);
        if (!(first >= 'a' && first <= 'z')) {
            return Status::error("theory name must start with a lowercase letter: " +
                                 std::string(name));
        }
        for (char ch : name) {
            const unsigned char c = static_cast<unsigned char>(ch);
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
                continue;
            }
            return Status::error("theory name has illegal character: " + std::string(name));
        }
        return std::string(name);
    }

    [[nodiscard]] static StatusOr<std::uint32_t> validate_version(std::uint32_t version) {
        if (version == 0) {
            return Status::error("theory version must be a positive integer");
        }
        return version;
    }

    [[nodiscard]] static StatusOr<TheoryUri> make(std::string_view name, std::uint32_t version) {
        StatusOr<std::string> n = validate_name(name);
        if (!n.ok()) {
            return n.status();
        }
        StatusOr<std::uint32_t> v = validate_version(version);
        if (!v.ok()) {
            return v.status();
        }
        return TheoryUri{std::move(n.value()), v.value()};
    }

    [[nodiscard]] static StatusOr<TheoryUri> parse(std::string_view text) {
        constexpr std::string_view kPrefix = "parcae://theories/";
        if (text.size() < kPrefix.size() || text.substr(0, kPrefix.size()) != kPrefix) {
            return Status::error("theory URI must start with parcae://theories/: " +
                                 std::string(text));
        }
        const std::string_view rest = text.substr(kPrefix.size());
        const std::size_t at = rest.find('@');
        if (at == std::string_view::npos || at == 0 || at + 1 >= rest.size()) {
            return Status::error("theory URI must be parcae://theories/<name>@<version>: " +
                                 std::string(text));
        }
        const std::string_view name = rest.substr(0, at);
        const std::string_view ver_text = rest.substr(at + 1);
        if (ver_text.empty() || (ver_text.size() > 1 && ver_text[0] == '0') ||
            ver_text.find_first_not_of("0123456789") != std::string_view::npos) {
            return Status::error("theory URI version must be a positive decimal integer: " +
                                 std::string(text));
        }
        unsigned long value = 0;
        for (char ch : ver_text) {
            value = value * 10u + static_cast<unsigned long>(ch - '0');
            if (value > 1'000'000'000ul) {
                return Status::error("theory URI version out of range: " + std::string(text));
            }
        }
        if (value == 0) {
            return Status::error("theory URI version must be positive: " + std::string(text));
        }
        return make(name, static_cast<std::uint32_t>(value));
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] std::uint32_t version() const noexcept { return version_; }

    [[nodiscard]] std::string to_string() const {
        return std::string("parcae://theories/") + name_ + "@" + std::to_string(version_);
    }

private:
    TheoryUri(std::string name, std::uint32_t version) noexcept
        : name_(std::move(name)), version_(version) {}

    std::string name_;
    std::uint32_t version_ = 0;
};

#endif // THEORY_URI_HPP
