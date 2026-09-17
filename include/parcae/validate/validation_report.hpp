#ifndef VALIDATION_REPORT_HPP
#define VALIDATION_REPORT_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

class ValidationCheck {
public:
    ValidationCheck(std::string name, bool ok, std::string message)
        : name_(std::move(name)), ok_(ok), message_(std::move(message)) {}

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    [[nodiscard]] bool ok() const noexcept {
        return ok_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

private:
    std::string name_;
    bool ok_ = false;
    std::string message_;
};

class ValidationReport {
public:
    ValidationReport() = default;

    [[nodiscard]] bool ok() const noexcept {
        return ok_;
    }

    [[nodiscard]] const std::string& fixture_id() const noexcept {
        return fixture_id_;
    }

    [[nodiscard]] const std::vector<ValidationCheck>& checks() const noexcept {
        return checks_;
    }

    [[nodiscard]] const std::optional<std::string>& diff_excerpt() const noexcept {
        return diff_excerpt_;
    }

    void set_fixture_id(std::string id) {
        fixture_id_ = std::move(id);
    }

    void add_check(std::string name, bool passed, std::string message) {
        if (!passed) {
            ok_ = false;
        }
        checks_.emplace_back(std::move(name), passed, std::move(message));
    }

    void set_diff_excerpt(std::string excerpt) {
        diff_excerpt_ = std::move(excerpt);
    }

private:
    bool ok_ = true;
    std::string fixture_id_;
    std::vector<ValidationCheck> checks_;
    std::optional<std::string> diff_excerpt_;
};

#endif // VALIDATION_REPORT_HPP
