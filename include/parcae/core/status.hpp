#ifndef STATUS_HPP
#define STATUS_HPP

#include <string>
#include <utility>

class Status {
public:
    [[nodiscard]] static Status success() {
        return Status{true, {}};
    }

    [[nodiscard]] static Status error(std::string message) {
        return Status{false, std::move(message)};
    }

    [[nodiscard]] bool ok() const noexcept {
        return ok_;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return ok_;
    }

    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }

private:
    Status(bool ok, std::string message) : ok_(ok), message_(std::move(message)) {}

    bool ok_;
    std::string message_;
};

#endif // STATUS_HPP
