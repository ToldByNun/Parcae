#ifndef STATUS_OR_HPP
#define STATUS_OR_HPP

#include "parcae/core/status.hpp"

#include <cstdlib>
#include <optional>
#include <utility>

template <typename T>
class StatusOr {
public:
    StatusOr(const T& value) : status_(Status::success()), value_(value) {}

    StatusOr(T&& value) : status_(Status::success()), value_(std::move(value)) {}

    StatusOr(Status status) : status_(std::move(status)), value_(std::nullopt) {
        if (status_.ok()) {
            fatal_invalid();
        }
    }

    [[nodiscard]] bool ok() const noexcept { return status_.ok() && value_.has_value(); }

    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

    [[nodiscard]] const Status& status() const noexcept { return status_; }

    [[nodiscard]] T& value() & {
        require_value();
        return *value_;
    }

    [[nodiscard]] const T& value() const& {
        require_value();
        return *value_;
    }

    [[nodiscard]] T&& value() && {
        require_value();
        return std::move(*value_);
    }

private:
    void require_value() const {
        if (!ok()) {
            fatal_invalid();
        }
    }

    [[noreturn]] static void fatal_invalid() noexcept { std::abort(); }

    Status status_;
    std::optional<T> value_;
};

#endif // STATUS_OR_HPP
