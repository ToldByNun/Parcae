#ifndef TRANSFORM_DIRECTION_HPP
#define TRANSFORM_DIRECTION_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <string_view>

enum class TransformDirection {
    Encrypt,
    Decrypt,
};

class TransformDirectionUtil {
public:
    [[nodiscard]] static constexpr std::string_view to_string(TransformDirection direction) noexcept {
        switch (direction) {
            case TransformDirection::Encrypt:
                return "encrypt";
            case TransformDirection::Decrypt:
                return "decrypt";
        }
        return "unknown";
    }

    [[nodiscard]] static StatusOr<TransformDirection> from_string(std::string_view text) {
        if (text == "encrypt") {
            return TransformDirection::Encrypt;
        }
        if (text == "decrypt") {
            return TransformDirection::Decrypt;
        }
        return Status::error("direction must be encrypt or decrypt");
    }

private:
    TransformDirectionUtil() = delete;
};

#endif // TRANSFORM_DIRECTION_HPP
