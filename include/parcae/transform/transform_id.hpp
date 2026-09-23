#ifndef TRANSFORM_ID_HPP
#define TRANSFORM_ID_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <string>
#include <string_view>
#include <utility>

/// Stable string id for a transform family (`identity`, `atbash`, …).
class TransformId {
public:
    [[nodiscard]] static TransformId identity() {
        return TransformId{"identity"};
    }

    [[nodiscard]] static TransformId atbash() {
        return TransformId{"atbash"};
    }

    [[nodiscard]] static TransformId caesar() {
        return TransformId{"caesar"};
    }

    [[nodiscard]] static TransformId affine() {
        return TransformId{"affine"};
    }

    [[nodiscard]] static TransformId compose() {
        return TransformId{"compose"};
    }

    [[nodiscard]] static TransformId vigenere_key() {
        return TransformId{"vigenere_key"};
    }

    [[nodiscard]] static TransformId beaufort_key() {
        return TransformId{"beaufort_key"};
    }

    [[nodiscard]] static TransformId totient_prime_stream() {
        return TransformId{"totient_prime_stream"};
    }

    /// Non-catalog id (theory URI). Callers MUST validate the URI separately.
    [[nodiscard]] static TransformId unchecked(std::string value) {
        return TransformId{std::move(value)};
    }

    /// Parse a catalog id. Unknown ids fail.
    [[nodiscard]] static StatusOr<TransformId> from_string(std::string_view text) {
        if (text == "identity") {
            return identity();
        }
        if (text == "atbash") {
            return atbash();
        }
        if (text == "caesar") {
            return caesar();
        }
        if (text == "affine") {
            return affine();
        }
        if (text == "compose") {
            return compose();
        }
        if (text == "vigenere_key") {
            return vigenere_key();
        }
        if (text == "beaufort_key") {
            return beaufort_key();
        }
        if (text == "totient_prime_stream") {
            return totient_prime_stream();
        }
        return Status::error("Unknown transform_id");
    }

    [[nodiscard]] const std::string& str() const noexcept {
        return value_;
    }

    [[nodiscard]] bool operator==(const TransformId&) const noexcept = default;

private:
    explicit TransformId(std::string value) : value_(std::move(value)) {}

    std::string value_;
};

#endif // TRANSFORM_ID_HPP
