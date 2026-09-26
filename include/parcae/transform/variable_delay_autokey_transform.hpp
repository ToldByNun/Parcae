#ifndef VARIABLE_DELAY_AUTOKEY_TRANSFORM_HPP
#define VARIABLE_DELAY_AUTOKEY_TRANSFORM_HPP

#include "parcae/core/z29.hpp"
#include "parcae/math/primes.hpp"
#include "parcae/transform/transform.hpp"
#include "parcae/transform/transform_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/// Variable-delay autokey over \(\mathbb{Z}_{29}\) with a **prime** lag.
///
/// Primer `key` of length \(L\); lag \(p\) MUST be prime (\(\ge 2\)).
/// For consumed index \(j\): \(j < p\) → `key[j % L]`; else prior stream at lag \(p\).
///
/// Mode selects the feedback stream:
/// - `ciphertext` (CTAK): encrypt uses prior ciphertext; decrypt uses input ciphertext
/// - `plaintext` (PTAK): encrypt uses prior plaintext; decrypt uses recovered plaintext
///
/// Dense (no interrupts) with \(L = p\) and mode `ciphertext` matches
/// `CiphertextAutokeyTransform`; mode `plaintext` matches `PlaintextAutokeyTransform`.
///
/// With interrupts: skip positions pass through and do not consume primer/feedback
/// (Vigenère-style cursor over consumed runes only).
class VariableDelayAutokeyTransform : public Transform {
public:
    enum class Mode : std::uint8_t { Ciphertext, Plaintext };

    VariableDelayAutokeyTransform() = default;

    [[nodiscard]] TransformId id() const override {
        return TransformId::variable_delay_autokey();
    }

    /// `key` MUST be non-empty; `lag` MUST be prime \(\ge 2\);
    /// `skip_indices_sorted` MUST be sorted unique (caller-validated). In-place OK.
    [[nodiscard]] static Status kernel(std::span<const Index29> input, std::span<Index29> output,
                                       std::span<const Index29> key, std::size_t lag, Mode mode,
                                       std::span<const std::size_t> skip_indices_sorted,
                                       TransformDirection direction) {
        Status sizes = TransformBuffer::require_same_length(input, output);
        if (!sizes.ok()) {
            return sizes;
        }
        if (key.empty()) {
            return Status::error("variable_delay_autokey key must be non-empty");
        }
        if (lag < 2 || !is_prime(lag)) {
            return Status::error("variable_delay_autokey lag must be a prime >= 2");
        }

        const std::size_t primer_len = key.size();
        const bool ciphertext_mode = (mode == Mode::Ciphertext);

        if (skip_indices_sorted.empty()) {
            for (std::size_t i = 0; i < input.size(); ++i) {
                Index29 key_symbol;
                if (i < lag) {
                    key_symbol = key[i % primer_len];
                } else if (ciphertext_mode) {
                    key_symbol = (direction == TransformDirection::Encrypt) ? output[i - lag]
                                                                            : input[i - lag];
                } else {
                    key_symbol = (direction == TransformDirection::Encrypt) ? input[i - lag]
                                                                            : output[i - lag];
                }
                if (direction == TransformDirection::Encrypt) {
                    output[i] = Z29::add(input[i], key_symbol);
                } else {
                    output[i] = Z29::sub(input[i], key_symbol);
                }
            }
            return Status::success();
        }

        std::vector<Index29> feedback;
        feedback.reserve(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (TransformBuffer::should_skip(skip_indices_sorted, i)) {
                output[i] = input[i];
                continue;
            }
            Index29 key_symbol;
            if (feedback.size() < lag) {
                key_symbol = key[feedback.size() % primer_len];
            } else {
                key_symbol = feedback[feedback.size() - lag];
            }
            if (direction == TransformDirection::Encrypt) {
                output[i] = Z29::add(input[i], key_symbol);
                feedback.push_back(ciphertext_mode ? output[i] : input[i]);
            } else {
                output[i] = Z29::sub(input[i], key_symbol);
                feedback.push_back(ciphertext_mode ? input[i] : output[i]);
            }
        }
        return Status::success();
    }

    [[nodiscard]] Status
    apply_into(std::span<const Index29> input, std::span<Index29> output,
               const nlohmann::json& params, TransformDirection direction,
               const InterruptPolicy& interrupt = InterruptPolicy::none()) const override {
        StatusOr<ParsedParams> parsed = parse_params(params);
        if (!parsed.ok()) {
            return parsed.status();
        }
        Status range = TransformBuffer::validate_interrupt_range(interrupt, input.size(),
                                                                 "variable_delay_autokey");
        if (!range.ok()) {
            return range;
        }
        return kernel(input, output, parsed.value().key, parsed.value().lag, parsed.value().mode,
                      TransformBuffer::skip_span(interrupt), direction);
    }

private:
    struct ParsedParams {
        std::vector<Index29> key;
        std::size_t lag = 0;
        Mode mode = Mode::Ciphertext;
    };

    [[nodiscard]] static bool is_prime(std::size_t n) {
        if (n < 2) {
            return false;
        }
        const std::vector<std::uint64_t> sieved = Primes::sieve_upto(static_cast<std::uint64_t>(n));
        return !sieved.empty() && sieved.back() == static_cast<std::uint64_t>(n);
    }

    [[nodiscard]] static StatusOr<ParsedParams> parse_params(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("variable_delay_autokey params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error("variable_delay_autokey params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("variable_delay_autokey params.key_indices must be an array");
        }

        const bool has_lag = params.contains("lag");
        const bool has_lag_prime_index = params.contains("lag_prime_index");
        if (has_lag == has_lag_prime_index) {
            return Status::error(
                "variable_delay_autokey requires exactly one of lag or lag_prime_index");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin" && it.key() != "lag" &&
                it.key() != "lag_prime_index" && it.key() != "mode") {
                return Status::error("variable_delay_autokey params contains unknown field");
            }
        }

        ParsedParams out;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("variable_delay_autokey key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("variable_delay_autokey key_indices entries must be in 0..28");
            }
            out.key.push_back(Index29{static_cast<std::uint8_t>(raw)});
        }
        if (out.key.empty()) {
            return Status::error("variable_delay_autokey key_indices must be non-empty");
        }

        if (has_lag) {
            if (!params.at("lag").is_number_integer()) {
                return Status::error("variable_delay_autokey lag must be an integer");
            }
            const auto raw = params.at("lag").get<std::int64_t>();
            if (raw < 2) {
                return Status::error("variable_delay_autokey lag must be a prime >= 2");
            }
            out.lag = static_cast<std::size_t>(raw);
            if (!is_prime(out.lag)) {
                return Status::error("variable_delay_autokey lag must be a prime >= 2");
            }
        } else {
            if (!params.at("lag_prime_index").is_number_integer()) {
                return Status::error("variable_delay_autokey lag_prime_index must be an integer");
            }
            const auto raw = params.at("lag_prime_index").get<std::int64_t>();
            if (raw < 0) {
                return Status::error("variable_delay_autokey lag_prime_index must be non-negative");
            }
            StatusOr<std::uint64_t> prime = Primes::nth(static_cast<std::size_t>(raw));
            if (!prime.ok()) {
                return prime.status();
            }
            out.lag = static_cast<std::size_t>(prime.value());
        }

        out.mode = Mode::Ciphertext;
        if (params.contains("mode")) {
            if (!params.at("mode").is_string()) {
                return Status::error("variable_delay_autokey mode must be a string");
            }
            const std::string mode = params.at("mode").get<std::string>();
            if (mode == "ciphertext") {
                out.mode = Mode::Ciphertext;
            } else if (mode == "plaintext") {
                out.mode = Mode::Plaintext;
            } else {
                return Status::error(
                    "variable_delay_autokey mode must be ciphertext or plaintext");
            }
        }

        return out;
    }
};

#endif // VARIABLE_DELAY_AUTOKEY_TRANSFORM_HPP
