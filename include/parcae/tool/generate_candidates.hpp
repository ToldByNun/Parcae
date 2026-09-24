#ifndef GENERATE_CANDIDATES_HPP
#define GENERATE_CANDIDATES_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/generate/generator_registry.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cctype>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// Tool-facing `generate_candidates` primitive.
///
/// Dispatches `generator_id` via `GeneratorRegistry`. Prefer this over calling
/// individual `*CandidateGenerator` classes from CLIs / agents.
class GenerateCandidates {
public:
    [[nodiscard]] static std::vector<std::string> list_generator_ids() {
        return GeneratorRegistry::list_generator_ids();
    }

    /// Generate from an Index29 ciphertext span.
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    from_indices(std::string_view generator_id, std::span<const Index29> ciphertext,
                 TransformDirection direction = TransformDirection::Decrypt,
                 const nlohmann::json& params = nlohmann::json::object()) {
        return GeneratorRegistry::generate(generator_id, ciphertext, direction, params);
    }

    /// Generate from a tokenized stream (consumable runes only).
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    from_stream(std::string_view generator_id, const TokenStream& stream,
                TransformDirection direction = TransformDirection::Decrypt,
                const nlohmann::json& params = nlohmann::json::object()) {
        if (stream.consumable_count() == 0) {
            return Status::error("generate_candidates: no consumable runes in stream");
        }
        return from_indices(generator_id, stream.consumable_indices(), direction, params);
    }

    /// Load ciphertext from UTF-8 text then generate.
    ///
    /// `input_mode`: `indices` (0..28 list), `runes` (tokenize Liber Primus text),
    /// or `latin` (letters → delatinize).
    [[nodiscard]] static StatusOr<std::vector<TransformCandidate>>
    from_source(const Context& ctx, std::string_view generator_id, const std::string& source_utf8,
                std::string_view input_mode = "runes",
                TransformDirection direction = TransformDirection::Decrypt,
                const nlohmann::json& params = nlohmann::json::object()) {
        StatusOr<std::vector<Index29>> cipher = load_indices(ctx, source_utf8, input_mode);
        if (!cipher.ok()) {
            return cipher.status();
        }
        return from_indices(generator_id, cipher.value(), direction, params);
    }

private:
    GenerateCandidates() = delete;

    [[nodiscard]] static StatusOr<std::vector<Index29>>
    load_indices(const Context& ctx, const std::string& source, std::string_view mode) {
        if (mode == "indices") {
            return parse_indices_text(source);
        }
        if (mode == "runes") {
            StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, source);
            if (!stream.ok()) {
                return stream.status();
            }
            if (stream.value().consumable_count() == 0) {
                return Status::error("generate_candidates: no consumable runes in input");
            }
            return stream.value().consumable_indices();
        }
        if (mode == "latin") {
            StatusOr<GematriaProfile> profile = ctx.load_gematria();
            if (!profile.ok()) {
                return profile.status();
            }
            const LatinCodec codec(profile.value());
            const std::string letters = letters_only_upper(source);
            if (letters.empty()) {
                return Status::error("generate_candidates: no Latin letters in input");
            }
            return codec.delatinize(letters);
        }
        return Status::error("generate_candidates: input_mode must be indices|runes|latin");
    }

    [[nodiscard]] static std::string letters_only_upper(std::string_view text) {
        std::string out;
        out.reserve(text.size());
        for (unsigned char ch : text) {
            if (std::isalpha(ch) != 0) {
                out.push_back(static_cast<char>(std::toupper(ch)));
            }
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> parse_indices_text(std::string_view text) {
        std::vector<Index29> out;
        std::size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && (text[i] == ' ' || text[i] == ',' || text[i] == '\t' ||
                                       text[i] == '\n' || text[i] == '\r')) {
                ++i;
            }
            if (i >= text.size()) {
                break;
            }
            std::size_t j = i;
            while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
                ++j;
            }
            if (j == i) {
                return Status::error(
                    "generate_candidates: invalid indices input (expected digits)");
            }
            const unsigned long value = std::stoul(std::string(text.substr(i, j - i)));
            if (value >= Index29::modulus) {
                return Status::error("generate_candidates: Index29 out of range [0,28]");
            }
            out.push_back(Index29{static_cast<std::uint8_t>(value)});
            i = j;
        }
        if (out.empty()) {
            return Status::error("generate_candidates: indices input is empty");
        }
        return out;
    }
};

#endif // GENERATE_CANDIDATES_HPP
