#ifndef TOOL_API_HPP
#define TOOL_API_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/token.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/corpus/tokenizer.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/gematria/rune_codec.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/score/score_request.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/tool/transform_envelope.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/validate/fixture_validator.hpp"
#include "parcae/validate/validation_report.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "backend.hpp"
#include "cuda_score.hpp"
#endif

#include <cstddef>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class ToolApi {
public:
    /// Tokenize UTF-8 source with the named separator grammar (+ default Gematria).
    [[nodiscard]] static StatusOr<TokenStream>
    tokenize(const Context& ctx, const std::string& source_utf8,
             std::string_view grammar_id = "rtkd-separator-grammar-v0", bool strict = true) {
        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            return profile.status();
        }
        StatusOr<SeparatorGrammar> grammar = ctx.load_grammar(grammar_id);
        if (!grammar.ok()) {
            return grammar.status();
        }
        return Tokenizer{profile.value(), grammar.value()}.tokenize(source_utf8, strict);
    }

    /// Apply envelope to a consumable Index29 span (`Backend::Cpu` default).
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_to_indices(std::span<const Index29> input, const TransformEnvelope& envelope,
                     Backend backend = Backend::Cpu) {
        Status usable = BackendUtil::ensure_usable(backend);
        if (!usable.ok()) {
            return usable;
        }

        if (backend == Backend::Cpu) {
            return ApplyTransform::apply(envelope.transform_id(), input, envelope.params(),
                                         envelope.direction(), envelope.interrupt());
        }

#if defined(PARCAE_HAS_CUDA)
        if (!CudaBackend::available()) {
            return Status::error("CUDA backend requested but CUDA is not available");
        }
        return CudaBackend::apply(envelope.transform_id(), input, envelope.params(),
                                  envelope.direction(), envelope.interrupt());
#else
        return Status::error(
            "CUDA backend requested but Parcae was built without CUDA (PARCAE_BUILD_CUDA)");
#endif
    }

    /// Theory-aware apply: catalog ids → ApplyTransform; `parcae://theories/…` →
    /// TheoryDispatch under `theories_root` (typically `data/theories`).
    /// CUDA for theory URIs is not supported in this slice (CPU IR path only).
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_to_indices(std::span<const Index29> input, const TheoryEnvelopeBridge::Envelope& envelope,
                     const std::filesystem::path& theories_root, Backend backend = Backend::Cpu) {
        Status usable = BackendUtil::ensure_usable(backend);
        if (!usable.ok()) {
            return usable;
        }
        if (envelope.is_catalog()) {
            StatusOr<TransformEnvelope> catalog = envelope.to_catalog_envelope();
            if (!catalog.ok()) {
                return catalog.status();
            }
            return apply_to_indices(input, catalog.value(), backend);
        }
        if (backend != Backend::Cpu) {
            return Status::error("TheoryDispatch theory URIs are CPU-only in this slice "
                                 "(CUDA path uses emitted twins / [cuda][dsl][smoke])");
        }
        return TheoryDispatch::apply(theories_root, envelope, input);
    }

    /// Apply envelope to consumable runes of a token stream (indices only).
    [[nodiscard]] static StatusOr<std::vector<Index29>>
    apply_to_indices(const TokenStream& stream, const TransformEnvelope& envelope,
                     Backend backend = Backend::Cpu) {
        return apply_to_indices(stream.consumable_indices(), envelope, backend);
    }

    /// Apply to consumable runes and rebuild UTF-8 text, preserving non-rune tokens.
    [[nodiscard]] static StatusOr<std::string>
    apply_and_rebuild_text(const Context& ctx, const TokenStream& stream,
                           const TransformEnvelope& envelope, Backend backend = Backend::Cpu) {
        StatusOr<std::vector<Index29>> plain = apply_to_indices(stream, envelope, backend);
        if (!plain.ok()) {
            return plain.status();
        }
        if (plain.value().size() != stream.consumable_count()) {
            return Status::error("apply_and_rebuild_text: transform changed consumable length");
        }

        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            return profile.status();
        }
        const RuneCodec rune_codec(profile.value());

        std::vector<Token> rebuilt;
        rebuilt.reserve(stream.size());
        std::size_t byte_pos = 0;
        std::size_t consumable = 0;

        for (std::size_t i = 0; i < stream.size(); ++i) {
            const Token& token = stream.at(i);
            if (token.is_rune()) {
                StatusOr<std::string> utf8 = rune_codec.encode(plain.value()[consumable]);
                if (!utf8.ok()) {
                    return utf8.status();
                }
                const std::size_t begin = byte_pos;
                byte_pos += utf8.value().size();
                rebuilt.push_back(Token::rune(plain.value()[consumable], begin, byte_pos,
                                              std::move(utf8.value()), consumable));
                ++consumable;
            } else {
                const std::size_t begin = byte_pos;
                byte_pos += token.text().size();
                rebuilt.push_back(Token::non_rune(token.kind(), begin, byte_pos, token.text()));
            }
        }

        StatusOr<TokenStream> out = TokenStream::create(std::move(rebuilt));
        if (!out.ok()) {
            return out.status();
        }
        return out.value().text();
    }

    /// Preferred-label Latin concatenation (no inserted spaces).
    [[nodiscard]] static StatusOr<std::string>
    to_latin(const Context& ctx, std::span<const Index29> indices,
             std::string_view label_profile = "gematria-primus-v0-preferred") {
        // v0 preferred labels come from the gematria-primus-v0 profile.
        if (label_profile != "gematria-primus-v0-preferred" &&
            label_profile != "gematria-primus-v0") {
            return Status::error("Unsupported label_profile for to_latin");
        }
        StatusOr<GematriaProfile> profile = ctx.load_gematria("gematria-primus-v0");
        if (!profile.ok()) {
            return profile.status();
        }
        const LatinCodec codec(profile.value());
        return codec.latinize(std::vector<Index29>(indices.begin(), indices.end()));
    }

    /// Score via CPU `ScoreRegistry` or CUDA `CudaScore` (`Backend::Cpu` default).
    [[nodiscard]] static StatusOr<double>
    score(const Context& ctx, std::span<const Index29> indices, std::string_view score_id,
          std::string_view score_version = "v0",
          const nlohmann::json& params = nlohmann::json::object(),
          ScoreRequest request = ScoreRequest(), Backend backend = Backend::Cpu) {
        Status usable = BackendUtil::ensure_usable(backend);
        if (!usable.ok()) {
            return usable;
        }

        std::optional<ExpectedFrequencyTable> owned_table;
        if (score_id == ScoreId::chi2_english_gp_v0().str() &&
            request.expected_frequencies == nullptr) {
            StatusOr<ExpectedFrequencyTable> table = ctx.load_english_gp_expected();
            if (!table.ok()) {
                return table.status();
            }
            owned_table = std::move(table.value());
            request.expected_frequencies = &owned_table.value();
        }

        std::optional<BigramModelTable> owned_bigram;
        if (score_id == ScoreId::log_bigram_gp_v0().str() && request.bigram_model == nullptr) {
            StatusOr<BigramModelTable> table = ctx.load_english_gp_bigram();
            if (!table.ok()) {
                return Status::error(std::string("log_bigram_gp_v0 model missing: ") +
                                     table.status().message());
            }
            owned_bigram = std::move(table.value());
            request.bigram_model = &owned_bigram.value();
        }

        if (backend == Backend::Cpu) {
            return ScoreRegistry::score(score_id, indices, score_version, params, request);
        }

#if defined(PARCAE_HAS_CUDA)
        if (!CudaScore::available()) {
            return Status::error("CUDA backend requested but CUDA is not available");
        }
        return CudaScore::score(score_id, indices, score_version, params, request);
#else
        return Status::error(
            "CUDA backend requested but Parcae was built without CUDA (PARCAE_BUILD_CUDA)");
#endif
    }

    /// Validate a fixture directory path or solved-fixture id.
    [[nodiscard]] static ValidationReport validate_fixture(const Context& ctx,
                                                           std::string_view fixture_dir_or_id,
                                                           bool require_locked = false) {
        StatusOr<std::filesystem::path> dir = ctx.resolve_fixture_dir(fixture_dir_or_id);
        if (!dir.ok()) {
            ValidationReport report;
            report.set_fixture_id(std::string(fixture_dir_or_id));
            report.add_check("resolve", false, dir.status().message());
            return report;
        }

        StatusOr<GematriaProfile> profile = ctx.load_gematria();
        if (!profile.ok()) {
            ValidationReport report;
            report.set_fixture_id(dir.value().filename().string());
            report.add_check("gematria", false, profile.status().message());
            return report;
        }
        StatusOr<SeparatorGrammar> grammar = ctx.load_grammar();
        if (!grammar.ok()) {
            ValidationReport report;
            report.set_fixture_id(dir.value().filename().string());
            report.add_check("grammar", false, grammar.status().message());
            return report;
        }

        const FixtureValidator validator(profile.value(), grammar.value());
        return validator.validate_directory(dir.value().string(), require_locked);
    }

    [[nodiscard]] static std::vector<std::string> list_transform_ids() {
        return {
            TransformId::identity().str(),     TransformId::atbash().str(),
            TransformId::caesar().str(),       TransformId::affine().str(),
            TransformId::hill_2().str(),       TransformId::hill_3().str(),
            TransformId::compose().str(),      TransformId::vigenere_key().str(),
            TransformId::ciphertext_autokey().str(), TransformId::plaintext_autokey().str(),
            TransformId::variable_delay_autokey().str(), TransformId::beaufort_key().str(),
            TransformId::totient_prime_stream().str(),
        };
    }

    [[nodiscard]] static std::vector<std::string> list_score_ids() {
        return ScoreRegistry::known_ids();
    }

private:
    ToolApi() = delete;
};

#endif // TOOL_API_HPP
