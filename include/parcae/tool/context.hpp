#ifndef TOOL_CONTEXT_HPP
#define TOOL_CONTEXT_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/separator_grammar.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/gematria_profile_loader.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/expected_frequency_table.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

/// Resolves profile / fixture paths under a Parcae `data/` root.
class Context {
public:
    explicit Context(std::filesystem::path data_root) : data_root_(std::move(data_root)) {}

    [[nodiscard]] const std::filesystem::path& data_root() const noexcept {
        return data_root_;
    }

    [[nodiscard]] std::filesystem::path gematria_profile_path(
        std::string_view profile_id = "gematria-primus-v0") const {
        return data_root_ / "profiles" / "gematria" / (std::string(profile_id) + ".json");
    }

    [[nodiscard]] std::filesystem::path separator_grammar_path(
        std::string_view grammar_id = "rtkd-separator-grammar-v0") const {
        return data_root_ / "profiles" / "separators" / (std::string(grammar_id) + ".json");
    }

    [[nodiscard]] std::filesystem::path english_gp_expected_path() const {
        return data_root_ / "profiles" / "scores" / "english-gp-expected-v0.json";
    }

    /// Absolute/existing directory wins; otherwise `fixtures/solved/<id>`.
    [[nodiscard]] StatusOr<std::filesystem::path> resolve_fixture_dir(
        std::string_view fixture_dir_or_id) const {
        const std::filesystem::path as_path(fixture_dir_or_id);
        if (std::filesystem::is_directory(as_path)) {
            return as_path;
        }
        const std::filesystem::path solved =
            data_root_ / "fixtures" / "solved" / std::string(fixture_dir_or_id);
        if (std::filesystem::is_directory(solved)) {
            return solved;
        }
        return Status::error(
            "Fixture path not found (not a directory and not under fixtures/solved): " +
            std::string(fixture_dir_or_id));
    }

    [[nodiscard]] StatusOr<GematriaProfile> load_gematria(
        std::string_view profile_id = "gematria-primus-v0") const {
        return GematriaProfileLoader::load_from_file(gematria_profile_path(profile_id).string());
    }

    [[nodiscard]] StatusOr<SeparatorGrammar> load_grammar(
        std::string_view grammar_id = "rtkd-separator-grammar-v0") const {
        return SeparatorGrammar::load_from_file(separator_grammar_path(grammar_id).string());
    }

    [[nodiscard]] StatusOr<ExpectedFrequencyTable> load_english_gp_expected() const {
        return ExpectedFrequencyLoader::load_from_file(english_gp_expected_path().string());
    }

private:
    std::filesystem::path data_root_;
};

#endif // TOOL_CONTEXT_HPP
