#ifndef FIXTURE_VALIDATOR_HPP
#define FIXTURE_VALIDATOR_HPP

#include "parcae/core/sha256.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/corpus/separator_grammar.hpp"
#include "parcae/corpus/tokenizer.hpp"
#include "parcae/gematria/gematria_profile.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"
#include "parcae/validate/plaintext_normalizer.hpp"
#include "parcae/validate/validation_report.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/// Load fixture → tokenize → apply method → latinize → compare + optional hash lock.
class FixtureValidator {
public:
    FixtureValidator(const GematriaProfile& profile, const SeparatorGrammar& grammar)
        : codec_(profile), normalizer_(codec_), tokenizer_(profile, grammar) {}

    [[nodiscard]] ValidationReport validate_directory(
        const std::string& fixture_dir,
        bool require_locked = false) const {
        ValidationReport report;

        StatusOr<Fixture> loaded = FixtureLoader::load_directory(fixture_dir);
        if (!loaded.ok()) {
            report.set_fixture_id(std::filesystem::path(fixture_dir).filename().string());
            report.add_check("manifest", false, loaded.status().message());
            return report;
        }

        const Fixture& fixture = loaded.value();
        report.set_fixture_id(fixture.id());
        report.add_check("manifest", true, "loaded");

        if (require_locked && fixture.verification_status() != "locked") {
            report.add_check(
                "locked",
                false,
                "require_locked=true but verification.status is not locked");
            return report;
        }
        if (fixture.verification_status() == "locked") {
            report.add_check("locked", true, "fixture is locked");
        }

        StatusOr<TokenStream> stream = tokenizer_.tokenize(fixture.ciphertext(), true);
        if (!stream.ok()) {
            report.add_check("tokenize", false, stream.status().message());
            return report;
        }
        report.add_check(
            "tokenize",
            true,
            "consumable_runes=" + std::to_string(stream.value().consumable_count()));

        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(fixture.skip_indices());
        if (!interrupt.ok()) {
            report.add_check("interrupt", false, interrupt.status().message());
            return report;
        }
        if (!fixture.skip_indices().empty()) {
            for (std::size_t skip : fixture.skip_indices()) {
                if (skip >= stream.value().consumable_count()) {
                    report.add_check(
                        "interrupt",
                        false,
                        "skip_indices out of range for consumable rune count");
                    return report;
                }
            }
        }
        report.add_check("interrupt", true, "policy ok");

        StatusOr<TransformId> transform_id = TransformId::from_string(fixture.transform_id());
        if (!transform_id.ok()) {
            report.add_check("transform", false, transform_id.status().message());
            return report;
        }
        StatusOr<TransformDirection> direction =
            TransformDirectionUtil::from_string(fixture.direction());
        if (!direction.ok()) {
            report.add_check("transform", false, direction.status().message());
            return report;
        }

        const std::vector<Index29> cipher_indices = stream.value().consumable_indices();
        StatusOr<std::vector<Index29>> plain_indices = ApplyTransform::apply(
            transform_id.value(),
            cipher_indices,
            fixture.params(),
            direction.value(),
            interrupt.value());
        if (!plain_indices.ok()) {
            report.add_check("transform", false, plain_indices.status().message());
            return report;
        }
        report.add_check("transform", true, fixture.transform_id());

        const std::string actual_latin = normalizer_.from_indices(plain_indices.value());
        StatusOr<std::string> expected_latin = normalizer_.normalize(fixture.plaintext());
        if (!expected_latin.ok()) {
            report.add_check("latinize", false, expected_latin.status().message());
            return report;
        }
        report.add_check("latinize", true, "preferred labels");

        if (actual_latin == expected_latin.value()) {
            report.add_check("plaintext_compare", true, "normalized Latin match");
        } else {
            report.add_check("plaintext_compare", false, "normalized Latin mismatch");
            report.set_diff_excerpt(make_diff_excerpt(actual_latin, expected_latin.value()));
        }

        Status literals = check_literal_files(fixture_dir, fixture);
        report.add_check(
            "literal_regions",
            literals.ok(),
            literals.ok() ? "declared files present" : literals.message());

        check_hashes(report, fixture, actual_latin);
        return report;
    }

private:
    void check_hashes(
        ValidationReport& report,
        const Fixture& fixture,
        const std::string& normalized_latin) const {
        const auto& hashes = fixture.hashes();
        const bool any_expected = hashes.ciphertext_sha256().has_value() ||
                                  hashes.plaintext_sha256().has_value() ||
                                  hashes.normalized_plaintext_sha256().has_value();
        if (!any_expected && fixture.verification_status() != "locked") {
            report.add_check("hashes", true, "draft hashes unset");
            return;
        }

        bool ok = true;
        std::ostringstream msg;

        const std::string cipher_digest = Sha256::hex_digest(fixture.ciphertext());
        const std::string plain_digest = Sha256::hex_digest(fixture.plaintext());
        const std::string normalized_digest = Sha256::hex_digest(normalized_latin);

        if (hashes.ciphertext_sha256().has_value()) {
            if (hashes.ciphertext_sha256().value() != cipher_digest) {
                ok = false;
                msg << "ciphertext_sha256 mismatch; ";
            }
        } else if (fixture.verification_status() == "locked") {
            ok = false;
            msg << "ciphertext_sha256 missing; ";
        }

        if (hashes.plaintext_sha256().has_value()) {
            if (hashes.plaintext_sha256().value() != plain_digest) {
                ok = false;
                msg << "plaintext_sha256 mismatch; ";
            }
        } else if (fixture.verification_status() == "locked") {
            ok = false;
            msg << "plaintext_sha256 missing; ";
        }

        if (hashes.normalized_plaintext_sha256().has_value()) {
            if (hashes.normalized_plaintext_sha256().value() != normalized_digest) {
                ok = false;
                msg << "normalized_plaintext_sha256 mismatch; ";
            }
        } else if (fixture.verification_status() == "locked") {
            ok = false;
            msg << "normalized_plaintext_sha256 missing; ";
        }

        if (ok) {
            report.add_check("hashes", true, "digests match");
        } else {
            report.add_check("hashes", false, msg.str());
        }
    }

    [[nodiscard]] static std::string make_diff_excerpt(
        const std::string& actual,
        const std::string& expected) {
        std::size_t i = 0;
        while (i < actual.size() && i < expected.size() && actual[i] == expected[i]) {
            ++i;
        }
        constexpr std::size_t window = 48;
        const std::size_t begin = i > 16 ? i - 16 : 0;
        std::ostringstream out;
        out << "at=" << i << " actual=" << actual.substr(begin, window)
            << " expected=" << expected.substr(begin, window);
        return out.str();
    }

    [[nodiscard]] static Status check_literal_files(
        const std::string& fixture_dir,
        const Fixture& fixture) {
        for (const FixtureLiteralRegion& region : fixture.literal_regions()) {
            if (region.value_file().empty()) {
                return Status::error("literal region missing value_file");
            }
            const std::filesystem::path path =
                std::filesystem::path(fixture_dir) / region.value_file();
            std::ifstream input(path, std::ios::binary);
            if (!input) {
                return Status::error("literal value_file missing: " + path.string());
            }
        }
        return Status::success();
    }

    LatinCodec codec_;
    PlaintextNormalizer normalizer_;
    Tokenizer tokenizer_;
};

#endif // FIXTURE_VALIDATOR_HPP
