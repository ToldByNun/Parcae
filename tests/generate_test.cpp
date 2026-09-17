#include <parcae/core/index29.hpp>
#include <parcae/core/z29.hpp>
#include <parcae/generate/caesar_candidate_generator.hpp>
#include <parcae/score/exact_match.hpp>
#include <parcae/transform/caesar_transform.hpp>
#include <parcae/transform/transform_direction.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

}  // namespace

TEST_CASE("CaesarCandidateGenerator emits all 29 shifts in order", "[generate][caesar]") {
    const std::vector<Index29> cipher = {I(0), I(5), I(10), I(28)};

    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(candidates.ok());
    REQUIRE(candidates.value().size() == 29);
    REQUIRE(CaesarCandidateGenerator::candidate_count == 29);
    REQUIRE(CaesarCandidateGenerator::generator_id == "gen_caesar");

    const CaesarTransform caesar;
    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        const TransformCandidate& c = candidates.value()[shift];
        REQUIRE(c.candidate_id() == CaesarCandidateGenerator::make_candidate_id(shift));
        REQUIRE(c.transform_id() == TransformId::caesar());
        REQUIRE(c.direction() == TransformDirection::Decrypt);
        REQUIRE(c.params().at("shift").get<int>() == static_cast<int>(shift));
        REQUIRE(c.output_indices().size() == cipher.size());

        StatusOr<std::vector<Index29>> expected = caesar.apply(
            cipher,
            nlohmann::json{{"shift", static_cast<int>(shift)}},
            TransformDirection::Decrypt);
        REQUIRE(expected.ok());
        REQUIRE(c.output_indices() == expected.value());

        const nlohmann::json env = c.envelope();
        REQUIRE(env.at("transform_id") == "caesar");
        REQUIRE(env.at("direction") == "decrypt");
        REQUIRE(env.at("params").at("shift") == static_cast<int>(shift));
    }

    // shift=0 decrypt is identity.
    REQUIRE(candidates.value()[0].output_indices() == cipher);
}

TEST_CASE("CaesarCandidateGenerator encrypt path and round-trip", "[generate][caesar]") {
    const std::vector<Index29> plain = {I(1), I(2), I(3), I(4), I(5)};

    StatusOr<std::vector<TransformCandidate>> encrypted =
        CaesarCandidateGenerator::generate(plain, TransformDirection::Encrypt);
    REQUIRE(encrypted.ok());
    REQUIRE(encrypted.value().size() == 29);

    constexpr std::uint8_t kShift = 7;
    const std::vector<Index29>& cipher = encrypted.value()[kShift].output_indices();
    REQUIRE(encrypted.value()[kShift].candidate_id() == "caesar:shift=7");
    REQUIRE(encrypted.value()[kShift].envelope().at("direction") == "encrypt");

    StatusOr<std::vector<TransformCandidate>> decrypted =
        CaesarCandidateGenerator::generate(cipher, TransformDirection::Decrypt);
    REQUIRE(decrypted.ok());
    REQUIRE(decrypted.value()[kShift].output_indices() == plain);

    StatusOr<double> exact =
        ExactMatch::score(decrypted.value()[kShift].output_indices(), plain);
    REQUIRE(exact.ok());
    REQUIRE(exact.value() == 1.0);
}

TEST_CASE("CaesarCandidateGenerator JSON record shape", "[generate][caesar]") {
    const std::vector<Index29> cipher = {I(3), I(4)};
    StatusOr<std::vector<TransformCandidate>> candidates =
        CaesarCandidateGenerator::generate(cipher);
    REQUIRE(candidates.ok());

    const nlohmann::json record = candidates.value()[3].to_json();
    REQUIRE(record.at("candidate_id") == "caesar:shift=3");
    REQUIRE(record.contains("envelope"));
    REQUIRE(record.contains("output_indices"));
    REQUIRE(record.at("output_indices").is_array());
    REQUIRE(record.at("output_indices").size() == 2);
    REQUIRE(record.at("output_indices")[0].get<int>() ==
            static_cast<int>(Z29::sub(I(3), I(3)).value()));
    REQUIRE(record.at("output_indices")[1].get<int>() ==
            static_cast<int>(Z29::sub(I(4), I(3)).value()));
}
