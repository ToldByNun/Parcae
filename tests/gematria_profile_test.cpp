#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <parcae/gematria/gematria_profile.hpp>
#include <parcae/gematria/gematria_profile_loader.hpp>
#include <sstream>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined for profile tests"
#endif

namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::string valid_profile_path() {
    return std::string(PARCAE_TEST_DATA_DIR) + "/profiles/gematria/gematria-primus-v0.json";
}

} // namespace

TEST_CASE("Load canonical gematria-primus-v0 profile", "[gematria]") {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_file(valid_profile_path());
    REQUIRE(profile.ok());
    REQUIRE(profile.value().profile_id() == "gematria-primus-v0");
    REQUIRE(profile.value().size() == 29);

    REQUIRE(profile.value().entry_at(Index29{0}).rune() == "ᚠ");
    REQUIRE(profile.value().entry_at(Index29{0}).prime() == 2);
    REQUIRE(profile.value().entry_at(Index29{28}).rune() == "ᛠ");
    REQUIRE(profile.value().entry_at(Index29{28}).prime() == 109);

    StatusOr<Index29> f_index = profile.value().index_for_rune("ᚠ");
    REQUIRE(f_index.ok());
    REQUIRE(f_index.value() == Index29{0});

    StatusOr<std::string> rune = profile.value().rune_for_index(Index29{23});
    REQUIRE(rune.ok());
    REQUIRE(rune.value() == "ᛞ");

    StatusOr<Index29> by_prime = profile.value().index_for_prime(89);
    REQUIRE(by_prime.ok());
    REQUIRE(by_prime.value() == Index29{23});
}

TEST_CASE("Reject profile with wrong entry count", "[gematria]") {
    std::string json = read_file(valid_profile_path());
    nlohmann::json root = nlohmann::json::parse(json);
    root.at("entries").erase(root.at("entries").begin());

    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string(root.dump());
    REQUIRE_FALSE(profile.ok());
    REQUIRE(profile.status().message().find("29 entries") != std::string::npos);
}

TEST_CASE("Reject profile with non-contiguous indices", "[gematria]") {
    std::string json = read_file(valid_profile_path());
    nlohmann::json root = nlohmann::json::parse(json);
    root.at("entries").at(5).at("index") = 4; // duplicate 4, missing 5

    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string(root.dump());
    REQUIRE_FALSE(profile.ok());
}

TEST_CASE("Reject profile with wrong prime for index", "[gematria]") {
    std::string json = read_file(valid_profile_path());
    nlohmann::json root = nlohmann::json::parse(json);
    root.at("entries").at(0).at("prime") = 4; // not prime / not first-29 table

    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string(root.dump());
    REQUIRE_FALSE(profile.ok());
    REQUIRE(profile.status().message().find("first 29 primes") != std::string::npos);
}

TEST_CASE("Reject profile with duplicate rune", "[gematria]") {
    std::string json = read_file(valid_profile_path());
    nlohmann::json root = nlohmann::json::parse(json);
    root.at("entries").at(2).at("rune") = root.at("entries").at(0).at("rune");

    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string(root.dump());
    REQUIRE_FALSE(profile.ok());
    REQUIRE(profile.status().message().find("runes are not unique") != std::string::npos);
}

TEST_CASE("Reject profile with bad schema", "[gematria]") {
    std::string json = read_file(valid_profile_path());
    nlohmann::json root = nlohmann::json::parse(json);
    root.at("schema") = "not-a-real-schema";

    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string(root.dump());
    REQUIRE_FALSE(profile.ok());
    REQUIRE(profile.status().message().find("schema") != std::string::npos);
}

TEST_CASE("Reject invalid JSON", "[gematria]") {
    StatusOr<GematriaProfile> profile = GematriaProfileLoader::load_from_string("{not json");
    REQUIRE_FALSE(profile.ok());
}
