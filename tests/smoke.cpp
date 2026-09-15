#include <parcae/core/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

TEST_CASE("parcae_core version macros are wired", "[smoke]") {
    REQUIRE(parcae::core::version_major == PARCAE_VERSION_MAJOR);
    REQUIRE(parcae::core::version_minor == PARCAE_VERSION_MINOR);
    REQUIRE(parcae::core::version_patch == PARCAE_VERSION_PATCH);
}

TEST_CASE("nlohmann_json is available through parcae::core", "[smoke]") {
    const nlohmann::json value = {{"modulus", 29}};
    REQUIRE(value.at("modulus") == 29);
}
