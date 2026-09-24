#include <catch2/catch_test_macros.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/dsl/dsl.hpp>
#include <parcae/dsl/dsl_ast_json_version.hpp>
#include <parcae/dsl/dsl_spec_version.hpp>

TEST_CASE("DslSpecVersion current matches dsl.md 1.0.0", "[dsl][spec_version]") {
    REQUIRE(DslSpecVersion::current_major == 1);
    REQUIRE(DslSpecVersion::current_minor == 0);
    REQUIRE(DslSpecVersion::current_patch == 0);
    REQUIRE(DslSpecVersion::current_string == "1.0.0");

    const DslSpecVersion cur = DslSpecVersion::current();
    REQUIRE(cur.to_string() == "1.0.0");
    REQUIRE(cur.equals_current());
    REQUIRE(cur.check_compatible_with_current().ok());
    REQUIRE_FALSE(cur.stale_spec_relative_to_current());
}

TEST_CASE("DslSpecVersion parse accepts SemVer", "[dsl][spec_version]") {
    const StatusOr<DslSpecVersion> v = DslSpecVersion::parse("1.2.3");
    REQUIRE(v.ok());
    REQUIRE(v.value().major() == 1);
    REQUIRE(v.value().minor() == 2);
    REQUIRE(v.value().patch() == 3);
    REQUIRE(v.value().to_string() == "1.2.3");
}

TEST_CASE("DslSpecVersion parse rejects garbage", "[dsl][spec_version]") {
    REQUIRE_FALSE(DslSpecVersion::parse("").ok());
    REQUIRE_FALSE(DslSpecVersion::parse("1").ok());
    REQUIRE_FALSE(DslSpecVersion::parse("1.0").ok());
    REQUIRE_FALSE(DslSpecVersion::parse("1.0.0-beta").ok());
    REQUIRE_FALSE(DslSpecVersion::parse("01.0.0").ok());
    REQUIRE_FALSE(DslSpecVersion::parse("1.0.0 ").ok());
}

TEST_CASE("DslSpecVersion compatibility policy", "[dsl][spec_version]") {
    const DslSpecVersion same = DslSpecVersion::parse("1.0.0").value();
    REQUIRE(same.check_compatible_with_current().ok());

    // Forward-incompatible: newer minor on same major.
    const DslSpecVersion newer_minor = DslSpecVersion::parse("1.1.0").value();
    REQUIRE_FALSE(newer_minor.check_compatible_with_current().ok());
    REQUIRE(newer_minor.is_newer_than_current());
    REQUIRE_FALSE(newer_minor.stale_spec_relative_to_current());

    // Major mismatch — stale / recompile.
    const DslSpecVersion old_major = DslSpecVersion::parse("0.9.0").value();
    REQUIRE_FALSE(old_major.check_compatible_with_current().ok());
    REQUIRE(old_major.stale_spec_relative_to_current());
    REQUIRE(old_major.major_mismatch_with_current());

    const DslSpecVersion future_major = DslSpecVersion::parse("2.0.0").value();
    REQUIRE_FALSE(future_major.check_compatible_with_current().ok());
    REQUIRE(future_major.stale_spec_relative_to_current());
}

TEST_CASE("DslAstJsonVersion current and schema id", "[dsl][ast_json_version]") {
    REQUIRE(DslAstJsonVersion::current_string == "1.1.0");
    REQUIRE(DslAstJsonVersion::current_minor == 1);
    REQUIRE(DslAstJsonVersion::schema_id == "parcae.dsl_ast_json.v0");
    REQUIRE(DslAstJsonVersion::current().check_compatible_with_current().ok());

    const StatusOr<DslAstJsonVersion> parsed = DslAstJsonVersion::parse("1.1.0");
    REQUIRE(parsed.ok());
    REQUIRE(parsed.value() == DslAstJsonVersion::current());

    // Older minor documents remain compatible.
    const StatusOr<DslAstJsonVersion> older = DslAstJsonVersion::parse("1.0.0");
    REQUIRE(older.ok());
    REQUIRE(older.value().check_compatible_with_current().ok());

    const StatusOr<DslAstJsonVersion> major2 = DslAstJsonVersion::parse("2.0.0");
    REQUIRE(major2.ok());
    REQUIRE_FALSE(major2.value().check_compatible_with_current().ok());
}
