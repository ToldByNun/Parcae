#include <parcae/dsl/dsl_diag.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("DslDiag formats path:line:col: RULE message", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(
        DslRuleId::E013_tier_structural_claim,
        "tier B requires structural_claim()",
        "theories/x.py",
        42,
        4);

    REQUIRE(d.rule_id() == "E013");
    REQUIRE(d.has_location());
    REQUIRE(d.format() == "theories/x.py:42:4: E013 tier B requires structural_claim()");
    REQUIRE(d.to_status().ok() == false);
    REQUIRE(d.to_status().message() == d.format());
}

TEST_CASE("DslDiag formats import whitelist example", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(
        DslRuleId::E021_import_whitelist,
        "import 'numpy' not in parcae.dsl.* whitelist",
        "theories/x.py",
        3,
        1);
    REQUIRE(d.format() == "theories/x.py:3:1: E021 import 'numpy' not in parcae.dsl.* whitelist");
}

TEST_CASE("DslDiag formats interrupt example with hint", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(
        DslRuleId::E030_interrupt_policy,
        "interrupt_policy missing",
        "theories/x.py",
        88,
        4,
        "set interrupts='none_by_design' if intentional");
    REQUIRE(
        d.format() ==
        "theories/x.py:88:4: E030 interrupt_policy missing");
    REQUIRE(
        d.format_with_hint() ==
        "theories/x.py:88:4: E030 interrupt_policy missing\n"
        "note: set interrupts='none_by_design' if intentional");
    REQUIRE(d.to_status().message() == d.format_with_hint());
}

TEST_CASE("DslDiag without path still prints rule", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(DslRuleId::E101_source_size, "source too large");
    REQUIRE_FALSE(d.has_location());
    REQUIRE(d.format() == "E101 source too large");
}

TEST_CASE("DslDiag lineno without path", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(
        DslRuleId::E031_forbidden_construct,
        "for-loop not allowed in primitive body",
        {},
        10,
        0);
    REQUIRE(d.format() == "10:0: E031 for-loop not allowed in primitive body");
}

TEST_CASE("DslDiag missing col defaults to 0 when lineno set", "[dsl][diag]") {
    const DslDiag d = DslDiag::make(
        DslRuleId::E100_json_schema,
        "missing schema field",
        "dump.json",
        1,
        std::nullopt);
    REQUIRE(d.format() == "dump.json:1:0: E100 missing schema field");
}
