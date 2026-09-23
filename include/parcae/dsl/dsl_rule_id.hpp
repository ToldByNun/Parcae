#ifndef DSL_RULE_ID_HPP
#define DSL_RULE_ID_HPP

#include <string_view>

/// Stable diagnostic rule ids for the theory DSL compiler (docs/spec/dsl.md).
/// E0xx = language / semantic gate; E1xx = AST-JSON ingest limits / protocol.
class DslRuleId {
public:
    // --- Language / semantic (E0xx) ---
    static constexpr std::string_view E013_tier_structural_claim = "E013";
    static constexpr std::string_view E021_import_whitelist = "E021";
    static constexpr std::string_view E030_interrupt_policy = "E030";
    static constexpr std::string_view E031_forbidden_construct = "E031";
    static constexpr std::string_view E032_primitive_body = "E032";
    static constexpr std::string_view E033_divergent_branch = "E033";
    static constexpr std::string_view E034_hotloop_control = "E034";
    static constexpr std::string_view E035_host_loop_unbounded = "E035";
    static constexpr std::string_view E040_param_domain = "E040";
    static constexpr std::string_view E050_verify_failed = "E050";

    // --- Warnings (W0xx; never alone fail compile) ---
    static constexpr std::string_view W010_dsl_ignore_used = "W010";
    static constexpr std::string_view W011_relaxed_branch = "W011";

    // --- Ingest / protocol (E1xx) ---
    static constexpr std::string_view E100_json_schema = "E100";
    static constexpr std::string_view E101_source_size = "E101";
    static constexpr std::string_view E102_json_size = "E102";
    static constexpr std::string_view E103_node_count = "E103";
    static constexpr std::string_view E104_tree_depth = "E104";
    static constexpr std::string_view E105_string_length = "E105";
    static constexpr std::string_view E106_list_length = "E106";
    static constexpr std::string_view E110_utf8 = "E110";
    static constexpr std::string_view E111_trailing_garbage = "E111";

private:
    DslRuleId() = delete;
};

#endif // DSL_RULE_ID_HPP
