#ifndef DSL_AST_LIMITS_HPP
#define DSL_AST_LIMITS_HPP

#include <cstddef>

/// Normative v0 ceilings from docs/spec/dsl-ast-json.md (C++ ingest MUST enforce).
class DslAstLimits {
public:
    static constexpr std::size_t max_source_bytes = 1'048'576;
    static constexpr std::size_t max_json_bytes = 8'388'608;
    static constexpr std::size_t max_node_count = 50'000;
    static constexpr std::size_t max_tree_depth = 64;
    static constexpr std::size_t max_string_bytes = 16'384;
    static constexpr std::size_t max_list_length = 4'096;

private:
    DslAstLimits() = delete;
};

#endif // DSL_AST_LIMITS_HPP
