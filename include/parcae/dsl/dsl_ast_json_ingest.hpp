#ifndef DSL_AST_JSON_INGEST_HPP
#define DSL_AST_JSON_INGEST_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/utf8.hpp"
#include "parcae/dsl/dsl_ast.hpp"
#include "parcae/dsl/dsl_ast_json_version.hpp"
#include "parcae/dsl/dsl_ast_limits.hpp"
#include "parcae/dsl/dsl_diag.hpp"
#include "parcae/dsl/dsl_rule_id.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Hardened JSON → DslAstDocument ingest (docs/spec/dsl-ast-json.md).
/// Strict schema mode: unknown top-level keys rejected. Never throws to callers.
class DslAstJsonIngest {
public:
    [[nodiscard]] static StatusOr<DslAstDocument> parse_text(std::string_view text) {
        if (text.size() > DslAstLimits::max_json_bytes) {
            return fail(DslRuleId::E102_json_size,
                        "JSON document exceeds " + std::to_string(DslAstLimits::max_json_bytes) +
                            " bytes (" + std::to_string(text.size()) + ")");
        }

        Status utf8 = validate_utf8(text);
        if (!utf8.ok()) {
            return fail(DslRuleId::E110_utf8, utf8.message());
        }

        nlohmann::json root;
        try {
            std::istringstream stream{std::string(text)};
            stream >> root;
            if (stream.fail() && !stream.eof()) {
                return fail(DslRuleId::E100_json_schema, "invalid JSON");
            }
            stream >> std::ws;
            const int peek = stream.peek();
            if (peek != std::char_traits<char>::eof()) {
                return fail(DslRuleId::E111_trailing_garbage,
                            "JSON document has trailing non-whitespace garbage");
            }
        } catch (const nlohmann::json::exception& ex) {
            return fail(DslRuleId::E100_json_schema, std::string("invalid JSON: ") + ex.what());
        }

        return parse_root(root);
    }

    [[nodiscard]] static StatusOr<DslAstDocument> parse_root(const nlohmann::json& root) {
        if (!root.is_object()) {
            return fail(DslRuleId::E100_json_schema, "document root must be a JSON object");
        }

        // Failure envelope from ast_dump — not an ingestable Module document.
        if (root.contains("ok") && root.at("ok").is_boolean() &&
            root.at("ok").get<bool>() == false) {
            std::string msg = "received dsl_ast_json failure envelope (ok=false)";
            if (root.contains("error") && root.at("error").is_object() &&
                root.at("error").contains("message") &&
                root.at("error").at("message").is_string()) {
                msg += ": ";
                msg += root.at("error").at("message").get<std::string>();
            }
            return fail(DslRuleId::E100_json_schema, std::move(msg));
        }

        for (auto it = root.begin(); it != root.end(); ++it) {
            const std::string& key = it.key();
            if (key != "schema" && key != "dsl_ast_json_version" && key != "source_path" &&
                key != "source_sha256" && key != "python_version" && key != "module" &&
                key != "ok" && key != "directives") {
                return fail(DslRuleId::E100_json_schema,
                            "unknown top-level key '" + key + "' (strict schema mode)");
            }
        }

        if (!root.contains("schema") || !root.at("schema").is_string() ||
            root.at("schema").get<std::string>() != DslAstJsonVersion::schema_id) {
            return fail(DslRuleId::E100_json_schema,
                        std::string("schema must be ") + std::string(DslAstJsonVersion::schema_id));
        }

        if (!root.contains("dsl_ast_json_version") ||
            !root.at("dsl_ast_json_version").is_string()) {
            return fail(DslRuleId::E100_json_schema, "dsl_ast_json_version must be a string");
        }
        const std::string version_text = root.at("dsl_ast_json_version").get<std::string>();
        StatusOr<DslAstJsonVersion> version = DslAstJsonVersion::parse(version_text);
        if (!version.ok()) {
            return fail(DslRuleId::E100_json_schema, version.status().message());
        }
        Status compat = version.value().check_compatible_with_current();
        if (!compat.ok()) {
            return fail(DslRuleId::E100_json_schema, compat.message());
        }

        if (!root.contains("source_path") || !root.at("source_path").is_string()) {
            return fail(DslRuleId::E100_json_schema, "source_path must be a string");
        }
        if (!root.contains("source_sha256") || !root.at("source_sha256").is_string()) {
            return fail(DslRuleId::E100_json_schema, "source_sha256 must be a string");
        }
        if (!root.contains("module") || !root.at("module").is_object()) {
            return fail(DslRuleId::E100_json_schema, "module must be a JSON object");
        }

        if (root.contains("ok") && root.at("ok").is_boolean() &&
            root.at("ok").get<bool>() != true) {
            return fail(DslRuleId::E100_json_schema,
                        "ok must be true when present on success docs");
        }

        IngestState state;
        StatusOr<std::shared_ptr<DslAstNode>> module =
            convert_node(root.at("module"), state, /*depth=*/0, /*path_hint=*/{});
        if (!module.ok()) {
            return module.status();
        }
        if (module.value()->kind() != "Module") {
            return fail_at(DslRuleId::E100_json_schema, "module.kind must be Module",
                           root.at("source_path").get<std::string>(), module.value()->lineno(),
                           module.value()->col_offset());
        }

        DslAstDocument doc;
        doc.set_source_path(root.at("source_path").get<std::string>());
        doc.set_source_sha256(root.at("source_sha256").get<std::string>());
        doc.set_dsl_ast_json_version(version_text);
        if (root.contains("python_version")) {
            if (!root.at("python_version").is_string()) {
                return fail(DslRuleId::E100_json_schema,
                            "python_version must be a string when present");
            }
            doc.set_python_version(root.at("python_version").get<std::string>());
        }
        doc.set_module(std::move(module.value()));

        if (root.contains("directives")) {
            StatusOr<std::vector<DslAstDirective>> dirs = parse_directives(root.at("directives"));
            if (!dirs.ok()) {
                return dirs.status();
            }
            doc.set_directives(std::move(dirs.value()));
        }

        return doc;
    }

private:
    struct IngestState {
        std::size_t node_count = 0;
    };

    DslAstJsonIngest() = delete;

    [[nodiscard]] static Status fail(std::string_view rule_id, std::string message) {
        return DslDiag::make(rule_id, std::move(message)).to_status();
    }

    [[nodiscard]] static StatusOr<std::vector<DslAstDirective>>
    parse_directives(const nlohmann::json& arr) {
        if (!arr.is_array()) {
            return fail(DslRuleId::E100_json_schema, "directives must be a JSON array");
        }
        if (arr.size() > DslAstLimits::max_list_length) {
            return fail(DslRuleId::E106_list_length,
                        "directives length exceeds " +
                            std::to_string(DslAstLimits::max_list_length));
        }
        std::vector<DslAstDirective> out;
        out.reserve(arr.size());
        for (const nlohmann::json& item : arr) {
            if (!item.is_object()) {
                return fail(DslRuleId::E100_json_schema, "directives[] entries must be objects");
            }
            for (auto it = item.begin(); it != item.end(); ++it) {
                const std::string& key = it.key();
                if (key != "lineno" && key != "flag" && key != "raw") {
                    return fail(DslRuleId::E100_json_schema,
                                "unknown directives[] key '" + key + "'");
                }
            }
            if (!item.contains("lineno") || !item.at("lineno").is_number_integer()) {
                return fail(DslRuleId::E100_json_schema, "directives[].lineno must be an integer");
            }
            if (!item.contains("flag") || !item.at("flag").is_string()) {
                return fail(DslRuleId::E100_json_schema, "directives[].flag must be a string");
            }
            if (!item.contains("raw") || !item.at("raw").is_string()) {
                return fail(DslRuleId::E100_json_schema, "directives[].raw must be a string");
            }
            const std::string flag = item.at("flag").get<std::string>();
            const std::string raw = item.at("raw").get<std::string>();
            Status flag_len = check_string(flag);
            if (!flag_len.ok()) {
                return flag_len;
            }
            Status raw_len = check_string(raw);
            if (!raw_len.ok()) {
                return raw_len;
            }
            // Grammar: FLAG_NAME := [a-z][a-z0-9_]*
            if (flag.empty() || flag[0] < 'a' || flag[0] > 'z') {
                return fail(DslRuleId::E100_json_schema,
                            "directives[].flag must match [a-z][a-z0-9_]*");
            }
            for (char c : flag) {
                const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
                if (!ok) {
                    return fail(DslRuleId::E100_json_schema,
                                "directives[].flag must match [a-z][a-z0-9_]*");
                }
            }
            const std::int64_t lineno = item.at("lineno").get<std::int64_t>();
            if (lineno < 1 || lineno > 2'000'000'000) {
                return fail(DslRuleId::E100_json_schema, "directives[].lineno out of range");
            }
            out.emplace_back(static_cast<int>(lineno), flag, raw);
        }
        return out;
    }

    [[nodiscard]] static Status fail_at(std::string_view rule_id, std::string message,
                                        std::string path, std::optional<int> lineno,
                                        std::optional<int> col) {
        return DslDiag::make(rule_id, std::move(message), std::move(path), lineno, col).to_status();
    }

    [[nodiscard]] static Status validate_utf8(std::string_view text) {
        // Walk via string copy for Utf8::decode_at API.
        const std::string owned(text);
        std::size_t i = 0;
        while (i < owned.size()) {
            StatusOr<Utf8::Codepoint> cp = Utf8::decode_at(owned, i);
            if (!cp.ok()) {
                return Status::error(cp.status().message());
            }
            i += cp.value().size;
        }
        return Status::success();
    }

    [[nodiscard]] static Status check_string(std::string_view s) {
        // Byte length of UTF-8 string content (already validated as UTF-8 at doc level;
        // individual JSON strings are UTF-8 by JSON rules when parsed from UTF-8 text).
        if (s.size() > DslAstLimits::max_string_bytes) {
            return fail(DslRuleId::E105_string_length,
                        "string field exceeds " + std::to_string(DslAstLimits::max_string_bytes) +
                            " UTF-8 bytes (" + std::to_string(s.size()) + ")");
        }
        return Status::success();
    }

    [[nodiscard]] static std::optional<int> opt_int(const nlohmann::json& obj, const char* key) {
        if (!obj.contains(key) || obj.at(key).is_null()) {
            return std::nullopt;
        }
        if (!obj.at(key).is_number_integer()) {
            return std::nullopt; // caller validates type when key present and non-null
        }
        return obj.at(key).get<int>();
    }

    [[nodiscard]] static StatusOr<std::shared_ptr<DslAstNode>>
    convert_node(const nlohmann::json& obj, IngestState& state, std::size_t depth,
                 const std::string& source_path) {
        if (depth > DslAstLimits::max_tree_depth) {
            return fail(DslRuleId::E104_tree_depth, "AST depth exceeds limit");
        }
        if (!obj.is_object()) {
            return fail(DslRuleId::E100_json_schema, "AST node must be a JSON object");
        }
        if (!obj.contains("kind") || !obj.at("kind").is_string()) {
            return fail(DslRuleId::E100_json_schema, "AST node missing string kind");
        }

        ++state.node_count;
        if (state.node_count > DslAstLimits::max_node_count) {
            return fail(DslRuleId::E103_node_count, "AST node count exceeds limit");
        }

        auto node = std::make_shared<DslAstNode>(obj.at("kind").get<std::string>());
        Status kind_len = check_string(node->kind());
        if (!kind_len.ok()) {
            return kind_len;
        }

        // Location fields (optional / nullable).
        for (const char* key : {"lineno", "col_offset", "end_lineno", "end_col_offset"}) {
            if (!obj.contains(key)) {
                continue;
            }
            const nlohmann::json& v = obj.at(key);
            if (v.is_null()) {
                continue;
            }
            if (!v.is_number_integer()) {
                return fail_at(DslRuleId::E100_json_schema,
                               std::string(key) + " must be integer or null", source_path,
                               node->lineno(), node->col_offset());
            }
        }
        node->set_lineno(opt_int(obj, "lineno"));
        node->set_col_offset(opt_int(obj, "col_offset"));
        node->set_end_lineno(opt_int(obj, "end_lineno"));
        node->set_end_col_offset(opt_int(obj, "end_col_offset"));

        for (auto it = obj.begin(); it != obj.end(); ++it) {
            const std::string& key = it.key();
            if (key == "kind" || key == "lineno" || key == "col_offset" || key == "end_lineno" ||
                key == "end_col_offset") {
                continue;
            }
            StatusOr<DslAstValue> child =
                convert_value(it.value(), state, depth + 1, source_path, node.get());
            if (!child.ok()) {
                return child.status();
            }
            node->set_field(key, std::move(child.value()));
        }
        return node;
    }

    [[nodiscard]] static StatusOr<DslAstValue> convert_value(const nlohmann::json& value,
                                                             IngestState& state, std::size_t depth,
                                                             const std::string& source_path,
                                                             const DslAstNode* parent) {
        if (value.is_null()) {
            return DslAstValue::null();
        }
        if (value.is_boolean()) {
            return DslAstValue::boolean(value.get<bool>());
        }
        if (value.is_number_integer()) {
            return DslAstValue::integer(value.get<std::int64_t>());
        }
        if (value.is_number_float()) {
            return DslAstValue::floating(value.get<double>());
        }
        if (value.is_string()) {
            // Compact op/ctx ("Add", "Load") or normal string field.
            const std::string s = value.get<std::string>();
            Status chk = check_string(s);
            if (!chk.ok()) {
                return chk;
            }
            return DslAstValue::string(s);
        }
        if (value.is_array()) {
            if (value.size() > DslAstLimits::max_list_length) {
                return fail(DslRuleId::E106_list_length,
                            "list length exceeds " + std::to_string(DslAstLimits::max_list_length));
            }
            std::vector<DslAstValue> items;
            items.reserve(value.size());
            for (const nlohmann::json& item : value) {
                StatusOr<DslAstValue> converted =
                    convert_value(item, state, depth + 1, source_path, parent);
                if (!converted.ok()) {
                    return converted.status();
                }
                items.push_back(std::move(converted.value()));
            }
            return DslAstValue::array(std::move(items));
        }
        if (value.is_object()) {
            // Nested AST node, or compact `{ "kind": "Add" }` operator object.
            StatusOr<std::shared_ptr<DslAstNode>> node =
                convert_node(value, state, depth, source_path);
            if (!node.ok()) {
                return node.status();
            }
            return DslAstValue::node(std::move(node.value()));
        }
        return fail(DslRuleId::E100_json_schema, "unsupported JSON value type in AST");
    }
};

#endif // DSL_AST_JSON_INGEST_HPP
