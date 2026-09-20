#ifndef DSL_AST_HPP
#define DSL_AST_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class DslAstNode;

/// Recursive value carried under DslAstNode fields (JSON child after ingest).
class DslAstValue {
public:
    enum class Type : std::uint8_t {
        Null = 0,
        Bool,
        Int,
        Float,
        String,
        Node,
        Array,
    };

    [[nodiscard]] static DslAstValue null() {
        return DslAstValue{Type::Null};
    }

    [[nodiscard]] static DslAstValue boolean(bool v) {
        DslAstValue out{Type::Bool};
        out.bool_ = v;
        return out;
    }

    [[nodiscard]] static DslAstValue integer(std::int64_t v) {
        DslAstValue out{Type::Int};
        out.int_ = v;
        return out;
    }

    [[nodiscard]] static DslAstValue floating(double v) {
        DslAstValue out{Type::Float};
        out.float_ = v;
        return out;
    }

    [[nodiscard]] static DslAstValue string(std::string v) {
        DslAstValue out{Type::String};
        out.string_ = std::move(v);
        return out;
    }

    [[nodiscard]] static DslAstValue node(std::shared_ptr<DslAstNode> v) {
        DslAstValue out{Type::Node};
        out.node_ = std::move(v);
        return out;
    }

    [[nodiscard]] static DslAstValue array(std::vector<DslAstValue> v) {
        DslAstValue out{Type::Array};
        out.array_ = std::move(v);
        return out;
    }

    [[nodiscard]] Type type() const noexcept {
        return type_;
    }

    [[nodiscard]] bool is_null() const noexcept {
        return type_ == Type::Null;
    }

    [[nodiscard]] bool as_bool() const noexcept {
        return bool_;
    }

    [[nodiscard]] std::int64_t as_int() const noexcept {
        return int_;
    }

    [[nodiscard]] double as_float() const noexcept {
        return float_;
    }

    [[nodiscard]] const std::string& as_string() const noexcept {
        return string_;
    }

    [[nodiscard]] const std::shared_ptr<DslAstNode>& as_node() const noexcept {
        return node_;
    }

    [[nodiscard]] const std::vector<DslAstValue>& as_array() const noexcept {
        return array_;
    }

private:
    explicit DslAstValue(Type type) : type_(type) {}

    Type type_;
    bool bool_ = false;
    std::int64_t int_ = 0;
    double float_ = 0.0;
    std::string string_;
    std::shared_ptr<DslAstNode> node_;
    std::vector<DslAstValue> array_;
};

/// One AST node from parcae.dsl_ast_json.v0 (kind + location + named children).
class DslAstNode {
public:
    DslAstNode() = default;

    explicit DslAstNode(std::string kind) : kind_(std::move(kind)) {}

    [[nodiscard]] const std::string& kind() const noexcept {
        return kind_;
    }

    void set_kind(std::string kind) {
        kind_ = std::move(kind);
    }

    [[nodiscard]] std::optional<int> lineno() const noexcept {
        return lineno_;
    }

    [[nodiscard]] std::optional<int> col_offset() const noexcept {
        return col_offset_;
    }

    [[nodiscard]] std::optional<int> end_lineno() const noexcept {
        return end_lineno_;
    }

    [[nodiscard]] std::optional<int> end_col_offset() const noexcept {
        return end_col_offset_;
    }

    void set_lineno(std::optional<int> v) {
        lineno_ = v;
    }

    void set_col_offset(std::optional<int> v) {
        col_offset_ = v;
    }

    void set_end_lineno(std::optional<int> v) {
        end_lineno_ = v;
    }

    void set_end_col_offset(std::optional<int> v) {
        end_col_offset_ = v;
    }

    void set_field(std::string name, DslAstValue value) {
        fields_.emplace_back(std::move(name), std::move(value));
    }

    [[nodiscard]] const std::vector<std::pair<std::string, DslAstValue>>& fields() const noexcept {
        return fields_;
    }

    [[nodiscard]] const DslAstValue* find_field(std::string_view name) const noexcept {
        for (const auto& entry : fields_) {
            if (entry.first == name) {
                return &entry.second;
            }
        }
        return nullptr;
    }

private:
    std::string kind_;
    std::optional<int> lineno_;
    std::optional<int> col_offset_;
    std::optional<int> end_lineno_;
    std::optional<int> end_col_offset_;
    std::vector<std::pair<std::string, DslAstValue>> fields_;
};

/// Top-level success document after JSON ingest (not a failure envelope).
class DslAstDocument {
public:
    DslAstDocument() = default;

    void set_source_path(std::string path) {
        source_path_ = std::move(path);
    }

    void set_source_sha256(std::string hash) {
        source_sha256_ = std::move(hash);
    }

    void set_python_version(std::string version) {
        python_version_ = std::move(version);
    }

    void set_dsl_ast_json_version(std::string version) {
        dsl_ast_json_version_ = std::move(version);
    }

    void set_module(std::shared_ptr<DslAstNode> module) {
        module_ = std::move(module);
    }

    [[nodiscard]] const std::string& source_path() const noexcept {
        return source_path_;
    }

    [[nodiscard]] const std::string& source_sha256() const noexcept {
        return source_sha256_;
    }

    [[nodiscard]] const std::string& python_version() const noexcept {
        return python_version_;
    }

    [[nodiscard]] const std::string& dsl_ast_json_version() const noexcept {
        return dsl_ast_json_version_;
    }

    [[nodiscard]] const std::shared_ptr<DslAstNode>& module() const noexcept {
        return module_;
    }

private:
    std::string source_path_;
    std::string source_sha256_;
    std::string python_version_;
    std::string dsl_ast_json_version_;
    std::shared_ptr<DslAstNode> module_;
};

#endif // DSL_AST_HPP
