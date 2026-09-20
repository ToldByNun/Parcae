#include <parcae/dsl/dsl_ast.hpp>
#include <parcae/dsl/dsl_ast_json_ingest.hpp>
#include <parcae/dsl/dsl_ast_limits.hpp>
#include <parcae/dsl/dsl_rule_id.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

/// Fixed seed shared with other Parcae property tests / future DslVerifier fuzz.
constexpr std::uint32_t k_ingest_fuzz_seed = 0xC1CADAu;

[[nodiscard]] std::string minimal_success_doc(const std::string& module_json) {
    return std::string("{") +
           R"("schema":"parcae.dsl_ast_json.v0",)" +
           R"("dsl_ast_json_version":"1.0.0",)" +
           R"("source_path":"fuzz.py",)" +
           R"("source_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",)" +
           R"("python_version":"3.12.0",)" +
           R"("ok":true,)" +
           R"("module":)" + module_json + "}";
}

[[nodiscard]] nlohmann::json valid_root_json() {
    return nlohmann::json{
        {"schema", "parcae.dsl_ast_json.v0"},
        {"dsl_ast_json_version", "1.0.0"},
        {"source_path", "fuzz.py"},
        {"source_sha256",
         "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
        {"python_version", "3.12.0"},
        {"ok", true},
        {"module",
         {{"kind", "Module"},
          {"lineno", 1},
          {"col_offset", 0},
          {"body", nlohmann::json::array()},
          {"type_ignores", nlohmann::json::array()}}},
    };
}

/// Bushy tree: fanout 40, depth 3 → >50k nodes with list lengths ≪ max_list_length.
[[nodiscard]] nlohmann::json bush_expr(int depth) {
    if (depth <= 0) {
        return nlohmann::json{
            {"kind", "Name"},
            {"id", "x"},
            {"ctx", "Load"},
            {"lineno", 1},
            {"col_offset", 0},
        };
    }
    nlohmann::json elts = nlohmann::json::array();
    for (int i = 0; i < 40; ++i) {
        elts.push_back(bush_expr(depth - 1));
    }
    return nlohmann::json{
        {"kind", "Tuple"},
        {"lineno", 1},
        {"col_offset", 0},
        {"elts", std::move(elts)},
        {"ctx", "Load"},
    };
}

[[nodiscard]] std::string mutate_bytes(std::string text, std::mt19937& rng) {
    if (text.empty()) {
        return text;
    }
    std::uniform_int_distribution<int> mode_dist(0, 4);
    std::uniform_int_distribution<std::size_t> idx_dist(0, text.size() - 1);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    switch (mode_dist(rng)) {
    case 0: { // flip one byte
        const std::size_t i = idx_dist(rng);
        text[i] = static_cast<char>(byte_dist(rng));
        break;
    }
    case 1: { // truncate
        std::uniform_int_distribution<std::size_t> cut(0, text.size());
        text.resize(cut(rng));
        break;
    }
    case 2: { // insert garbage
        const std::size_t i = idx_dist(rng);
        text.insert(i, 1, static_cast<char>(byte_dist(rng)));
        break;
    }
    case 3: { // append trailing junk
        text += static_cast<char>(byte_dist(rng));
        text += "{";
        break;
    }
    default: { // duplicate a slice / scramble
        const std::size_t i = idx_dist(rng);
        text.push_back(text[i]);
        break;
    }
    }
    return text;
}

/// Ingest must never throw; result is either ok or a Status error message.
void require_no_crash_ingest(std::string_view text) {
    StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    if (doc.ok()) {
        REQUIRE(doc.value().module() != nullptr);
        REQUIRE_FALSE(doc.value().module()->kind().empty());
    } else {
        REQUIRE_FALSE(doc.status().ok());
        REQUIRE_FALSE(doc.status().message().empty());
    }
}

}  // namespace

TEST_CASE("ingest limit E103 node count reject (no crash)", "[dsl][ingest][fuzz]") {
    nlohmann::json root = valid_root_json();
    root["module"] = {
        {"kind", "Module"},
        {"lineno", 1},
        {"col_offset", 0},
        {"body",
         nlohmann::json::array(
             {nlohmann::json{
                 {"kind", "Expr"},
                 {"lineno", 1},
                 {"col_offset", 0},
                 {"value", bush_expr(/*depth=*/3)},
             }})},
        {"type_ignores", nlohmann::json::array()},
    };

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_root(root);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find(DslRuleId::E103_node_count) != std::string::npos);
}

TEST_CASE("ingest limit E110 invalid UTF-8 reject", "[dsl][ingest][fuzz]") {
    std::string text = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})");
    // Inject an invalid UTF-8 byte sequence into the middle of the document.
    const auto pos = text.find("fuzz.py");
    REQUIRE(pos != std::string::npos);
    text.insert(pos, "\xff", 1);

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(text);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find(DslRuleId::E110_utf8) != std::string::npos);
}

TEST_CASE("ingest limit E104 depth still rejects under fuzz tag", "[dsl][ingest][fuzz]") {
    nlohmann::json value = {
        {"kind", "Name"}, {"id", "x"}, {"ctx", "Load"}, {"lineno", 1}, {"col_offset", 0}};
    for (std::size_t i = 0; i < DslAstLimits::max_tree_depth + 2; ++i) {
        value = nlohmann::json{
            {"kind", "Expr"}, {"lineno", 1}, {"col_offset", 0}, {"value", std::move(value)}};
    }
    nlohmann::json root = valid_root_json();
    root["module"]["body"] = nlohmann::json::array({value});

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_root(root);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find(DslRuleId::E104_tree_depth) != std::string::npos);
}

TEST_CASE("ingest limit E106 list length via parse_root", "[dsl][ingest][fuzz]") {
    nlohmann::json body = nlohmann::json::array();
    for (std::size_t i = 0; i < DslAstLimits::max_list_length + 1; ++i) {
        body.push_back(
            {{"kind", "Pass"}, {"lineno", 1}, {"col_offset", 0}});
    }
    nlohmann::json root = valid_root_json();
    root["module"]["body"] = std::move(body);

    const StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_root(root);
    REQUIRE_FALSE(doc.ok());
    REQUIRE(doc.status().message().find(DslRuleId::E106_list_length) != std::string::npos);
}

TEST_CASE("ingest adversarial corpus never crashes", "[dsl][ingest][fuzz]") {
    const std::vector<std::string> corpus = {
        "",
        " ",
        "{}",
        "[]",
        "null",
        "true",
        "\"str\"",
        "{",
        "{]",
        "{{}",
        std::string(1, '\0'),
        std::string("\xff\xfe\xfd", 3),
        minimal_success_doc(
            R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})"),
        minimal_success_doc(
            R"({"kind":"Module","lineno":1,"col_offset":0,"body":[],"type_ignores":[]})") +
            " trailing",
        R"({"schema":"parcae.dsl_ast_json.v0","ok":false,"error":{"message":"boom"}})",
        R"({"schema":"nope","dsl_ast_json_version":"1.0.0","source_path":"x",
            "source_sha256":"00","module":{"kind":"Module","body":[]}})",
        R"({"schema":"parcae.dsl_ast_json.v0","dsl_ast_json_version":"1.0.0",
            "source_path":"x","source_sha256":"00","module":null})",
        R"({"schema":"parcae.dsl_ast_json.v0","dsl_ast_json_version":"1.0.0",
            "source_path":"x","source_sha256":"00","module":{"kind":"NotModule","body":[]}})",
    };

    for (const std::string& sample : corpus) {
        require_no_crash_ingest(sample);
    }
}

TEST_CASE("ingest seeded mutation fuzz (0xC1CADA) never crashes", "[dsl][ingest][fuzz]") {
    std::mt19937 rng(k_ingest_fuzz_seed);
    const std::string base = minimal_success_doc(
        R"({"kind":"Module","lineno":1,"col_offset":0,"body":[{
            "kind":"Expr","lineno":1,"col_offset":0,
            "value":{"kind":"Constant","value":1,"lineno":1,"col_offset":0}
          }],"type_ignores":[]})");

    // Sanity: base document is valid.
    {
        const StatusOr<DslAstDocument> ok = DslAstJsonIngest::parse_text(base);
        REQUIRE(ok.ok());
    }

    constexpr int trials = 256;
    int accepted = 0;
    int rejected = 0;
    for (int t = 0; t < trials; ++t) {
        std::string mutant = base;
        const int mutations = 1 + static_cast<int>(rng() % 4);
        for (int m = 0; m < mutations; ++m) {
            mutant = mutate_bytes(std::move(mutant), rng);
        }
        // Cap size so we do not spend CI time on multi-MiB accidents.
        if (mutant.size() > 256 * 1024) {
            mutant.resize(256 * 1024);
        }
        StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(mutant);
        if (doc.ok()) {
            ++accepted;
            REQUIRE(doc.value().module() != nullptr);
        } else {
            ++rejected;
            REQUIRE_FALSE(doc.status().message().empty());
        }
    }
    // Mutations should almost always break the document; allow rare survivals.
    REQUIRE(rejected + accepted == trials);
    REQUIRE(rejected > trials / 2);
}

TEST_CASE("ingest random byte blobs never crash", "[dsl][ingest][fuzz]") {
    std::mt19937 rng(k_ingest_fuzz_seed ^ 0x9E3779B9u);
    std::uniform_int_distribution<int> len_dist(0, 512);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (int trial = 0; trial < 128; ++trial) {
        const int len = len_dist(rng);
        std::string blob;
        blob.resize(static_cast<std::size_t>(len));
        for (char& ch : blob) {
            ch = static_cast<char>(byte_dist(rng));
        }
        require_no_crash_ingest(blob);
    }
}

TEST_CASE("ingest empty module still gates cleanly after fuzz helpers", "[dsl][ingest][fuzz]") {
    const StatusOr<DslAstDocument> doc =
        DslAstJsonIngest::parse_root(valid_root_json());
    REQUIRE(doc.ok());
    REQUIRE(doc.value().module()->kind() == "Module");
}
