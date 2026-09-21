#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include "parcae/dsl/theory_registry.hpp"
#include "parcae/generate/generator_registry.hpp"
#include "parcae/score/score_registry.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/tool_backend.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "catalog";

void print_help() {
    std::cerr
        << "Usage: parcae-catalog [--all] [--transforms] [--scores] [--generators]\n"
        << "                      [--backends] [--theories] [--json] [--data-dir <path>]\n"
        << "\n"
        << "List agent-facing registries (transforms, scores, generators, backends,\n"
        << "compiled theories). With no section flags, --all is implied.\n"
        << "\n"
        << "  --all          All sections (default when none selected)\n"
        << "  --transforms   Transform ids\n"
        << "  --scores       Score catalog (id, version, order, arity)\n"
        << "  --generators   Generator catalog (gen_*)\n"
        << "  --backends     cpu / cuda (cuda only if this build linked CUDA)\n"
        << "  --theories     Compiled theory URIs under data/theories/\n"
        << "                 (marks stale_spec; ready=false when incompatible)\n"
        << "  --json         JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir     Parcae data/ root (required for meaningful --theories)\n"
        << "  -h, --help     Show this help\n";
}

[[nodiscard]] int fail(
    bool json_mode,
    ToolErrorCode code,
    std::string message,
    int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(kTool, std::nullopt, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] nlohmann::json backends_json() {
    nlohmann::json backends = nlohmann::json::array();
    backends.push_back({
        {"id", "cpu"},
        {"available", true},
    });
    backends.push_back({
        {"id", "cuda"},
        {"available", parcae::tool::BackendUtil::cuda_built()},
    });
    return backends;
}

[[nodiscard]] StatusOr<std::vector<TheoryRegistry::CatalogEntry>> load_theories(
    const std::filesystem::path& theories_root) {
    return TheoryRegistry::list(theories_root);
}

void print_theories_human(const std::vector<TheoryRegistry::CatalogEntry>& entries) {
    std::cout << "theories:\n";
    if (entries.empty()) {
        std::cout << "  (none)\n";
        return;
    }
    for (const TheoryRegistry::CatalogEntry& e : entries) {
        std::cout << "  " << e.uri().to_string() << '\t' << "dsl_spec=" << e.dsl_spec_version()
                  << '\t' << (e.stale_spec() ? "stale_spec" : "current_major") << '\t'
                  << (e.ready() ? "ready" : "not_ready");
        if (!e.detail().empty()) {
            std::cout << '\t' << e.detail();
        }
        std::cout << '\n';
    }
}

void print_human(
    bool want_transforms,
    bool want_scores,
    bool want_generators,
    bool want_backends,
    bool want_theories,
    const std::vector<TheoryRegistry::CatalogEntry>* theories) {
    if (want_transforms) {
        std::cout << "transforms:\n";
        for (const std::string& id : parcae::tool::list_transform_ids()) {
            std::cout << "  " << id << '\n';
        }
    }
    if (want_scores) {
        std::cout << "scores:\n";
        for (const ScoreCatalogEntry& entry : ScoreRegistry::catalog()) {
            std::cout << "  " << entry.id() << '\t' << entry.version() << '\t'
                      << ScoreOrderUtil::to_string(entry.order()) << '\t'
                      << ScoreCatalogEntry::arity_string(entry.arity()) << '\n';
        }
    }
    if (want_generators) {
        std::cout << "generators:\n";
        for (const GeneratorCatalogEntry& entry : GeneratorRegistry::catalog()) {
            std::cout << "  " << entry.id() << '\t' << entry.transform_id() << '\t'
                      << "bounded_count=" << entry.bounded_count()
                      << (entry.requires_params() ? "\trequires_params" : "") << '\n';
        }
    }
    if (want_backends) {
        std::cout << "backends:\n";
        std::cout << "  cpu\tavailable\n";
        std::cout << "  cuda\t"
                  << (parcae::tool::BackendUtil::cuda_built() ? "available" : "not_built")
                  << '\n';
    }
    if (want_theories && theories != nullptr) {
        print_theories_human(*theories);
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help")) {
        print_help();
        return kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool flag_all = has_flag(args, "--all");
    const bool flag_transforms = has_flag(args, "--transforms");
    const bool flag_scores = has_flag(args, "--scores");
    const bool flag_generators = has_flag(args, "--generators");
    const bool flag_backends = has_flag(args, "--backends");
    const bool flag_theories = has_flag(args, "--theories");
    const std::string data_dir = optional_option(args, "--data-dir");

    const bool any_section = flag_transforms || flag_scores || flag_generators || flag_backends ||
                             flag_theories;
    if (flag_all && any_section) {
        return fail(
            json_mode,
            ToolErrorCode::Usage,
            "Use either --all or specific section flags, not both",
            kExitUsage);
    }

    const bool want_all = flag_all || !any_section;
    const bool want_transforms = want_all || flag_transforms;
    const bool want_scores = want_all || flag_scores;
    const bool want_generators = want_all || flag_generators;
    const bool want_backends = want_all || flag_backends;
    const bool want_theories = want_all || flag_theories;

    // Reject unknown options (other than known flags / --data-dir value).
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--json" || arg == "--all" || arg == "--transforms" || arg == "--scores" ||
            arg == "--generators" || arg == "--backends" || arg == "--theories" || arg == "-h" ||
            arg == "--help") {
            continue;
        }
        if (arg == "--data-dir") {
            ++i;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            print_help();
            return fail(json_mode, ToolErrorCode::Usage, "Unknown option: " + arg, kExitUsage);
        }
        print_help();
        return fail(json_mode, ToolErrorCode::Usage, "Unexpected argument: " + arg, kExitUsage);
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, ToolErrorCode::Io, ctx.status().message(), kExitUsage);
    }

    std::vector<TheoryRegistry::CatalogEntry> theories;
    if (want_theories) {
        StatusOr<std::vector<TheoryRegistry::CatalogEntry>> listed =
            load_theories(ctx.value().data_root() / "theories");
        if (!listed.ok()) {
            return fail(json_mode, ToolErrorCode::Io, listed.status().message(), kExitUsage);
        }
        theories = std::move(listed.value());
    }

    if (!json_mode) {
        print_human(
            want_transforms,
            want_scores,
            want_generators,
            want_backends,
            want_theories,
            want_theories ? &theories : nullptr);
        return kExitOk;
    }

    nlohmann::json result = nlohmann::json::object();
    if (want_transforms) {
        result["transforms"] = parcae::tool::list_transform_ids();
    }
    if (want_scores) {
        nlohmann::json scores = nlohmann::json::array();
        for (const ScoreCatalogEntry& entry : ScoreRegistry::catalog()) {
            scores.push_back(entry.to_json());
        }
        result["scores"] = std::move(scores);
        result["score_ids"] = ScoreRegistry::known_ids();
    }
    if (want_generators) {
        nlohmann::json generators = nlohmann::json::array();
        for (const GeneratorCatalogEntry& entry : GeneratorRegistry::catalog()) {
            generators.push_back(entry.to_json());
        }
        result["generators"] = std::move(generators);
        result["generator_ids"] = GeneratorRegistry::list_generator_ids();
    }
    if (want_backends) {
        result["backends"] = backends_json();
    }
    if (want_theories) {
        nlohmann::json arr = nlohmann::json::array();
        nlohmann::json uris = nlohmann::json::array();
        for (const TheoryRegistry::CatalogEntry& e : theories) {
            arr.push_back(e.to_json());
            uris.push_back(e.uri().to_string());
        }
        result["theories"] = std::move(arr);
        result["theory_uris"] = std::move(uris);
        result["theories_dir"] = (ctx.value().data_root() / "theories").string();
    }

    return ToolCliJson::ok(kTool, std::nullopt, std::move(result));
}
