#include "cli_io.hpp"
#include "tool_cli_json.hpp"
#include "agent_policy_cli.hpp"

#include "parcae/batch/batch_execution.hpp"
#include "parcae/batch/batch_result.hpp"
#include "parcae/core/index29.hpp"
#include "parcae/generate/transform_candidate.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/tool/rank_candidates.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "rank";
constexpr std::size_t kDefaultLatinMax = 64;

void print_help() {
    std::cerr
        << "Usage: parcae-rank --candidates <file|-> --score-id <id> --k <n>\n"
        << "                   [--params-json <json>] [--latin-max <n>] [--no-latin]\n"
        << "                   [--backend cpu|cuda] [--allow-cuda]\n"
        << "                   [--json] [--data-dir <path>]\n"
        << "\n"
        << "Score and rank TransformCandidate JSON (stable ties).\n"
        << "\n"
        << "  --candidates     JSON file or '-' (stdin). Accepts:\n"
        << "                   • generate --json envelope (result.candidates)\n"
        << "                   • {\"candidates\":[...]} object\n"
        << "                   • bare candidate array\n"
        << "  --score-id       Registry id (e.g. ic_mod29, chi2_english_gp_v0)\n"
        << "  --k              Top-k hits to keep (required, >= 1)\n"
        << "  --params-json    Score params (e.g. {\"reference\":[…]} for pairwise)\n"
        << "  --latin-max      Truncate latin preview (default 64; 0 = full)\n"
        << "  --no-latin       Omit latin preview in JSON / human output\n"
        << "  --backend        cpu|cuda (default cpu; exit 2 if cuda not built)\n"
        << "  --allow-cuda     Required with --backend cuda (AgentPolicy opt-in)\n"
        << "  --json           JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir       Parcae data/ root (needed for chi2 table / latin)\n"
        << "  -h, --help       Show this help\n";
}

[[nodiscard]] int fail(
    bool json_mode,
    const std::optional<std::string>& backend,
    ToolErrorCode code,
    std::string message,
    int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(kTool, backend, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] StatusOr<std::size_t> parse_size_option(
    const std::vector<std::string>& args,
    std::string_view flag,
    bool required) {
    if (required) {
        StatusOr<std::string> text = CliIo::require_option(args, flag);
        if (!text.ok()) {
            return text.status();
        }
        try {
            const unsigned long long value = std::stoull(text.value());
            return static_cast<std::size_t>(value);
        } catch (const std::exception&) {
            return Status::error(std::string(flag) + " must be a non-negative integer");
        }
    }
    const std::string text = CliIo::optional_option(args, flag);
    if (text.empty()) {
        return static_cast<std::size_t>(0);
    }
    try {
        const unsigned long long value = std::stoull(text);
        return static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        return Status::error(std::string(flag) + " must be a non-negative integer");
    }
}

[[nodiscard]] StatusOr<TransformCandidate> candidate_from_json(const nlohmann::json& root) {
    if (!root.is_object()) {
        return Status::error("TransformCandidate must be a JSON object");
    }
    if (!root.contains("candidate_id") || !root.at("candidate_id").is_string()) {
        return Status::error("TransformCandidate.candidate_id is required");
    }
    if (!root.contains("envelope") || !root.at("envelope").is_object()) {
        return Status::error("TransformCandidate.envelope is required");
    }
    if (!root.contains("output_indices") || !root.at("output_indices").is_array()) {
        return Status::error("TransformCandidate.output_indices is required");
    }

    const nlohmann::json& envelope = root.at("envelope");
    if (!envelope.contains("transform_id") || !envelope.at("transform_id").is_string()) {
        return Status::error("envelope.transform_id is required");
    }
    StatusOr<TransformId> transform_id =
        TransformId::from_string(envelope.at("transform_id").get<std::string>());
    if (!transform_id.ok()) {
        return transform_id.status();
    }

    TransformDirection direction = TransformDirection::Decrypt;
    if (envelope.contains("direction")) {
        if (!envelope.at("direction").is_string()) {
            return Status::error("envelope.direction must be a string");
        }
        StatusOr<TransformDirection> parsed =
            TransformDirectionUtil::from_string(envelope.at("direction").get<std::string>());
        if (!parsed.ok()) {
            return parsed.status();
        }
        direction = parsed.value();
    }

    nlohmann::json params = nlohmann::json::object();
    if (envelope.contains("params")) {
        if (!envelope.at("params").is_object()) {
            return Status::error("envelope.params must be an object");
        }
        params = envelope.at("params");
    }

    std::optional<nlohmann::json> interrupt;
    if (envelope.contains("interrupt")) {
        interrupt = envelope.at("interrupt");
    }

    std::vector<Index29> indices;
    indices.reserve(root.at("output_indices").size());
    for (const auto& item : root.at("output_indices")) {
        if (!item.is_number_integer()) {
            return Status::error("output_indices entries must be integers");
        }
        const auto value = item.get<std::int64_t>();
        if (value < 0 || value >= static_cast<std::int64_t>(Index29::modulus)) {
            return Status::error("output_indices entry out of range [0,28]");
        }
        indices.push_back(Index29{static_cast<std::uint8_t>(value)});
    }

    return TransformCandidate{
        root.at("candidate_id").get<std::string>(),
        transform_id.value(),
        direction,
        std::move(params),
        std::move(indices),
        std::move(interrupt),
    };
}

[[nodiscard]] StatusOr<std::vector<TransformCandidate>> load_candidates(
    const std::string& text) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& ex) {
        return Status::error(std::string("Invalid candidates JSON: ") + ex.what());
    }

    const nlohmann::json* array = nullptr;
    if (root.is_array()) {
        array = &root;
    } else if (root.is_object()) {
        if (root.contains("result") && root.at("result").is_object() &&
            root.at("result").contains("candidates")) {
            array = &root.at("result").at("candidates");
        } else if (root.contains("candidates")) {
            array = &root.at("candidates");
        } else {
            return Status::error(
                "Candidates JSON must be an array, {\"candidates\":[...]}, or a "
                "generate --json envelope");
        }
    } else {
        return Status::error("Candidates JSON must be an object or array");
    }

    if (!array->is_array()) {
        return Status::error("candidates must be a JSON array");
    }
    if (array->empty()) {
        return Status::error("candidates array is empty");
    }

    std::vector<TransformCandidate> out;
    out.reserve(array->size());
    for (std::size_t i = 0; i < array->size(); ++i) {
        StatusOr<TransformCandidate> candidate = candidate_from_json((*array)[i]);
        if (!candidate.ok()) {
            return Status::error(
                "candidates[" + std::to_string(i) + "]: " + candidate.status().message());
        }
        out.push_back(std::move(candidate.value()));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const bool no_latin = CliIo::has_flag(args, "--no-latin");
    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    std::optional<std::string> backend_label;

    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, std::nullopt, ToolErrorCode::Io, ctx.status().message(), CliIo::kExitUsage);
    }

    const AgentPolicy policy = AgentPolicyCli::make(ctx.value(), args);
    StatusOr<Backend> backend = AgentPolicyCli::resolve_backend(policy, args);
    if (!backend.ok()) {
        const ToolErrorCode code = AgentPolicyCli::backend_error_code(backend.status());
        if (code == ToolErrorCode::Usage) {
            print_help();
        }
        backend_label = CliIo::optional_option(args, "--backend", "cpu");
        return fail(json_mode, backend_label, code, backend.status().message(), CliIo::kExitUsage);
    }
    backend_label = std::string(BackendUtil::to_string(backend.value()));

    StatusOr<std::string> candidates_path = CliIo::require_option(args, "--candidates");
    if (!candidates_path.ok()) {
        print_help();
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, candidates_path.status().message(),
            CliIo::kExitUsage);
    }

    StatusOr<std::string> score_id = CliIo::require_option(args, "--score-id");
    if (!score_id.ok()) {
        print_help();
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, score_id.status().message(),
            CliIo::kExitUsage);
    }
    if (!ScoreId::from_string(score_id.value()).ok()) {
        return fail(
            json_mode,
            backend_label,
            ToolErrorCode::Usage,
            "Unknown score_id: " + score_id.value(),
            CliIo::kExitUsage);
    }

    StatusOr<std::size_t> k = parse_size_option(args, "--k", /*required=*/true);
    if (!k.ok()) {
        print_help();
        return fail(
            json_mode, backend_label, ToolErrorCode::Usage, k.status().message(), CliIo::kExitUsage);
    }
    if (k.value() == 0) {
        return fail(json_mode, backend_label, ToolErrorCode::Usage, "--k must be >= 1", CliIo::kExitUsage);
    }

    std::size_t latin_max = kDefaultLatinMax;
    const std::string latin_max_text = CliIo::optional_option(args, "--latin-max");
    if (!latin_max_text.empty()) {
        try {
            const unsigned long long value = std::stoull(latin_max_text);
            latin_max = static_cast<std::size_t>(value);
            if (latin_max == 0) {
                latin_max = static_cast<std::size_t>(-1);  // no truncation
            }
        } catch (const std::exception&) {
            return fail(
                json_mode,
                backend_label,
                ToolErrorCode::Usage,
                "--latin-max must be a non-negative integer",
                CliIo::kExitUsage);
        }
    }

    nlohmann::json params = nlohmann::json::object();
    const std::string params_json = CliIo::optional_option(args, "--params-json");
    if (!params_json.empty()) {
        try {
            params = nlohmann::json::parse(params_json);
        } catch (const nlohmann::json::exception& ex) {
            return fail(
                json_mode,
                backend_label,
                ToolErrorCode::Schema,
                std::string("Invalid --params-json: ") + ex.what(),
                CliIo::kExitUsage);
        }
        if (!params.is_object()) {
            return fail(
                json_mode,
                backend_label,
                ToolErrorCode::Schema,
                "--params-json must be an object",
                CliIo::kExitUsage);
        }
    }

    StatusOr<std::string> source = CliIo::read_all_utf8(candidates_path.value());
    if (!source.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Io, source.status().message(), CliIo::kExitUsage);
    }

    StatusOr<std::vector<TransformCandidate>> candidates = load_candidates(source.value());
    if (!candidates.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Schema, candidates.status().message(),
            CliIo::kExitUsage);
    }

    StatusOr<BatchResult> ranked = RankCandidates::run(
        candidates.value(),
        score_id.value(),
        k.value(),
        &ctx.value(),
        {},
        params,
        "v0",
        BatchExecution::Serial,
        backend.value());
    if (!ranked.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Internal, ranked.status().message(),
            CliIo::kExitFail);
    }

    const Context* latin_ctx = no_latin ? nullptr : &ctx.value();
    StatusOr<nlohmann::json> payload = RankCandidates::result_to_json(
        ranked.value(),
        candidates.value(),
        latin_ctx,
        latin_max,
        backend.value());
    if (!payload.ok()) {
        return fail(
            json_mode, backend_label, ToolErrorCode::Internal, payload.status().message(),
            CliIo::kExitFail);
    }

    if (!json_mode) {
        for (const auto& hit : payload.value().at("hits")) {
            std::ostringstream score;
            score << std::setprecision(17) << hit.at("score").get<double>();
            std::cout << hit.at("rank").get<std::size_t>() << '\t'
                      << hit.at("candidate_id").get<std::string>() << '\t' << score.str();
            if (!no_latin && hit.contains("latin") && hit.at("latin").is_string()) {
                std::cout << '\t' << hit.at("latin").get<std::string>();
            }
            std::cout << '\n';
        }
        return CliIo::kExitOk;
    }

    return ToolCliJson::ok(kTool, backend_label, std::move(payload.value()));
}
