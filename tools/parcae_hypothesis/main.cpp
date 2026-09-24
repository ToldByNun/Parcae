#include "parcae/core/index29.hpp"
#include "parcae/gematria/latin_codec.hpp"
#include "parcae/hypothesis/hypothesis_record.hpp"
#include "parcae/hypothesis/hypothesis_status.hpp"
#include "parcae/hypothesis/workspace_manifest.hpp"
#include "parcae/hypothesis/workspace_paths.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include "agent_policy_cli.hpp"
#include "cli_io.hpp"
#include "tool_cli_json.hpp"

#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

[[nodiscard]] std::string utc_now_rfc3339() {
    using clock = std::chrono::system_clock;
    const std::time_t t = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

void print_help() {
    std::cerr << "Usage:\n"
              << "  parcae-hypothesis init --workspace <id> --id <hid>\n"
              << "      [--title <text>] [--method-json <json>] [--utc <RFC3339>] [--json]\n"
              << "  parcae-hypothesis propose --workspace <id> --id <hid>\n"
              << "      --method-json <json> | --method-file <path>\n"
              << "      [--title <text>] [--rationale <text>] [--source-json <json>]\n"
              << "      [--utc <RFC3339>] [--json]\n"
              << "  parcae-hypothesis show --workspace <id> --id <hid> [--json]\n"
              << "  parcae-hypothesis list --workspace <id> [--json]\n"
              << "  parcae-hypothesis score --workspace <id> --id <hid> --input <file|->\n"
              << "      [--latin|--runes|--indices] [--score-id <id>] [--utc <RFC3339>] [--json]\n"
              << "  parcae-hypothesis set-status --workspace <id> --id <hid> --status <s>\n"
              << "      [--utc <RFC3339>] [--json]\n"
              << "\n"
              << "HypothesisRecord I/O under data/workspaces/<id>/hypotheses/.\n"
              << "Common: --data-dir <path>  [--allow-cuda]  -h/--help\n"
              << "JSON tool names: hypothesis_init|propose|show|list|score|set_status\n"
              << "Writes are gated by AgentPolicy (fixtures / path escape → policy).\n";
}

[[nodiscard]] int fail(std::string_view tool, bool json_mode, ToolErrorCode code,
                       std::string message, int plain_exit) {
    if (json_mode) {
        return ToolCliJson::err(tool, std::nullopt, code, std::move(message));
    }
    std::cerr << message << '\n';
    return plain_exit;
}

[[nodiscard]] std::string resolve_utc(const std::vector<std::string>& args) {
    const std::string flag = CliIo::optional_option(args, "--utc");
    if (!flag.empty()) {
        return flag;
    }
    return utc_now_rfc3339();
}

[[nodiscard]] StatusOr<nlohmann::json> parse_json_object(const std::string& text,
                                                         std::string_view label) {
    nlohmann::json root;
    try {
        root = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& ex) {
        return Status::error(std::string("Invalid ") + std::string(label) + ": " + ex.what());
    }
    if (!root.is_object()) {
        return Status::error(std::string(label) + " must be a JSON object");
    }
    return root;
}

[[nodiscard]] StatusOr<nlohmann::json> load_method_json(const std::vector<std::string>& args) {
    const std::string inline_json = CliIo::optional_option(args, "--method-json");
    const std::string file_path = CliIo::optional_option(args, "--method-file");
    if (!inline_json.empty() && !file_path.empty()) {
        return Status::error("Choose only one of --method-json or --method-file");
    }
    if (!inline_json.empty()) {
        return parse_json_object(inline_json, "--method-json");
    }
    if (!file_path.empty()) {
        StatusOr<std::string> text = CliIo::read_all_utf8(file_path);
        if (!text.ok()) {
            return text.status();
        }
        return parse_json_object(text.value(), "--method-file");
    }
    return Status::error("Missing --method-json or --method-file");
}

[[nodiscard]] int fail_status(std::string_view tool, bool json_mode, const Status& status,
                              int plain_exit_fallback) {
    const ToolErrorCode code = AgentPolicy::error_code_for(status);
    const int plain = (code == ToolErrorCode::Policy) ? CliIo::kExitUsage : plain_exit_fallback;
    return fail(tool, json_mode, code, status.message(), plain);
}

[[nodiscard]] Status require_hypothesis_write(const AgentPolicy& policy,
                                              std::string_view workspace_id,
                                              std::string_view hypothesis_id) {
    return policy.allow_workspace_write(workspace_id, std::filesystem::path("hypotheses") /
                                                          (std::string(hypothesis_id) + ".json"));
}

[[nodiscard]] Status require_workspace_manifest_write(const AgentPolicy& policy,
                                                      std::string_view workspace_id) {
    return policy.allow_workspace_write(workspace_id, "workspace.json");
}

[[nodiscard]] std::string letters_only_upper(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (std::isalpha(ch) != 0) {
            out.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    return out;
}

[[nodiscard]] StatusOr<std::vector<Index29>> parse_indices_text(std::string_view text) {
    std::vector<Index29> out;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == ',' || text[i] == '\t' ||
                                   text[i] == '\n' || text[i] == '\r')) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        std::size_t j = i;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
            ++j;
        }
        if (j == i) {
            return Status::error("Invalid --indices input (expected digits/commas/whitespace)");
        }
        const unsigned long value = std::stoul(std::string(text.substr(i, j - i)));
        if (value >= Index29::modulus) {
            return Status::error("Index29 out of range [0,28]");
        }
        out.push_back(Index29{static_cast<std::uint8_t>(value)});
        i = j;
    }
    if (out.empty()) {
        return Status::error("--indices input is empty");
    }
    return out;
}

[[nodiscard]] StatusOr<std::vector<Index29>>
load_input_indices(const Context& ctx, const std::string& source, std::string_view mode) {
    if (mode == "indices") {
        return parse_indices_text(source);
    }
    if (mode == "runes") {
        StatusOr<TokenStream> stream = ToolApi::tokenize(ctx, source);
        if (!stream.ok()) {
            return stream.status();
        }
        if (stream.value().consumable_count() == 0) {
            return Status::error("No consumable runes in input");
        }
        return stream.value().consumable_indices();
    }
    StatusOr<GematriaProfile> profile = ctx.load_gematria();
    if (!profile.ok()) {
        return profile.status();
    }
    const LatinCodec codec(profile.value());
    const std::string letters = letters_only_upper(source);
    if (letters.empty()) {
        return Status::error("No Latin letters in input");
    }
    return codec.delatinize(letters);
}

[[nodiscard]] int cmd_init(const std::vector<std::string>& args, const Context& ctx,
                           const AgentPolicy& policy, bool json_mode) {
    constexpr std::string_view tool = "hypothesis_init";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> id = CliIo::require_option(args, "--id");
    if (!id.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, id.status().message(),
                    CliIo::kExitUsage);
    }

    Status ws_write = require_workspace_manifest_write(policy, workspace.value());
    if (!ws_write.ok()) {
        return fail_status(tool, json_mode, ws_write, CliIo::kExitUsage);
    }
    Status hyp_write = require_hypothesis_write(policy, workspace.value(), id.value());
    if (!hyp_write.ok()) {
        return fail_status(tool, json_mode, hyp_write, CliIo::kExitUsage);
    }

    const std::string utc = resolve_utc(args);
    StatusOr<WorkspaceManifest> ws =
        WorkspaceManifest::ensure(ctx.data_root(), workspace.value(), utc);
    if (!ws.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, ws.status().message(), CliIo::kExitUsage);
    }

    StatusOr<std::filesystem::path> existing =
        WorkspacePaths::hypothesis_file(ctx.data_root(), workspace.value(), id.value());
    if (!existing.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, existing.status().message(),
                    CliIo::kExitUsage);
    }
    if (std::filesystem::exists(existing.value())) {
        return fail(tool, json_mode, ToolErrorCode::Usage,
                    "Hypothesis already exists: " + id.value(), CliIo::kExitUsage);
    }

    nlohmann::json method{
        {"transform_id", "identity"},
        {"direction", "decrypt"},
        {"params", nlohmann::json::object()},
    };
    const std::string method_json = CliIo::optional_option(args, "--method-json");
    if (!method_json.empty()) {
        StatusOr<nlohmann::json> parsed = parse_json_object(method_json, "--method-json");
        if (!parsed.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, parsed.status().message(),
                        CliIo::kExitUsage);
        }
        method = std::move(parsed.value());
    }

    StatusOr<HypothesisRecord> draft =
        HypothesisRecord::make_draft(workspace.value(), id.value(), utc,
                                     CliIo::optional_option(args, "--title"), std::move(method));
    if (!draft.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Schema, draft.status().message(),
                    CliIo::kExitUsage);
    }
    draft.value().recompute_method_digest();
    Status stored = draft.value().store(ctx.data_root());
    if (!stored.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, stored.message(), CliIo::kExitFail);
    }

    if (!json_mode) {
        std::cout << draft.value().workspace_id() << '/' << draft.value().id() << '\t'
                  << HypothesisStatusUtil::to_string(draft.value().status()) << '\n';
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::nullopt, draft.value().to_json());
}

[[nodiscard]] int cmd_propose(const std::vector<std::string>& args, const Context& ctx,
                              const AgentPolicy& policy, bool json_mode) {
    constexpr std::string_view tool = "hypothesis_propose";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> id = CliIo::require_option(args, "--id");
    if (!id.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, id.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<nlohmann::json> method = load_method_json(args);
    if (!method.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, method.status().message(),
                    CliIo::kExitUsage);
    }

    Status ws_write = require_workspace_manifest_write(policy, workspace.value());
    if (!ws_write.ok()) {
        return fail_status(tool, json_mode, ws_write, CliIo::kExitUsage);
    }
    Status hyp_write = require_hypothesis_write(policy, workspace.value(), id.value());
    if (!hyp_write.ok()) {
        return fail_status(tool, json_mode, hyp_write, CliIo::kExitUsage);
    }

    const std::string utc = resolve_utc(args);
    StatusOr<WorkspaceManifest> ws =
        WorkspaceManifest::ensure(ctx.data_root(), workspace.value(), utc);
    if (!ws.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, ws.status().message(), CliIo::kExitUsage);
    }

    StatusOr<std::filesystem::path> path =
        WorkspacePaths::hypothesis_file(ctx.data_root(), workspace.value(), id.value());
    if (!path.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, path.status().message(),
                    CliIo::kExitUsage);
    }

    HypothesisRecord record;
    if (std::filesystem::exists(path.value())) {
        StatusOr<HypothesisRecord> loaded =
            HypothesisRecord::load(ctx.data_root(), workspace.value(), id.value());
        if (!loaded.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, loaded.status().message(),
                        CliIo::kExitFail);
        }
        record = std::move(loaded.value());
    } else {
        StatusOr<HypothesisRecord> draft =
            HypothesisRecord::make_draft(workspace.value(), id.value(), utc,
                                         CliIo::optional_option(args, "--title"), method.value());
        if (!draft.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, draft.status().message(),
                        CliIo::kExitUsage);
        }
        record = std::move(draft.value());
    }

    Status method_ok = record.set_method(method.value());
    if (!method_ok.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Schema, method_ok.message(), CliIo::kExitUsage);
    }
    const std::string title = CliIo::optional_option(args, "--title");
    if (!title.empty()) {
        record.set_title(title);
    }
    const std::string rationale = CliIo::optional_option(args, "--rationale");
    if (!rationale.empty()) {
        record.set_rationale(rationale);
    }
    const std::string source_json = CliIo::optional_option(args, "--source-json");
    if (!source_json.empty()) {
        StatusOr<nlohmann::json> source = parse_json_object(source_json, "--source-json");
        if (!source.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, source.status().message(),
                        CliIo::kExitUsage);
        }
        Status source_ok = record.set_source(std::move(source.value()));
        if (!source_ok.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, source_ok.message(),
                        CliIo::kExitUsage);
        }
    }
    record.set_updated_utc(utc);
    record.recompute_method_digest();

    if (record.status() == HypothesisStatus::Draft) {
        Status tr = record.set_status(HypothesisStatus::Proposed);
        if (!tr.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, tr.message(), CliIo::kExitFail);
        }
    }

    Status stored = record.store(ctx.data_root());
    if (!stored.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, stored.message(), CliIo::kExitFail);
    }

    if (!json_mode) {
        std::cout << record.workspace_id() << '/' << record.id() << '\t'
                  << HypothesisStatusUtil::to_string(record.status()) << '\n';
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::nullopt, record.to_json());
}

[[nodiscard]] int cmd_show(const std::vector<std::string>& args, const Context& ctx,
                           bool json_mode) {
    constexpr std::string_view tool = "hypothesis_show";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> id = CliIo::require_option(args, "--id");
    if (!id.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, id.status().message(),
                    CliIo::kExitUsage);
    }

    StatusOr<HypothesisRecord> record =
        HypothesisRecord::load(ctx.data_root(), workspace.value(), id.value());
    if (!record.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, record.status().message(),
                    CliIo::kExitFail);
    }

    if (!json_mode) {
        std::cout << record.value().to_json().dump(2) << '\n';
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::nullopt, record.value().to_json());
}

[[nodiscard]] int cmd_list(const std::vector<std::string>& args, const Context& ctx,
                           bool json_mode) {
    constexpr std::string_view tool = "hypothesis_list";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }

    StatusOr<std::vector<std::string>> ids =
        HypothesisRecord::list_ids(ctx.data_root(), workspace.value());
    if (!ids.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, ids.status().message(), CliIo::kExitFail);
    }

    nlohmann::json rows = nlohmann::json::array();
    for (const std::string& id : ids.value()) {
        StatusOr<HypothesisRecord> record =
            HypothesisRecord::load(ctx.data_root(), workspace.value(), id);
        if (!record.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, record.status().message(),
                        CliIo::kExitFail);
        }
        rows.push_back({
            {"id", record.value().id()},
            {"status", std::string(HypothesisStatusUtil::to_string(record.value().status()))},
            {"title", record.value().title()},
            {"updated_utc", record.value().updated_utc()},
        });
        if (!json_mode) {
            std::cout << record.value().id() << '\t'
                      << HypothesisStatusUtil::to_string(record.value().status()) << '\t'
                      << record.value().title() << '\n';
        }
    }

    if (!json_mode) {
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::nullopt,
                           nlohmann::json{
                               {"workspace_id", workspace.value()},
                               {"count", rows.size()},
                               {"hypotheses", std::move(rows)},
                           });
}

[[nodiscard]] int cmd_score(const std::vector<std::string>& args, const Context& ctx,
                            const AgentPolicy& policy, bool json_mode) {
    constexpr std::string_view tool = "hypothesis_score";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> id = CliIo::require_option(args, "--id");
    if (!id.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, id.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> input_path = CliIo::require_option(args, "--input");
    if (!input_path.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, input_path.status().message(),
                    CliIo::kExitUsage);
    }

    Status hyp_write = require_hypothesis_write(policy, workspace.value(), id.value());
    if (!hyp_write.ok()) {
        return fail_status(tool, json_mode, hyp_write, CliIo::kExitUsage);
    }

    const bool mode_latin = CliIo::has_flag(args, "--latin");
    const bool mode_runes = CliIo::has_flag(args, "--runes");
    const bool mode_indices = CliIo::has_flag(args, "--indices");
    const int modes = static_cast<int>(mode_latin) + static_cast<int>(mode_runes) +
                      static_cast<int>(mode_indices);
    if (modes > 1) {
        return fail(tool, json_mode, ToolErrorCode::Usage,
                    "Choose at most one of --latin, --runes, --indices", CliIo::kExitUsage);
    }
    const std::string_view input_mode = mode_latin ? "latin" : (mode_indices ? "indices" : "runes");

    StatusOr<HypothesisRecord> loaded =
        HypothesisRecord::load(ctx.data_root(), workspace.value(), id.value());
    if (!loaded.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, loaded.status().message(),
                    CliIo::kExitFail);
    }
    HypothesisRecord record = std::move(loaded.value());

    std::string score_id = CliIo::optional_option(args, "--score-id");
    if (score_id.empty()) {
        StatusOr<WorkspaceManifest> ws =
            WorkspaceManifest::load(ctx.data_root(), workspace.value());
        if (ws.ok() && !ws.value().default_score_id().empty()) {
            score_id = ws.value().default_score_id();
        } else {
            score_id = "chi2_english_gp_v0";
        }
    }
    if (!ScoreId::from_string(score_id).ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, "Unknown score_id: " + score_id,
                    CliIo::kExitUsage);
    }

    StatusOr<std::string> source = CliIo::read_all_utf8(input_path.value());
    if (!source.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, source.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::vector<Index29>> cipher = load_input_indices(ctx, source.value(), input_mode);
    if (!cipher.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Internal, cipher.status().message(),
                    CliIo::kExitFail);
    }

    StatusOr<TransformEnvelope> envelope = TransformEnvelope::from_json(record.method());
    if (!envelope.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Schema, envelope.status().message(),
                    CliIo::kExitFail);
    }
    StatusOr<std::vector<Index29>> plain =
        ToolApi::apply_to_indices(cipher.value(), envelope.value());
    if (!plain.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Internal, plain.status().message(),
                    CliIo::kExitFail);
    }

    StatusOr<double> value = ToolApi::score(ctx, plain.value(), score_id);
    if (!value.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Internal, value.status().message(),
                    CliIo::kExitFail);
    }

    const std::string utc = resolve_utc(args);
    StatusOr<std::string> latin = ToolApi::to_latin(ctx, plain.value());
    if (!latin.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Internal, latin.status().message(),
                    CliIo::kExitFail);
    }
    std::size_t max_chars = 64;
    if (record.preview().contains("max_chars") &&
        record.preview().at("max_chars").is_number_unsigned()) {
        max_chars = record.preview().at("max_chars").get<std::size_t>();
    }
    std::string prefix = latin.value();
    if (prefix.size() > max_chars) {
        prefix.resize(max_chars);
    }

    nlohmann::json score_entry{
        {"score_id", score_id}, {"score_version", "v0"}, {"value", value.value()},
        {"backend", "cpu"},     {"scored_utc", utc},
    };
    Status append_ok = record.append_score(std::move(score_entry));
    if (!append_ok.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Schema, append_ok.message(), CliIo::kExitFail);
    }

    record.set_preview(nlohmann::json{{"latin_prefix", prefix}, {"max_chars", max_chars}});
    record.set_output_indices_digest(plain.value());
    record.recompute_method_digest();
    record.set_updated_utc(utc);

    if (record.status() == HypothesisStatus::Draft) {
        Status to_proposed = record.set_status(HypothesisStatus::Proposed);
        if (!to_proposed.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, to_proposed.message(),
                        CliIo::kExitFail);
        }
    }
    if (record.status() == HypothesisStatus::Proposed) {
        Status to_scored = record.set_status(HypothesisStatus::Scored);
        if (!to_scored.ok()) {
            return fail(tool, json_mode, ToolErrorCode::Schema, to_scored.message(),
                        CliIo::kExitFail);
        }
    }

    Status stored = record.store(ctx.data_root());
    if (!stored.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, stored.message(), CliIo::kExitFail);
    }

    if (!json_mode) {
        std::cout << score_id << '\t' << std::setprecision(17) << value.value() << '\t' << prefix
                  << '\n';
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::string("cpu"),
                           nlohmann::json{
                               {"hypothesis", record.to_json()},
                               {"score_id", score_id},
                               {"value", value.value()},
                               {"latin_prefix", prefix},
                           });
}

[[nodiscard]] int cmd_set_status(const std::vector<std::string>& args, const Context& ctx,
                                 const AgentPolicy& policy, bool json_mode) {
    constexpr std::string_view tool = "hypothesis_set_status";

    StatusOr<std::string> workspace = CliIo::require_option(args, "--workspace");
    if (!workspace.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, workspace.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> id = CliIo::require_option(args, "--id");
    if (!id.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, id.status().message(),
                    CliIo::kExitUsage);
    }
    StatusOr<std::string> status_text = CliIo::require_option(args, "--status");
    if (!status_text.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, status_text.status().message(),
                    CliIo::kExitUsage);
    }

    Status hyp_write = require_hypothesis_write(policy, workspace.value(), id.value());
    if (!hyp_write.ok()) {
        return fail_status(tool, json_mode, hyp_write, CliIo::kExitUsage);
    }

    StatusOr<HypothesisStatus> next = HypothesisStatusUtil::from_string(status_text.value());
    if (!next.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Usage, next.status().message(),
                    CliIo::kExitUsage);
    }

    StatusOr<HypothesisRecord> loaded =
        HypothesisRecord::load(ctx.data_root(), workspace.value(), id.value());
    if (!loaded.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, loaded.status().message(),
                    CliIo::kExitFail);
    }
    HypothesisRecord record = std::move(loaded.value());
    Status tr = record.set_status(next.value());
    if (!tr.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Schema, tr.message(), CliIo::kExitFail);
    }
    record.set_updated_utc(resolve_utc(args));
    Status stored = record.store(ctx.data_root());
    if (!stored.ok()) {
        return fail(tool, json_mode, ToolErrorCode::Io, stored.message(), CliIo::kExitFail);
    }

    if (!json_mode) {
        std::cout << record.id() << '\t' << HypothesisStatusUtil::to_string(record.status())
                  << '\n';
        return CliIo::kExitOk;
    }
    return ToolCliJson::ok(tool, std::nullopt, record.to_json());
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");
    const std::string data_dir = CliIo::optional_option(args, "--data-dir");
    StatusOr<Context> ctx = CliIo::make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail("hypothesis", json_mode, ToolErrorCode::Io, ctx.status().message(),
                    CliIo::kExitUsage);
    }
    const AgentPolicy policy = AgentPolicyCli::make(ctx.value(), args);

    // First positional token is the subcommand (flags may appear before/after).
    std::string cmd;
    std::vector<std::string> rest;
    rest.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--data-dir" || arg == "--utc" || arg == "--workspace" || arg == "--id" ||
            arg == "--title" || arg == "--method-json" || arg == "--method-file" ||
            arg == "--rationale" || arg == "--source-json" || arg == "--input" ||
            arg == "--score-id" || arg == "--status") {
            rest.push_back(arg);
            if (i + 1 < args.size()) {
                rest.push_back(args[++i]);
            }
            continue;
        }
        if (arg == "--allow-cuda" || arg == "--json" || arg == "--latin" || arg == "--runes" ||
            arg == "--indices" || arg == "-h" || arg == "--help") {
            rest.push_back(arg);
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            rest.push_back(arg);
            continue;
        }
        if (cmd.empty()) {
            cmd = arg;
            continue;
        }
        rest.push_back(arg);
    }

    if (cmd.empty()) {
        print_help();
        return fail("hypothesis", json_mode, ToolErrorCode::Usage,
                    "Missing subcommand (init|propose|show|list|score|set-status)",
                    CliIo::kExitUsage);
    }

    if (cmd == "init") {
        return cmd_init(rest, ctx.value(), policy, json_mode);
    }
    if (cmd == "propose") {
        return cmd_propose(rest, ctx.value(), policy, json_mode);
    }
    if (cmd == "show") {
        return cmd_show(rest, ctx.value(), json_mode);
    }
    if (cmd == "list") {
        return cmd_list(rest, ctx.value(), json_mode);
    }
    if (cmd == "score") {
        return cmd_score(rest, ctx.value(), policy, json_mode);
    }
    if (cmd == "set-status") {
        return cmd_set_status(rest, ctx.value(), policy, json_mode);
    }

    print_help();
    return fail("hypothesis", json_mode, ToolErrorCode::Usage, "Unknown subcommand: " + cmd,
                CliIo::kExitUsage);
}
