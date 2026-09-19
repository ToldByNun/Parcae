#include "cli_io.hpp"
#include "tool_cli_json.hpp"
#include "agent_policy_cli.hpp"

#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

constexpr std::string_view kTool = "decode";

void print_help() {
    std::cerr
        << "Usage:\n"
        << "  parcae-decode --manifest <fixture_dir> [--json] [--data-dir <path>]\n"
        << "  parcae-decode --transform-json <path> --input <file|-> [--json] [--data-dir <path>]\n"
        << "  parcae-decode --input <file|-> --transform-id <id> [method flags...] [--json]\n"
        << "\n"
        << "Method flags (flag mode):\n"
        << "  --transform-id <id>     Required in flag mode (e.g. atbash, vigenere_key)\n"
        << "  --direction <dir>       encrypt|decrypt (default decrypt)\n"
        << "  --params-json <json>    Full params object (overrides key flags)\n"
        << "  --key-indices <list>    Comma-separated 0..28 (into params.key_indices)\n"
        << "  --key-latin <text>      Optional metadata / params.key_latin\n"
        << "  --skip-indices <list>   Comma-separated consumable skip indices\n"
        << "  --shift <n>             Shortcut for caesar params.shift\n"
        << "\n"
        << "Global:\n"
        << "  --backend    cpu|cuda (default cpu; exit 2 if cuda not built)\n"
        << "  --allow-cuda Required with --backend cuda (AgentPolicy opt-in)\n"
        << "  --rebuild-text  Also rebuild UTF-8 with separators preserved\n"
        << "  --json       JSON envelope on stdout (parcae.tool_response.v0)\n"
        << "  --data-dir   Parcae data/ root\n"
        << "  -h, --help   Show this help\n";
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

[[nodiscard]] StatusOr<parcae::tool::TransformEnvelope> envelope_from_fixture(
    const Fixture& fixture) {
    nlohmann::json root{
        {"transform_id", fixture.transform_id()},
        {"direction", fixture.direction()},
        {"params", fixture.params()},
    };
    if (!fixture.skip_indices().empty()) {
        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(fixture.skip_indices());
        if (!interrupt.ok()) {
            return interrupt.status();
        }
        root["interrupt"] = interrupt.value().to_json();
    }
    return parcae::tool::TransformEnvelope::from_json(root);
}

[[nodiscard]] StatusOr<parcae::tool::TransformEnvelope> envelope_from_flags(
    const std::vector<std::string>& args) {
    StatusOr<std::string> transform_id = parcae::cli::require_option(args, "--transform-id");
    if (!transform_id.ok()) {
        return transform_id.status();
    }

    const std::string direction = parcae::cli::optional_option(args, "--direction", "decrypt");
    nlohmann::json params = nlohmann::json::object();

    const std::string params_json = parcae::cli::optional_option(args, "--params-json");
    if (!params_json.empty()) {
        try {
            params = nlohmann::json::parse(params_json);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid --params-json: ") + ex.what());
        }
        if (!params.is_object()) {
            return Status::error("--params-json must be an object");
        }
    } else {
        const std::string key_indices = parcae::cli::optional_option(args, "--key-indices");
        if (!key_indices.empty()) {
            StatusOr<std::vector<int>> indices = parcae::cli::parse_int_list(key_indices);
            if (!indices.ok()) {
                return indices.status();
            }
            params["key_indices"] = indices.value();
        }
        const std::string key_latin = parcae::cli::optional_option(args, "--key-latin");
        if (!key_latin.empty()) {
            params["key_latin"] = key_latin;
        }
        const std::string shift = parcae::cli::optional_option(args, "--shift");
        if (!shift.empty()) {
            try {
                params["shift"] = std::stoi(shift);
            } catch (...) {
                return Status::error("Invalid --shift");
            }
        }
    }

    nlohmann::json root{
        {"transform_id", transform_id.value()},
        {"direction", direction},
        {"params", params},
    };

    const std::string skips = parcae::cli::optional_option(args, "--skip-indices");
    if (!skips.empty()) {
        StatusOr<std::vector<std::size_t>> skip_indices = parcae::cli::parse_size_list(skips);
        if (!skip_indices.ok()) {
            return skip_indices.status();
        }
        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(std::move(skip_indices.value()));
        if (!interrupt.ok()) {
            return interrupt.status();
        }
        root["interrupt"] = interrupt.value().to_json();
    }

    return parcae::tool::TransformEnvelope::from_json(root);
}

[[nodiscard]] int emit_result(
    const parcae::tool::Context& ctx,
    const TokenStream& stream,
    const parcae::tool::TransformEnvelope& envelope,
    parcae::tool::Backend backend,
    bool json_mode,
    bool rebuild_text,
    const std::string& backend_label) {
    StatusOr<std::vector<Index29>> indices =
        parcae::tool::apply_to_indices(stream, envelope, backend);
    if (!indices.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Internal, indices.status().message(),
                    parcae::cli::kExitFail);
    }

    StatusOr<std::string> latin = parcae::tool::to_latin(ctx, indices.value());
    if (!latin.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Internal, latin.status().message(),
                    parcae::cli::kExitFail);
    }

    std::optional<std::string> rebuilt;
    if (rebuild_text) {
        StatusOr<std::string> text =
            parcae::tool::apply_and_rebuild_text(ctx, stream, envelope, backend);
        if (!text.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Internal, text.status().message(),
                        parcae::cli::kExitFail);
        }
        rebuilt = std::move(text.value());
    }

    if (!json_mode) {
        if (rebuilt.has_value()) {
            std::cout << rebuilt.value() << '\n';
        } else {
            std::cout << latin.value() << '\n';
        }
        return parcae::cli::kExitOk;
    }

    nlohmann::json index_json = nlohmann::json::array();
    for (Index29 idx : indices.value()) {
        index_json.push_back(idx.value());
    }
    nlohmann::json result{
        {"indices", std::move(index_json)},
        {"latin", latin.value()},
    };
    if (rebuilt.has_value()) {
        result["text"] = std::move(*rebuilt);
    } else {
        result["text"] = nullptr;
    }
    return ToolCliJson::ok(kTool, backend_label, std::move(result));
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? kExitUsage : kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const bool rebuild_text = has_flag(args, "--rebuild-text");
    const std::string data_dir = optional_option(args, "--data-dir");
    std::optional<std::string> backend_label;

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        return fail(json_mode, std::nullopt, ToolErrorCode::Io, ctx.status().message(),
                    kExitUsage);
    }

    const AgentPolicy policy = AgentPolicyCli::make(ctx.value(), args);
    StatusOr<parcae::tool::Backend> backend = AgentPolicyCli::resolve_backend(policy, args);
    if (!backend.ok()) {
        const ToolErrorCode code = AgentPolicyCli::backend_error_code(backend.status());
        if (code == ToolErrorCode::Usage) {
            print_help();
        }
        backend_label = optional_option(args, "--backend", "cpu");
        return fail(json_mode, backend_label, code, backend.status().message(), kExitUsage);
    }
    backend_label = std::string(parcae::tool::BackendUtil::to_string(backend.value()));

    const bool has_manifest = has_flag(args, "--manifest");
    const bool has_transform_json = has_flag(args, "--transform-json");
    const bool has_transform_id = has_flag(args, "--transform-id");
    const int modes =
        static_cast<int>(has_manifest) + static_cast<int>(has_transform_json) +
        static_cast<int>(has_transform_id);
    if (modes != 1) {
        print_help();
        return fail(
            json_mode,
            backend_label,
            ToolErrorCode::Usage,
            "Choose exactly one of --manifest, --transform-json, or --transform-id",
            kExitUsage);
    }

    if (has_manifest) {
        StatusOr<std::string> manifest_path = require_option(args, "--manifest");
        if (!manifest_path.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Usage,
                        manifest_path.status().message(), kExitUsage);
        }

        std::filesystem::path fixture_dir(manifest_path.value());
        if (fixture_dir.filename() == "manifest.json") {
            fixture_dir = fixture_dir.parent_path();
        }

        StatusOr<Fixture> fixture = FixtureLoader::load_directory(fixture_dir.string());
        if (!fixture.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Io, fixture.status().message(),
                        kExitFail);
        }

        StatusOr<parcae::tool::TransformEnvelope> envelope = envelope_from_fixture(fixture.value());
        if (!envelope.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Schema,
                        envelope.status().message(), kExitFail);
        }

        StatusOr<TokenStream> stream =
            parcae::tool::tokenize(ctx.value(), fixture.value().ciphertext());
        if (!stream.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Internal,
                        stream.status().message(), kExitFail);
        }

        return emit_result(
            ctx.value(),
            stream.value(),
            envelope.value(),
            backend.value(),
            json_mode,
            rebuild_text,
            *backend_label);
    }

    StatusOr<std::string> input_path = require_option(args, "--input");
    if (!input_path.ok()) {
        print_help();
        return fail(json_mode, backend_label, ToolErrorCode::Usage, input_path.status().message(),
                    kExitUsage);
    }
    StatusOr<std::string> source = read_all_utf8(input_path.value());
    if (!source.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Io, source.status().message(),
                    kExitUsage);
    }

    StatusOr<parcae::tool::TransformEnvelope> envelope{Status::error("unset")};
    if (has_transform_json) {
        StatusOr<std::string> path = require_option(args, "--transform-json");
        if (!path.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Usage, path.status().message(),
                        kExitUsage);
        }
        StatusOr<std::string> json_text = read_file_utf8(path.value());
        if (!json_text.ok()) {
            return fail(json_mode, backend_label, ToolErrorCode::Io, json_text.status().message(),
                        kExitUsage);
        }
        envelope = parcae::tool::TransformEnvelope::from_string(json_text.value());
    } else {
        envelope = envelope_from_flags(args);
    }
    if (!envelope.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Schema, envelope.status().message(),
                    kExitFail);
    }

    StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx.value(), source.value());
    if (!stream.ok()) {
        return fail(json_mode, backend_label, ToolErrorCode::Internal, stream.status().message(),
                    kExitFail);
    }

    return emit_result(
        ctx.value(),
        stream.value(),
        envelope.value(),
        backend.value(),
        json_mode,
        rebuild_text,
        *backend_label);
}
