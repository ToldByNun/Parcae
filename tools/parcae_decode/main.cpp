#include "cli_io.hpp"

#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/tool/api.hpp"
#include "parcae/tool/transform_envelope.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

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
        << "  --json       JSON with indices + latin on stdout\n"
        << "  --data-dir   Parcae data/ root\n"
        << "  -h, --help   Show this help\n";
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
    const std::vector<Index29>& indices,
    bool json_mode) {
    StatusOr<std::string> latin = parcae::tool::to_latin(ctx, indices);
    if (!latin.ok()) {
        std::cerr << latin.status().message() << '\n';
        return parcae::cli::kExitFail;
    }

    if (json_mode) {
        nlohmann::json index_json = nlohmann::json::array();
        for (Index29 idx : indices) {
            index_json.push_back(idx.value());
        }
        std::cout << nlohmann::json{
                         {"indices", std::move(index_json)},
                         {"latin", latin.value()},
                     }
                         .dump(2)
                  << '\n';
    } else {
        std::cout << latin.value() << '\n';
    }
    return parcae::cli::kExitOk;
}

} // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    const std::vector<std::string> args = argv_tail(argc, argv);
    if (has_flag(args, "-h") || has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? kExitUsage : kExitOk;
    }

    const bool json_mode = has_flag(args, "--json");
    const std::string data_dir = optional_option(args, "--data-dir");

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    const bool has_manifest = has_flag(args, "--manifest");
    const bool has_transform_json = has_flag(args, "--transform-json");
    const bool has_transform_id = has_flag(args, "--transform-id");
    const int modes =
        static_cast<int>(has_manifest) + static_cast<int>(has_transform_json) +
        static_cast<int>(has_transform_id);
    if (modes != 1) {
        std::cerr << "Choose exactly one of --manifest, --transform-json, or --transform-id\n";
        print_help();
        return kExitUsage;
    }

    if (has_manifest) {
        StatusOr<std::string> manifest_path = require_option(args, "--manifest");
        if (!manifest_path.ok()) {
            std::cerr << manifest_path.status().message() << '\n';
            return kExitUsage;
        }

        std::filesystem::path fixture_dir(manifest_path.value());
        if (fixture_dir.filename() == "manifest.json") {
            fixture_dir = fixture_dir.parent_path();
        }

        StatusOr<Fixture> fixture = FixtureLoader::load_directory(fixture_dir.string());
        if (!fixture.ok()) {
            std::cerr << fixture.status().message() << '\n';
            return kExitFail;
        }

        StatusOr<parcae::tool::TransformEnvelope> envelope = envelope_from_fixture(fixture.value());
        if (!envelope.ok()) {
            std::cerr << envelope.status().message() << '\n';
            return kExitFail;
        }

        StatusOr<TokenStream> stream =
            parcae::tool::tokenize(ctx.value(), fixture.value().ciphertext());
        if (!stream.ok()) {
            std::cerr << stream.status().message() << '\n';
            return kExitFail;
        }

        StatusOr<std::vector<Index29>> plain =
            parcae::tool::apply_to_indices(stream.value(), envelope.value());
        if (!plain.ok()) {
            std::cerr << plain.status().message() << '\n';
            return kExitFail;
        }
        return emit_result(ctx.value(), plain.value(), json_mode);
    }

    StatusOr<std::string> input_path = require_option(args, "--input");
    if (!input_path.ok()) {
        std::cerr << input_path.status().message() << '\n';
        print_help();
        return kExitUsage;
    }
    StatusOr<std::string> source = read_all_utf8(input_path.value());
    if (!source.ok()) {
        std::cerr << source.status().message() << '\n';
        return kExitUsage;
    }

    StatusOr<parcae::tool::TransformEnvelope> envelope{Status::error("unset")};
    if (has_transform_json) {
        StatusOr<std::string> path = require_option(args, "--transform-json");
        if (!path.ok()) {
            std::cerr << path.status().message() << '\n';
            return kExitUsage;
        }
        StatusOr<std::string> json_text = read_file_utf8(path.value());
        if (!json_text.ok()) {
            std::cerr << json_text.status().message() << '\n';
            return kExitUsage;
        }
        envelope = parcae::tool::TransformEnvelope::from_string(json_text.value());
    } else {
        envelope = envelope_from_flags(args);
    }
    if (!envelope.ok()) {
        std::cerr << envelope.status().message() << '\n';
        return kExitFail;
    }

    StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx.value(), source.value());
    if (!stream.ok()) {
        std::cerr << stream.status().message() << '\n';
        return kExitFail;
    }

    StatusOr<std::vector<Index29>> plain =
        parcae::tool::apply_to_indices(stream.value(), envelope.value());
    if (!plain.ok()) {
        std::cerr << plain.status().message() << '\n';
        return kExitFail;
    }
    return emit_result(ctx.value(), plain.value(), json_mode);
}
