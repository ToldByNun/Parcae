#include "cli_io.hpp"

#include "parcae/gematria/latin_codec.hpp"
#include "parcae/score/score_id.hpp"
#include "parcae/tool/api.hpp"

#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
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
        << "Usage: parcae-score --score-id <id> --input <file|->\n"
        << "                    [--latin|--runes|--indices] [--params-json <json>]\n"
        << "                    [--json] [--data-dir <path>]\n"
        << "       parcae-score --list [--json] [--data-dir <path>]\n"
        << "\n"
        << "Input modes (exactly one; default --latin):\n"
        << "  --latin     Preferred/alias Latin letters (non-letters stripped)\n"
        << "  --runes     Tokenize Liber Primus UTF-8; score consumable runes\n"
        << "  --indices   Comma/whitespace-separated Index29 values 0..28\n"
        << "\n"
        << "  --score-id       Registry id (e.g. ic_mod29, chi2_english_gp_v0)\n"
        << "  --params-json    Optional params object (e.g. {\"reference\":[…]})\n"
        << "  --backend        cpu|cuda (default cpu; exit 2 if cuda not built)\n"
        << "  --list           Print known score ids and exit\n"
        << "  --json           Machine-readable JSON on stdout\n"
        << "  --data-dir       Parcae data/ root\n"
        << "  -h, --help       Show this help\n";
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
        while (i < text.size() &&
               (text[i] == ' ' || text[i] == ',' || text[i] == '\t' || text[i] == '\n' ||
                text[i] == '\r')) {
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
        const unsigned long value =
            std::stoul(std::string(text.substr(i, j - i)));
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

[[nodiscard]] StatusOr<std::vector<Index29>> load_candidate(
    const parcae::tool::Context& ctx,
    const std::string& source,
    std::string_view mode) {
    if (mode == "indices") {
        return parse_indices_text(source);
    }
    if (mode == "runes") {
        StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx, source);
        if (!stream.ok()) {
            return stream.status();
        }
        if (stream.value().consumable_count() == 0) {
            return Status::error("No consumable runes in input");
        }
        return stream.value().consumable_indices();
    }

    // latin (default)
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

    StatusOr<parcae::tool::Backend> backend = parcae::tool::BackendUtil::from_string(
        optional_option(args, "--backend", "cpu"));
    if (!backend.ok()) {
        std::cerr << backend.status().message() << '\n';
        print_help();
        return kExitUsage;
    }
    Status backend_ok = parcae::tool::BackendUtil::ensure_usable(backend.value());
    if (!backend_ok.ok()) {
        std::cerr << backend_ok.message() << '\n';
        return kExitUsage;
    }

    StatusOr<parcae::tool::Context> ctx = make_context(data_dir, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return kExitUsage;
    }

    if (has_flag(args, "--list")) {
        const std::vector<std::string> ids = parcae::tool::list_score_ids();
        if (json_mode) {
            std::cout << nlohmann::json{{"score_ids", ids}}.dump(2) << '\n';
        } else {
            for (const std::string& id : ids) {
                std::cout << id << '\n';
            }
        }
        return kExitOk;
    }

    StatusOr<std::string> score_id = require_option(args, "--score-id");
    if (!score_id.ok()) {
        std::cerr << score_id.status().message() << '\n';
        print_help();
        return kExitUsage;
    }
    if (!ScoreId::from_string(score_id.value()).ok()) {
        std::cerr << "Unknown score_id: " << score_id.value() << '\n';
        return kExitUsage;
    }

    StatusOr<std::string> input_path = require_option(args, "--input");
    if (!input_path.ok()) {
        std::cerr << input_path.status().message() << '\n';
        print_help();
        return kExitUsage;
    }

    const bool mode_latin = has_flag(args, "--latin");
    const bool mode_runes = has_flag(args, "--runes");
    const bool mode_indices = has_flag(args, "--indices");
    const int modes =
        static_cast<int>(mode_latin) + static_cast<int>(mode_runes) +
        static_cast<int>(mode_indices);
    if (modes > 1) {
        std::cerr << "Choose at most one of --latin, --runes, --indices\n";
        return kExitUsage;
    }
    const std::string_view mode =
        mode_runes ? "runes" : (mode_indices ? "indices" : "latin");

    StatusOr<std::string> source = read_all_utf8(input_path.value());
    if (!source.ok()) {
        std::cerr << source.status().message() << '\n';
        return kExitUsage;
    }

    StatusOr<std::vector<Index29>> candidate =
        load_candidate(ctx.value(), source.value(), mode);
    if (!candidate.ok()) {
        std::cerr << candidate.status().message() << '\n';
        return kExitFail;
    }

    nlohmann::json params = nlohmann::json::object();
    const std::string params_json = optional_option(args, "--params-json");
    if (!params_json.empty()) {
        try {
            params = nlohmann::json::parse(params_json);
        } catch (const nlohmann::json::exception& ex) {
            std::cerr << "Invalid --params-json: " << ex.what() << '\n';
            return kExitUsage;
        }
        if (!params.is_object()) {
            std::cerr << "--params-json must be an object\n";
            return kExitUsage;
        }
    }

    StatusOr<double> value = parcae::tool::score(
        ctx.value(),
        candidate.value(),
        score_id.value(),
        "v0",
        params,
        {},
        backend.value());
    if (!value.ok()) {
        std::cerr << value.status().message() << '\n';
        return kExitFail;
    }

    if (json_mode) {
        std::cout << nlohmann::json{
                         {"score_id", score_id.value()},
                         {"score_version", "v0"},
                         {"backend", parcae::tool::BackendUtil::to_string(backend.value())},
                         {"value", value.value()},
                     }
                         .dump(2)
                  << '\n';
    } else {
        std::ostringstream out;
        out << std::setprecision(17) << value.value();
        std::cout << out.str() << '\n';
    }
    return kExitOk;
}
