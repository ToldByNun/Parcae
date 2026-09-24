#include "cli_io.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/tool/tool_backend.hpp"
#include "parcae/transform/apply_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(PARCAE_HAS_CUDA)
#include "backend.hpp"
#endif

#ifndef PARCAE_DEFAULT_DATA_DIR
#define PARCAE_DEFAULT_DATA_DIR ""
#endif

namespace {

void print_help() {
    std::cerr
        << "Usage:\n"
        << "  parcae-parity check [--all|--name <golden>] [--compare-cuda]\n"
        << "                      [--parity-dir <path>] [--data-dir <path>] [--json]\n"
        << "  parcae-parity dump  --name <golden> [--backend cpu|cuda]\n"
        << "                      [--parity-dir <path>] [--data-dir <path>]\n"
        << "\n"
        << "  check  Replay case recipes; compare digests to committed ParityRecords.\n"
        << "         --compare-cuda also runs CudaBackend (exit 2 if CUDA not built).\n"
        << "  dump   Apply one golden on cpu|cuda and print ParityRecord JSON.\n"
        << "\n"
        << "Default --parity-dir is <data>/parity. Goldens are produced by parcae-parity-gen.\n";
}

[[nodiscard]] StatusOr<std::vector<std::uint8_t>> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status::error("Failed to open: " + path.string());
    }
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

[[nodiscard]] StatusOr<nlohmann::json> read_json(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status::error("Failed to open: " + path.string());
    }
    nlohmann::json root;
    try {
        in >> root;
    } catch (const nlohmann::json::exception& ex) {
        return Status::error(std::string("Invalid JSON in ") + path.string() + ": " + ex.what());
    }
    return root;
}

[[nodiscard]] StatusOr<std::vector<Index29>> indices_from_bytes(
    const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        if (value >= Index29::modulus) {
            return Status::error("Index29 byte out of range [0,28]");
        }
        out.push_back(Index29{value});
    }
    return out;
}

struct LoadedCase {
    std::string name;
    TransformId id;
    TransformDirection direction{};
    nlohmann::json params;
    InterruptPolicy interrupt;
    std::vector<Index29> input;
    ParityRecord golden;
};

[[nodiscard]] StatusOr<LoadedCase> load_case(
    const std::filesystem::path& parity_dir,
    const std::string& name) {
    StatusOr<nlohmann::json> case_json = read_json(parity_dir / (name + ".case.json"));
    if (!case_json.ok()) {
        return case_json.status();
    }
    if (case_json.value().at("name").get<std::string>() != name) {
        return Status::error("case.json name mismatch for " + name);
    }

    StatusOr<TransformId> id =
        TransformId::from_string(case_json.value().at("transform_id").get<std::string>());
    if (!id.ok()) {
        return id.status();
    }
    StatusOr<TransformDirection> direction =
        TransformDirectionUtil::from_string(case_json.value().at("direction").get<std::string>());
    if (!direction.ok()) {
        return direction.status();
    }
    StatusOr<InterruptPolicy> interrupt =
        InterruptPolicy::from_json(case_json.value().at("interrupt"));
    if (!interrupt.ok()) {
        return interrupt.status();
    }

    StatusOr<nlohmann::json> record_json = read_json(parity_dir / (name + ".json"));
    if (!record_json.ok()) {
        return record_json.status();
    }
    StatusOr<ParityRecord> golden = ParityRecord::from_json(record_json.value());
    if (!golden.ok()) {
        return golden.status();
    }

    const std::string input_file = case_json.value().at("input_file").get<std::string>();
    StatusOr<std::vector<std::uint8_t>> input_bytes = read_bytes(parity_dir / input_file);
    if (!input_bytes.ok()) {
        return input_bytes.status();
    }
    StatusOr<std::vector<Index29>> input = indices_from_bytes(input_bytes.value());
    if (!input.ok()) {
        return input.status();
    }

    return LoadedCase{
        name,
        id.value(),
        direction.value(),
        case_json.value().at("params"),
        interrupt.value(),
        std::move(input.value()),
        std::move(golden.value()),
    };
}

[[nodiscard]] StatusOr<std::vector<std::string>> load_manifest_names(
    const std::filesystem::path& parity_dir) {
    StatusOr<nlohmann::json> manifest = read_json(parity_dir / "manifest.json");
    if (!manifest.ok()) {
        return manifest.status();
    }
    if (manifest.value().at("schema").get<std::string>() != "parcae.parity_manifest.v0") {
        return Status::error("unsupported parity manifest schema");
    }
    if (!manifest.value().at("goldens").is_array() || manifest.value().at("goldens").empty()) {
        return Status::error("parity manifest goldens must be a non-empty array");
    }
    std::vector<std::string> names;
    for (const nlohmann::json& item : manifest.value().at("goldens")) {
        names.push_back(item.get<std::string>());
    }
    return names;
}

[[nodiscard]] bool digests_match_except_backend(
    const ParityRecord& expected,
    const ParityRecord& actual) {
    return expected.transform_id() == actual.transform_id() &&
           expected.params_hash_sha256() == actual.params_hash_sha256() &&
           expected.input_sha256() == actual.input_sha256() &&
           expected.output_sha256() == actual.output_sha256() &&
           expected.interrupt_sha256() == actual.interrupt_sha256();
}

[[nodiscard]] StatusOr<std::pair<std::vector<Index29>, ParityRecord>> apply_and_capture(
    const LoadedCase& loaded,
    Backend backend) {
    Status usable = BackendUtil::ensure_usable(backend);
    if (!usable.ok()) {
        return usable;
    }

    if (backend == Backend::Cpu) {
        return ParityRecord::apply_and_capture(
            loaded.id,
            loaded.input,
            loaded.params,
            loaded.direction,
            loaded.interrupt,
            "cpu");
    }

#if defined(PARCAE_HAS_CUDA)
    if (!CudaBackend::available()) {
        return Status::error("CUDA backend requested but CUDA is not available");
    }
    return CudaBackend::apply_and_capture(
        loaded.id,
        loaded.input,
        loaded.params,
        loaded.direction,
        loaded.interrupt);
#else
    return Status::error(
        "CUDA backend requested but Parcae was built without CUDA (PARCAE_BUILD_CUDA)");
#endif
}

struct CheckResult {
    std::string name;
    bool cpu_ok = false;
    bool cuda_ok = false;  // meaningful only when compare_cuda
    std::string message;
};

[[nodiscard]] CheckResult check_one(
    const std::filesystem::path& parity_dir,
    const std::string& name,
    bool compare_cuda) {
    CheckResult result;
    result.name = name;

    StatusOr<LoadedCase> loaded = load_case(parity_dir, name);
    if (!loaded.ok()) {
        result.message = loaded.status().message();
        return result;
    }
    if (loaded.value().golden.backend() != "cpu") {
        result.message = "golden backend must be cpu";
        return result;
    }

    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cpu =
        apply_and_capture(loaded.value(), Backend::Cpu);
    if (!cpu.ok()) {
        result.message = cpu.status().message();
        return result;
    }
    if (!digests_match_except_backend(loaded.value().golden, cpu.value().second) ||
        cpu.value().second.backend() != "cpu") {
        result.message = "CPU replay digests mismatch golden";
        return result;
    }
    result.cpu_ok = true;

    if (!compare_cuda) {
        result.message = "ok";
        return result;
    }

    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cuda =
        apply_and_capture(loaded.value(), Backend::Cuda);
    if (!cuda.ok()) {
        result.message = cuda.status().message();
        return result;
    }
    if (!digests_match_except_backend(loaded.value().golden, cuda.value().second) ||
        cuda.value().second.backend() != "cuda") {
        result.message = "CUDA digests mismatch golden (or backend != cuda)";
        return result;
    }
    result.cuda_ok = true;
    result.message = "ok";
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = CliIo::argv_tail(argc, argv);
    if (CliIo::has_flag(args, "-h") || CliIo::has_flag(args, "--help") || args.empty()) {
        print_help();
        return args.empty() ? CliIo::kExitUsage : CliIo::kExitOk;
    }

    std::string command;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "check" || arg == "dump") {
            if (!command.empty()) {
                std::cerr << "Multiple commands specified\n";
                return CliIo::kExitUsage;
            }
            command = arg;
            continue;
        }
        if (arg == "--data-dir" || arg == "--parity-dir" || arg == "--name" || arg == "--backend") {
            if (i + 1 >= args.size()) {
                std::cerr << "Missing value for " << arg << '\n';
                return CliIo::kExitUsage;
            }
            ++i;  // skip value
            continue;
        }
        if (arg == "--all" || arg == "--json" || arg == "--compare-cuda" || arg == "-h" ||
            arg == "--help") {
            continue;
        }
        std::cerr << "Unknown argument: " << arg << '\n';
        print_help();
        return CliIo::kExitUsage;
    }
    if (command.empty()) {
        std::cerr << "Missing command (expected check|dump)\n";
        print_help();
        return CliIo::kExitUsage;
    }

    const std::string data_dir_flag = CliIo::optional_option(args, "--data-dir");
    StatusOr<Context> ctx = CliIo::make_context(data_dir_flag, PARCAE_DEFAULT_DATA_DIR);
    if (!ctx.ok()) {
        std::cerr << ctx.status().message() << '\n';
        return CliIo::kExitUsage;
    }

    std::string parity_dir = CliIo::optional_option(args, "--parity-dir");
    if (parity_dir.empty()) {
        parity_dir = (ctx.value().data_root() / "parity").string();
    }
    const std::filesystem::path parity_path(parity_dir);
    if (!std::filesystem::is_directory(parity_path)) {
        std::cerr << "Parity dir is not a directory: " << parity_dir << '\n';
        return CliIo::kExitUsage;
    }

    const bool json_mode = CliIo::has_flag(args, "--json");

    if (command == "dump") {
        StatusOr<std::string> name = CliIo::require_option(args, "--name");
        if (!name.ok()) {
            std::cerr << name.status().message() << '\n';
            print_help();
            return CliIo::kExitUsage;
        }
        StatusOr<Backend> backend = BackendUtil::from_string(
            CliIo::optional_option(args, "--backend", "cpu"));
        if (!backend.ok()) {
            std::cerr << backend.status().message() << '\n';
            return CliIo::kExitUsage;
        }
        Status backend_ok = BackendUtil::ensure_usable(backend.value());
        if (!backend_ok.ok()) {
            std::cerr << backend_ok.message() << '\n';
            return CliIo::kExitUsage;
        }

        StatusOr<LoadedCase> loaded = load_case(parity_path, name.value());
        if (!loaded.ok()) {
            std::cerr << loaded.status().message() << '\n';
            return CliIo::kExitFail;
        }
        StatusOr<std::pair<std::vector<Index29>, ParityRecord>> captured =
            apply_and_capture(loaded.value(), backend.value());
        if (!captured.ok()) {
            std::cerr << captured.status().message() << '\n';
            return CliIo::kExitFail;
        }
        std::cout << captured.value().second.to_json().dump(2) << '\n';
        return CliIo::kExitOk;
    }

    // check
    const bool compare_cuda = CliIo::has_flag(args, "--compare-cuda");
    if (compare_cuda) {
        Status backend_ok = BackendUtil::ensure_usable(Backend::Cuda);
        if (!backend_ok.ok()) {
            std::cerr << backend_ok.message() << '\n';
            return CliIo::kExitUsage;
        }
    }

    const bool has_name = CliIo::has_flag(args, "--name");
    const bool has_all = CliIo::has_flag(args, "--all");
    if (has_name && has_all) {
        std::cerr << "Choose at most one of --name and --all\n";
        return CliIo::kExitUsage;
    }

    std::vector<std::string> names;
    if (has_name) {
        StatusOr<std::string> name = CliIo::require_option(args, "--name");
        if (!name.ok()) {
            std::cerr << name.status().message() << '\n';
            return CliIo::kExitUsage;
        }
        names.push_back(name.value());
    } else {
        StatusOr<std::vector<std::string>> manifest = load_manifest_names(parity_path);
        if (!manifest.ok()) {
            std::cerr << manifest.status().message() << '\n';
            return CliIo::kExitFail;
        }
        names = std::move(manifest.value());
    }

    std::vector<CheckResult> results;
    results.reserve(names.size());
    bool all_ok = true;
    for (const std::string& name : names) {
        CheckResult one = check_one(parity_path, name, compare_cuda);
        if (!one.cpu_ok || (compare_cuda && !one.cuda_ok)) {
            all_ok = false;
        }
        results.push_back(std::move(one));
    }

    if (json_mode) {
        nlohmann::json cases = nlohmann::json::array();
        for (const CheckResult& r : results) {
            nlohmann::json row{
                {"name", r.name},
                {"cpu_ok", r.cpu_ok},
                {"message", r.message},
            };
            if (compare_cuda) {
                row["cuda_ok"] = r.cuda_ok;
            }
            cases.push_back(std::move(row));
        }
        std::cout << nlohmann::json{
                         {"ok", all_ok},
                         {"compare_cuda", compare_cuda},
                         {"cases", std::move(cases)},
                     }
                         .dump(2)
                  << '\n';
    } else {
        for (const CheckResult& r : results) {
            const char* tag = (r.cpu_ok && (!compare_cuda || r.cuda_ok)) ? "PASS" : "FAIL";
            std::cout << tag << ' ' << r.name;
            if (compare_cuda) {
                std::cout << " cpu=" << (r.cpu_ok ? "ok" : "fail")
                          << " cuda=" << (r.cuda_ok ? "ok" : "fail");
            }
            if (r.message != "ok") {
                std::cout << " (" << r.message << ')';
            }
            std::cout << '\n';
        }
    }

    return all_ok ? CliIo::kExitOk : CliIo::kExitFail;
}
