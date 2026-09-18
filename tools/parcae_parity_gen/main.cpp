#include "cli_io.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/corpus/fixture.hpp"
#include "parcae/corpus/fixture_loader.hpp"
#include "parcae/corpus/token_stream.hpp"
#include "parcae/corpus/tokenizer.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/tool/context.hpp"
#include "parcae/transform/compose_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
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
        << "Usage: parcae-parity-gen --out <dir> [--data-dir <data>]\n"
        << "\n"
        << "  Write CPU parity goldens (ParityRecord + case + idx29) under --out.\n"
        << "  Default --out is <data>/parity.\n";
}

[[nodiscard]] std::vector<Index29> random_indices(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 28);
    std::vector<Index29> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(Index29{static_cast<std::uint8_t>(dist(rng))});
    }
    return out;
}

[[nodiscard]] Status write_bytes(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Status::error("Failed to write: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        return Status::error("Failed while writing: " + path.string());
    }
    return Status::success();
}

[[nodiscard]] std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
    std::vector<std::uint8_t> bytes(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        bytes[i] = indices[i].value();
    }
    return bytes;
}

[[nodiscard]] Status write_golden(
    const std::filesystem::path& out_dir,
    const std::string& name,
    const TransformId& id,
    const std::vector<Index29>& input,
    const nlohmann::json& params,
    TransformDirection direction,
    const InterruptPolicy& interrupt) {
    StatusOr<std::pair<std::vector<Index29>, ParityRecord>> captured =
        ParityRecord::apply_and_capture(id, input, params, direction, interrupt, "cpu");
    if (!captured.ok()) {
        return captured.status();
    }

    const auto& output = captured.value().first;
    const auto& record = captured.value().second;

    nlohmann::json case_json{
        {"name", name},
        {"transform_id", id.str()},
        {"direction", TransformDirectionUtil::to_string(direction)},
        {"params", params},
        {"interrupt", interrupt.to_json()},
        {"input_file", name + ".in.idx29"},
        {"output_file", name + ".out.idx29"},
        {"record_file", name + ".json"},
    };

    {
        std::ofstream json_out(out_dir / (name + ".json"), std::ios::binary | std::ios::trunc);
        if (!json_out) {
            return Status::error("Failed to write record JSON for " + name);
        }
        json_out << record.to_json().dump(2) << '\n';
    }
    {
        std::ofstream case_out(out_dir / (name + ".case.json"), std::ios::binary | std::ios::trunc);
        if (!case_out) {
            return Status::error("Failed to write case JSON for " + name);
        }
        case_out << case_json.dump(2) << '\n';
    }

    Status in_status = write_bytes(out_dir / (name + ".in.idx29"), to_bytes(input));
    if (!in_status.ok()) {
        return in_status;
    }
    return write_bytes(out_dir / (name + ".out.idx29"), to_bytes(output));
}

[[nodiscard]] StatusOr<std::vector<Index29>> a_warning_consumable(
    const parcae::tool::Context& ctx) {
    StatusOr<std::filesystem::path> dir = ctx.resolve_fixture_dir("a-warning");
    if (!dir.ok()) {
        return dir.status();
    }
    StatusOr<Fixture> fixture = FixtureLoader::load_directory(dir.value().string());
    if (!fixture.ok()) {
        return fixture.status();
    }
    StatusOr<GematriaProfile> profile = ctx.load_gematria();
    if (!profile.ok()) {
        return profile.status();
    }
    StatusOr<SeparatorGrammar> grammar = ctx.load_grammar();
    if (!grammar.ok()) {
        return grammar.status();
    }
    StatusOr<TokenStream> stream =
        Tokenizer{profile.value(), grammar.value()}.tokenize(fixture.value().ciphertext(), true);
    if (!stream.ok()) {
        return stream.status();
    }
    return stream.value().consumable_indices();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace parcae::cli;

    std::string data_dir = PARCAE_DEFAULT_DATA_DIR;
    std::string out_dir;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help();
            return kExitOk;
        }
        if (arg == "--data-dir" && i + 1 < argc) {
            data_dir = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
            continue;
        }
        std::cerr << "Unknown argument: " << arg << '\n';
        print_help();
        return kExitUsage;
    }

    if (data_dir.empty()) {
        std::cerr << "--data-dir required (or build with PARCAE_DEFAULT_DATA_DIR)\n";
        return kExitUsage;
    }
    if (out_dir.empty()) {
        out_dir = (std::filesystem::path(data_dir) / "parity").string();
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create " << out_dir << ": " << ec.message() << '\n';
        return kExitFail;
    }

    const parcae::tool::Context ctx{std::filesystem::path(data_dir)};
    const std::filesystem::path out_path(out_dir);

    struct Job {
        std::string name;
        Status (*run)(const std::filesystem::path&, const parcae::tool::Context&);
    };

    const auto caesar = [](const std::filesystem::path& out,
                           const parcae::tool::Context&) -> Status {
        return write_golden(
            out,
            "caesar-s3-len64",
            TransformId::caesar(),
            random_indices(64, 0xA71A001u),
            nlohmann::json{{"shift", 3}},
            TransformDirection::Decrypt,
            InterruptPolicy::none());
    };
    const auto atbash = [](const std::filesystem::path& out,
                           const parcae::tool::Context&) -> Status {
        return write_golden(
            out,
            "atbash-len64",
            TransformId::atbash(),
            random_indices(64, 0xA71A002u),
            nlohmann::json::object(),
            TransformDirection::Decrypt,
            InterruptPolicy::none());
    };
    const auto vigenere = [](const std::filesystem::path& out,
                             const parcae::tool::Context&) -> Status {
        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(std::vector<std::size_t>{7, 31});
        if (!interrupt.ok()) {
            return interrupt.status();
        }
        return write_golden(
            out,
            "vigenere-key8-skips2",
            TransformId::vigenere_key(),
            random_indices(64, 0xA71A003u),
            nlohmann::json{{"key_indices", {23, 10, 1, 10, 9, 10, 16, 26}}, {"key_latin", "DIVINITY"}},
            TransformDirection::Decrypt,
            interrupt.value());
    };
    const auto totient = [](const std::filesystem::path& out,
                            const parcae::tool::Context&) -> Status {
        StatusOr<InterruptPolicy> interrupt =
            InterruptPolicy::from_skip_indices(std::vector<std::size_t>{5});
        if (!interrupt.ok()) {
            return interrupt.status();
        }
        return write_golden(
            out,
            "totient-start0-skip1",
            TransformId::totient_prime_stream(),
            random_indices(32, 0xA71A004u),
            nlohmann::json{{"prime_start_index", 0}},
            TransformDirection::Decrypt,
            interrupt.value());
    };
    const auto affine = [](const std::filesystem::path& out,
                           const parcae::tool::Context&) -> Status {
        return write_golden(
            out,
            "affine-a2-b5",
            TransformId::affine(),
            random_indices(48, 0xA71A005u),
            nlohmann::json{{"a", 2}, {"b", 5}},
            TransformDirection::Decrypt,
            InterruptPolicy::none());
    };
    const auto compose = [](const std::filesystem::path& out,
                            const parcae::tool::Context&) -> Status {
        return write_golden(
            out,
            "compose-atbash-caesar3",
            TransformId::compose(),
            random_indices(40, 0xA71A006u),
            ComposeTransform::atbash_then_caesar_params(3),
            TransformDirection::Decrypt,
            InterruptPolicy::none());
    };
    const auto a_warning = [](const std::filesystem::path& out,
                              const parcae::tool::Context& context) -> Status {
        StatusOr<std::vector<Index29>> input = a_warning_consumable(context);
        if (!input.ok()) {
            return input.status();
        }
        return write_golden(
            out,
            "a-warning-atbash",
            TransformId::atbash(),
            input.value(),
            nlohmann::json::object(),
            TransformDirection::Decrypt,
            InterruptPolicy::none());
    };

    const std::vector<Job> jobs{
        {"caesar-s3-len64", caesar},
        {"atbash-len64", atbash},
        {"vigenere-key8-skips2", vigenere},
        {"totient-start0-skip1", totient},
        {"affine-a2-b5", affine},
        {"compose-atbash-caesar3", compose},
        {"a-warning-atbash", a_warning},
    };

    nlohmann::json manifest = nlohmann::json::array();
    for (const Job& job : jobs) {
        Status status = job.run(out_path, ctx);
        if (!status.ok()) {
            std::cerr << "FAIL " << job.name << ": " << status.message() << '\n';
            return kExitFail;
        }
        std::cout << "Wrote " << job.name << '\n';
        manifest.push_back(job.name);
    }

    {
        std::ofstream manifest_out(out_path / "manifest.json", std::ios::binary | std::ios::trunc);
        if (!manifest_out) {
            std::cerr << "Failed to write manifest.json\n";
            return kExitFail;
        }
        nlohmann::json root{
            {"schema", "parcae.parity_manifest.v0"},
            {"goldens", std::move(manifest)},
        };
        manifest_out << root.dump(2) << '\n';
    }

    return kExitOk;
}
