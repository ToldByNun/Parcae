#include "parcae/core/index29.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include "backend.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path parity_dir() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR) / "parity";
}

[[nodiscard]] std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>());
}

[[nodiscard]] nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in);
    nlohmann::json root;
    in >> root;
    REQUIRE(in);
    return root;
}

[[nodiscard]] std::vector<Index29> indices_from_bytes(const std::vector<std::uint8_t>& bytes) {
    std::vector<Index29> out;
    out.reserve(bytes.size());
    for (std::uint8_t value : bytes) {
        REQUIRE(value < Index29::modulus);
        out.push_back(Index29{value});
    }
    return out;
}

[[nodiscard]] bool digests_match_except_backend(const ParityRecord& golden_cpu,
                                                const ParityRecord& actual) {
    return golden_cpu.transform_id() == actual.transform_id() &&
           golden_cpu.params_hash_sha256() == actual.params_hash_sha256() &&
           golden_cpu.input_sha256() == actual.input_sha256() &&
           golden_cpu.output_sha256() == actual.output_sha256() &&
           golden_cpu.interrupt_sha256() == actual.interrupt_sha256() &&
           golden_cpu.backend() == "cpu";
}

struct LoadedGolden {
    std::string name;
    TransformId id;
    TransformDirection direction{};
    nlohmann::json params;
    InterruptPolicy interrupt;
    std::vector<Index29> input;
    std::vector<std::uint8_t> expected_out_bytes;
    ParityRecord record;
};

[[nodiscard]] LoadedGolden load_golden(const std::string& name) {
    const std::filesystem::path dir = parity_dir();
    const nlohmann::json case_json = read_json(dir / (name + ".case.json"));
    REQUIRE(case_json.at("name").get<std::string>() == name);

    StatusOr<TransformId> id =
        TransformId::from_string(case_json.at("transform_id").get<std::string>());
    REQUIRE(id.ok());
    StatusOr<TransformDirection> direction =
        TransformDirectionUtil::from_string(case_json.at("direction").get<std::string>());
    REQUIRE(direction.ok());
    StatusOr<InterruptPolicy> interrupt = InterruptPolicy::from_json(case_json.at("interrupt"));
    REQUIRE(interrupt.ok());

    StatusOr<ParityRecord> record = ParityRecord::from_json(read_json(dir / (name + ".json")));
    REQUIRE(record.ok());
    REQUIRE(record.value().backend() == "cpu");

    const std::string input_file = case_json.at("input_file").get<std::string>();
    const std::string output_file = case_json.at("output_file").get<std::string>();
    REQUIRE(input_file == name + ".in.idx29");
    REQUIRE(output_file == name + ".out.idx29");

    return LoadedGolden{
        name,
        id.value(),
        direction.value(),
        case_json.at("params"),
        interrupt.value(),
        indices_from_bytes(read_bytes(dir / input_file)),
        read_bytes(dir / output_file),
        record.value(),
    };
}

[[nodiscard]] std::vector<std::string> load_manifest_names() {
    const nlohmann::json manifest = read_json(parity_dir() / "manifest.json");
    REQUIRE(manifest.at("schema").get<std::string>() == "parcae.parity_manifest.v0");
    REQUIRE(manifest.at("goldens").is_array());
    REQUIRE_FALSE(manifest.at("goldens").empty());

    std::vector<std::string> names;
    names.reserve(manifest.at("goldens").size());
    for (const nlohmann::json& item : manifest.at("goldens")) {
        names.push_back(item.get<std::string>());
    }
    return names;
}

} // namespace

TEST_CASE("CPU parity goldens lock output_sha256", "[cuda][parity][golden]") {
    for (const std::string& name : load_manifest_names()) {
        SECTION(name) {
            const LoadedGolden golden = load_golden(name);

            REQUIRE(ParityRecord::hash_indices(golden.input) == golden.record.input_sha256());
            REQUIRE(ParityRecord::hash_json(golden.params) == golden.record.params_hash_sha256());
            REQUIRE(ParityRecord::hash_json(golden.interrupt.to_json()) ==
                    golden.record.interrupt_sha256());

            StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cpu =
                ParityRecord::apply_and_capture(golden.id, golden.input, golden.params,
                                                golden.direction, golden.interrupt, "cpu");
            REQUIRE(cpu.ok());
            REQUIRE(cpu.value().second == golden.record);
            REQUIRE(ParityRecord::hash_indices(cpu.value().first) == golden.record.output_sha256());

            std::vector<std::uint8_t> out_bytes(cpu.value().first.size());
            for (std::size_t i = 0; i < cpu.value().first.size(); ++i) {
                out_bytes[i] = cpu.value().first[i].value();
            }
            REQUIRE(out_bytes == golden.expected_out_bytes);
        }
    }
}

TEST_CASE("CUDA vs golden output_sha256 suite", "[cuda][parity][golden]") {
#if !defined(PARCAE_HAS_CUDA)
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise CudaBackend vs data/parity goldens");
    return;
#else
    if (!CudaBackend::available()) {
        SUCCEED("CUDA Toolkit build present but no usable device; skipping golden CUDA apply");
        return;
    }

    for (const std::string& name : load_manifest_names()) {
        SECTION(name) {
            const LoadedGolden golden = load_golden(name);

            StatusOr<std::pair<std::vector<Index29>, ParityRecord>> cuda =
                CudaBackend::apply_and_capture(golden.id, golden.input, golden.params,
                                               golden.direction, golden.interrupt);
            REQUIRE(cuda.ok());
            REQUIRE(cuda.value().second.backend() == "cuda");
            REQUIRE(digests_match_except_backend(golden.record, cuda.value().second));
            REQUIRE(cuda.value().second.output_sha256() == golden.record.output_sha256());

            std::vector<std::uint8_t> out_bytes(cuda.value().first.size());
            for (std::size_t i = 0; i < cuda.value().first.size(); ++i) {
                out_bytes[i] = cuda.value().first[i].value();
            }
            REQUIRE(out_bytes == golden.expected_out_bytes);
        }
    }
#endif
}
