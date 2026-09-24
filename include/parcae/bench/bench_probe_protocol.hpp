#ifndef BENCH_PROBE_PROTOCOL_HPP
#define BENCH_PROBE_PROTOCOL_HPP

#include "parcae/bench/bench_report.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>

/// Parse / validate external bench probe JSON (`probe_schema_version` 1.0.0).
///
/// Normative wire format: `docs/spec/bench-probe.md`.
class BenchProbeProtocol {
public:
    static constexpr std::string_view schema_version = "1.0.0";

    class Accuracy {
    public:
        bool oracle_cracked = false;
        std::int64_t top_rank = 0;
        std::string notes;
    };

    class Config {
    public:
        std::size_t candidates = 0; // C
        std::size_t tokens = 0;     // T
        std::size_t repeats = 0;    // reps
    };

    class Result {
    public:
        std::string tool_id;
        std::string tier;
        double runes_per_sec = 0.0;
        double keys_per_sec = 0.0;
        double wall_seconds = 0.0;
        BenchReport::Backend backend = BenchReport::Backend::Unknown;
        Accuracy accuracy;
        Config config;
    };

    /// Parse a single JSON object from stdout text. Rejects wrong version,
    /// missing fields, non-finite numbers, and empty `tool_id`.
    [[nodiscard]] static StatusOr<Result> parse(std::string_view text) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("BenchProbeProtocol: invalid JSON (") + ex.what() +
                                 ")");
        }
        if (!j.is_object()) {
            return Status::error("BenchProbeProtocol: stdout must be one JSON object");
        }
        return parse_object(j);
    }

    [[nodiscard]] static StatusOr<Result> parse_object(const nlohmann::json& j) {
        if (!j.contains("probe_schema_version") || !j.at("probe_schema_version").is_string()) {
            return Status::error("BenchProbeProtocol: missing probe_schema_version string");
        }
        const std::string version = j.at("probe_schema_version").get<std::string>();
        if (version != schema_version) {
            return Status::error("BenchProbeProtocol: unsupported probe_schema_version '" +
                                 version + "' (need " + std::string(schema_version) + ")");
        }

        Result out;
        StatusOr<std::string> tool_id = require_nonempty_string(j, "tool_id");
        if (!tool_id.ok()) {
            return tool_id.status();
        }
        out.tool_id = std::move(tool_id.value());

        StatusOr<std::string> tier = require_nonempty_string(j, "tier");
        if (!tier.ok()) {
            return tier.status();
        }
        out.tier = std::move(tier.value());

        StatusOr<double> rps = require_finite_nonneg(j, "runes_per_sec");
        if (!rps.ok()) {
            return rps.status();
        }
        out.runes_per_sec = rps.value();

        StatusOr<double> kps = require_finite_nonneg(j, "keys_per_sec");
        if (!kps.ok()) {
            return kps.status();
        }
        out.keys_per_sec = kps.value();

        StatusOr<double> wall = require_finite_nonneg(j, "wall_seconds");
        if (!wall.ok()) {
            return wall.status();
        }
        out.wall_seconds = wall.value();

        StatusOr<BenchReport::Backend> backend = parse_backend_field(j);
        if (!backend.ok()) {
            return backend.status();
        }
        out.backend = backend.value();

        StatusOr<Accuracy> accuracy = parse_accuracy(j);
        if (!accuracy.ok()) {
            return accuracy.status();
        }
        out.accuracy = std::move(accuracy.value());

        StatusOr<Config> config = parse_config(j);
        if (!config.ok()) {
            return config.status();
        }
        out.config = config.value();
        return out;
    }

private:
    BenchProbeProtocol() = delete;

    [[nodiscard]] static StatusOr<std::string> require_nonempty_string(const nlohmann::json& j,
                                                                       const char* key) {
        if (!j.contains(key) || !j.at(key).is_string()) {
            return Status::error(std::string("BenchProbeProtocol: missing string field '") + key +
                                 "'");
        }
        std::string value = j.at(key).get<std::string>();
        if (value.empty()) {
            return Status::error(std::string("BenchProbeProtocol: empty string field '") + key +
                                 "'");
        }
        return value;
    }

    [[nodiscard]] static StatusOr<double> require_finite_nonneg(const nlohmann::json& j,
                                                                const char* key) {
        if (!j.contains(key) || !j.at(key).is_number()) {
            return Status::error(std::string("BenchProbeProtocol: missing number field '") + key +
                                 "'");
        }
        const double value = j.at(key).get<double>();
        if (!std::isfinite(value) || value < 0.0) {
            return Status::error(std::string("BenchProbeProtocol: field '") + key +
                                 "' must be finite and >= 0");
        }
        return value;
    }

    [[nodiscard]] static StatusOr<BenchReport::Backend>
    parse_backend_field(const nlohmann::json& j) {
        StatusOr<std::string> raw = require_nonempty_string(j, "backend");
        if (!raw.ok()) {
            return raw.status();
        }
        if (raw.value() == "cpu") {
            return BenchReport::Backend::Cpu;
        }
        if (raw.value() == "cuda") {
            return BenchReport::Backend::Cuda;
        }
        if (raw.value() == "both") {
            return BenchReport::Backend::Both;
        }
        if (raw.value() == "unknown") {
            return BenchReport::Backend::Unknown;
        }
        return Status::error("BenchProbeProtocol: backend must be cpu|cuda|both|unknown (got '" +
                             raw.value() + "')");
    }

    [[nodiscard]] static StatusOr<Accuracy> parse_accuracy(const nlohmann::json& j) {
        if (!j.contains("accuracy") || !j.at("accuracy").is_object()) {
            return Status::error("BenchProbeProtocol: missing accuracy object");
        }
        const nlohmann::json& a = j.at("accuracy");
        if (!a.contains("oracle_cracked") || !a.at("oracle_cracked").is_boolean()) {
            return Status::error("BenchProbeProtocol: accuracy.oracle_cracked must be bool");
        }
        if (!a.contains("top_rank") || !a.at("top_rank").is_number_integer()) {
            return Status::error("BenchProbeProtocol: accuracy.top_rank must be integer");
        }
        if (!a.contains("notes") || !a.at("notes").is_string()) {
            return Status::error("BenchProbeProtocol: accuracy.notes must be string");
        }
        Accuracy out;
        out.oracle_cracked = a.at("oracle_cracked").get<bool>();
        out.top_rank = a.at("top_rank").get<std::int64_t>();
        if (out.top_rank < 0) {
            return Status::error("BenchProbeProtocol: accuracy.top_rank must be >= 0");
        }
        out.notes = a.at("notes").get<std::string>();
        return out;
    }

    [[nodiscard]] static StatusOr<Config> parse_config(const nlohmann::json& j) {
        if (!j.contains("config") || !j.at("config").is_object()) {
            return Status::error("BenchProbeProtocol: missing config object");
        }
        const nlohmann::json& c = j.at("config");
        StatusOr<std::size_t> candidates = require_positive_size(c, "C");
        if (!candidates.ok()) {
            return candidates.status();
        }
        StatusOr<std::size_t> tokens = require_positive_size(c, "T");
        if (!tokens.ok()) {
            return tokens.status();
        }
        StatusOr<std::size_t> repeats = require_positive_size(c, "reps");
        if (!repeats.ok()) {
            return repeats.status();
        }
        Config out;
        out.candidates = candidates.value();
        out.tokens = tokens.value();
        out.repeats = repeats.value();
        return out;
    }

    [[nodiscard]] static StatusOr<std::size_t> require_positive_size(const nlohmann::json& j,
                                                                     const char* key) {
        if (!j.contains(key) || !j.at(key).is_number_integer()) {
            return Status::error(std::string("BenchProbeProtocol: config.") + key +
                                 " must be integer");
        }
        const std::int64_t raw = j.at(key).get<std::int64_t>();
        if (raw < 1) {
            return Status::error(std::string("BenchProbeProtocol: config.") + key +
                                 " must be >= 1");
        }
        return static_cast<std::size_t>(raw);
    }
};

#endif // BENCH_PROBE_PROTOCOL_HPP
