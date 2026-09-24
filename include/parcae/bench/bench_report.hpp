#ifndef BENCH_REPORT_HPP
#define BENCH_REPORT_HPP

#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/version.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Unified bench / diagnostics report model (all suites).
///
/// JSON export supports `--omit-timing`: drops non-deterministic rate / wall
/// fields so agent digests stay stable (same contract as SearchRunMetrics).
class BenchReport {
public:
    enum class Suite : std::uint8_t {
        Slo = 0,
        Accuracy,
        Hardware,
        Probe,
        All,
    };

    enum class Backend : std::uint8_t {
        Cpu = 0,
        Cuda,
        Both,
        Unknown,
    };

    enum class RowStatus : std::uint8_t {
        Pass = 0,
        Fail,
        Skipped,
    };

    /// One measured or validated row (tier / check / probe).
    class Row {
    public:
        Row() = default;

        [[nodiscard]] static Row make(
            std::string name,
            std::string workload,
            Suite suite,
            Backend backend,
            RowStatus status,
            double runes_per_sec,
            double keys_per_sec,
            double wall_seconds,
            double target_min,
            double target_max,
            double estimated_peak,
            std::size_t candidates,
            std::size_t tokens,
            std::size_t repeats,
            std::string detail = {}) {
            Row row;
            row.name_ = std::move(name);
            row.workload_ = std::move(workload);
            row.suite_ = suite;
            row.backend_ = backend;
            row.status_ = status;
            row.runes_per_sec_ = runes_per_sec;
            row.keys_per_sec_ = keys_per_sec;
            row.wall_seconds_ = wall_seconds;
            row.target_min_ = target_min;
            row.target_max_ = target_max;
            row.estimated_peak_ = estimated_peak;
            row.candidates_ = candidates;
            row.tokens_ = tokens;
            row.repeats_ = repeats;
            row.detail_ = std::move(detail);
            return row;
        }

        /// Convenience: SLO row from `BenchTierSpec::Tier` + measured sample.
        [[nodiscard]] static Row from_tier_spec(
            const BenchTierSpec::Tier& tier,
            Backend backend,
            double runes_per_sec,
            double keys_per_sec,
            double wall_seconds,
            bool pass,
            std::string detail = {}) {
            return make(
                tier.id,
                tier.workload,
                Suite::Slo,
                backend,
                pass ? RowStatus::Pass : RowStatus::Fail,
                runes_per_sec,
                keys_per_sec,
                wall_seconds,
                tier.slo_min,
                tier.slo_max,
                tier.estimated_peak,
                tier.candidates,
                tier.tokens,
                tier.repeats,
                std::move(detail));
        }

        [[nodiscard]] const std::string& name() const noexcept {
            return name_;
        }

        [[nodiscard]] const std::string& workload() const noexcept {
            return workload_;
        }

        [[nodiscard]] Suite suite() const noexcept {
            return suite_;
        }

        [[nodiscard]] Backend backend() const noexcept {
            return backend_;
        }

        [[nodiscard]] RowStatus status() const noexcept {
            return status_;
        }

        [[nodiscard]] bool passed() const noexcept {
            return status_ == RowStatus::Pass;
        }

        [[nodiscard]] double runes_per_sec() const noexcept {
            return runes_per_sec_;
        }

        [[nodiscard]] double keys_per_sec() const noexcept {
            return keys_per_sec_;
        }

        [[nodiscard]] double wall_seconds() const noexcept {
            return wall_seconds_;
        }

        [[nodiscard]] double target_min() const noexcept {
            return target_min_;
        }

        [[nodiscard]] double target_max() const noexcept {
            return target_max_;
        }

        [[nodiscard]] double estimated_peak() const noexcept {
            return estimated_peak_;
        }

        [[nodiscard]] double percent_peak() const noexcept {
            return BenchTierSpec::percent_peak(runes_per_sec_, estimated_peak_);
        }

        [[nodiscard]] std::size_t candidates() const noexcept {
            return candidates_;
        }

        [[nodiscard]] std::size_t tokens() const noexcept {
            return tokens_;
        }

        [[nodiscard]] std::size_t repeats() const noexcept {
            return repeats_;
        }

        [[nodiscard]] const std::string& detail() const noexcept {
            return detail_;
        }

        /// Row JSON. When `omit_timing`, drops runes/s, keys/s, wall, %peak.
        [[nodiscard]] nlohmann::json to_json(bool omit_timing = false) const {
            nlohmann::json out = {
                {"name", name_},
                {"workload", workload_},
                {"suite", suite_str(suite_)},
                {"backend", backend_str(backend_)},
                {"status", status_str(status_)},
                {"pass", passed()},
                {"target_min", target_min_},
                {"target_max", target_max_},
                {"estimated_peak", estimated_peak_},
                {"candidates", candidates_},
                {"tokens", tokens_},
                {"repeats", repeats_},
            };
            if (!detail_.empty()) {
                out["detail"] = detail_;
            }
            if (!omit_timing) {
                out["runes_per_sec"] = runes_per_sec_;
                out["keys_per_sec"] = keys_per_sec_;
                out["wall_seconds"] = wall_seconds_;
                out["percent_peak"] = percent_peak();
            }
            return out;
        }

    private:
        std::string name_;
        std::string workload_;
        Suite suite_ = Suite::Slo;
        Backend backend_ = Backend::Unknown;
        RowStatus status_ = RowStatus::Fail;
        double runes_per_sec_ = 0.0;
        double keys_per_sec_ = 0.0;
        double wall_seconds_ = 0.0;
        double target_min_ = 0.0;
        double target_max_ = 0.0;
        double estimated_peak_ = 0.0;
        std::size_t candidates_ = 0;
        std::size_t tokens_ = 0;
        std::size_t repeats_ = 0;
        std::string detail_;
    };

    class Document {
    public:
        Document() = default;

        explicit Document(Suite suite, std::string toolkit_version = PARCAE_VERSION_STRING)
            : suite_(suite), toolkit_version_(std::move(toolkit_version)) {}

        void set_suite(Suite suite) noexcept {
            suite_ = suite;
        }

        void set_toolkit_version(std::string version) {
            toolkit_version_ = std::move(version);
        }

        void add_row(Row row) {
            if (row.status() == RowStatus::Fail) {
                all_pass_ = false;
            }
            rows_.push_back(std::move(row));
        }

        void recompute_all_pass() noexcept {
            all_pass_ = true;
            for (const Row& row : rows_) {
                if (row.status() == RowStatus::Fail) {
                    all_pass_ = false;
                    return;
                }
            }
        }

        [[nodiscard]] Suite suite() const noexcept {
            return suite_;
        }

        [[nodiscard]] const std::string& toolkit_version() const noexcept {
            return toolkit_version_;
        }

        [[nodiscard]] bool all_pass() const noexcept {
            return all_pass_;
        }

        [[nodiscard]] const std::vector<Row>& rows() const noexcept {
            return rows_;
        }

        [[nodiscard]] std::vector<Row>& rows() noexcept {
            return rows_;
        }

        /// Top-level JSON. `omit_timing` strips rate/wall fields on every row
        /// and records `"omit_timing": true` for digest replay.
        [[nodiscard]] nlohmann::json to_json(bool omit_timing = false) const {
            nlohmann::json rows = nlohmann::json::array();
            for (const Row& row : rows_) {
                rows.push_back(row.to_json(omit_timing));
            }
            return nlohmann::json{
                {"ok", all_pass_},
                {"toolkit_version", toolkit_version_},
                {"suite", suite_str(suite_)},
                {"omit_timing", omit_timing},
                {"all_pass", all_pass_},
                {"row_count", rows_.size()},
                {"rows", std::move(rows)},
            };
        }

    private:
        Suite suite_ = Suite::Slo;
        std::string toolkit_version_{PARCAE_VERSION_STRING};
        bool all_pass_ = true;
        std::vector<Row> rows_;
    };

    [[nodiscard]] static constexpr const char* suite_str(Suite suite) noexcept {
        switch (suite) {
        case Suite::Slo:
            return "slo";
        case Suite::Accuracy:
            return "accuracy";
        case Suite::Hardware:
            return "hardware";
        case Suite::Probe:
            return "probe";
        case Suite::All:
            return "all";
        }
        return "unknown";
    }

    [[nodiscard]] static constexpr const char* backend_str(Backend backend) noexcept {
        switch (backend) {
        case Backend::Cpu:
            return "cpu";
        case Backend::Cuda:
            return "cuda";
        case Backend::Both:
            return "both";
        case Backend::Unknown:
            return "unknown";
        }
        return "unknown";
    }

    [[nodiscard]] static constexpr const char* status_str(RowStatus status) noexcept {
        switch (status) {
        case RowStatus::Pass:
            return "pass";
        case RowStatus::Fail:
            return "fail";
        case RowStatus::Skipped:
            return "skipped";
        }
        return "unknown";
    }

private:
    BenchReport() = delete;
};

#endif // BENCH_REPORT_HPP
