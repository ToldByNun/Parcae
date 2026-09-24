#ifndef BENCH_CONFIG_HPP
#define BENCH_CONFIG_HPP

#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// Suite / probe flags shared by `parcae-bench` orchestration.
class BenchConfig {
public:
    static constexpr std::uint32_t default_probe_timeout_ms = 120000u;

    BenchConfig() {
        probe_tiers_ = {"T1", "T2", "T3"};
    }

    [[nodiscard]] const std::string& probe_cmd() const noexcept {
        return probe_cmd_;
    }

    void set_probe_cmd(std::string cmd) {
        probe_cmd_ = std::move(cmd);
    }

    [[nodiscard]] bool has_probe_cmd() const noexcept {
        return !probe_cmd_.empty();
    }

    [[nodiscard]] const std::vector<std::string>& probe_tiers() const noexcept {
        return probe_tiers_;
    }

    void set_probe_tiers(std::vector<std::string> tiers) {
        probe_tiers_ = std::move(tiers);
    }

    [[nodiscard]] std::uint32_t probe_timeout_ms() const noexcept {
        return probe_timeout_ms_;
    }

    void set_probe_timeout_ms(std::uint32_t ms) noexcept {
        probe_timeout_ms_ = ms;
    }

    [[nodiscard]] bool compare_builtin() const noexcept {
        return compare_builtin_;
    }

    void set_compare_builtin(bool enabled) noexcept {
        compare_builtin_ = enabled;
    }

    [[nodiscard]] std::uint32_t seed() const noexcept {
        return seed_;
    }

    void set_seed(std::uint32_t seed) noexcept {
        seed_ = seed;
    }

    /// Parse `T1,T2,T3` (whitespace around commas ignored). Empty → default trio.
    [[nodiscard]] static StatusOr<std::vector<std::string>> parse_probe_tiers(
        std::string_view text) {
        std::vector<std::string> out;
        std::string current;
        auto flush = [&]() -> Status {
            // trim
            std::size_t b = 0;
            while (b < current.size() &&
                   (current[b] == ' ' || current[b] == '\t')) {
                ++b;
            }
            std::size_t e = current.size();
            while (e > b && (current[e - 1] == ' ' || current[e - 1] == '\t')) {
                --e;
            }
            if (b >= e) {
                current.clear();
                return Status::success();
            }
            const std::string id = current.substr(b, e - b);
            if (BenchTierSpec::find_primary(id) == nullptr) {
                return Status::error(
                    "BenchConfig: unknown probe tier '" + id + "' (use T1|T2|T3)");
            }
            out.push_back(id);
            current.clear();
            return Status::success();
        };

        for (char ch : text) {
            if (ch == ',') {
                Status flushed = flush();
                if (!flushed.ok()) {
                    return flushed;
                }
            } else {
                current.push_back(ch);
            }
        }
        Status flushed = flush();
        if (!flushed.ok()) {
            return flushed;
        }
        if (out.empty()) {
            out = {"T1", "T2", "T3"};
        }
        return out;
    }

    [[nodiscard]] static StatusOr<std::uint32_t> parse_timeout_ms(std::string_view text) {
        if (text.empty()) {
            return default_probe_timeout_ms;
        }
        std::uint32_t value = 0;
        for (char ch : text) {
            if (ch < '0' || ch > '9') {
                return Status::error(
                    "BenchConfig: --probe-timeout-ms must be a non-negative integer");
            }
            const std::uint32_t digit = static_cast<std::uint32_t>(ch - '0');
            if (value > (UINT32_MAX - digit) / 10u) {
                return Status::error("BenchConfig: --probe-timeout-ms overflow");
            }
            value = value * 10u + digit;
        }
        if (value == 0) {
            return Status::error("BenchConfig: --probe-timeout-ms must be >= 1");
        }
        return value;
    }

private:
    std::string probe_cmd_;
    std::vector<std::string> probe_tiers_;
    std::uint32_t probe_timeout_ms_ = default_probe_timeout_ms;
    bool compare_builtin_ = false;
    std::uint32_t seed_ = 0u;
};

#endif // BENCH_CONFIG_HPP
