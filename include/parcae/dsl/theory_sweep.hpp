#ifndef THEORY_SWEEP_HPP
#define THEORY_SWEEP_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Expand `TheoryArtifact.sweep` into a candidate param plan (docs/spec/theory-artifact.md).
/// Does **not** apply transforms or invent solved-corpus defaults. Apply/score needs
/// TheoryDispatch (later). Stale `dsl_spec_version` → hard reject via TheoryRegistry.
class TheorySweep {
public:
    class Options {
    public:
        Options() = default;

        /// Cap expanded candidates (0 = unlimited). Default 100000.
        void set_limit(std::size_t limit) {
            limit_ = limit;
        }

        [[nodiscard]] std::size_t limit() const noexcept {
            return limit_;
        }

    private:
        std::size_t limit_ = 100000;
    };

    class Plan {
    public:
        Plan(
            std::string uri,
            std::string corpus,
            std::vector<std::string> record_metrics,
            std::optional<std::string> compare_against,
            std::vector<nlohmann::json> candidates,
            std::size_t total_before_limit,
            bool truncated)
            : uri_(std::move(uri)),
              corpus_(std::move(corpus)),
              record_metrics_(std::move(record_metrics)),
              compare_against_(std::move(compare_against)),
              candidates_(std::move(candidates)),
              total_before_limit_(total_before_limit),
              truncated_(truncated) {}

        [[nodiscard]] const std::string& uri() const noexcept {
            return uri_;
        }

        [[nodiscard]] const std::string& corpus() const noexcept {
            return corpus_;
        }

        [[nodiscard]] const std::vector<std::string>& record_metrics() const noexcept {
            return record_metrics_;
        }

        [[nodiscard]] const std::optional<std::string>& compare_against() const noexcept {
            return compare_against_;
        }

        [[nodiscard]] const std::vector<nlohmann::json>& candidates() const noexcept {
            return candidates_;
        }

        [[nodiscard]] std::size_t total_before_limit() const noexcept {
            return total_before_limit_;
        }

        [[nodiscard]] bool truncated() const noexcept {
            return truncated_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json metrics = nlohmann::json::array();
            for (const std::string& m : record_metrics_) {
                metrics.push_back(m);
            }
            nlohmann::json cands = nlohmann::json::array();
            for (const nlohmann::json& c : candidates_) {
                cands.push_back(c);
            }
            nlohmann::json out{
                {"uri", uri_},
                {"corpus", corpus_},
                {"record_metrics", std::move(metrics)},
                {"candidates", std::move(cands)},
                {"count", candidates_.size()},
                {"total_before_limit", total_before_limit_},
                {"truncated", truncated_},
                {"execute", false},
                {"note", "plan-only; apply/score requires TheoryDispatch"},
            };
            if (compare_against_.has_value()) {
                out["compare_against"] = *compare_against_;
            } else {
                out["compare_against"] = nullptr;
            }
            return out;
        }

    private:
        std::string uri_;
        std::string corpus_;
        std::vector<std::string> record_metrics_;
        std::optional<std::string> compare_against_;
        std::vector<nlohmann::json> candidates_;
        std::size_t total_before_limit_ = 0;
        bool truncated_ = false;
    };

    /// Build a sweep plan from a loaded (spec-compatible) artifact.
    [[nodiscard]] static StatusOr<Plan> plan(const TheoryArtifact& artifact, Options opt = {}) {
        const nlohmann::json& sweep = artifact.sweep();
        if (sweep.is_null()) {
            return Status::error(
                "theory '" + artifact.name() +
                "' has sweep=null; nothing to sweep (configure sweep in the theory source / recompile)");
        }
        if (!sweep.is_object()) {
            return Status::error("TheoryArtifact.sweep must be an object when set");
        }

        Status cfg = validate_config(artifact, sweep);
        if (!cfg.ok()) {
            return cfg;
        }

        const std::string corpus = sweep.at("corpus").get<std::string>();
        Status corpus_ok = check_corpus(artifact.tier(), corpus);
        if (!corpus_ok.ok()) {
            return corpus_ok;
        }

        std::vector<std::string> metrics;
        for (const auto& m : sweep.at("record_metrics")) {
            if (!m.is_string() || m.get<std::string>().empty()) {
                return Status::error("sweep.record_metrics entries must be non-empty strings");
            }
            metrics.push_back(m.get<std::string>());
        }

        std::optional<std::string> compare;
        if (sweep.contains("compare_against") && !sweep.at("compare_against").is_null()) {
            if (!sweep.at("compare_against").is_string() ||
                sweep.at("compare_against").get<std::string>().empty()) {
                return Status::error("sweep.compare_against must be a non-empty string when set");
            }
            compare = sweep.at("compare_against").get<std::string>();
        }

        StatusOr<std::vector<std::vector<std::int64_t>>> axes =
            expand_axes(artifact, sweep.at("param_grid"));
        if (!axes.ok()) {
            return axes.status();
        }

        std::vector<nlohmann::json> candidates;
        std::size_t total = 0;
        bool truncated = false;
        Status cart = cartesian(artifact.params(), axes.value(), opt.limit(), candidates, total, truncated);
        if (!cart.ok()) {
            return cart;
        }

        return Plan{
            artifact.uri().to_string(),
            corpus,
            std::move(metrics),
            std::move(compare),
            std::move(candidates),
            total,
            truncated,
        };
    }

    /// Load via TheoryRegistry (stale dsl_spec → error) then plan.
    [[nodiscard]] static StatusOr<Plan> plan_uri(
        const std::filesystem::path& theories_root,
        std::string_view uri_or_ref,
        Options opt = {}) {
        StatusOr<TheoryUri> uri = parse_ref(uri_or_ref);
        if (!uri.ok()) {
            return uri.status();
        }
        StatusOr<TheoryArtifact> artifact =
            TheoryRegistry::load(theories_root, uri.value().name(), uri.value().version());
        if (!artifact.ok()) {
            return artifact.status();
        }
        return plan(artifact.value(), opt);
    }

private:
    TheorySweep() = delete;

    [[nodiscard]] static StatusOr<TheoryUri> parse_ref(std::string_view text) {
        if (text.rfind("parcae://", 0) == 0) {
            return TheoryUri::parse(text);
        }
        const auto at = text.find('@');
        if (at == std::string_view::npos || at == 0 || at + 1 >= text.size()) {
            return Status::error(
                "theory ref must be parcae://theories/<name>@<ver> or <name>@<ver>");
        }
        const std::string_view name = text.substr(0, at);
        const std::string_view ver = text.substr(at + 1);
        if (ver.empty() || (ver.size() > 1 && ver[0] == '0') ||
            ver.find_first_not_of("0123456789") != std::string_view::npos) {
            return Status::error("invalid theory version in ref: " + std::string(text));
        }
        std::uint32_t v = 0;
        for (char c : ver) {
            v = v * 10u + static_cast<std::uint32_t>(c - '0');
        }
        if (v == 0) {
            return Status::error("version must be >= 1");
        }
        return TheoryUri::make(std::string(name), v);
    }

    [[nodiscard]] static Status validate_config(
        const TheoryArtifact& artifact,
        const nlohmann::json& sweep) {
        if (!sweep.contains("theory") || !sweep.at("theory").is_string()) {
            return Status::error("sweep.theory must be a string equal to the artifact name");
        }
        if (sweep.at("theory").get<std::string>() != artifact.name()) {
            return Status::error(
                "sweep.theory '" + sweep.at("theory").get<std::string>() +
                "' must equal artifact name '" + artifact.name() + "'");
        }
        if (!sweep.contains("corpus") || !sweep.at("corpus").is_string() ||
            sweep.at("corpus").get<std::string>().empty()) {
            return Status::error("sweep.corpus must be a non-empty string");
        }
        if (!sweep.contains("param_grid") || !sweep.at("param_grid").is_object()) {
            return Status::error("sweep.param_grid must be an object");
        }
        if (!sweep.contains("record_metrics") || !sweep.at("record_metrics").is_array() ||
            sweep.at("record_metrics").empty()) {
            return Status::error("sweep.record_metrics must be a non-empty array of strings");
        }
        if ((artifact.tier() == TheoryIr::Tier::B || artifact.tier() == TheoryIr::Tier::C) &&
            (!sweep.contains("compare_against") || sweep.at("compare_against").is_null())) {
            return Status::error(
                "sweep.compare_against required for tier B/C when sweep is set");
        }
        return Status::success();
    }

    /// Solved-fixture ids must not be silent Tier B/C discovery corpora.
    [[nodiscard]] static bool is_solved_oracle_corpus(std::string_view corpus) {
        static constexpr std::string_view kSolved[] = {
            "a-warning",
            "an-end",
            "an-instruction",
            "koan-1",
            "koan-2",
            "loss-of-divinity",
            "lp2-57-identity",
            "some-wisdom",
            "welcome",
            "synth-identity",
            "synth-vigenere-draft",
        };
        for (std::string_view id : kSolved) {
            if (corpus == id) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] static Status check_corpus(TheoryIr::Tier tier, std::string_view corpus) {
        if ((tier == TheoryIr::Tier::B || tier == TheoryIr::Tier::C) &&
            is_solved_oracle_corpus(corpus)) {
            return Status::error(
                "sweep.corpus '" + std::string(corpus) +
                "' is a solved-oracle fixture id; Tier B/C MUST NOT use it as a silent "
                "discovery default (set an explicit research corpus / compare_against)");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::vector<std::int64_t>> parse_axis(
        const TheoryArtifact::Param& param,
        const nlohmann::json& spec) {
        std::vector<std::int64_t> values;
        if (spec.is_string() && spec.get<std::string>() == "full") {
            for (std::int64_t v = param.min(); v <= param.max(); ++v) {
                values.push_back(v);
            }
            return values;
        }
        if (spec.is_array()) {
            for (const auto& item : spec) {
                if (!item.is_number_integer()) {
                    return Status::error(
                        "param_grid." + param.name() + " array entries must be integers");
                }
                const std::int64_t v = item.get<std::int64_t>();
                if (v < param.min() || v > param.max()) {
                    return Status::error(
                        "param_grid." + param.name() + " value " + std::to_string(v) +
                        " outside declared domain [" + std::to_string(param.min()) + "," +
                        std::to_string(param.max()) + "]");
                }
                values.push_back(v);
            }
            if (values.empty()) {
                return Status::error("param_grid." + param.name() + " array must be non-empty");
            }
            return values;
        }
        if (spec.is_object()) {
            if (spec.contains("values")) {
                if (!spec.at("values").is_array()) {
                    return Status::error(
                        "param_grid." + param.name() + ".values must be an array");
                }
                return parse_axis(param, spec.at("values"));
            }
            if (!spec.contains("min") || !spec.contains("max") || !spec.at("min").is_number_integer() ||
                !spec.at("max").is_number_integer()) {
                return Status::error(
                    "param_grid." + param.name() +
                    " object must have integer min/max or a values array (or \"full\")");
            }
            const std::int64_t lo = spec.at("min").get<std::int64_t>();
            const std::int64_t hi = spec.at("max").get<std::int64_t>();
            if (lo > hi) {
                return Status::error("param_grid." + param.name() + " min > max");
            }
            if (lo < param.min() || hi > param.max()) {
                return Status::error(
                    "param_grid." + param.name() + " range outside declared domain [" +
                    std::to_string(param.min()) + "," + std::to_string(param.max()) + "]");
            }
            for (std::int64_t v = lo; v <= hi; ++v) {
                values.push_back(v);
            }
            return values;
        }
        return Status::error(
            "param_grid." + param.name() +
            " must be \"full\", an int array, or {min,max}/{values}");
    }

    [[nodiscard]] static StatusOr<std::vector<std::vector<std::int64_t>>> expand_axes(
        const TheoryArtifact& artifact,
        const nlohmann::json& grid) {
        std::vector<std::vector<std::int64_t>> axes;
        axes.reserve(artifact.params().size());
        for (const TheoryArtifact::Param& p : artifact.params()) {
            if (!grid.contains(p.name())) {
                return Status::error(
                    "param_grid missing theory param '" + p.name() + "'");
            }
            StatusOr<std::vector<std::int64_t>> axis = parse_axis(p, grid.at(p.name()));
            if (!axis.ok()) {
                return axis.status();
            }
            axes.push_back(std::move(axis.value()));
        }
        // Reject unknown keys in grid.
        for (auto it = grid.begin(); it != grid.end(); ++it) {
            bool known = false;
            for (const TheoryArtifact::Param& p : artifact.params()) {
                if (p.name() == it.key()) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                return Status::error("param_grid has unknown param '" + it.key() + "'");
            }
        }
        return axes;
    }

    [[nodiscard]] static Status cartesian(
        const std::vector<TheoryArtifact::Param>& params,
        const std::vector<std::vector<std::int64_t>>& axes,
        std::size_t limit,
        std::vector<nlohmann::json>& out,
        std::size_t& total,
        bool& truncated) {
        if (params.size() != axes.size()) {
            return Status::error("internal: param/axis size mismatch");
        }
        if (params.empty()) {
            total = 1;
            if (limit != 0 && limit < 1) {
                truncated = true;
                return Status::success();
            }
            out.push_back(nlohmann::json::object());
            return Status::success();
        }

        // Precompute total with overflow guard.
        total = 1;
        for (const auto& axis : axes) {
            if (axis.empty()) {
                return Status::error("empty param axis");
            }
            if (total > (static_cast<std::size_t>(-1) / axis.size())) {
                return Status::error("param_grid cartesian product overflows size_t");
            }
            total *= axis.size();
        }

        std::vector<std::size_t> idx(axes.size(), 0);
        const std::size_t emit_cap = (limit == 0) ? total : (limit < total ? limit : total);
        truncated = limit != 0 && limit < total;
        out.reserve(emit_cap);

        for (std::size_t n = 0; n < emit_cap; ++n) {
            nlohmann::json cand = nlohmann::json::object();
            for (std::size_t i = 0; i < params.size(); ++i) {
                cand[params[i].name()] = axes[i][idx[i]];
            }
            out.push_back(std::move(cand));

            // odometer increment
            for (std::size_t i = 0; i < idx.size(); ++i) {
                ++idx[i];
                if (idx[i] < axes[i].size()) {
                    break;
                }
                idx[i] = 0;
            }
        }
        return Status::success();
    }
};

#endif // THEORY_SWEEP_HPP
