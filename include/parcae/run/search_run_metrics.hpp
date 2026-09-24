#ifndef SEARCH_RUN_METRICS_HPP
#define SEARCH_RUN_METRICS_HPP

#include "parcae/core/sha256.hpp"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// One step along a parameter sweep (AI-"training step" analogue).
class SearchRunStep {
public:
    SearchRunStep(std::size_t step_id, std::string transform_id, nlohmann::json params,
                  double score)
        : step_id_(step_id), transform_id_(std::move(transform_id)), params_(std::move(params)),
          param_hash_(Sha256::hex_digest(params_.dump())), score_(score) {}

    [[nodiscard]] std::size_t step_id() const noexcept { return step_id_; }

    [[nodiscard]] const std::string& transform_id() const noexcept { return transform_id_; }

    /// Transform params for this step (replayable into decode / generators).
    [[nodiscard]] const nlohmann::json& params() const noexcept { return params_; }

    [[nodiscard]] const std::string& param_hash() const noexcept { return param_hash_; }

    [[nodiscard]] double score() const noexcept { return score_; }

    [[nodiscard]] nlohmann::json to_json() const {
        return nlohmann::json{
            {"step_id", step_id_}, {"transform_id", transform_id_},
            {"params", params_},   {"param_hash", param_hash_},
            {"score", score_},
        };
    }

private:
    std::size_t step_id_ = 0;
    std::string transform_id_;
    nlohmann::json params_ = nlohmann::json::object();
    std::string param_hash_;
    double score_ = 0.0;
};

/// Aggregate tracking for a CUDA/CPU search run (throughput + sweep + fixture eval).
class SearchRunMetrics {
public:
    SearchRunMetrics() = default;

    void set_seed(std::uint32_t seed) noexcept { seed_ = seed; }

    void set_transform_id(std::string id) { transform_id_ = std::move(id); }

    void set_parameters_label(std::string label) { parameters_label_ = std::move(label); }

    void set_score_id(std::string id) { score_id_ = std::move(id); }

    void set_backend(std::string backend) { backend_ = std::move(backend); }

    void set_tok_per_sec(double value) noexcept { tok_per_sec_ = value; }

    void set_score_mean(double value) noexcept { score_mean_ = value; }

    void set_score_std(double value) noexcept { score_std_ = value; }

    void set_eval(std::size_t passed, std::size_t total) noexcept {
        eval_passed_ = passed;
        eval_total_ = total;
        eval_set_pass_rate_ =
            total == 0 ? 0.0 : static_cast<double>(passed) / static_cast<double>(total);
    }

    void set_cpu_cuda_pass(std::optional<bool> value) noexcept { cpu_cuda_pass_ = value; }

    void set_steps(std::vector<SearchRunStep> steps) { steps_ = std::move(steps); }

    [[nodiscard]] std::uint32_t seed() const noexcept { return seed_; }

    [[nodiscard]] const std::string& transform_id() const noexcept { return transform_id_; }

    [[nodiscard]] const std::string& parameters_label() const noexcept { return parameters_label_; }

    [[nodiscard]] const std::string& score_id() const noexcept { return score_id_; }

    [[nodiscard]] const std::string& backend() const noexcept { return backend_; }

    [[nodiscard]] double tok_per_sec() const noexcept { return tok_per_sec_; }

    [[nodiscard]] double score_mean() const noexcept { return score_mean_; }

    [[nodiscard]] double score_std() const noexcept { return score_std_; }

    [[nodiscard]] double eval_set_pass_rate() const noexcept { return eval_set_pass_rate_; }

    [[nodiscard]] std::size_t eval_passed() const noexcept { return eval_passed_; }

    [[nodiscard]] std::size_t eval_total() const noexcept { return eval_total_; }

    [[nodiscard]] const std::optional<bool>& cpu_cuda_pass() const noexcept {
        return cpu_cuda_pass_;
    }

    [[nodiscard]] const std::vector<SearchRunStep>& steps() const noexcept { return steps_; }

    /// Agent/JSON export. When `omit_timing`, drops non-deterministic `tok_per_sec`.
    [[nodiscard]] nlohmann::json to_json(bool omit_timing = false) const {
        nlohmann::json steps = nlohmann::json::array();
        for (const SearchRunStep& step : steps_) {
            steps.push_back(step.to_json());
        }
        nlohmann::json out = {
            {"transform_id", transform_id_},
            {"parameters", parameters_label_},
            {"seed", seed_},
            {"backend", backend_},
            {"score_id", score_id_},
            {"score_mean", score_mean_},
            {"score_std", score_std_},
            {"eval_passed", eval_passed_},
            {"eval_total", eval_total_},
            {"eval_set_pass_rate", eval_set_pass_rate_},
            {"steps", std::move(steps)},
        };
        if (!omit_timing) {
            out["tok_per_sec"] = tok_per_sec_;
        }
        if (cpu_cuda_pass_.has_value()) {
            out["cpu_cuda_pass"] = cpu_cuda_pass_.value();
        }
        return out;
    }

private:
    std::uint32_t seed_ = 0;
    std::string transform_id_;
    std::string parameters_label_;
    std::string score_id_;
    std::string backend_{"cpu"};
    double tok_per_sec_ = 0.0;
    double score_mean_ = 0.0;
    double score_std_ = 0.0;
    double eval_set_pass_rate_ = 0.0;
    std::size_t eval_passed_ = 0;
    std::size_t eval_total_ = 0;
    std::optional<bool> cpu_cuda_pass_;
    std::vector<SearchRunStep> steps_;
};

#endif // SEARCH_RUN_METRICS_HPP
