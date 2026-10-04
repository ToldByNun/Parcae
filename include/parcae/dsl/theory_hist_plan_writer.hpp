#ifndef THEORY_HIST_PLAN_WRITER_HPP
#define THEORY_HIST_PLAN_WRITER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_hist_chi2_emit.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_shape_match.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/// Persist fused-hist classify/emit into `hist/hist_plan.json` (+ optional sources).
///
/// Schema: `parcae.theory_hist_plan.v0` (`docs/spec/theory-artifact.md`).
/// Soft-fallback (`specialized==false`) still writes a plan so tools see the classify
/// reason; search remains correct via runtime S0. Emit failure → caller leaves
/// `hist` / `paths.hist_*` null (safe shipping default). No C++ namespaces.
class TheoryHistPlanWriter {
public:
    static constexpr std::string_view schema_id = "parcae.theory_hist_plan.v0";

    /// Compact top-level `manifest.hist` summary.
    class Summary {
    public:
        Summary() = default;

        Summary(std::string intended, std::string emitted, bool specialized, std::string reason,
                std::optional<std::string> shape_peak_tier = std::nullopt)
            : intended_(std::move(intended)), emitted_(std::move(emitted)),
              specialized_(specialized), reason_(std::move(reason)),
              shape_peak_tier_(std::move(shape_peak_tier)) {}

        [[nodiscard]] const std::string& intended_strategy() const noexcept { return intended_; }

        [[nodiscard]] const std::string& emitted_strategy() const noexcept { return emitted_; }

        [[nodiscard]] bool specialized() const noexcept { return specialized_; }

        [[nodiscard]] const std::string& reason() const noexcept { return reason_; }

        [[nodiscard]] const std::optional<std::string>& shape_peak_tier() const noexcept {
            return shape_peak_tier_;
        }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json j{{"intended_strategy", intended_},
                             {"emitted_strategy", emitted_},
                             {"specialized", specialized_},
                             {"reason", reason_}};
            if (shape_peak_tier_.has_value()) {
                j["shape_peak_tier"] = *shape_peak_tier_;
            } else {
                j["shape_peak_tier"] = nullptr;
            }
            return j;
        }

        [[nodiscard]] static StatusOr<Summary> from_json(const nlohmann::json& j) {
            if (!j.is_object()) {
                return Status::error("TheoryArtifact.hist must be an object when set");
            }
            for (const char* key : {"intended_strategy", "emitted_strategy", "reason"}) {
                if (!j.contains(key) || !j.at(key).is_string() || j.at(key).get<std::string>().empty()) {
                    return Status::error(std::string("TheoryArtifact.hist.") + key +
                                         " must be a non-empty string");
                }
            }
            if (!j.contains("specialized") || !j.at("specialized").is_boolean()) {
                return Status::error("TheoryArtifact.hist.specialized must be a boolean");
            }
            std::optional<std::string> peak;
            if (j.contains("shape_peak_tier") && !j.at("shape_peak_tier").is_null()) {
                if (!j.at("shape_peak_tier").is_string()) {
                    return Status::error("TheoryArtifact.hist.shape_peak_tier must be string or null");
                }
                peak = j.at("shape_peak_tier").get<std::string>();
            }
            return Summary{j.at("intended_strategy").get<std::string>(),
                           j.at("emitted_strategy").get<std::string>(),
                           j.at("specialized").get<bool>(), j.at("reason").get<std::string>(),
                           std::move(peak)};
        }

    private:
        std::string intended_;
        std::string emitted_;
        bool specialized_ = false;
        std::string reason_;
        std::optional<std::string> shape_peak_tier_;
    };

    /// Prepared hist artifact payload (plan JSON + optional sources).
    class Bundle {
    public:
        Bundle(nlohmann::json plan_json, Summary summary, bool specialized,
               std::string header_text, std::string cu_text, std::string header_rel,
               std::string source_rel)
            : plan_json_(std::move(plan_json)), summary_(std::move(summary)),
              specialized_(specialized), header_text_(std::move(header_text)),
              cu_text_(std::move(cu_text)), header_rel_(std::move(header_rel)),
              source_rel_(std::move(source_rel)) {}

        [[nodiscard]] const nlohmann::json& plan_json() const noexcept { return plan_json_; }

        [[nodiscard]] const Summary& summary() const noexcept { return summary_; }

        [[nodiscard]] bool specialized() const noexcept { return specialized_; }

        [[nodiscard]] bool write_sources() const noexcept {
            return specialized_ && !header_text_.empty() && !cu_text_.empty();
        }

        [[nodiscard]] const std::string& header_text() const noexcept { return header_text_; }

        [[nodiscard]] const std::string& cu_text() const noexcept { return cu_text_; }

        [[nodiscard]] const std::string& header_rel() const noexcept { return header_rel_; }

        [[nodiscard]] const std::string& source_rel() const noexcept { return source_rel_; }

        [[nodiscard]] static constexpr std::string_view plan_rel() noexcept {
            return "hist/hist_plan.json";
        }

    private:
        nlohmann::json plan_json_;
        Summary summary_;
        bool specialized_ = false;
        std::string header_text_;
        std::string cu_text_;
        std::string header_rel_;
        std::string source_rel_;
    };

    /// Classify+emit decrypt hist and build a persistable plan (does not touch disk).
    [[nodiscard]] static StatusOr<Bundle> prepare(const TheoryIr& theory, std::string_view theory_uri,
                                                  std::string_view cipher_var = "x") {
        if (theory_uri.empty()) {
            return Status::error("TheoryHistPlanWriter: empty theory_uri");
        }
        StatusOr<TheoryHistChi2Emit::EmitBundle> emit =
            TheoryHistChi2Emit::emit_decrypt_hist(theory, cipher_var);
        if (!emit.ok()) {
            return emit.status();
        }
        return from_emit_bundle(emit.value(), theory_uri, cipher_var, theory.name());
    }

    [[nodiscard]] static Bundle from_emit_bundle(const TheoryHistChi2Emit::EmitBundle& emit,
                                                 std::string_view theory_uri,
                                                 std::string_view cipher_var,
                                                 std::string_view theory_name) {
        const std::string intended =
            std::string(TheoryHistChi2Emit::strategy_str(emit.intended_strategy()));
        const std::string emitted =
            std::string(TheoryHistChi2Emit::strategy_str(emit.emitted_strategy()));
        const bool specialized = emit.specialized();
        std::string reason = emit.reason();
        if (reason.empty()) {
            reason = specialized ? "specialized hist plan" : "soft-fallback S0";
        }

        nlohmann::json plan{{"schema", std::string(schema_id)},
                            {"theory_uri", std::string(theory_uri)},
                            {"intended_strategy", intended},
                            {"emitted_strategy", emitted},
                            {"specialized", specialized},
                            {"cipher_var", std::string(cipher_var)},
                            {"reason", reason},
                            {"s1_lut", nullptr},
                            {"s2_linear", nullptr},
                            {"s3", nullptr},
                            {"s4_autokey", nullptr},
                            {"shape", nullptr}};

        if (emit.s1_lut().has_value()) {
            plan["s1_lut"] = nlohmann::json{{"param_names", emit.s1_lut()->param_names()}};
        }
        if (emit.s2_linear().has_value()) {
            const auto& s2 = *emit.s2_linear();
            nlohmann::json s2j{{"b0_name", s2.b0_name()},
                               {"b1_name", s2.b1_name()},
                               {"cipher_minus_ks", s2.cipher_minus_ks()},
                               {"has_const_b0", s2.has_const_b0()},
                               {"has_const_b1", s2.has_const_b1()}};
            if (s2.has_const_b0()) {
                s2j["const_b0"] = s2.const_b0();
            }
            if (s2.has_const_b1()) {
                s2j["const_b1"] = s2.const_b1();
            }
            plan["s2_linear"] = std::move(s2j);
        }
        if (emit.s3_scalar().has_value()) {
            const auto& s3 = *emit.s3_scalar();
            plan["s3"] = nlohmann::json{{"op_count", s3.op_count()},
                                        {"max_stack", s3.max_stack()},
                                        {"slot_count", s3.slot_count()},
                                        {"binds_index_i", s3.binds_index_i()},
                                        {"device_cpp", s3.device_cpp()}};
        }
        if (emit.shape().has_value()) {
            const auto& sh = *emit.shape();
            plan["shape"] = nlohmann::json{
                {"shape_id", TheoryShapeMatch::shape_str(sh.shape())},
                {"reason", sh.reason()},
                {"shift_name", sh.shift_name()},
                {"a_name", sh.a_name()},
                {"b_name", sh.b_name()},
                {"b0_name", sh.b0_name()},
                {"b1_name", sh.b1_name()},
                {"cipher_minus_ks", sh.cipher_minus_ks()},
                {"affine_decrypt", sh.affine_decrypt()},
            };
        }

        std::optional<std::string> peak = peak_tier_for(emit.emitted_strategy());
        Summary summary{intended, emitted, specialized, reason, std::move(peak)};

        std::string header_rel;
        std::string source_rel;
        if (specialized && !emit.header_text().empty() && !emit.cu_text().empty()) {
            const std::string stem = std::string(theory_name) + "_hist";
            header_rel = "hist/" + stem + ".hpp";
            source_rel = "hist/" + stem + ".cu";
        }

        return Bundle{std::move(plan), std::move(summary), specialized, emit.header_text(),
                      emit.cu_text(), std::move(header_rel), std::move(source_rel)};
    }

    /// Validate `parcae.theory_hist_plan.v0` document (optionally match URI).
    [[nodiscard]] static Status validate_plan_json(const nlohmann::json& plan,
                                                   std::string_view expected_uri = {}) {
        if (!plan.is_object()) {
            return Status::error("hist_plan must be a JSON object");
        }
        if (!plan.contains("schema") || !plan.at("schema").is_string() ||
            plan.at("schema").get<std::string>() != schema_id) {
            return Status::error("hist_plan.schema must be parcae.theory_hist_plan.v0");
        }
        if (!plan.contains("theory_uri") || !plan.at("theory_uri").is_string() ||
            plan.at("theory_uri").get<std::string>().empty()) {
            return Status::error("hist_plan.theory_uri must be a non-empty string");
        }
        if (!expected_uri.empty() && plan.at("theory_uri").get<std::string>() != expected_uri) {
            return Status::error("hist_plan.theory_uri must equal manifest uri");
        }
        for (const char* key : {"intended_strategy", "emitted_strategy", "reason", "cipher_var"}) {
            if (!plan.contains(key) || !plan.at(key).is_string() ||
                plan.at(key).get<std::string>().empty()) {
                return Status::error(std::string("hist_plan.") + key +
                                     " must be a non-empty string");
            }
        }
        if (!plan.contains("specialized") || !plan.at("specialized").is_boolean()) {
            return Status::error("hist_plan.specialized must be a boolean");
        }
        for (const char* key : {"s1_lut", "s2_linear", "s3", "s4_autokey", "shape"}) {
            if (plan.contains(key) && !plan.at(key).is_null() && !plan.at(key).is_object()) {
                return Status::error(std::string("hist_plan.") + key +
                                     " must be null or an object");
            }
        }
        if (plan.contains("s1_lut") && plan.at("s1_lut").is_object()) {
            if (!plan.at("s1_lut").contains("param_names") ||
                !plan.at("s1_lut").at("param_names").is_array()) {
                return Status::error("hist_plan.s1_lut.param_names must be an array");
            }
        }
        if (plan.contains("s2_linear") && plan.at("s2_linear").is_object()) {
            const auto& s2 = plan.at("s2_linear");
            if (!s2.contains("cipher_minus_ks") || !s2.at("cipher_minus_ks").is_boolean()) {
                return Status::error("hist_plan.s2_linear.cipher_minus_ks must be a boolean");
            }
        }
        if (plan.contains("s3") && plan.at("s3").is_object()) {
            const auto& s3 = plan.at("s3");
            for (const char* key : {"op_count", "max_stack", "slot_count"}) {
                if (!s3.contains(key) || !s3.at(key).is_number_unsigned()) {
                    return Status::error(std::string("hist_plan.s3.") + key +
                                         " must be an unsigned integer");
                }
            }
            if (!s3.contains("binds_index_i") || !s3.at("binds_index_i").is_boolean()) {
                return Status::error("hist_plan.s3.binds_index_i must be a boolean");
            }
        }
        // Soft-fall invariant: specialized false ⇒ emitted is S0 (safe search default).
        if (!plan.at("specialized").get<bool>() &&
            plan.at("emitted_strategy").get<std::string>() != "S0_bytecode") {
            return Status::error(
                "hist_plan: when specialized is false, emitted_strategy must be S0_bytecode");
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<nlohmann::json>
    load_plan(const std::filesystem::path& path, std::string_view expected_uri = {}) {
        std::ifstream in(path);
        if (!in) {
            return Status::error("Failed to open hist_plan.json: " + path.string());
        }
        nlohmann::json plan;
        try {
            in >> plan;
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid hist_plan.json: ") + ex.what());
        }
        Status v = validate_plan_json(plan, expected_uri);
        if (!v.ok()) {
            return v;
        }
        return plan;
    }

    /// Write `hist/hist_plan.json` and optional specialized sources under `artifact_dir`.
    [[nodiscard]] static Status write(const std::filesystem::path& artifact_dir,
                                      const Bundle& bundle) {
        Status v = validate_plan_json(bundle.plan_json());
        if (!v.ok()) {
            return v;
        }
        std::error_code ec;
        std::filesystem::create_directories(artifact_dir / "hist", ec);
        if (ec) {
            return Status::error("failed to create hist/: " + ec.message());
        }
        const std::filesystem::path plan_path = artifact_dir / "hist" / "hist_plan.json";
        {
            std::ofstream out(plan_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                return Status::error("Failed to write hist_plan.json: " + plan_path.string());
            }
            out << bundle.plan_json().dump(2) << '\n';
            if (!out) {
                return Status::error("Failed while writing hist_plan.json");
            }
        }
        if (bundle.write_sources()) {
            Status wh = write_text(artifact_dir / bundle.header_rel(), bundle.header_text());
            if (!wh.ok()) {
                return wh;
            }
            Status ws = write_text(artifact_dir / bundle.source_rel(), bundle.cu_text());
            if (!ws.ok()) {
                return ws;
            }
        }
        return Status::success();
    }

private:
    TheoryHistPlanWriter() = delete;

    [[nodiscard]] static std::optional<std::string>
    peak_tier_for(TheoryHistChi2Emit::Strategy emitted) {
        switch (emitted) {
        case TheoryHistChi2Emit::Strategy::S0Bytecode:
            return std::string{"T.theory.caesar_bytecode"};
        case TheoryHistChi2Emit::Strategy::S1Lut29:
            return std::string{"T.theory.s1_lut29"};
        case TheoryHistChi2Emit::Strategy::S2Uchar4Inline:
            return std::string{"T.theory.s2_linear"};
        case TheoryHistChi2Emit::Strategy::S3ScalarInline:
            return std::nullopt; // Spec row TBD
        case TheoryHistChi2Emit::Strategy::ShapeInline:
            return std::string{"T.theory.s1_lut29"};
        case TheoryHistChi2Emit::Strategy::ModuleLoaded:
            return std::string{"T.theory.caesar_bytecode"};
        }
        return std::nullopt;
    }

    [[nodiscard]] static Status write_text(const std::filesystem::path& path,
                                           const std::string& text) {
        if (path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(path.parent_path(), ec);
            if (ec) {
                return Status::error("failed to create parent for " + path.string() + ": " +
                                     ec.message());
            }
        }
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("Failed to write " + path.string());
        }
        out << text;
        if (!out) {
            return Status::error("Failed while writing " + path.string());
        }
        return Status::success();
    }
};

#endif // THEORY_HIST_PLAN_WRITER_HPP
