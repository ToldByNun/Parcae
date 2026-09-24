#ifndef THEORY_ENVELOPE_BRIDGE_HPP
#define THEORY_ENVELOPE_BRIDGE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/tool/transform_envelope.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

/// Theory-aware envelope bridge (docs/spec/theory-artifact.md § Envelope bridge).
///
/// `envelope.json` MAY be either:
///   1. a frozen-catalog TransformEnvelope (`caesar`, `affine`, …), or
///   2. a documented extension whose `transform_id` is a
///      `parcae://theories/<name>@<ver>` URI.
///
/// Catalog-only consumers MUST call `Envelope::to_catalog_envelope()` — theory
/// URIs fail clearly (no silent no-op decode).
class TheoryEnvelopeBridge {
public:
    enum class Kind : std::uint8_t {
        Catalog = 0,
        Theory,
    };

    class Envelope {
    public:
        Envelope(Kind kind, std::string transform_id, TransformDirection direction,
                 nlohmann::json params, InterruptPolicy interrupt,
                 std::optional<TheoryUri> theory_uri = std::nullopt)
            : kind_(kind), transform_id_(std::move(transform_id)), direction_(direction),
              params_(std::move(params)), interrupt_(std::move(interrupt)),
              theory_uri_(std::move(theory_uri)) {}

        [[nodiscard]] Kind kind() const noexcept { return kind_; }

        [[nodiscard]] bool is_catalog() const noexcept { return kind_ == Kind::Catalog; }

        [[nodiscard]] bool is_theory() const noexcept { return kind_ == Kind::Theory; }

        [[nodiscard]] const std::string& transform_id() const noexcept { return transform_id_; }

        [[nodiscard]] TransformDirection direction() const noexcept { return direction_; }

        [[nodiscard]] const nlohmann::json& params() const noexcept { return params_; }

        [[nodiscard]] const InterruptPolicy& interrupt() const noexcept { return interrupt_; }

        [[nodiscard]] const std::optional<TheoryUri>& theory_uri() const noexcept {
            return theory_uri_;
        }

        /// Lower to a frozen-catalog TransformEnvelope. Theory URIs fail loudly.
        [[nodiscard]] StatusOr<TransformEnvelope> to_catalog_envelope() const {
            if (kind_ == Kind::Theory) {
                return Status::error(
                    "envelope transform_id is a theory URI and cannot be lowered to the "
                    "frozen catalog (use TheoryDispatch): " +
                    transform_id_);
            }
            StatusOr<TransformId> id = TransformId::from_string(transform_id_);
            if (!id.ok()) {
                return Status::error("envelope transform_id is not a catalog id: " + transform_id_ +
                                     " (" + id.status().message() + ")");
            }
            return TransformEnvelope{id.value(), direction_, params_, interrupt_};
        }

        [[nodiscard]] nlohmann::json to_json() const {
            nlohmann::json root{
                {"transform_id", transform_id_},
                {"direction", std::string(TransformDirectionUtil::to_string(direction_))},
                {"params", params_},
            };
            if (!interrupt_.skip_indices().empty()) {
                root["interrupt"] = interrupt_.to_json();
            }
            return root;
        }

    private:
        Kind kind_ = Kind::Catalog;
        std::string transform_id_;
        TransformDirection direction_ = TransformDirection::Decrypt;
        nlohmann::json params_ = nlohmann::json::object();
        InterruptPolicy interrupt_ = InterruptPolicy::none();
        std::optional<TheoryUri> theory_uri_;
    };

    /// Parse catalog or theory-URI envelope JSON.
    [[nodiscard]] static StatusOr<Envelope> from_json(const nlohmann::json& root) {
        if (!root.is_object()) {
            return Status::error("TheoryEnvelope must be a JSON object");
        }
        if (!root.contains("transform_id") || !root.at("transform_id").is_string()) {
            return Status::error("TheoryEnvelope.transform_id is required");
        }
        const std::string id_text = root.at("transform_id").get<std::string>();

        TransformDirection direction = TransformDirection::Decrypt;
        if (root.contains("direction")) {
            if (!root.at("direction").is_string()) {
                return Status::error("TheoryEnvelope.direction must be a string");
            }
            StatusOr<TransformDirection> parsed =
                TransformDirectionUtil::from_string(root.at("direction").get<std::string>());
            if (!parsed.ok()) {
                return parsed.status();
            }
            direction = parsed.value();
        }

        nlohmann::json params = nlohmann::json::object();
        if (root.contains("params")) {
            if (!root.at("params").is_object()) {
                return Status::error("TheoryEnvelope.params must be an object");
            }
            params = root.at("params");
        }

        InterruptPolicy interrupt = InterruptPolicy::none();
        if (root.contains("interrupt")) {
            StatusOr<InterruptPolicy> parsed = InterruptPolicy::from_json(root.at("interrupt"));
            if (!parsed.ok()) {
                return parsed.status();
            }
            interrupt = std::move(parsed.value());
        }

        // Theory URI extension (parcae://theories/<name>@<ver>).
        constexpr std::string_view kUriScheme = "parcae://";
        if (id_text.size() >= kUriScheme.size() &&
            id_text.compare(0, kUriScheme.size(), kUriScheme) == 0) {
            StatusOr<TheoryUri> uri = TheoryUri::parse(id_text);
            if (!uri.ok()) {
                return Status::error(
                    "TheoryEnvelope.transform_id looks like a URI but is not a valid "
                    "theory URI: " +
                    uri.status().message());
            }
            return Envelope{
                Kind::Theory,          id_text, direction, std::move(params), std::move(interrupt),
                std::move(uri.value())};
        }

        // Frozen catalog id.
        StatusOr<TransformId> catalog = TransformId::from_string(id_text);
        if (!catalog.ok()) {
            return Status::error("TheoryEnvelope.transform_id is neither a catalog id nor a "
                                 "parcae://theories/ URI: " +
                                 id_text);
        }
        return Envelope{Kind::Catalog,     catalog.value().str(), direction,
                        std::move(params), std::move(interrupt),  std::nullopt};
    }

    [[nodiscard]] static StatusOr<Envelope> from_string(const std::string& json_text) {
        nlohmann::json root;
        try {
            root = nlohmann::json::parse(json_text);
        } catch (const nlohmann::json::exception& ex) {
            return Status::error(std::string("Invalid TheoryEnvelope JSON: ") + ex.what());
        }
        return from_json(root);
    }

    [[nodiscard]] static StatusOr<Envelope> load(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("failed to open envelope: " + path.string());
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        if (!in && !in.eof()) {
            return Status::error("failed while reading envelope: " + path.string());
        }
        return from_string(ss.str());
    }

    [[nodiscard]] static Status write(const std::filesystem::path& path, const Envelope& envelope) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("failed to write envelope: " + path.string());
        }
        out << envelope.to_json().dump(2) << '\n';
        if (!out) {
            return Status::error("failed while writing envelope: " + path.string());
        }
        return Status::success();
    }

    /// Default compile-time template: `transform_id` = artifact URI, params at
    /// each declared param's `min` (placeholder binding for sweep/dispatch).
    [[nodiscard]] static StatusOr<Envelope>
    template_for(const TheoryArtifact& artifact,
                 TransformDirection direction = TransformDirection::Decrypt) {
        nlohmann::json params = nlohmann::json::object();
        for (const TheoryArtifact::Param& p : artifact.params()) {
            params[p.name()] = p.min();
        }
        return Envelope{Kind::Theory,      artifact.uri().to_string(), direction,
                        std::move(params), InterruptPolicy::none(),    artifact.uri()};
    }

    /// Validate that a loaded envelope is consistent with a compiled artifact
    /// (theory URI must match; catalog envelopes are accepted as-is).
    [[nodiscard]] static Status check_against_artifact(const Envelope& envelope,
                                                       const TheoryArtifact& artifact) {
        if (envelope.is_catalog()) {
            StatusOr<TransformEnvelope> lowered = envelope.to_catalog_envelope();
            if (!lowered.ok()) {
                return lowered.status();
            }
            return Status::success();
        }
        if (!envelope.theory_uri().has_value()) {
            return Status::error("theory envelope missing parsed TheoryUri");
        }
        if (envelope.theory_uri()->to_string() != artifact.uri().to_string()) {
            return Status::error("envelope theory URI does not match artifact URI: envelope=" +
                                 envelope.theory_uri()->to_string() +
                                 " artifact=" + artifact.uri().to_string());
        }
        // Every declared artifact param MUST appear in params (extra keys allowed
        // for forward-compat metadata; unknown required names fail).
        for (const TheoryArtifact::Param& p : artifact.params()) {
            if (!envelope.params().contains(p.name())) {
                return Status::error("envelope params missing theory param '" + p.name() + "'");
            }
            const nlohmann::json& v = envelope.params().at(p.name());
            if (!v.is_number_integer()) {
                return Status::error("envelope params." + p.name() + " must be an integer");
            }
            const std::int64_t iv = v.get<std::int64_t>();
            if (iv < p.min() || iv > p.max()) {
                return Status::error("envelope params." + p.name() + " out of declared domain [" +
                                     std::to_string(p.min()) + "," + std::to_string(p.max()) + "]");
            }
        }
        return Status::success();
    }

private:
    TheoryEnvelopeBridge() = delete;
};

#endif // THEORY_ENVELOPE_BRIDGE_HPP
