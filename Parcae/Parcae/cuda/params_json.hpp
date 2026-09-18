#ifndef PARAMS_JSON_HPP
#define PARAMS_JSON_HPP

#include "params.hpp"

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// Host-side key bytes (Index29 values as uint8) plus KeyParams view.
class KeyParamsHost {
public:
    std::vector<std::uint8_t> key;
    KeyParams view{};

    void sync_view() noexcept {
        view.key_ptr = key.empty() ? nullptr : key.data();
        view.key_len = static_cast<std::uint32_t>(key.size());
    }
};

/// Host-owned compose recipe: stages + flat key arena for keyed stages.
class ComposeParamsHost {
public:
    std::vector<ComposeStageParams> stages;
    std::vector<std::uint8_t> key_arena;
};

/// Host JSON ↔ POD converters for CUDA twin params.
class CudaParamsJson {
public:
    [[nodiscard]] static StatusOr<EmptyParams> empty_from_json(const nlohmann::json& params) {
        if (params.is_null()) {
            return EmptyParams{};
        }
        if (!params.is_object()) {
            return Status::error("empty params must be an object");
        }
        if (!params.empty()) {
            return Status::error("empty params must be empty");
        }
        return EmptyParams{};
    }

    [[nodiscard]] static StatusOr<CaesarParams> caesar_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("caesar params must be an object");
        }
        if (!params.contains("shift") || !params.at("shift").is_number_integer()) {
            return Status::error("caesar params.shift must be an integer");
        }
        if (params.size() != 1) {
            return Status::error("caesar params may only contain shift");
        }
        const auto raw = params.at("shift").get<std::int64_t>();
        if (raw < 0 || raw > 28) {
            return Status::error("caesar params.shift must be in 0..28");
        }
        CaesarParams out;
        out.shift = static_cast<std::uint8_t>(raw);
        return out;
    }

    [[nodiscard]] static nlohmann::json caesar_to_json(const CaesarParams& params) {
        return nlohmann::json{{"shift", params.shift}};
    }

    [[nodiscard]] static StatusOr<AffineParams> affine_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("affine params must be an object");
        }
        if (params.size() != 2) {
            return Status::error("affine params may only contain a and b");
        }
        if (!params.contains("a") || !params.at("a").is_number_integer()) {
            return Status::error("affine params.a must be an integer");
        }
        if (!params.contains("b") || !params.at("b").is_number_integer()) {
            return Status::error("affine params.b must be an integer");
        }
        const auto a = params.at("a").get<std::int64_t>();
        const auto b = params.at("b").get<std::int64_t>();
        if (a < 1 || a > 28) {
            return Status::error("affine params.a must be in 1..28");
        }
        if (b < 0 || b > 28) {
            return Status::error("affine params.b must be in 0..28");
        }
        AffineParams out;
        out.a = static_cast<std::uint8_t>(a);
        out.b = static_cast<std::uint8_t>(b);
        return out;
    }

    [[nodiscard]] static nlohmann::json affine_to_json(const AffineParams& params) {
        return nlohmann::json{{"a", params.a}, {"b", params.b}};
    }

    [[nodiscard]] static StatusOr<KeyParamsHost> key_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("key params must be an object");
        }
        if (!params.contains("key_indices") || !params.at("key_indices").is_array()) {
            return Status::error("key params.key_indices must be an array");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("key params contains unknown field");
            }
        }

        KeyParamsHost out;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("key_indices entries must be in 0..28");
            }
            out.key.push_back(static_cast<std::uint8_t>(raw));
        }
        if (out.key.empty()) {
            return Status::error("key_indices must be non-empty");
        }
        out.sync_view();
        return out;
    }

    [[nodiscard]] static nlohmann::json key_to_json(const KeyParamsHost& params) {
        nlohmann::json indices = nlohmann::json::array();
        for (std::uint8_t v : params.key) {
            indices.push_back(v);
        }
        return nlohmann::json{{"key_indices", std::move(indices)}};
    }

    [[nodiscard]] static StatusOr<TotientParams> totient_from_json(const nlohmann::json& params) {
        if (params.is_null()) {
            return TotientParams{};
        }
        if (!params.is_object()) {
            return Status::error("totient params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "prime_start_index" && it.key() != "shift_mode") {
                return Status::error("totient params contains unknown field");
            }
        }
        if (params.contains("shift_mode")) {
            if (!params.at("shift_mode").is_string() ||
                params.at("shift_mode").get<std::string>() != "prime_minus_one_mod_29") {
                return Status::error("totient shift_mode must be prime_minus_one_mod_29");
            }
        }
        TotientParams out{};
        if (params.contains("prime_start_index")) {
            if (!params.at("prime_start_index").is_number_integer()) {
                return Status::error("totient prime_start_index must be an integer");
            }
            const auto raw = params.at("prime_start_index").get<std::int64_t>();
            if (raw < 0) {
                return Status::error("totient prime_start_index must be non-negative");
            }
            out.prime_start_index = static_cast<std::uint32_t>(raw);
        }
        return out;
    }

    [[nodiscard]] static nlohmann::json totient_to_json(const TotientParams& params) {
        return nlohmann::json{{"prime_start_index", params.prime_start_index}};
    }

    [[nodiscard]] static StatusOr<ComposeParamsHost> compose_from_json(
        const nlohmann::json& params) {
        if (!params.is_object() || !params.contains("stages") || !params.at("stages").is_array()) {
            return Status::error("compose params.stages must be an array");
        }
        if (params.size() != 1) {
            return Status::error("compose params may only contain stages");
        }
        const nlohmann::json& stages = params.at("stages");
        if (stages.empty()) {
            return Status::error("compose params.stages must be non-empty");
        }

        ComposeParamsHost out;
        out.stages.reserve(stages.size());

        for (const nlohmann::json& stage : stages) {
            if (!stage.is_object() || !stage.contains("transform_id")) {
                return Status::error("compose stage requires transform_id");
            }
            if (!stage.at("transform_id").is_string()) {
                return Status::error("compose stage transform_id must be a string");
            }
            StatusOr<TransformId> id =
                TransformId::from_string(stage.at("transform_id").get<std::string>());
            if (!id.ok()) {
                return id.status();
            }
            StatusOr<CudaFamilyId> family = CudaFamilyIdUtil::from_transform_id(id.value());
            if (!family.ok()) {
                return family.status();
            }
            if (family.value() == CudaFamilyId::Compose) {
                return Status::error("compose params_json does not flatten nested compose");
            }

            StatusOr<CudaDir> direction = dir_from_stage(stage, CudaDir::Decrypt);
            if (!direction.ok()) {
                return direction.status();
            }

            const nlohmann::json stage_params =
                stage.contains("params") ? stage.at("params") : nlohmann::json::object();

            ComposeStageParams pod;
            pod.family = family.value();
            pod.direction = direction.value();

            switch (family.value()) {
                case CudaFamilyId::Identity:
                case CudaFamilyId::Atbash: {
                    StatusOr<EmptyParams> empty = empty_from_json(stage_params);
                    if (!empty.ok()) {
                        return empty.status();
                    }
                    break;
                }
                case CudaFamilyId::Caesar: {
                    StatusOr<CaesarParams> caesar = caesar_from_json(stage_params);
                    if (!caesar.ok()) {
                        return caesar.status();
                    }
                    pod.caesar = caesar.value();
                    break;
                }
                case CudaFamilyId::Affine: {
                    StatusOr<AffineParams> affine = affine_from_json(stage_params);
                    if (!affine.ok()) {
                        return affine.status();
                    }
                    pod.affine = affine.value();
                    break;
                }
                case CudaFamilyId::VigenereKey:
                case CudaFamilyId::BeaufortKey: {
                    StatusOr<KeyParamsHost> key = key_from_json(stage_params);
                    if (!key.ok()) {
                        return key.status();
                    }
                    pod.key_begin = static_cast<std::uint32_t>(out.key_arena.size());
                    pod.key_len = static_cast<std::uint32_t>(key.value().key.size());
                    out.key_arena.insert(
                        out.key_arena.end(), key.value().key.begin(), key.value().key.end());
                    break;
                }
                case CudaFamilyId::TotientPrimeStream: {
                    StatusOr<TotientParams> totient = totient_from_json(stage_params);
                    if (!totient.ok()) {
                        return totient.status();
                    }
                    pod.totient = totient.value();
                    break;
                }
                case CudaFamilyId::Compose:
                    return Status::error("compose nesting not expanded in params_json");
            }

            out.stages.push_back(pod);
        }

        return out;
    }

private:
    CudaParamsJson() = delete;

    [[nodiscard]] static StatusOr<CudaDir> dir_from_stage(
        const nlohmann::json& stage,
        CudaDir default_dir) {
        if (!stage.contains("direction")) {
            return default_dir;
        }
        if (!stage.at("direction").is_string()) {
            return Status::error("compose stage direction must be a string");
        }
        StatusOr<TransformDirection> parsed =
            TransformDirectionUtil::from_string(stage.at("direction").get<std::string>());
        if (!parsed.ok()) {
            return parsed.status();
        }
        return CudaDirUtil::from_transform_direction(parsed.value());
    }
};

#endif // PARAMS_JSON_HPP
