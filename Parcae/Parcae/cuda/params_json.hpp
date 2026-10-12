#ifndef PARAMS_JSON_HPP
#define PARAMS_JSON_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/math/primes.hpp"
#include "parcae/math/z29_matrix2.hpp"
#include "parcae/math/z29_matrix3.hpp"
#include "parcae/transform/grid_read_order.hpp"

#include "params.hpp"

#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

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

/// Host-owned compose recipe: stages + flat arenas for keyed / matrix / order stages.
///
/// - `key_arena`: Index29 key bytes (Vigenère, Beaufort, CTAK, PTAK, VDA, columnar)
/// - `matrix_arena`: row-major ℤ₂₉ matrix entries (Hill2 len 4, Hill3 len 9)
/// - `order_arena`: uint32 gather indices into row-major slots (grid-read families)
class ComposeParamsHost {
public:
    std::vector<ComposeStageParams> stages;
    std::vector<std::uint8_t> key_arena;
    std::vector<std::uint8_t> matrix_arena;
    std::vector<std::uint32_t> order_arena;
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

    /// Host hill matrix bytes (row-major ℤ₂₉). Length 4 (Hill2) or 9 (Hill3).
    class HillMatrixHost {
    public:
        std::vector<std::uint8_t> entries;
    };

    /// VDA primer key + lag/mode POD.
    class VariableDelayAutokeyHost {
    public:
        KeyParamsHost key;
        VariableDelayAutokeyParams vda{};
    };

    /// Grid-read dims/flags + precomputed gather order (+ columnar key when used).
    class GridReadHost {
    public:
        GridReadParams grid{};
        std::vector<std::uint32_t> order;
        KeyParamsHost columnar_key;
    };

    [[nodiscard]] static StatusOr<HillMatrixHost> hill2_from_json(const nlohmann::json& params) {
        return hill_matrix_from_json(params, 2, "hill_2");
    }

    [[nodiscard]] static StatusOr<HillMatrixHost> hill3_from_json(const nlohmann::json& params) {
        return hill_matrix_from_json(params, 3, "hill_3");
    }

    [[nodiscard]] static nlohmann::json hill_to_json(const HillMatrixHost& params) {
        nlohmann::json matrix = nlohmann::json::array();
        for (std::uint8_t v : params.entries) {
            matrix.push_back(v);
        }
        return nlohmann::json{{"matrix", std::move(matrix)}};
    }

    /// CTAK / PTAK primer (same wire shape as keyed Vigenère: `key_indices`).
    [[nodiscard]] static StatusOr<KeyParamsHost>
    ciphertext_autokey_from_json(const nlohmann::json& params) {
        return autokey_key_from_json(params, "ciphertext_autokey");
    }

    [[nodiscard]] static StatusOr<KeyParamsHost>
    plaintext_autokey_from_json(const nlohmann::json& params) {
        return autokey_key_from_json(params, "plaintext_autokey");
    }

    [[nodiscard]] static StatusOr<VariableDelayAutokeyHost>
    variable_delay_autokey_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("variable_delay_autokey params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error("variable_delay_autokey params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("variable_delay_autokey params.key_indices must be an array");
        }

        const bool has_lag = params.contains("lag");
        const bool has_lag_prime_index = params.contains("lag_prime_index");
        if (has_lag == has_lag_prime_index) {
            return Status::error(
                "variable_delay_autokey requires exactly one of lag or lag_prime_index");
        }

        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin" && it.key() != "lag" &&
                it.key() != "lag_prime_index" && it.key() != "mode") {
                return Status::error("variable_delay_autokey params contains unknown field");
            }
        }

        VariableDelayAutokeyHost out;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error("variable_delay_autokey key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error("variable_delay_autokey key_indices entries must be in 0..28");
            }
            out.key.key.push_back(static_cast<std::uint8_t>(raw));
        }
        if (out.key.key.empty()) {
            return Status::error("variable_delay_autokey key_indices must be non-empty");
        }
        out.key.sync_view();

        if (has_lag) {
            if (!params.at("lag").is_number_integer()) {
                return Status::error("variable_delay_autokey lag must be an integer");
            }
            const auto raw = params.at("lag").get<std::int64_t>();
            if (raw < 2 || !is_prime_u32(static_cast<std::uint32_t>(raw))) {
                return Status::error("variable_delay_autokey lag must be a prime >= 2");
            }
            out.vda.lag = static_cast<std::uint32_t>(raw);
        } else {
            if (!params.at("lag_prime_index").is_number_integer()) {
                return Status::error("variable_delay_autokey lag_prime_index must be an integer");
            }
            const auto raw = params.at("lag_prime_index").get<std::int64_t>();
            if (raw < 0) {
                return Status::error("variable_delay_autokey lag_prime_index must be non-negative");
            }
            StatusOr<std::uint64_t> prime = Primes::nth(static_cast<std::size_t>(raw));
            if (!prime.ok()) {
                return prime.status();
            }
            if (prime.value() > 0xffffffffull) {
                return Status::error("variable_delay_autokey lag overflows uint32");
            }
            out.vda.lag = static_cast<std::uint32_t>(prime.value());
        }

        out.vda.mode = CudaAutokeyMode::Ciphertext;
        if (params.contains("mode")) {
            if (!params.at("mode").is_string()) {
                return Status::error("variable_delay_autokey mode must be a string");
            }
            const std::string mode = params.at("mode").get<std::string>();
            if (mode == "ciphertext") {
                out.vda.mode = CudaAutokeyMode::Ciphertext;
            } else if (mode == "plaintext") {
                out.vda.mode = CudaAutokeyMode::Plaintext;
            } else {
                return Status::error(
                    "variable_delay_autokey mode must be ciphertext or plaintext");
            }
        }
        return out;
    }

    [[nodiscard]] static StatusOr<GridReadHost> spiral_read_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("spiral_read params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "cols" && it.key() != "spiral") {
                return Status::error("spiral_read params contains unknown field");
            }
        }
        StatusOr<std::size_t> rows = parse_positive_dim(params, "rows", "spiral_read");
        if (!rows.ok()) {
            return rows.status();
        }
        StatusOr<std::size_t> cols = parse_positive_dim(params, "cols", "spiral_read");
        if (!cols.ok()) {
            return cols.status();
        }

        bool outward = false;
        if (params.contains("spiral")) {
            if (!params.at("spiral").is_string()) {
                return Status::error("spiral_read spiral must be a string");
            }
            const std::string spiral = params.at("spiral").get<std::string>();
            if (spiral == "inward") {
                outward = false;
            } else if (spiral == "outward") {
                outward = true;
            } else {
                return Status::error("spiral_read spiral must be inward or outward");
            }
        }

        GridReadHost out;
        out.grid.rows = static_cast<std::uint32_t>(rows.value());
        out.grid.cols = static_cast<std::uint32_t>(cols.value());
        out.grid.spiral_outward = outward ? 1u : 0u;
        std::vector<std::size_t> order(rows.value() * cols.value());
        Status filled =
            outward ? GridReadOrder::fill_spiral_outward(order, rows.value(), cols.value())
                    : GridReadOrder::fill_spiral_inward(order, rows.value(), cols.value());
        if (!filled.ok()) {
            return filled;
        }
        Status copied = copy_order_u32(out.order, order);
        if (!copied.ok()) {
            return copied;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<GridReadHost>
    boustrophedon_read_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("boustrophedon_read params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "cols" && it.key() != "first_row") {
                return Status::error("boustrophedon_read params contains unknown field");
            }
        }
        StatusOr<std::size_t> rows = parse_positive_dim(params, "rows", "boustrophedon_read");
        if (!rows.ok()) {
            return rows.status();
        }
        StatusOr<std::size_t> cols = parse_positive_dim(params, "cols", "boustrophedon_read");
        if (!cols.ok()) {
            return cols.status();
        }

        bool first_row_ltr = true;
        if (params.contains("first_row")) {
            if (!params.at("first_row").is_string()) {
                return Status::error("boustrophedon_read first_row must be a string");
            }
            const std::string first = params.at("first_row").get<std::string>();
            if (first == "ltr") {
                first_row_ltr = true;
            } else if (first == "rtl") {
                first_row_ltr = false;
            } else {
                return Status::error("boustrophedon_read first_row must be ltr or rtl");
            }
        }

        GridReadHost out;
        out.grid.rows = static_cast<std::uint32_t>(rows.value());
        out.grid.cols = static_cast<std::uint32_t>(cols.value());
        out.grid.boustrophedon_rtl = first_row_ltr ? 0u : 1u;
        std::vector<std::size_t> order(rows.value() * cols.value());
        Status filled = GridReadOrder::fill_boustrophedon(order, rows.value(), cols.value(),
                                                          first_row_ltr);
        if (!filled.ok()) {
            return filled;
        }
        Status copied = copy_order_u32(out.order, order);
        if (!copied.ok()) {
            return copied;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<GridReadHost>
    diagonal_read_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("diagonal_read params must be an object");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "cols" && it.key() != "anti") {
                return Status::error("diagonal_read params contains unknown field");
            }
        }
        StatusOr<std::size_t> rows = parse_positive_dim(params, "rows", "diagonal_read");
        if (!rows.ok()) {
            return rows.status();
        }
        StatusOr<std::size_t> cols = parse_positive_dim(params, "cols", "diagonal_read");
        if (!cols.ok()) {
            return cols.status();
        }

        bool anti = false;
        if (params.contains("anti")) {
            if (!params.at("anti").is_boolean()) {
                return Status::error("diagonal_read anti must be a boolean");
            }
            anti = params.at("anti").get<bool>();
        }

        GridReadHost out;
        out.grid.rows = static_cast<std::uint32_t>(rows.value());
        out.grid.cols = static_cast<std::uint32_t>(cols.value());
        out.grid.diagonal_anti = anti ? 1u : 0u;
        std::vector<std::size_t> order(rows.value() * cols.value());
        Status filled =
            GridReadOrder::fill_diagonal(order, rows.value(), cols.value(), anti);
        if (!filled.ok()) {
            return filled;
        }
        Status copied = copy_order_u32(out.order, order);
        if (!copied.ok()) {
            return copied;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<GridReadHost>
    columnar_transposition_from_json(const nlohmann::json& params) {
        if (!params.is_object()) {
            return Status::error("columnar_transposition params must be an object");
        }
        if (!params.contains("rows")) {
            return Status::error("columnar_transposition params.rows is required");
        }
        if (!params.contains("key_indices")) {
            return Status::error("columnar_transposition params.key_indices is required");
        }
        if (!params.at("rows").is_number_integer()) {
            return Status::error("columnar_transposition rows must be an integer");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error("columnar_transposition key_indices must be an array");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "rows" && it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error("columnar_transposition params contains unknown field");
            }
        }

        const auto rows_raw = params.at("rows").get<std::int64_t>();
        if (rows_raw < 1) {
            return Status::error("columnar_transposition rows must be >= 1");
        }

        GridReadHost out;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error(
                    "columnar_transposition key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error(
                    "columnar_transposition key_indices entries must be in 0..28");
            }
            out.columnar_key.key.push_back(static_cast<std::uint8_t>(raw));
        }
        if (out.columnar_key.key.empty()) {
            return Status::error("columnar_transposition key_indices must be non-empty");
        }
        out.columnar_key.sync_view();

        const std::size_t rows = static_cast<std::size_t>(rows_raw);
        const std::size_t cols = out.columnar_key.key.size();
        out.grid.rows = static_cast<std::uint32_t>(rows);
        out.grid.cols = static_cast<std::uint32_t>(cols);
        std::vector<std::size_t> order(rows * cols);
        Status filled =
            GridReadOrder::fill_columnar(order, rows, cols, out.columnar_key.key);
        if (!filled.ok()) {
            return filled;
        }
        Status copied = copy_order_u32(out.order, order);
        if (!copied.ok()) {
            return copied;
        }
        return out;
    }

    [[nodiscard]] static StatusOr<ComposeParamsHost>
    compose_from_json(const nlohmann::json& params) {
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
                out.key_arena.insert(out.key_arena.end(), key.value().key.begin(),
                                     key.value().key.end());
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
            case CudaFamilyId::Hill2: {
                StatusOr<HillMatrixHost> matrix = hill2_from_json(stage_params);
                if (!matrix.ok()) {
                    return matrix.status();
                }
                append_matrix(out, pod, matrix.value());
                break;
            }
            case CudaFamilyId::Hill3: {
                StatusOr<HillMatrixHost> matrix = hill3_from_json(stage_params);
                if (!matrix.ok()) {
                    return matrix.status();
                }
                append_matrix(out, pod, matrix.value());
                break;
            }
            case CudaFamilyId::CiphertextAutokey: {
                StatusOr<KeyParamsHost> key = ciphertext_autokey_from_json(stage_params);
                if (!key.ok()) {
                    return key.status();
                }
                append_key(out, pod, key.value());
                break;
            }
            case CudaFamilyId::PlaintextAutokey: {
                StatusOr<KeyParamsHost> key = plaintext_autokey_from_json(stage_params);
                if (!key.ok()) {
                    return key.status();
                }
                append_key(out, pod, key.value());
                break;
            }
            case CudaFamilyId::VariableDelayAutokey: {
                StatusOr<VariableDelayAutokeyHost> vda =
                    variable_delay_autokey_from_json(stage_params);
                if (!vda.ok()) {
                    return vda.status();
                }
                pod.vda = vda.value().vda;
                append_key(out, pod, vda.value().key);
                break;
            }
            case CudaFamilyId::SpiralRead: {
                StatusOr<GridReadHost> grid = spiral_read_from_json(stage_params);
                if (!grid.ok()) {
                    return grid.status();
                }
                append_grid(out, pod, grid.value());
                break;
            }
            case CudaFamilyId::BoustrophedonRead: {
                StatusOr<GridReadHost> grid = boustrophedon_read_from_json(stage_params);
                if (!grid.ok()) {
                    return grid.status();
                }
                append_grid(out, pod, grid.value());
                break;
            }
            case CudaFamilyId::DiagonalRead: {
                StatusOr<GridReadHost> grid = diagonal_read_from_json(stage_params);
                if (!grid.ok()) {
                    return grid.status();
                }
                append_grid(out, pod, grid.value());
                break;
            }
            case CudaFamilyId::ColumnarTransposition: {
                StatusOr<GridReadHost> grid = columnar_transposition_from_json(stage_params);
                if (!grid.ok()) {
                    return grid.status();
                }
                append_key(out, pod, grid.value().columnar_key);
                append_grid(out, pod, grid.value());
                break;
            }
            }

            out.stages.push_back(pod);
        }

        return out;
    }

private:
    CudaParamsJson() = delete;

    [[nodiscard]] static StatusOr<CudaDir> dir_from_stage(const nlohmann::json& stage,
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

    [[nodiscard]] static bool is_prime_u32(std::uint32_t n) {
        if (n < 2) {
            return false;
        }
        const std::vector<std::uint64_t> sieved = Primes::sieve_upto(n);
        return !sieved.empty() && sieved.back() == n;
    }

    [[nodiscard]] static StatusOr<std::size_t>
    parse_positive_dim(const nlohmann::json& params, const char* key, const char* family) {
        if (!params.contains(key)) {
            return Status::error(std::string(family) + " params." + key + " is required");
        }
        if (!params.at(key).is_number_integer()) {
            return Status::error(std::string(family) + " params." + key + " must be an integer");
        }
        const auto raw = params.at(key).get<std::int64_t>();
        if (raw < 1) {
            return Status::error(std::string(family) + " params." + key + " must be >= 1");
        }
        return static_cast<std::size_t>(raw);
    }

    [[nodiscard]] static Status copy_order_u32(std::vector<std::uint32_t>& out,
                                               const std::vector<std::size_t>& order) {
        out.clear();
        out.reserve(order.size());
        for (std::size_t idx : order) {
            if (idx > 0xffffffffull) {
                return Status::error("CudaParamsJson: grid order index overflows uint32");
            }
            out.push_back(static_cast<std::uint32_t>(idx));
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<KeyParamsHost>
    autokey_key_from_json(const nlohmann::json& params, const char* family) {
        if (!params.is_object()) {
            return Status::error(std::string(family) + " params must be an object");
        }
        if (!params.contains("key_indices")) {
            return Status::error(std::string(family) + " params.key_indices is required");
        }
        if (!params.at("key_indices").is_array()) {
            return Status::error(std::string(family) + " params.key_indices must be an array");
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it.key() != "key_indices" && it.key() != "key_latin") {
                return Status::error(std::string(family) + " params contains unknown field");
            }
        }

        KeyParamsHost out;
        for (const nlohmann::json& item : params.at("key_indices")) {
            if (!item.is_number_integer()) {
                return Status::error(std::string(family) +
                                     " key_indices entries must be integers");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error(std::string(family) +
                                     " key_indices entries must be in 0..28");
            }
            out.key.push_back(static_cast<std::uint8_t>(raw));
        }
        if (out.key.empty()) {
            return Status::error(std::string(family) + " key_indices must be non-empty");
        }
        out.sync_view();
        return out;
    }

    [[nodiscard]] static StatusOr<HillMatrixHost>
    hill_matrix_from_json(const nlohmann::json& params, int dim, const char* family) {
        const std::size_t expected = static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim);
        if (!params.is_object()) {
            return Status::error(std::string(family) + " params must be an object");
        }
        if (!params.contains("matrix")) {
            return Status::error(std::string(family) + " params.matrix is required");
        }
        if (!params.at("matrix").is_array()) {
            return Status::error(std::string(family) + " params.matrix must be an array of " +
                                 std::to_string(expected) + " integers");
        }
        if (params.at("matrix").size() != expected) {
            return Status::error(std::string(family) + " params.matrix must contain exactly " +
                                 std::to_string(expected) + " entries");
        }
        if (params.size() != 1) {
            return Status::error(std::string(family) + " params may only contain matrix");
        }

        HillMatrixHost out;
        out.entries.reserve(expected);
        for (std::size_t i = 0; i < expected; ++i) {
            const nlohmann::json& item = params.at("matrix").at(i);
            if (!item.is_number_integer()) {
                return Status::error(std::string(family) + " params.matrix[" +
                                     std::to_string(i) + "] must be an integer");
            }
            const auto raw = item.get<std::int64_t>();
            if (raw < 0 || raw > 28) {
                return Status::error(std::string(family) + " params.matrix[" +
                                     std::to_string(i) + "] must be in 0..28");
            }
            out.entries.push_back(static_cast<std::uint8_t>(raw));
        }

        if (dim == 2) {
            std::array<Index29, 4> entries{};
            for (std::size_t i = 0; i < 4; ++i) {
                entries[i] = Index29{out.entries[i]};
            }
            const Z29Matrix2 matrix = Z29Matrix2::from_row_major(entries);
            if (matrix.det().value() == 0) {
                return Status::error(std::string(family) +
                                     " matrix must be invertible (det ≢ 0 mod 29)");
            }
        } else {
            std::array<Index29, 9> entries{};
            for (std::size_t i = 0; i < 9; ++i) {
                entries[i] = Index29{out.entries[i]};
            }
            const Z29Matrix3 matrix = Z29Matrix3::from_row_major(entries);
            if (matrix.det().value() == 0) {
                return Status::error(std::string(family) +
                                     " matrix must be invertible (det ≢ 0 mod 29)");
            }
        }
        return out;
    }

    static void append_key(ComposeParamsHost& host, ComposeStageParams& pod,
                           const KeyParamsHost& key) {
        pod.key_begin = static_cast<std::uint32_t>(host.key_arena.size());
        pod.key_len = static_cast<std::uint32_t>(key.key.size());
        host.key_arena.insert(host.key_arena.end(), key.key.begin(), key.key.end());
    }

    static void append_matrix(ComposeParamsHost& host, ComposeStageParams& pod,
                              const HillMatrixHost& matrix) {
        pod.matrix_begin = static_cast<std::uint32_t>(host.matrix_arena.size());
        pod.matrix_len = static_cast<std::uint32_t>(matrix.entries.size());
        host.matrix_arena.insert(host.matrix_arena.end(), matrix.entries.begin(),
                                 matrix.entries.end());
    }

    static void append_grid(ComposeParamsHost& host, ComposeStageParams& pod,
                            const GridReadHost& grid) {
        pod.grid = grid.grid;
        pod.order_begin = static_cast<std::uint32_t>(host.order_arena.size());
        pod.order_len = static_cast<std::uint32_t>(grid.order.size());
        host.order_arena.insert(host.order_arena.end(), grid.order.begin(), grid.order.end());
    }
};

#endif // PARAMS_JSON_HPP
