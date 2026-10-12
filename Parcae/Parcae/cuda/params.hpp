#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#include <cstdint>
#include <type_traits>

/// Device / SoA direction codes (`docs/architecture/cuda-abi.md`).
enum class CudaDir : std::uint8_t {
    Decrypt = 0,
    Encrypt = 1,
};

/// Catalog family id for compose stages and backend dispatch.
/// Values 0–7 are stable (pre-1.1.0); 8–16 are toolkit 1.0 catalog twins for
/// `v1.1.0` CUDA catalog parity (Hill / autokey / grid-read).
enum class CudaFamilyId : std::uint8_t {
    Identity = 0,
    Atbash = 1,
    Caesar = 2,
    Affine = 3,
    VigenereKey = 4,
    BeaufortKey = 5,
    TotientPrimeStream = 6,
    Compose = 7,
    Hill2 = 8,
    Hill3 = 9,
    CiphertextAutokey = 10,
    PlaintextAutokey = 11,
    VariableDelayAutokey = 12,
    SpiralRead = 13,
    BoustrophedonRead = 14,
    DiagonalRead = 15,
    ColumnarTransposition = 16,
};

class CudaDirUtil {
public:
    [[nodiscard]] static CudaDir from_transform_direction(TransformDirection direction) noexcept {
        return direction == TransformDirection::Encrypt ? CudaDir::Encrypt : CudaDir::Decrypt;
    }

    [[nodiscard]] static TransformDirection to_transform_direction(CudaDir direction) noexcept {
        return direction == CudaDir::Encrypt ? TransformDirection::Encrypt
                                             : TransformDirection::Decrypt;
    }

private:
    CudaDirUtil() = delete;
};

class CudaFamilyIdUtil {
public:
    [[nodiscard]] static StatusOr<CudaFamilyId> from_transform_id(const TransformId& id) {
        if (id == TransformId::identity()) {
            return CudaFamilyId::Identity;
        }
        if (id == TransformId::atbash()) {
            return CudaFamilyId::Atbash;
        }
        if (id == TransformId::caesar()) {
            return CudaFamilyId::Caesar;
        }
        if (id == TransformId::affine()) {
            return CudaFamilyId::Affine;
        }
        if (id == TransformId::vigenere_key()) {
            return CudaFamilyId::VigenereKey;
        }
        if (id == TransformId::beaufort_key()) {
            return CudaFamilyId::BeaufortKey;
        }
        if (id == TransformId::totient_prime_stream()) {
            return CudaFamilyId::TotientPrimeStream;
        }
        if (id == TransformId::compose()) {
            return CudaFamilyId::Compose;
        }
        if (id == TransformId::hill_2()) {
            return CudaFamilyId::Hill2;
        }
        if (id == TransformId::hill_3()) {
            return CudaFamilyId::Hill3;
        }
        if (id == TransformId::ciphertext_autokey()) {
            return CudaFamilyId::CiphertextAutokey;
        }
        if (id == TransformId::plaintext_autokey()) {
            return CudaFamilyId::PlaintextAutokey;
        }
        if (id == TransformId::variable_delay_autokey()) {
            return CudaFamilyId::VariableDelayAutokey;
        }
        if (id == TransformId::spiral_read()) {
            return CudaFamilyId::SpiralRead;
        }
        if (id == TransformId::boustrophedon_read()) {
            return CudaFamilyId::BoustrophedonRead;
        }
        if (id == TransformId::diagonal_read()) {
            return CudaFamilyId::DiagonalRead;
        }
        if (id == TransformId::columnar_transposition()) {
            return CudaFamilyId::ColumnarTransposition;
        }
        return Status::error("CudaFamilyIdUtil: unsupported transform_id");
    }

private:
    CudaFamilyIdUtil() = delete;
};

/// Empty params for identity / atbash.
class EmptyParams {
public:
    EmptyParams() = default;
};

/// Caesar: `params.shift` in 0..28.
class CaesarParams {
public:
    std::uint8_t shift = 0;
};

/// Affine: `a` in 1..28, `b` in 0..28.
class AffineParams {
public:
    std::uint8_t a = 1;
    std::uint8_t b = 0;
};

/// Keyed families: length in the POD; bytes live in a host/device key buffer.
class KeyParams {
public:
    const std::uint8_t* key_ptr = nullptr;
    std::uint32_t key_len = 0;
};

/// Totient: stream start index; shifts buffer is host-materialized separately.
class TotientParams {
public:
    std::uint32_t prime_start_index = 0;
};

/// VDA feedback mode (matches `VariableDelayAutokeyTransform::Mode`).
/// CTAK / PTAK stages ignore this field (key arena only).
enum class CudaAutokeyMode : std::uint8_t {
    Ciphertext = 0,
    Plaintext = 1,
};

/// Variable-delay autokey: prime lag ≥ 2 + ciphertext/plaintext feedback mode.
class VariableDelayAutokeyParams {
public:
    std::uint32_t lag = 2;
    CudaAutokeyMode mode = CudaAutokeyMode::Ciphertext;
};

/// Grid-read dimensions + option flags (spiral / boustrophedon / diagonal).
/// Gather/scatter order indices live in `ComposeParamsHost::order_arena` at
/// `[order_begin, order_begin + order_len)`. Columnar key bytes use `key_arena`.
class GridReadParams {
public:
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    /// `spiral_read`: 0 = inward (default), 1 = outward.
    std::uint8_t spiral_outward = 0;
    /// `boustrophedon_read`: 0 = first row L→R, 1 = first row R→L.
    std::uint8_t boustrophedon_rtl = 0;
    /// `diagonal_read`: 0 = main diagonals, 1 = anti-diagonals.
    std::uint8_t diagonal_anti = 0;
    std::uint8_t pad = 0;
};

/// One compose stage. Arena bytes referenced by offsets into `ComposeParamsHost`.
///
/// | Family | Arenas / PODs |
/// |--------|----------------|
/// | Vigenère / Beaufort / CTAK / PTAK / columnar key | `key_begin`/`key_len` → `key_arena` |
/// | Hill2 / Hill3 | `matrix_begin`/`matrix_len` (4 or 9) → `matrix_arena` |
/// | Spiral / boustrophedon / diagonal / columnar | `order_*` → `order_arena`; `grid` dims/flags |
/// | VDA | `key_*` + `vda.lag` / `vda.mode` |
class ComposeStageParams {
public:
    CudaFamilyId family = CudaFamilyId::Identity;
    CudaDir direction = CudaDir::Decrypt;
    CaesarParams caesar{};
    AffineParams affine{};
    TotientParams totient{};
    VariableDelayAutokeyParams vda{};
    GridReadParams grid{};
    std::uint32_t key_begin = 0;
    std::uint32_t key_len = 0;
    std::uint32_t matrix_begin = 0;
    std::uint32_t matrix_len = 0;
    std::uint32_t order_begin = 0;
    std::uint32_t order_len = 0;
};

static_assert(sizeof(CaesarParams) == 1, "CaesarParams must stay POD-sized");
static_assert(sizeof(AffineParams) == 2, "AffineParams must stay tightly packed");
static_assert(sizeof(VariableDelayAutokeyParams) == 8, "VDA params must stay tightly packed");
static_assert(sizeof(GridReadParams) == 12, "GridReadParams must stay tightly packed");
static_assert(std::is_trivially_copyable_v<CaesarParams>);
static_assert(std::is_trivially_copyable_v<AffineParams>);
static_assert(std::is_trivially_copyable_v<KeyParams>);
static_assert(std::is_trivially_copyable_v<TotientParams>);
static_assert(std::is_trivially_copyable_v<VariableDelayAutokeyParams>);
static_assert(std::is_trivially_copyable_v<GridReadParams>);
static_assert(std::is_trivially_copyable_v<ComposeStageParams>);

#endif // PARAMS_HPP
