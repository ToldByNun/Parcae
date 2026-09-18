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
enum class CudaFamilyId : std::uint8_t {
    Identity = 0,
    Atbash = 1,
    Caesar = 2,
    Affine = 3,
    VigenereKey = 4,
    BeaufortKey = 5,
    TotientPrimeStream = 6,
    Compose = 7,
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

/// One compose stage. Key bytes referenced by offset into `ComposeParamsHost` arena.
class ComposeStageParams {
public:
    CudaFamilyId family = CudaFamilyId::Identity;
    CudaDir direction = CudaDir::Decrypt;
    CaesarParams caesar{};
    AffineParams affine{};
    TotientParams totient{};
    std::uint32_t key_begin = 0;
    std::uint32_t key_len = 0;
};

static_assert(sizeof(CaesarParams) == 1, "CaesarParams must stay POD-sized");
static_assert(sizeof(AffineParams) == 2, "AffineParams must stay tightly packed");
static_assert(std::is_trivially_copyable_v<CaesarParams>);
static_assert(std::is_trivially_copyable_v<AffineParams>);
static_assert(std::is_trivially_copyable_v<KeyParams>);
static_assert(std::is_trivially_copyable_v<TotientParams>);
static_assert(std::is_trivially_copyable_v<ComposeStageParams>);

#endif // PARAMS_HPP
