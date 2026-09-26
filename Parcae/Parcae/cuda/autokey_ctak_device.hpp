#ifndef AUTOKEY_CTAK_DEVICE_HPP
#define AUTOKEY_CTAK_DEVICE_HPP

#include "hist_fast.hpp"
#include "z29_device.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_CTAK_HD __host__ __device__
#else
#define PARCAE_CTAK_HD
#endif

/// Dense ciphertext-autokey (CTAK) helpers shared by `DeepScoreBatch` hist and
/// `CiphertextAutokeyKernel`. Matches CPU `CiphertextAutokeyTransform` with empty skips:
/// primer while `t < L`, else key = ciphertext `in[t - L]`; decrypt `out = in - key`.
class AutokeyCtakDevice {
public:
    /// Key symbol at absolute index `t` for dense CTAK **decrypt** (ciphertext feedback).
    [[nodiscard]] PARCAE_CTAK_HD static std::uint8_t
    decrypt_key(const std::uint8_t* cipher, const std::uint8_t* primer, std::uint32_t primer_len,
                std::size_t t) noexcept {
        if (static_cast<std::uint32_t>(t) < primer_len) {
            return primer[t];
        }
        return cipher[t - static_cast<std::size_t>(primer_len)];
    }

    /// Dense CTAK decrypt of one symbol: `cipher - key` (mod 29).
    [[nodiscard]] PARCAE_CTAK_HD static std::uint8_t decrypt_symbol(std::uint8_t cipher,
                                                                    std::uint8_t key) noexcept {
        return HistFast::dec_sub(cipher, key);
    }

    /// Dense CTAK encrypt of one symbol: `plain + key` (mod 29).
    [[nodiscard]] PARCAE_CTAK_HD static std::uint8_t encrypt_symbol(std::uint8_t plain,
                                                                    std::uint8_t key) noexcept {
        return Z29Device::add(plain, key);
    }

private:
    AutokeyCtakDevice() = delete;
};

#undef PARCAE_CTAK_HD

#endif // AUTOKEY_CTAK_DEVICE_HPP
