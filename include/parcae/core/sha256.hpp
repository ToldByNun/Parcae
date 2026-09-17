#ifndef SHA256_HPP
#define SHA256_HPP

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

/// Portable SHA-256 (FIPS 180-4). Digests are lowercase hex.
class Sha256 {
public:
    [[nodiscard]] static std::string hex_digest(std::string_view data) {
        Hasher hasher;
        hasher.update(reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
        const std::array<std::uint8_t, 32> digest = hasher.finalize();
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out(64, '0');
        for (std::size_t i = 0; i < digest.size(); ++i) {
            out[2 * i] = kHex[(digest[i] >> 4) & 0xF];
            out[2 * i + 1] = kHex[digest[i] & 0xF];
        }
        return out;
    }

private:
    class Hasher {
    public:
        Hasher() {
            state_ = {
                0x6a09e667u,
                0xbb67ae85u,
                0x3c6ef372u,
                0xa54ff53au,
                0x510e527fu,
                0x9b05688cu,
                0x1f83d9abu,
                0x5be0cd19u,
            };
        }

        void update(const std::uint8_t* data, std::size_t len) {
            for (std::size_t i = 0; i < len; ++i) {
                buffer_[buffer_len_++] = data[i];
                if (buffer_len_ == 64) {
                    transform(buffer_.data());
                    bit_len_ += 512;
                    buffer_len_ = 0;
                }
            }
        }

        [[nodiscard]] std::array<std::uint8_t, 32> finalize() {
            const std::uint64_t total_bits = bit_len_ + static_cast<std::uint64_t>(buffer_len_) * 8;
            buffer_[buffer_len_++] = 0x80;
            if (buffer_len_ > 56) {
                while (buffer_len_ < 64) {
                    buffer_[buffer_len_++] = 0;
                }
                transform(buffer_.data());
                buffer_len_ = 0;
            }
            while (buffer_len_ < 56) {
                buffer_[buffer_len_++] = 0;
            }
            for (int i = 7; i >= 0; --i) {
                buffer_[buffer_len_++] = static_cast<std::uint8_t>((total_bits >> (8 * i)) & 0xFF);
            }
            transform(buffer_.data());

            std::array<std::uint8_t, 32> out{};
            for (std::size_t i = 0; i < state_.size(); ++i) {
                out[4 * i] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFF);
                out[4 * i + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFF);
                out[4 * i + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFF);
                out[4 * i + 3] = static_cast<std::uint8_t>(state_[i] & 0xFF);
            }
            return out;
        }

    private:
        static constexpr std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept {
            return (x >> n) | (x << (32 - n));
        }

        void transform(const std::uint8_t* chunk) {
            static constexpr std::uint32_t k[64] = {
                0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
                0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
                0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
                0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
                0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
                0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
                0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
                0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
                0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
                0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
                0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
            };

            std::uint32_t w[64];
            for (std::size_t i = 0; i < 16; ++i) {
                w[i] = (static_cast<std::uint32_t>(chunk[4 * i]) << 24) |
                       (static_cast<std::uint32_t>(chunk[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(chunk[4 * i + 2]) << 8) |
                       static_cast<std::uint32_t>(chunk[4 * i + 3]);
            }
            for (std::size_t i = 16; i < 64; ++i) {
                const std::uint32_t s0 =
                    rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const std::uint32_t s1 =
                    rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            std::uint32_t a = state_[0];
            std::uint32_t b = state_[1];
            std::uint32_t c = state_[2];
            std::uint32_t d = state_[3];
            std::uint32_t e = state_[4];
            std::uint32_t f = state_[5];
            std::uint32_t g = state_[6];
            std::uint32_t h = state_[7];

            for (std::size_t i = 0; i < 64; ++i) {
                const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                const std::uint32_t ch = (e & f) ^ ((~e) & g);
                const std::uint32_t temp1 = h + s1 + ch + k[i] + w[i];
                const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const std::uint32_t temp2 = s0 + maj;

                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            state_[0] += a;
            state_[1] += b;
            state_[2] += c;
            state_[3] += d;
            state_[4] += e;
            state_[5] += f;
            state_[6] += g;
            state_[7] += h;
        }

        std::array<std::uint32_t, 8> state_{};
        std::array<std::uint8_t, 64> buffer_{};
        std::size_t buffer_len_ = 0;
        std::uint64_t bit_len_ = 0;
    };
};

#endif // SHA256_HPP
