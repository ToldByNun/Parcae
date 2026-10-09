#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "parcae/core/index29.hpp"
#include "parcae/core/z29.hpp"
#include "parcae/score/chi2_english_gp.hpp"
#include "parcae/score/expected_frequency_loader.hpp"
#include "parcae/score/hist_alphabet_map.hpp"
#include "parcae/transform/affine_transform.hpp"
#include "parcae/transform/atbash_transform.hpp"
#include "parcae/transform/caesar_transform.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/beaufort_key_transform.hpp"
#include "parcae/transform/vigenere_key_transform.hpp"

#include <array>
#include <cstdint>
#include <numeric>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

class HistAlphabetMapTestSupport {
public:
    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(const std::vector<Index29>& indices) {
        std::vector<std::uint8_t> out(indices.size());
        for (std::size_t i = 0; i < indices.size(); ++i) {
            out[i] = indices[i].value();
        }
        return out;
    }

    [[nodiscard]] static HistAlphabetMap::Hist
    hist_from_indices(const std::vector<Index29>& indices) {
        HistAlphabetMap::Hist hist{};
        for (const Index29 idx : indices) {
            ++hist[idx.value()];
        }
        return hist;
    }

    [[nodiscard]] static std::vector<Index29>
    make_plain(std::size_t n, std::uint8_t seed_mod = 0) {
        std::vector<Index29> plain;
        plain.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            plain.push_back(Index29{static_cast<std::uint8_t>((i + seed_mod) % 29u)});
        }
        return plain;
    }

    [[nodiscard]] static std::array<float, HistAlphabetMap::bigram_bins> make_ll_table() {
        std::array<float, HistAlphabetMap::bigram_bins> ll{};
        for (std::size_t i = 0; i < ll.size(); ++i) {
            // Distinct, non-zero values so rotation cannot accidentally cancel.
            ll[i] = static_cast<float>(0.01 + 0.001 * static_cast<double>(i));
        }
        return ll;
    }

private:
    HistAlphabetMapTestSupport() = delete;
};

TEST_CASE("HistAlphabetMap rotate decrypt matches stream Caesar decrypt hist",
          "[score][hist_map][caesar]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(200);
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 11}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    const auto cipher_bytes = HistAlphabetMapTestSupport::to_bytes(cipher.value());
    StatusOr<HistAlphabetMap::Hist> H = HistAlphabetMap::count_stream_hist(cipher_bytes);
    REQUIRE(H.ok());
    REQUIRE(HistAlphabetMap::hist_total(H.value()) == cipher_bytes.size());

    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        StatusOr<std::vector<Index29>> decoded = CaesarTransform{}.apply(
            cipher.value(), nlohmann::json{{"shift", shift}}, TransformDirection::Decrypt);
        REQUIRE(decoded.ok());
        const HistAlphabetMap::Hist expected =
            HistAlphabetMapTestSupport::hist_from_indices(decoded.value());

        StatusOr<HistAlphabetMap::Hist> remapped =
            HistAlphabetMap::rotate_decrypt(H.value(), shift);
        REQUIRE(remapped.ok());
        REQUIRE(remapped.value() == expected);
    }
}

TEST_CASE("HistAlphabetMap rotate encrypt is inverse of rotate decrypt on bins",
          "[score][hist_map][caesar]") {
    HistAlphabetMap::Hist H{};
    for (std::size_t i = 0; i < 29; ++i) {
        H[i] = static_cast<std::uint32_t>(i + 1u);
    }
    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        StatusOr<HistAlphabetMap::Hist> dec = HistAlphabetMap::rotate_decrypt(H, shift);
        REQUIRE(dec.ok());
        StatusOr<HistAlphabetMap::Hist> back = HistAlphabetMap::rotate_encrypt(dec.value(), shift);
        REQUIRE(back.ok());
        REQUIRE(back.value() == H);
    }
}

TEST_CASE("HistAlphabetMap mirror_atbash matches stream Atbash hist",
          "[score][hist_map][atbash]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(150, 3);
    StatusOr<std::vector<Index29>> cipher =
        AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(cipher.ok());

    StatusOr<HistAlphabetMap::Hist> H =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(cipher.value()));
    REQUIRE(H.ok());

    StatusOr<std::vector<Index29>> decoded =
        AtbashTransform{}.apply(cipher.value(), nlohmann::json::object(),
                                TransformDirection::Decrypt);
    REQUIRE(decoded.ok());
    REQUIRE(HistAlphabetMap::mirror_atbash(H.value()) ==
            HistAlphabetMapTestSupport::hist_from_indices(decoded.value()));
}

TEST_CASE("HistAlphabetMap affine permute matches stream Affine decrypt hist",
          "[score][hist_map][affine]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(180, 5);
    constexpr std::uint8_t a = 7;
    constexpr std::uint8_t b = 13;
    StatusOr<std::vector<Index29>> cipher = AffineTransform{}.apply(
        plain, nlohmann::json{{"a", a}, {"b", b}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<HistAlphabetMap::Hist> H =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(cipher.value()));
    REQUIRE(H.ok());

    StatusOr<std::vector<Index29>> decoded = AffineTransform{}.apply(
        cipher.value(), nlohmann::json{{"a", a}, {"b", b}}, TransformDirection::Decrypt);
    REQUIRE(decoded.ok());

    StatusOr<HistAlphabetMap::Hist> remapped =
        HistAlphabetMap::permute_affine_decrypt(H.value(), a, b);
    REQUIRE(remapped.ok());
    REQUIRE(remapped.value() == HistAlphabetMapTestSupport::hist_from_indices(decoded.value()));
}

TEST_CASE("HistAlphabetMap compose atbash caesar encrypt matches stream hist",
          "[score][hist_map][compose]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(120, 2);
    StatusOr<std::vector<Index29>> after_atbash =
        AtbashTransform{}.apply(plain, nlohmann::json::object(), TransformDirection::Decrypt);
    REQUIRE(after_atbash.ok());

    constexpr std::uint8_t shift = 9;
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        after_atbash.value(), nlohmann::json{{"shift", shift}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    // Inverse search path: decrypt Caesar then Atbash ≡ Atbash then decrypt-Caesar on bins
    // Catalog fused kernel uses enc_caesar(dec_atbash(x), shift) as the *forward* compose
    // applied to ciphertext bins when scoring that family — match that bin map:
    // plain_guess = enc_caesar(dec_atbash(cipher), shift).
    std::vector<Index29> composed(cipher.value().size());
    for (std::size_t i = 0; i < cipher.value().size(); ++i) {
        const Index29 at = Z29::atbash(cipher.value()[i]);
        composed[i] = Z29::add(at, Index29{shift});
    }

    StatusOr<HistAlphabetMap::Hist> H =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(cipher.value()));
    REQUIRE(H.ok());
    StatusOr<HistAlphabetMap::Hist> remapped =
        HistAlphabetMap::compose_atbash_caesar_encrypt(H.value(), shift);
    REQUIRE(remapped.ok());
    REQUIRE(remapped.value() == HistAlphabetMapTestSupport::hist_from_indices(composed));
}

TEST_CASE("HistAlphabetMap period-29 linear/poly keystream matches stream hist",
          "[score][hist_map][s2][s5]") {
    const std::vector<std::uint8_t> cipher =
        HistAlphabetMapTestSupport::to_bytes(HistAlphabetMapTestSupport::make_plain(87, 9));
    constexpr std::uint8_t b0 = 5;
    constexpr std::uint8_t b1 = 3;
    constexpr std::uint8_t b2 = 2;

    for (bool minus : {true, false}) {
        HistAlphabetMap::Hist brute_lin{};
        HistAlphabetMap::Hist brute_poly{};
        for (std::size_t t = 0; t < cipher.size(); ++t) {
            const unsigned r = static_cast<unsigned>(t % 29u);
            const unsigned ks_lin = (static_cast<unsigned>(b0) + static_cast<unsigned>(b1) * r) % 29u;
            const unsigned ks_poly =
                (static_cast<unsigned>(b0) + static_cast<unsigned>(b1) * r +
                 static_cast<unsigned>(b2) * ((r * r) % 29u)) %
                29u;
            if (minus) {
                ++brute_lin[(static_cast<unsigned>(cipher[t]) + 29u - ks_lin) % 29u];
                ++brute_poly[(static_cast<unsigned>(cipher[t]) + 29u - ks_poly) % 29u];
            } else {
                ++brute_lin[(static_cast<unsigned>(cipher[t]) + ks_lin) % 29u];
                ++brute_poly[(static_cast<unsigned>(cipher[t]) + ks_poly) % 29u];
            }
        }
        StatusOr<HistAlphabetMap::Hist> once_lin =
            HistAlphabetMap::linear_period29_plain_hist_from_once(cipher, b0, b1, minus);
        REQUIRE(once_lin.ok());
        REQUIRE(once_lin.value() == brute_lin);

        StatusOr<HistAlphabetMap::Hist> once_poly =
            HistAlphabetMap::poly_period29_plain_hist_from_once(cipher, b0, b1, b2, minus);
        REQUIRE(once_poly.ok());
        REQUIRE(once_poly.value() == brute_poly);
    }
}

TEST_CASE("HistAlphabetMap apply_bin_map rejects bad maps", "[score][hist_map][edge]") {
    HistAlphabetMap::Hist H{};
    H[0] = 1;
    std::array<std::uint8_t, 28> short_map{};
    REQUIRE_FALSE(HistAlphabetMap::apply_bin_map(H, short_map).ok());

    std::array<std::uint8_t, 29> bad{};
    bad[3] = 29;
    REQUIRE_FALSE(HistAlphabetMap::apply_bin_map(H, bad).ok());
}

TEST_CASE("HistAlphabetMap CTAK lag-prefix merge matches brute decrypt hist",
          "[score][hist_map][autokey][ctak]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(97, 4);
    const std::vector<std::uint8_t> primer = {3, 8, 14, 21, 2};
    const std::uint32_t L = static_cast<std::uint32_t>(primer.size());

    // Dense CTAK encrypt: key = primer for t < L else cipher[t-L]; out = plain + key.
    std::vector<std::uint8_t> cipher(plain.size());
    for (std::size_t t = 0; t < plain.size(); ++t) {
        const std::uint8_t key =
            t < primer.size() ? primer[t] : cipher[t - static_cast<std::size_t>(L)];
        cipher[t] = Z29::add(plain[t], Index29{key}).value();
    }

    // Brute CTAK decrypt hist.
    HistAlphabetMap::Hist brute{};
    for (std::size_t t = 0; t < cipher.size(); ++t) {
        const std::uint8_t key =
            t < primer.size() ? primer[t] : cipher[t - static_cast<std::size_t>(L)];
        const std::uint8_t out = Z29::sub(Index29{cipher[t]}, Index29{key}).value();
        ++brute[out];
    }

    StatusOr<HistAlphabetMap::Hist> once =
        HistAlphabetMap::ctak_plain_hist_from_once(cipher, primer);
    REQUIRE(once.ok());
    REQUIRE(once.value() == brute);
    REQUIRE(HistAlphabetMap::hist_total(once.value()) == cipher.size());

    StatusOr<HistAlphabetMap::Hist> lag = HistAlphabetMap::count_lag_diff_hist(cipher, L);
    REQUIRE(lag.ok());
    REQUIRE(HistAlphabetMap::hist_total(lag.value()) == cipher.size() - primer.size());
}

TEST_CASE("HistAlphabetMap ring lag-prefix merge matches AutokeyRing brute hist",
          "[score][hist_map][autokey][ring]") {
    const std::vector<std::uint8_t> cipher =
        HistAlphabetMapTestSupport::to_bytes(HistAlphabetMapTestSupport::make_plain(64, 6));
    constexpr std::uint32_t lag = 5;

    HistAlphabetMap::Hist brute_minus{};
    HistAlphabetMap::Hist brute_add{};
    for (std::size_t t = 0; t < cipher.size(); ++t) {
        const std::uint8_t key = t < lag ? 0u : cipher[t - lag];
        ++brute_minus[Z29::sub(Index29{cipher[t]}, Index29{key}).value()];
        ++brute_add[Z29::add(Index29{cipher[t]}, Index29{key}).value()];
    }

    StatusOr<HistAlphabetMap::Hist> once_minus =
        HistAlphabetMap::ring_plain_hist_from_once(cipher, lag, /*cipher_minus_ks=*/true);
    REQUIRE(once_minus.ok());
    REQUIRE(once_minus.value() == brute_minus);

    StatusOr<HistAlphabetMap::Hist> once_add =
        HistAlphabetMap::ring_plain_hist_from_once(cipher, lag, /*cipher_minus_ks=*/false);
    REQUIRE(once_add.ok());
    REQUIRE(once_add.value() == brute_add);

    // lag == 0 → identity (key always 0).
    StatusOr<HistAlphabetMap::Hist> once_zero =
        HistAlphabetMap::ring_plain_hist_from_once(cipher, 0u, true);
    REQUIRE(once_zero.ok());
    REQUIRE(once_zero.value() == HistAlphabetMap::count_stream_hist(cipher).value());
}

TEST_CASE("HistAlphabetMap lag edge cases L>=T and L==0", "[score][hist_map][autokey][edge]") {
    const std::vector<std::uint8_t> cipher = {1, 2, 3};
    REQUIRE_FALSE(HistAlphabetMap::count_lag_diff_hist(cipher, 0).ok());

    StatusOr<HistAlphabetMap::Hist> empty_tail = HistAlphabetMap::count_lag_diff_hist(cipher, 3);
    REQUIRE(empty_tail.ok());
    REQUIRE(HistAlphabetMap::hist_total(empty_tail.value()) == 0u);

    StatusOr<HistAlphabetMap::Hist> empty_tail_gt =
        HistAlphabetMap::count_lag_diff_hist(cipher, 10);
    REQUIRE(empty_tail_gt.ok());
    REQUIRE(HistAlphabetMap::hist_total(empty_tail_gt.value()) == 0u);

    // CTAK with L == T: only prefix contributes.
    const std::vector<std::uint8_t> primer = {4, 5, 6};
    StatusOr<HistAlphabetMap::Hist> only_prefix =
        HistAlphabetMap::ctak_plain_hist_from_once(cipher, primer);
    REQUIRE(only_prefix.ok());
    REQUIRE(HistAlphabetMap::hist_total(only_prefix.value()) == cipher.size());

    // CTAK with L > T: prefix over whole stream (first T primer symbols).
    const std::vector<std::uint8_t> primer_gt = {4, 5, 6, 7, 8};
    StatusOr<HistAlphabetMap::Hist> prefix_gt =
        HistAlphabetMap::ctak_plain_hist_from_once(cipher, primer_gt);
    REQUIRE(prefix_gt.ok());
    REQUIRE(HistAlphabetMap::hist_total(prefix_gt.value()) == cipher.size());
    HistAlphabetMap::Hist brute{};
    for (std::size_t t = 0; t < cipher.size(); ++t) {
        ++brute[Z29::sub(Index29{cipher[t]}, Index29{primer_gt[t]}).value()];
    }
    REQUIRE(prefix_gt.value() == brute);
}

TEST_CASE("HistAlphabetMap bigram rotate-dot matches brute force",
          "[score][hist_map][bigram]") {
    const std::vector<std::uint8_t> cipher =
        HistAlphabetMapTestSupport::to_bytes(HistAlphabetMapTestSupport::make_plain(300, 7));
    const auto ll = HistAlphabetMapTestSupport::make_ll_table();

    StatusOr<HistAlphabetMap::BigramHist> B = HistAlphabetMap::count_bigrams(cipher);
    REQUIRE(B.ok());
    std::uint64_t pairs = 0;
    for (std::uint32_t c : B.value()) {
        pairs += c;
    }
    REQUIRE(pairs == cipher.size() - 1u);

    for (std::uint8_t shift = 0; shift < 29; ++shift) {
        StatusOr<double> rotated =
            HistAlphabetMap::bigram_ll_dot_rotated(B.value(), ll, shift);
        StatusOr<double> brute =
            HistAlphabetMap::bigram_ll_brute_force(cipher, ll, shift);
        REQUIRE(rotated.ok());
        REQUIRE(brute.ok());
        REQUIRE(rotated.value() == Catch::Approx(brute.value()).margin(1e-9));
    }
}

TEST_CASE("HistAlphabetMap bigram T<2 is zero score", "[score][hist_map][bigram][edge]") {
    const auto ll = HistAlphabetMapTestSupport::make_ll_table();
    const std::vector<std::uint8_t> one = {3};
    StatusOr<HistAlphabetMap::BigramHist> B = HistAlphabetMap::count_bigrams(one);
    REQUIRE(B.ok());
    REQUIRE(HistAlphabetMap::bigram_ll_dot_rotated(B.value(), ll, 0).value() ==
            Catch::Approx(0.0));
    REQUIRE(HistAlphabetMap::bigram_ll_brute_force(one, ll, 4).value() == Catch::Approx(0.0));
}

TEST_CASE("HistAlphabetMap remapped Caesar hist chi2 matches Chi2EnglishGp",
          "[score][hist_map][chi2]") {
    StatusOr<ExpectedFrequencyTable> freqs = ExpectedFrequencyLoader::load_from_file(
        std::string(PARCAE_TEST_DATA_DIR) + "/profiles/scores/english-gp-expected-v0.json");
    REQUIRE(freqs.ok());

    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(256, 1);
    StatusOr<std::vector<Index29>> cipher = CaesarTransform{}.apply(
        plain, nlohmann::json{{"shift", 5}}, TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    StatusOr<HistAlphabetMap::Hist> H =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(cipher.value()));
    REQUIRE(H.ok());

    StatusOr<HistAlphabetMap::Hist> P = HistAlphabetMap::rotate_decrypt(H.value(), 5);
    REQUIRE(P.ok());

    StatusOr<double> from_map =
        HistAlphabetMap::chi2_from_hist(P.value(), freqs.value().probabilities());
    StatusOr<double> from_stream = Chi2EnglishGp::score(plain, freqs.value());
    REQUIRE(from_map.ok());
    REQUIRE(from_stream.ok());
    REQUIRE(from_map.value() == Catch::Approx(from_stream.value()).margin(1e-12));
}

TEST_CASE("HistAlphabetMap column hist + Vigenère remap matches stream decrypt",
          "[score][hist_map][vigenere]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(180);
    StatusOr<std::vector<Index29>> cipher = VigenereKeyTransform{}.apply(
        plain, nlohmann::json{{"key_indices", nlohmann::json::array({4, 9, 15, 2})}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    const auto cipher_bytes = HistAlphabetMapTestSupport::to_bytes(cipher.value());
    constexpr std::uint32_t L = 4;
    const std::uint8_t key_arr[] = {4, 9, 15, 2};
    std::span<const std::uint8_t> key(key_arr, L);

    StatusOr<std::vector<std::uint32_t>> cols =
        HistAlphabetMap::count_column_hist(cipher_bytes, L);
    REQUIRE(cols.ok());
    REQUIRE(std::accumulate(cols.value().begin(), cols.value().end(), 0u) == cipher_bytes.size());

    StatusOr<HistAlphabetMap::Hist> from_cols =
        HistAlphabetMap::vigenere_plain_hist_from_columns(cols.value(), L, key);
    REQUIRE(from_cols.ok());

    StatusOr<std::vector<Index29>> decoded = VigenereKeyTransform{}.apply(
        cipher.value(), nlohmann::json{{"key_indices", nlohmann::json::array({4, 9, 15, 2})}},
        TransformDirection::Decrypt);
    REQUIRE(decoded.ok());
    StatusOr<HistAlphabetMap::Hist> from_stream =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(decoded.value()));
    REQUIRE(from_stream.ok());
    REQUIRE(from_cols.value() == from_stream.value());
    REQUIRE(from_cols.value() == HistAlphabetMapTestSupport::hist_from_indices(plain));
}

TEST_CASE("HistAlphabetMap column hist + Beaufort remap matches stream decrypt",
          "[score][hist_map][beaufort]") {
    const std::vector<Index29> plain = HistAlphabetMapTestSupport::make_plain(180, 2);
    StatusOr<std::vector<Index29>> cipher = BeaufortKeyTransform{}.apply(
        plain, nlohmann::json{{"key_indices", nlohmann::json::array({4, 9, 15, 2})}},
        TransformDirection::Encrypt);
    REQUIRE(cipher.ok());

    const auto cipher_bytes = HistAlphabetMapTestSupport::to_bytes(cipher.value());
    constexpr std::uint32_t L = 4;
    const std::uint8_t key_arr[] = {4, 9, 15, 2};
    std::span<const std::uint8_t> key(key_arr, L);

    StatusOr<std::vector<std::uint32_t>> cols =
        HistAlphabetMap::count_column_hist(cipher_bytes, L);
    REQUIRE(cols.ok());

    StatusOr<HistAlphabetMap::Hist> from_cols =
        HistAlphabetMap::beaufort_plain_hist_from_columns(cols.value(), L, key);
    REQUIRE(from_cols.ok());

    StatusOr<std::vector<Index29>> decoded = BeaufortKeyTransform{}.apply(
        cipher.value(), nlohmann::json{{"key_indices", nlohmann::json::array({4, 9, 15, 2})}},
        TransformDirection::Decrypt);
    REQUIRE(decoded.ok());
    StatusOr<HistAlphabetMap::Hist> from_stream =
        HistAlphabetMap::count_stream_hist(HistAlphabetMapTestSupport::to_bytes(decoded.value()));
    REQUIRE(from_stream.ok());
    REQUIRE(from_cols.value() == from_stream.value());
    REQUIRE(from_cols.value() == HistAlphabetMapTestSupport::hist_from_indices(plain));
}

TEST_CASE("HistAlphabetMap rejects out-of-range symbols and a=0", "[score][hist_map][edge]") {
    const std::vector<std::uint8_t> bad = {0, 29};
    REQUIRE_FALSE(HistAlphabetMap::count_stream_hist(bad).ok());
    REQUIRE_FALSE(HistAlphabetMap::rotate_decrypt(HistAlphabetMap::Hist{}, 29).ok());
    REQUIRE_FALSE(HistAlphabetMap::permute_affine_decrypt(HistAlphabetMap::Hist{}, 0, 1).ok());
    const std::vector<std::uint8_t> ok_two = {0, 1};
    REQUIRE_FALSE(HistAlphabetMap::count_column_hist(ok_two, 0u).ok());
}
