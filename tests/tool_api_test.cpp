#include <parcae/gematria/rune_codec.hpp>
#include <parcae/tool/api.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/transform_envelope.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] parcae::tool::Context test_ctx() {
    return parcae::tool::Context{std::filesystem::path(PARCAE_TEST_DATA_DIR)};
}

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

}  // namespace

TEST_CASE("tool::tokenize round-trips a-warning ciphertext", "[tool][tokenize]") {
    const auto ctx = test_ctx();
    StatusOr<std::string> ciphertext = [&]() -> StatusOr<std::string> {
        const auto path =
            ctx.data_root() / "fixtures" / "solved" / "a-warning" / "ciphertext.txt";
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::error("missing ciphertext");
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        return buf.str();
    }();
    REQUIRE(ciphertext.ok());

    StatusOr<TokenStream> stream =
        parcae::tool::tokenize(ctx, ciphertext.value(), "rtkd-separator-grammar-v0", true);
    REQUIRE(stream.ok());
    REQUIRE(stream.value().consumable_count() > 0);
    REQUIRE(stream.value().text() == ciphertext.value());
}

TEST_CASE("tool::apply_to_indices and apply_and_rebuild_text", "[tool][apply]") {
    const auto ctx = test_ctx();

    StatusOr<parcae::tool::TransformEnvelope> env = parcae::tool::TransformEnvelope::from_json(
        nlohmann::json{
            {"transform_id", "caesar"},
            {"direction", "encrypt"},
            {"params", {{"shift", 3}}},
        });
    REQUIRE(env.ok());

    const std::vector<Index29> plain = {I(0), I(1), I(2)};
    StatusOr<std::vector<Index29>> cipher =
        parcae::tool::apply_to_indices(plain, env.value());
    REQUIRE(cipher.ok());
    REQUIRE(cipher.value() == std::vector<Index29>{I(3), I(4), I(5)});

    StatusOr<parcae::tool::TransformEnvelope> decrypt = parcae::tool::TransformEnvelope::from_json(
        nlohmann::json{
            {"transform_id", "caesar"},
            {"direction", "decrypt"},
            {"params", {{"shift", 3}}},
        });
    REQUIRE(decrypt.ok());
    StatusOr<std::vector<Index29>> recovered =
        parcae::tool::apply_to_indices(cipher.value(), decrypt.value());
    REQUIRE(recovered.ok());
    REQUIRE(recovered.value() == plain);

    // Rebuild preserves separators in a tokenized page fragment.
    StatusOr<GematriaProfile> profile = ctx.load_gematria();
    REQUIRE(profile.ok());
    StatusOr<std::string> r0 = RuneCodec{profile.value()}.encode(I(0));
    StatusOr<std::string> r1 = RuneCodec{profile.value()}.encode(I(1));
    REQUIRE(r0.ok());
    REQUIRE(r1.ok());
    const std::string mixed = r0.value() + "-" + r1.value();
    StatusOr<TokenStream> stream = parcae::tool::tokenize(ctx, mixed);
    REQUIRE(stream.ok());

    StatusOr<std::string> rebuilt =
        parcae::tool::apply_and_rebuild_text(ctx, stream.value(), env.value());
    REQUIRE(rebuilt.ok());
    StatusOr<std::string> e3 = RuneCodec{profile.value()}.encode(I(3));
    StatusOr<std::string> e4 = RuneCodec{profile.value()}.encode(I(4));
    REQUIRE(e3.ok());
    REQUIRE(e4.ok());
    REQUIRE(rebuilt.value() == e3.value() + "-" + e4.value());
}

TEST_CASE("tool::to_latin preferred labels", "[tool][latin]") {
    const auto ctx = test_ctx();
    // Index 0 preferred is typically F; 1 is U — join without spaces.
    StatusOr<std::string> latin =
        parcae::tool::to_latin(ctx, std::vector<Index29>{I(0), I(1)});
    REQUIRE(latin.ok());
    REQUIRE_FALSE(latin.value().empty());
    REQUIRE(latin.value().find(' ') == std::string::npos);
}

TEST_CASE("tool::score ic and chi2", "[tool][score]") {
    const auto ctx = test_ctx();
    const std::vector<Index29> xs = {I(3), I(3), I(3), I(3)};
    StatusOr<double> ic = parcae::tool::score(ctx, xs, "ic_mod29");
    REQUIRE(ic.ok());
    REQUIRE(ic.value() == Catch::Approx(1.0).margin(0.0));

    StatusOr<double> chi2 = parcae::tool::score(ctx, xs, "chi2_english_gp_v0");
    REQUIRE(chi2.ok());
    REQUIRE(chi2.value() >= 0.0);
}

TEST_CASE("tool::validate_fixture by id and path", "[tool][validate]") {
    const auto ctx = test_ctx();

    const ValidationReport by_id =
        parcae::tool::validate_fixture(ctx, "a-warning", /*require_locked=*/true);
    REQUIRE(by_id.fixture_id() == "a-warning");
    REQUIRE(by_id.ok());

    const auto path = (ctx.data_root() / "fixtures" / "solved" / "welcome").string();
    const ValidationReport by_path =
        parcae::tool::validate_fixture(ctx, path, /*require_locked=*/true);
    REQUIRE(by_path.fixture_id() == "welcome");
    REQUIRE(by_path.ok());
}

TEST_CASE("tool::list registries", "[tool]") {
    const auto transforms = parcae::tool::list_transform_ids();
    REQUIRE(transforms.size() >= 8);
    REQUIRE(
        std::find(transforms.begin(), transforms.end(), "caesar") != transforms.end());

    const auto scores = parcae::tool::list_score_ids();
    REQUIRE(scores.size() == 5);
    REQUIRE(std::find(scores.begin(), scores.end(), "ic_mod29") != scores.end());
}
