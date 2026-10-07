#include <catch2/catch_test_macros.hpp>

#include "hist_fast.hpp"
#include "hist_tile_cap.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "caesar_chi2_batch.hpp"
#endif

TEST_CASE("HistTileCap slot defaults and clamps", "[cuda][hist][tile_cap]") {
    REQUIRE(HistTileCap::kSlotCount == 5);
    REQUIRE(HistTileCap::kCaesar == 0);

    HistTileCap::set(HistTileCap::kCaesar, 0);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == HistFast::production_tile_cap);
    REQUIRE(HistTileCap::tiles_for(HistTileCap::kCaesar, 1048576) == HistFast::production_tile_cap);

    HistTileCap::set(HistTileCap::kCaesar, 32);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 32);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == 32);
    REQUIRE(HistTileCap::tiles_for(HistTileCap::kCaesar, 1048576) == 32);

    HistTileCap::set(HistTileCap::kCaesar, HistFast::max_tiles);
    REQUIRE(HistTileCap::tiles_for(HistTileCap::kCaesar, 1048576) == 1024);

    // Negative → production (0).
    HistTileCap::set(HistTileCap::kCaesar, -7);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);
    REQUIRE(HistTileCap::effective(HistTileCap::kCaesar) == HistFast::production_tile_cap);

    // Unknown slot: set no-op, get 0, effective production.
    HistTileCap::set(-1, 128);
    HistTileCap::set(HistTileCap::kSlotCount, 128);
    REQUIRE(HistTileCap::get(-1) == 0);
    REQUIRE(HistTileCap::get(HistTileCap::kSlotCount) == 0);
    REQUIRE(HistTileCap::effective(-1) == HistFast::production_tile_cap);
    REQUIRE(HistTileCap::tiles_for(99, 1048576) == HistFast::production_tile_cap);

    // S2 slot: independent of Caesar; production default is 32 (not fat-64).
    HistTileCap::set(HistTileCap::kS2, 0);
    REQUIRE(HistTileCap::slot_default(HistTileCap::kS2) == 32);
    REQUIRE(HistTileCap::effective(HistTileCap::kS2) == 32);
    REQUIRE(HistTileCap::tiles_for(HistTileCap::kS2, 1048576) == 32);
    HistTileCap::set(HistTileCap::kS2, 128);
    REQUIRE(HistTileCap::get(HistTileCap::kS2) == 128);
    REQUIRE(HistTileCap::effective(HistTileCap::kS2) == 128);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);
    HistTileCap::set(HistTileCap::kS2, 0);

    HistTileCap::set(HistTileCap::kCaesar, 0);
}

#if defined(PARCAE_HAS_CUDA)

TEST_CASE("CaesarChi2Batch tile cap forwards to HistTileCap", "[cuda][hist][tile_cap][caesar]") {
    CaesarChi2Batch::set_hist_tile_cap(0);
    REQUIRE(CaesarChi2Batch::hist_tile_cap() == 0);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);
    REQUIRE(CaesarChi2Batch::tiles_for_public(1048576) == CaesarChi2Batch::kProductionTileCap);
    REQUIRE(CaesarChi2Batch::kProductionTileCap == HistFast::production_tile_cap);

    CaesarChi2Batch::set_hist_tile_cap(32);
    REQUIRE(CaesarChi2Batch::hist_tile_cap() == 32);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 32);
    REQUIRE(CaesarChi2Batch::tiles_for_public(1048576) == 32);

    CaesarChi2Batch::set_hist_tile_cap(-3);
    REQUIRE(CaesarChi2Batch::hist_tile_cap() == 0);
    REQUIRE(HistTileCap::get(HistTileCap::kCaesar) == 0);

    CaesarChi2Batch::set_hist_tile_cap(0);
}

#else

TEST_CASE("CaesarChi2Batch tile cap skipped without CUDA", "[cuda][hist][tile_cap][caesar]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON");
}

#endif
