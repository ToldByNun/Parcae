#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "hist_fast.hpp"
#include "launch_geom.hpp"

TEST_CASE("CUDA LaunchGeom defaults match HistFast threads", "[cuda][launch_geom]") {
    REQUIRE(LaunchGeom::kDefaultThreads == 256);
    REQUIRE(LaunchGeom::threads() == HistFast::threads);
    REQUIRE(LaunchGeom::threads_ab(LaunchGeom::AbThreads::k256) == 256);
    REQUIRE(LaunchGeom::threads_ab(LaunchGeom::AbThreads::k128) == 128);
    REQUIRE(LaunchGeom::threads_ab(LaunchGeom::AbThreads::k512) == 512);
}

TEST_CASE("CUDA LaunchGeom blocks_for ceiling", "[cuda][launch_geom]") {
    REQUIRE(LaunchGeom::blocks_for(0) == 0);
    REQUIRE(LaunchGeom::blocks_for(1) == 1);
    REQUIRE(LaunchGeom::blocks_for(256) == 1);
    REQUIRE(LaunchGeom::blocks_for(257) == 2);
    REQUIRE(LaunchGeom::blocks_for(1024, 256) == 4);
    REQUIRE(LaunchGeom::blocks_for(100, LaunchGeom::threads_ab(LaunchGeom::AbThreads::k128)) == 1);
}

#else

TEST_CASE("CUDA LaunchGeom skipped (PARCAE_HAS_CUDA unset)", "[cuda][launch_geom]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise LaunchGeom");
}

#endif
