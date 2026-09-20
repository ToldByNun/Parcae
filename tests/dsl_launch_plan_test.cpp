#include <parcae/dsl/dsl_launch_plan.hpp>
#include <parcae/dsl/param_ir.hpp>
#include <parcae/dsl/theory_ir.hpp>
#include <parcae/dsl/z29_expr.hpp>

#include <catch2/catch_test_macros.hpp>

#include "hist_fast.hpp"

#include <string>
TEST_CASE("DslLaunchPlan threads match HistFast / twin 256", "[dsl][launch]") {
    REQUIRE(DslLaunchPlan::threads_per_block == 256);
    REQUIRE(DslLaunchPlan::threads_per_block == HistFast::threads);
    REQUIRE(DslLaunchPlan::hist_max_tiles == HistFast::max_tiles);
    REQUIRE(DslLaunchPlan::hist_alphabet == HistFast::alphabet);
}

TEST_CASE("DslLaunchPlan blocks_1d matches CaesarKernel formula", "[dsl][launch]") {
    REQUIRE(DslLaunchPlan::blocks_1d(0) == 0);
    REQUIRE(DslLaunchPlan::blocks_1d(1) == 1);
    REQUIRE(DslLaunchPlan::blocks_1d(256) == 1);
    REQUIRE(DslLaunchPlan::blocks_1d(257) == 2);
    REQUIRE(DslLaunchPlan::blocks_1d(512) == 2);
    REQUIRE(DslLaunchPlan::blocks_1d(513) == 3);
}

TEST_CASE("DslLaunchPlan tiles_for matches HistFast", "[dsl][launch]") {
    for (std::size_t t : {0u, 1u, 4u, 255u, 256u, 1024u, 4096u, 100000u}) {
        REQUIRE(DslLaunchPlan::tiles_for(t) == HistFast::tiles_for(t));
    }
}

TEST_CASE("DslLaunchPlan elementwise_1d", "[dsl][launch]") {
    const StatusOr<DslLaunchPlan::Plan> empty = DslLaunchPlan::elementwise_1d(0);
    REQUIRE(empty.ok());
    REQUIRE(empty.value().kind() == DslLaunchPlan::Kind::Elementwise1D);
    REQUIRE(empty.value().is_empty_launch());
    REQUIRE(empty.value().grid_x() == 0);
    REQUIRE(empty.value().threads() == 256);

    const StatusOr<DslLaunchPlan::Plan> p = DslLaunchPlan::elementwise_1d(1000);
    REQUIRE(p.ok());
    REQUIRE(p.value().grid_x() == DslLaunchPlan::blocks_1d(1000));
    REQUIRE(p.value().grid_y() == 1);
    REQUIRE(p.value().work_items() == 1000);
    REQUIRE_FALSE(p.value().is_empty_launch());
}

TEST_CASE("DslLaunchPlan batch_flat_1d", "[dsl][launch]") {
    const StatusOr<DslLaunchPlan::Plan> p = DslLaunchPlan::batch_flat_1d(29, 1024);
    REQUIRE(p.ok());
    REQUIRE(p.value().kind() == DslLaunchPlan::Kind::BatchFlat1D);
    REQUIRE(p.value().work_items() == 29u * 1024u);
    REQUIRE(p.value().grid_x() == DslLaunchPlan::blocks_1d(29u * 1024u));
    REQUIRE(p.value().candidates() == 29);
    REQUIRE(p.value().tokens() == 1024);

    REQUIRE_FALSE(DslLaunchPlan::batch_flat_1d(0, 10).ok());
    REQUIRE_FALSE(DslLaunchPlan::batch_flat_1d(10, 0).ok());
}

TEST_CASE("DslLaunchPlan hist_chi2_2d", "[dsl][launch]") {
    const StatusOr<DslLaunchPlan::Plan> p = DslLaunchPlan::hist_chi2_2d(29, 4096);
    REQUIRE(p.ok());
    REQUIRE(p.value().kind() == DslLaunchPlan::Kind::HistChi2_2D);
    REQUIRE(p.value().grid_x() == 29);
    REQUIRE(p.value().grid_y() == DslLaunchPlan::tiles_for(4096));
    REQUIRE(p.value().threads() == 256);
    REQUIRE(std::string(DslLaunchPlan::kind_str(p.value().kind())) == "hist_chi2_2d");
}

TEST_CASE("DslLaunchPlan for_theory elementwise stream", "[dsl][launch]") {
    const StatusOr<TheoryIr> theory = TheoryIr::make(
        "dsl_caesar",
        TheoryIr::Family::Elementwise,
        TheoryIr::Tier::A,
        TheoryIr::InterruptMode::ElementwiseDefault,
        {},
        Z29Expr::var("x"),
        Z29Expr::var("x"));
    REQUIRE(theory.ok());

    const StatusOr<DslLaunchPlan::Plan> p =
        DslLaunchPlan::for_theory(theory.value(), /*stream_len=*/2048);
    REQUIRE(p.ok());
    REQUIRE(p.value().kind() == DslLaunchPlan::Kind::Elementwise1D);
    REQUIRE(p.value().grid_x() == DslLaunchPlan::blocks_1d(2048));

    const StatusOr<DslLaunchPlan::Plan> batch =
        DslLaunchPlan::for_theory(theory.value(), 512, /*candidates=*/29);
    REQUIRE(batch.ok());
    REQUIRE(batch.value().kind() == DslLaunchPlan::Kind::BatchFlat1D);
    REQUIRE(batch.value().work_items() == 29u * 512u);
}

TEST_CASE("DslLaunchPlan for_fused_chi2", "[dsl][launch]") {
    const StatusOr<DslLaunchPlan::Plan> p = DslLaunchPlan::for_fused_chi2(29, 8192);
    REQUIRE(p.ok());
    REQUIRE(p.value().kind() == DslLaunchPlan::Kind::HistChi2_2D);
    REQUIRE(p.value().grid_x() == 29);
    REQUIRE(p.value().grid_y() == HistFast::tiles_for(8192));
}
