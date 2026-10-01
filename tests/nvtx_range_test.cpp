#include <catch2/catch_test_macros.hpp>
#include <parcae/search/nvtx_range.hpp>

TEST_CASE("NvtxRange RAII does not throw", "[search][nvtx]") {
    {
        NvtxRange outer("prepare_theory");
        NvtxRange inner("bind_slots");
        (void)outer;
        (void)inner;
    }
    NvtxRange hist("hist_kernel");
    NvtxRange fin("finalize");
    NvtxRange d2h("d2h");
    NvtxRange mat("materialize");
    NvtxRange ingest("ingest");
    (void)hist;
    (void)fin;
    (void)d2h;
    (void)mat;
    (void)ingest;
}

TEST_CASE("NvtxRange available matches compile-time NVTX header", "[search][nvtx]") {
#if defined(PARCAE_HAS_NVTX)
    REQUIRE(NvtxRange::available());
#else
    REQUIRE_FALSE(NvtxRange::available());
#endif
}
