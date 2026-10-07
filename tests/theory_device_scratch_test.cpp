#include <catch2/catch_test_macros.hpp>

#if defined(PARCAE_HAS_CUDA)

#include "parcae_cuda.hpp"
#include "theory_device_scratch.hpp"
#include "theory_hist_chi2_s1.hpp"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

namespace {

Status sync_default() {
    return CudaError::to_status(cudaDeviceSynchronize(), "theory_device_scratch_test sync");
}

} // namespace

TEST_CASE("CUDA TheoryDeviceScratch ensure_capacity rejects zero and oversize",
          "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE_FALSE(scratch.ensure_capacity(0, 8).ok());
    REQUIRE_FALSE(scratch.ensure_capacity(4, 0).ok());
    REQUIRE_FALSE(scratch.ensure_capacity(TheoryDeviceScratch::kMaxCandidates + 1, 8).ok());
    REQUIRE_FALSE(scratch.ensure_capacity(4, TheoryDeviceScratch::kMaxTokens + 1).ok());
}

TEST_CASE("CUDA TheoryDeviceScratch ensure_capacity grow-only", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(4, 16).ok());
    REQUIRE(scratch.capacity_C() == 4);
    REQUIRE(scratch.capacity_T() == 16);
    REQUIRE(scratch.cipher() != nullptr);
    REQUIRE(scratch.probs() != nullptr);
    REQUIRE(scratch.counts() != nullptr);
    REQUIRE(scratch.scores() != nullptr);
    REQUIRE(scratch.slots() != nullptr);
    REQUIRE(scratch.luts() != nullptr);
    REQUIRE(scratch.b0() != nullptr);
    REQUIRE(scratch.b1() != nullptr);
    REQUIRE(scratch.lane_err() != nullptr);

    REQUIRE(scratch.ensure_capacity(4, 16).ok());
    REQUIRE(scratch.capacity_C() == 4);
    REQUIRE(scratch.capacity_T() == 16);

    REQUIRE(scratch.ensure_capacity(8, 16).ok());
    REQUIRE(scratch.capacity_C() == 8);
    REQUIRE(scratch.capacity_T() == 16);

    // Shrink request is a no-op (grow-only).
    REQUIRE(scratch.ensure_capacity(2, 8).ok());
    REQUIRE(scratch.capacity_C() == 8);
    REQUIRE(scratch.capacity_T() == 16);
}

TEST_CASE("CUDA TheoryDeviceScratch cipher upload skippable by generation", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(4, 8).ok());

    const std::vector<std::uint8_t> cipher{1, 2, 3, 4, 5, 6, 7, 8};
    StatusOr<bool> first = scratch.upload_cipher_async(cipher, /*generation=*/7);
    REQUIRE(first.ok());
    REQUIRE(first.value() == true);
    REQUIRE(scratch.live_T() == cipher.size());
    REQUIRE(scratch.cipher_generation() == 7);

    StatusOr<bool> skipped = scratch.upload_cipher_async(cipher, 7);
    REQUIRE(skipped.ok());
    REQUIRE(skipped.value() == false);

    StatusOr<bool> again = scratch.upload_cipher_async(cipher, 8);
    REQUIRE(again.ok());
    REQUIRE(again.value() == true);
    REQUIRE(scratch.cipher_generation() == 8);

    REQUIRE(sync_default().ok());
    std::vector<std::uint8_t> back(cipher.size());
    REQUIRE(CudaError::to_status(
                cudaMemcpy(back.data(), scratch.cipher(), cipher.size(), cudaMemcpyDeviceToHost),
                "cipher D2H")
                .ok());
    REQUIRE(back == cipher);
}

TEST_CASE("CUDA TheoryDeviceScratch probs upload skippable by generation", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 4).ok());

    std::vector<double> probs(TheoryDeviceScratch::alphabet_size, 1.0 / 29.0);
    probs[0] = 0.05;
    StatusOr<bool> first = scratch.upload_probs_async(probs, /*generation=*/1);
    REQUIRE(first.ok());
    REQUIRE(first.value() == true);
    StatusOr<bool> skipped = scratch.upload_probs_async(probs, 1);
    REQUIRE(skipped.ok());
    REQUIRE(skipped.value() == false);

    REQUIRE(sync_default().ok());
    std::vector<double> back(TheoryDeviceScratch::alphabet_size, 0.0);
    REQUIRE(CudaError::to_status(
                cudaMemcpy(back.data(), scratch.probs(),
                           TheoryDeviceScratch::alphabet_size * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "probs D2H")
                .ok());
    REQUIRE(back[0] == 0.05);
}

TEST_CASE("CUDA TheoryDeviceScratch growing T invalidates cipher generation",
          "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 4).ok());
    const std::vector<std::uint8_t> small{9, 8, 7, 6};
    REQUIRE(scratch.upload_cipher_async(small, 3).value() == true);

    REQUIRE(scratch.ensure_capacity(2, 8).ok());
    REQUIRE(scratch.cipher_generation() == 0);
    REQUIRE(scratch.live_T() == 0);

    const std::vector<std::uint8_t> bigger{1, 2, 3, 4, 5, 6, 7, 8};
    StatusOr<bool> reup = scratch.upload_cipher_async(bigger, 3);
    REQUIRE(reup.ok());
    REQUIRE(reup.value() == true);
}

TEST_CASE("CUDA TheoryDeviceScratch growing C keeps cipher resident", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 4).ok());
    const std::vector<std::uint8_t> cipher{4, 3, 2, 1};
    REQUIRE(scratch.upload_cipher_async(cipher, 11).value() == true);

    REQUIRE(scratch.ensure_capacity(8, 4).ok());
    REQUIRE(scratch.capacity_C() == 8);
    REQUIRE(scratch.cipher_generation() == 11);
    REQUIRE(scratch.live_T() == cipher.size());

    StatusOr<bool> skipped = scratch.upload_cipher_async(cipher, 11);
    REQUIRE(skipped.ok());
    REQUIRE(skipped.value() == false);
}

TEST_CASE("CUDA TheoryDeviceScratch slots and scores async round-trip", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(3, 4).ok());

    const std::vector<std::uint8_t> slots{1, 2, 3, 4, 5, 6}; // C=3, slot_count=2
    REQUIRE(scratch.upload_slots_async(slots, 3, 2).ok());
    REQUIRE(scratch.live_C() == 3);
    REQUIRE(scratch.live_slot_count() == 2);

    const std::vector<double> scores_in{1.5, 2.5, 3.5};
    REQUIRE(CudaError::to_status(
                cudaMemcpy(scratch.scores(), scores_in.data(), scores_in.size() * sizeof(double),
                           cudaMemcpyHostToDevice),
                "scores H2D seed")
                .ok());

    std::vector<double> scores_out(3, 0.0);
    REQUIRE(scratch.download_scores_async(scores_out).ok());
    REQUIRE(sync_default().ok());
    REQUIRE(scores_out == scores_in);

    REQUIRE(sync_default().ok());
    std::vector<std::uint8_t> slots_back(slots.size());
    REQUIRE(CudaError::to_status(
                cudaMemcpy(slots_back.data(), scratch.slots(), slots.size(),
                           cudaMemcpyDeviceToHost),
                "slots D2H")
                .ok());
    REQUIRE(slots_back == slots);
}

TEST_CASE("CUDA TheoryDeviceScratch lut and b0/b1 capacity guards", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 4).ok());

    std::vector<std::uint8_t> luts(2 * TheoryDeviceScratch::alphabet_size, 7);
    REQUIRE(scratch.upload_luts_async(luts, 2).ok());
    REQUIRE_FALSE(scratch.upload_luts_async(luts, 3).ok());

    const std::vector<std::uint8_t> b0{1, 2};
    const std::vector<std::uint8_t> b1{3, 4};
    REQUIRE(scratch.upload_b0_b1_async(b0, b1, 2).ok());
    REQUIRE_FALSE(scratch.upload_b0_b1_async(b0, b1, 3).ok());
    const std::vector<std::uint8_t> bad_slots{1, 2};
    REQUIRE_FALSE(scratch.upload_slots_async(bad_slots, 2, 2).ok()); // size mismatch
}

TEST_CASE("CUDA TheoryDeviceScratch param ping-pong slabs", "[cuda][scratch]") {
    REQUIRE(ParcaeCuda::available());

    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 4).ok());
    REQUIRE(scratch.write_slab() == 0);
    REQUIRE(scratch.launch_slab() == 0);

    const std::vector<std::uint8_t> slots_a{1, 2, 3, 4};
    REQUIRE(scratch.upload_slots_async(slots_a, 2, 2).ok());
    std::uint8_t* slab0 = scratch.slots();
    scratch.commit_param_slab();
    REQUIRE(scratch.launch_slab() == 0);
    REQUIRE(scratch.write_slab() == 1);
    REQUIRE(scratch.slots() == slab0);

    const std::vector<std::uint8_t> slots_b{5, 6, 7, 8};
    REQUIRE(scratch.upload_slots_async(slots_b, 2, 2).ok());
    scratch.commit_param_slab();
    REQUIRE(scratch.launch_slab() == 1);
    REQUIRE(scratch.write_slab() == 0);
    REQUIRE(scratch.slots() != slab0);

    REQUIRE(sync_default().ok());
    std::vector<std::uint8_t> back(4);
    REQUIRE(CudaError::to_status(
                cudaMemcpy(back.data(), scratch.slots(), 4, cudaMemcpyDeviceToHost), "slots D2H")
                .ok());
    REQUIRE(back == slots_b);
}

TEST_CASE("CUDA TheoryDeviceScratch S1 LUT residency key", "[cuda][scratch][s1]") {
    REQUIRE(ParcaeCuda::available());
    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(4, 16).ok());
    REQUIRE_FALSE(scratch.s1_lut_resident(1, 2, 4));

    scratch.note_s1_lut_baked(11, 22, 4);
    REQUIRE(scratch.s1_lut_resident(11, 22, 4));
    REQUIRE(scratch.s1_lut_bake_count() == 1u);
    REQUIRE_FALSE(scratch.s1_lut_resident(11, 99, 4));
    REQUIRE_FALSE(scratch.s1_lut_resident(11, 22, 2));
    REQUIRE_FALSE(scratch.s1_lut_resident(0, 22, 4));

    scratch.note_s1_lut_hit();
    REQUIRE(scratch.s1_lut_hit_count() == 1u);

    // Initial commit keeps launch on slab 0 (write catches up). The next commit
    // publishes the other slab, so the baked LUT is no longer the launch buffer.
    scratch.commit_param_slab();
    REQUIRE(scratch.s1_lut_resident(11, 22, 4));
    scratch.commit_param_slab();
    REQUIRE_FALSE(scratch.s1_lut_resident(11, 22, 4));

    scratch.note_s1_lut_baked(11, 22, 4);
    REQUIRE(scratch.s1_lut_resident(11, 22, 4));
    REQUIRE(scratch.ensure_capacity(8, 16).ok());
    REQUIRE_FALSE(scratch.s1_lut_resident(11, 22, 4));

    scratch.note_s1_lut_baked(3, 4, 8);
    scratch.reset();
    REQUIRE(scratch.s1_lut_bake_count() == 0u);
    REQUIRE_FALSE(scratch.s1_lut_resident(3, 4, 8));
}

TEST_CASE("TheoryHistChi2S1 rejects empty T and C", "[cuda][hist][s1][edge]") {
    REQUIRE(ParcaeCuda::available());
    TheoryDeviceScratch scratch;
    REQUIRE(scratch.ensure_capacity(2, 8).ok());
    REQUIRE_FALSE(TheoryHistChi2S1::launch_lut_async(scratch.cipher(), scratch.luts(), scratch.probs(),
                                                     scratch.counts(), scratch.scores(), 1, 0)
                      .ok());
    REQUIRE_FALSE(TheoryHistChi2S1::launch_lut_async(scratch.cipher(), scratch.luts(), scratch.probs(),
                                                     scratch.counts(), scratch.scores(), 0, 8)
                      .ok());
    REQUIRE_FALSE(TheoryHistChi2S1::launch_bake_async(scratch.cipher(), scratch.cipher(), 1,
                                                      scratch.slots(), 1, 0, 0, 0, 8, scratch.luts(),
                                                      scratch.lane_err(), 0)
                      .ok());
}

#else

TEST_CASE("CUDA TheoryDeviceScratch skipped (PARCAE_HAS_CUDA unset)", "[cuda][scratch]") {
    SUCCEED("Build with PARCAE_BUILD_CUDA=ON to exercise TheoryDeviceScratch");
}

#endif
