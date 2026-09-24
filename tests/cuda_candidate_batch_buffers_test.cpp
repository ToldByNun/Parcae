#include "parcae/core/index29.hpp"

#include "candidate_batch_buffers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>

namespace {

[[nodiscard]] Index29 I(std::uint8_t v) {
    return Index29{v};
}

} // namespace

TEST_CASE("CandidateBatchBuffers shared caesar SoA layout", "[cuda][batch][abi]") {
    CandidateBatchBuffers::AllocateOptions options;
    options.token_layout = CandidateBatchBuffers::TokenLayout::Shared;
    options.family = CudaFamilyId::Caesar;
    options.with_scores = true;

    StatusOr<CandidateBatchBuffers> buffers =
        CandidateBatchBuffers::allocate(Index29::modulus, 7, options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().candidate_count() == 29);
    REQUIRE(buffers.value().token_count() == 7);
    REQUIRE(buffers.value().token_layout() == CandidateBatchBuffers::TokenLayout::Shared);
    REQUIRE(buffers.value().token_index29().size() == 7);
    REQUIRE(buffers.value().out_index29().size() == 29 * 7);
    REQUIRE(buffers.value().scores().size() == 29);
    REQUIRE(buffers.value().caesar_shifts().size() == 29);
    REQUIRE(buffers.value().consume_mask().empty());

    const std::vector<Index29> cipher{I(0), I(1), I(2), I(3), I(10), I(14), I(28)};
    REQUIRE(buffers.value().set_shared_tokens(cipher).ok());
    REQUIRE(buffers.value().token_index29()[0] == 0);
    REQUIRE(buffers.value().token_index29()[6] == 28);

    REQUIRE(buffers.value().fill_caesar_shifts_iota().ok());
    for (std::size_t c = 0; c < 29; ++c) {
        REQUIRE(buffers.value().caesar_shifts()[c] == static_cast<std::uint8_t>(c));
    }

    REQUIRE(buffers.value().flat_index(2, 3) == 2 * 7 + 3);
    buffers.value().out_index29()[buffers.value().flat_index(2, 3)] = 11;

    std::vector<Index29> out(7);
    REQUIRE(buffers.value().copy_out_candidate(2, out).ok());
    REQUIRE(out[3].value() == 11);
}

TEST_CASE("CandidateBatchBuffers per-candidate tokens + consume mask", "[cuda][batch][abi]") {
    CandidateBatchBuffers::AllocateOptions options;
    options.token_layout = CandidateBatchBuffers::TokenLayout::PerCandidate;
    options.family = CudaFamilyId::Atbash;
    options.with_consume_mask = true;
    options.with_scores = false;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(2, 3, options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().token_index29().size() == 6);
    REQUIRE(buffers.value().consume_mask().size() == 6);
    REQUIRE(buffers.value().scores().empty());
    REQUIRE(buffers.value().caesar_shifts().empty());

    const std::vector<Index29> lane0{I(1), I(2), I(3)};
    const std::vector<Index29> lane1{I(4), I(5), I(6)};
    REQUIRE(buffers.value().set_candidate_tokens(0, lane0).ok());
    REQUIRE(buffers.value().set_candidate_tokens(1, lane1).ok());
    REQUIRE(buffers.value().token_index29()[0] == 1);
    REQUIRE(buffers.value().token_index29()[5] == 6);

    // Default consume_mask is all-consumable (1).
    for (std::uint8_t bit : buffers.value().consume_mask()) {
        REQUIRE(bit == 1);
    }
}

TEST_CASE("CandidateBatchBuffers affine nested 812 param lanes", "[cuda][batch][abi]") {
    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::Affine;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(812, 4, options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().fill_affine_ab_nested().ok());
    REQUIRE(buffers.value().affine_a()[0] == 1);
    REQUIRE(buffers.value().affine_b()[0] == 0);
    REQUIRE(buffers.value().affine_a()[28] == 1);
    REQUIRE(buffers.value().affine_b()[28] == 28);
    REQUIRE(buffers.value().affine_a()[29] == 2);
    REQUIRE(buffers.value().affine_b()[29] == 0);
    REQUIRE(buffers.value().affine_a()[811] == 28);
    REQUIRE(buffers.value().affine_b()[811] == 28);
}

TEST_CASE("CandidateBatchBuffers rejects oversize C/T and layout misuse", "[cuda][batch][abi]") {
    REQUIRE_FALSE(CandidateBatchBuffers::allocate(0, 1).ok());
    REQUIRE_FALSE(CandidateBatchBuffers::allocate(CandidateBatchBuffers::kMaxC + 1, 1).ok());
    REQUIRE_FALSE(CandidateBatchBuffers::allocate(1, CandidateBatchBuffers::kMaxT + 1).ok());

    const std::vector<Index29> two{I(0), I(1)};

    StatusOr<CandidateBatchBuffers> shared =
        CandidateBatchBuffers::allocate(1, 2, CandidateBatchBuffers::AllocateOptions{});
    REQUIRE(shared.ok());
    REQUIRE_FALSE(shared.value().set_candidate_tokens(0, two).ok());

    CandidateBatchBuffers::AllocateOptions per;
    per.token_layout = CandidateBatchBuffers::TokenLayout::PerCandidate;
    StatusOr<CandidateBatchBuffers> replicated = CandidateBatchBuffers::allocate(1, 2, per);
    REQUIRE(replicated.ok());
    REQUIRE_FALSE(replicated.value().set_shared_tokens(two).ok());
}

TEST_CASE("CandidateBatchBuffers keyed arena slots", "[cuda][batch][abi]") {
    CandidateBatchBuffers::AllocateOptions options;
    options.family = CudaFamilyId::VigenereKey;
    options.key_arena_capacity = 16;

    StatusOr<CandidateBatchBuffers> buffers = CandidateBatchBuffers::allocate(3, 5, options);
    REQUIRE(buffers.ok());
    REQUIRE(buffers.value().key_bytes().size() == 16);
    REQUIRE(buffers.value().key_begin().size() == 3);
    REQUIRE(buffers.value().key_len().size() == 3);

    buffers.value().key_begin()[0] = 0;
    buffers.value().key_len()[0] = 2;
    buffers.value().key_bytes()[0] = 7;
    buffers.value().key_bytes()[1] = 8;
    REQUIRE(buffers.value().key_bytes()[0] == 7);
}
