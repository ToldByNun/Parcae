#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <parcae/core/index29.hpp>
#include <parcae/math/matrix729_map.hpp>
#include <span>
#include <vector>

TEST_CASE("Matrix729Map cell_index and cell_rc round-trip", "[math][matrix729]") {
    REQUIRE(Matrix729Map::cells_729 == 729);
    REQUIRE(Matrix729Map::side_27 == 27);
    REQUIRE(Matrix729Map::side_29 == 29);

    REQUIRE(Matrix729Map::cell_index(0, 0) == 0);
    REQUIRE(Matrix729Map::cell_index(0, 26) == 26);
    REQUIRE(Matrix729Map::cell_index(1, 0) == 27);
    REQUIRE(Matrix729Map::cell_index(26, 26) == 728);

    for (std::size_t i = 0; i < Matrix729Map::cells_729; ++i) {
        const auto [r, c] = Matrix729Map::cell_rc(i);
        REQUIRE(Matrix729Map::cell_index(r, c) == i);
    }
}

TEST_CASE("Matrix729Map identity and transpose_729 are involutions", "[math][matrix729]") {
    const Matrix729Map::Perm729 id = Matrix729Map::identity_729();
    for (std::uint16_t i = 0; i < Matrix729Map::cells_729; ++i) {
        REQUIRE(id[i] == i);
    }

    StatusOr<Matrix729Map::Perm729> id_inv = Matrix729Map::try_inverse_729(id);
    REQUIRE(id_inv.ok());
    REQUIRE(id_inv.value() == id);

    const Matrix729Map::Perm729 tr = Matrix729Map::transpose_729();
    REQUIRE(tr[Matrix729Map::cell_index(2, 5)] ==
            static_cast<std::uint16_t>(Matrix729Map::cell_index(5, 2)));

    StatusOr<Matrix729Map::Perm729> tr_inv = Matrix729Map::try_inverse_729(tr);
    REQUIRE(tr_inv.ok());
    REQUIRE(tr_inv.value() == tr); // transpose is an involution

    std::vector<std::uint16_t> cells(Matrix729Map::cells_729);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        cells[i] = static_cast<std::uint16_t>(i);
    }
    std::vector<std::uint16_t> once(Matrix729Map::cells_729);
    std::vector<std::uint16_t> twice(Matrix729Map::cells_729);
    REQUIRE(Matrix729Map::apply_gather_729(tr, std::span<const std::uint16_t>(cells),
                                          std::span<std::uint16_t>(once))
                .ok());
    REQUIRE(Matrix729Map::apply_gather_729(tr, std::span<const std::uint16_t>(once),
                                          std::span<std::uint16_t>(twice))
                .ok());
    REQUIRE(twice == cells);
}

TEST_CASE("Matrix729Map from_perm_729 rejects non-bijections", "[math][matrix729]") {
    std::vector<std::uint16_t> bad(Matrix729Map::cells_729, 0);
    REQUIRE_FALSE(Matrix729Map::from_perm_729(bad).ok());

    std::vector<std::uint16_t> short_perm = {0, 1, 2};
    REQUIRE_FALSE(Matrix729Map::from_perm_729(short_perm).ok());

    Matrix729Map::Perm729 id = Matrix729Map::identity_729();
    id[0] = 1;
    id[1] = 1; // duplicate
    REQUIRE_FALSE(Matrix729Map::from_perm_729(id).ok());
}

TEST_CASE("Matrix729Map Perm29 identity inverse and map", "[math][matrix729]") {
    const Matrix729Map::Perm29 id = Matrix729Map::identity_29();
    StatusOr<Matrix729Map::Perm29> inv = Matrix729Map::try_inverse_29(id);
    REQUIRE(inv.ok());
    REQUIRE(inv.value() == id);

    // Reverse alphabet: perm[i] = 28 - i
    std::array<std::uint8_t, 29> rev{};
    for (std::uint8_t i = 0; i < 29; ++i) {
        rev[i] = static_cast<std::uint8_t>(28 - i);
    }
    StatusOr<Matrix729Map::Perm29> perm = Matrix729Map::from_perm_29(rev);
    REQUIRE(perm.ok());
    REQUIRE(Matrix729Map::map_29(perm.value(), Index29{0}).value() == 28);
    REQUIRE(Matrix729Map::map_29(perm.value(), Index29{28}).value() == 0);
    REQUIRE(Matrix729Map::map_29(perm.value(), Index29{14}).value() == 14);

    StatusOr<Matrix729Map::Perm29> rev_inv = Matrix729Map::try_inverse_29(perm.value());
    REQUIRE(rev_inv.ok());
    REQUIRE(rev_inv.value() == perm.value());

    const std::vector<Index29> in{Index29{0}, Index29{1}, Index29{14}, Index29{28}};
    std::vector<Index29> out(in.size());
    REQUIRE(Matrix729Map::apply_map_29(perm.value(), in, out).ok());
    REQUIRE(out == std::vector<Index29>{Index29{28}, Index29{27}, Index29{14}, Index29{0}});

    std::vector<Index29> back(in.size());
    REQUIRE(Matrix729Map::apply_map_29(rev_inv.value(), out, back).ok());
    REQUIRE(back == in);
}

TEST_CASE("Matrix729Map from_perm_29 rejects non-bijections", "[math][matrix729]") {
    std::array<std::uint8_t, 29> bad{};
    bad.fill(3);
    REQUIRE_FALSE(Matrix729Map::from_perm_29(bad).ok());
    std::array<std::uint8_t, 2> short_perm{0, 1};
    REQUIRE_FALSE(Matrix729Map::from_perm_29(short_perm).ok());
}

TEST_CASE("Matrix729Map project_to_27 / lift_from_27 round-trip", "[math][matrix729]") {
    const Index29 ea{5};
    const Index29 eb{10};

    REQUIRE_FALSE(Matrix729Map::project_to_27(Index29{5}, ea, eb).ok());
    REQUIRE_FALSE(Matrix729Map::project_to_27(Index29{10}, ea, eb).ok());
    REQUIRE_FALSE(Matrix729Map::project_to_27(Index29{0}, ea, ea).ok());

    for (std::uint8_t v = 0; v < 29; ++v) {
        if (v == 5 || v == 10) {
            continue;
        }
        StatusOr<std::uint8_t> rank = Matrix729Map::project_to_27(Index29{v}, ea, eb);
        REQUIRE(rank.ok());
        REQUIRE(rank.value() < 27);
        StatusOr<Index29> lifted = Matrix729Map::lift_from_27(rank.value(), ea, eb);
        REQUIRE(lifted.ok());
        REQUIRE(lifted.value().value() == v);
    }

    // All ranks 0..26 lift uniquely.
    std::array<bool, 29> used{};
    for (std::uint8_t r = 0; r < 27; ++r) {
        StatusOr<Index29> lifted = Matrix729Map::lift_from_27(r, ea, eb);
        REQUIRE(lifted.ok());
        REQUIRE_FALSE(used[lifted.value().value()]);
        used[lifted.value().value()] = true;
    }
    REQUIRE_FALSE(used[5]);
    REQUIRE_FALSE(used[10]);
}
