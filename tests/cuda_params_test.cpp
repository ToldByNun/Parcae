#include "parcae/transform/compose_transform.hpp"

#include "params.hpp"
#include "params_json.hpp"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

TEST_CASE("CUDA POD params caesar / affine from JSON", "[cuda][params]") {
    SECTION("caesar") {
        StatusOr<CaesarParams> p = CudaParamsJson::caesar_from_json(nlohmann::json{{"shift", 3}});
        REQUIRE(p.ok());
        REQUIRE(p.value().shift == 3);
        REQUIRE(CudaParamsJson::caesar_to_json(p.value()) == nlohmann::json{{"shift", 3}});
        REQUIRE_FALSE(CudaParamsJson::caesar_from_json(nlohmann::json{{"shift", 29}}).ok());
    }

    SECTION("affine") {
        StatusOr<AffineParams> p =
            CudaParamsJson::affine_from_json(nlohmann::json{{"a", 2}, {"b", 5}});
        REQUIRE(p.ok());
        REQUIRE(p.value().a == 2);
        REQUIRE(p.value().b == 5);
        REQUIRE(CudaParamsJson::affine_to_json(p.value()) == nlohmann::json{{"a", 2}, {"b", 5}});
        REQUIRE_FALSE(CudaParamsJson::affine_from_json(nlohmann::json{{"a", 0}, {"b", 1}}).ok());
    }
}

TEST_CASE("CUDA POD params vigenere key list", "[cuda][params]") {
    const nlohmann::json json{
        {"key_indices", {23, 10, 1, 10, 9, 10, 16, 26}},
        {"key_latin", "DIVINITY"},
    };
    StatusOr<KeyParamsHost> key = CudaParamsJson::key_from_json(json);
    REQUIRE(key.ok());
    REQUIRE(key.value().key.size() == 8);
    REQUIRE(key.value().view.key_len == 8);
    REQUIRE(key.value().view.key_ptr == key.value().key.data());
    REQUIRE(key.value().key[0] == 23);
    REQUIRE(key.value().key[7] == 26);

    const nlohmann::json back = CudaParamsJson::key_to_json(key.value());
    REQUIRE(back.at("key_indices") == json.at("key_indices"));
}

TEST_CASE("CUDA POD params totient start index", "[cuda][params]") {
    StatusOr<TotientParams> p =
        CudaParamsJson::totient_from_json(nlohmann::json{{"prime_start_index", 3}});
    REQUIRE(p.ok());
    REQUIRE(p.value().prime_start_index == 3);
}

TEST_CASE("CUDA POD params compose atbash_then_caesar", "[cuda][params]") {
    const nlohmann::json json = ComposeTransform::atbash_then_caesar_params(3);
    StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(json);
    REQUIRE(compose.ok());
    REQUIRE(compose.value().stages.size() == 2);
    REQUIRE(compose.value().stages[0].family == CudaFamilyId::Atbash);
    REQUIRE(compose.value().stages[0].direction == CudaDir::Decrypt);
    REQUIRE(compose.value().stages[1].family == CudaFamilyId::Caesar);
    REQUIRE(compose.value().stages[1].direction == CudaDir::Encrypt);
    REQUIRE(compose.value().stages[1].caesar.shift == 3);
    REQUIRE(compose.value().key_arena.empty());
    REQUIRE(compose.value().matrix_arena.empty());
    REQUIRE(compose.value().order_arena.empty());
    REQUIRE(compose.value().stages[0].matrix_len == 0);
    REQUIRE(compose.value().stages[0].order_len == 0);
    REQUIRE(compose.value().stages[0].grid.rows == 0);
    REQUIRE(compose.value().stages[0].vda.lag == 2);
    REQUIRE(compose.value().stages[0].vda.mode == CudaAutokeyMode::Ciphertext);
}

TEST_CASE("ComposeStageParams catalog arena fields default and pack", "[cuda][params]") {
    ComposeStageParams stage;
    REQUIRE(stage.matrix_begin == 0);
    REQUIRE(stage.matrix_len == 0);
    REQUIRE(stage.order_begin == 0);
    REQUIRE(stage.order_len == 0);
    REQUIRE(stage.grid.rows == 0);
    REQUIRE(stage.grid.cols == 0);
    REQUIRE(stage.grid.spiral_outward == 0);
    REQUIRE(stage.grid.boustrophedon_rtl == 0);
    REQUIRE(stage.grid.diagonal_anti == 0);
    REQUIRE(stage.vda.lag == 2);
    REQUIRE(stage.vda.mode == CudaAutokeyMode::Ciphertext);

    ComposeParamsHost host;
    host.matrix_arena = {1, 0, 0, 1};
    host.order_arena = {0u, 1u, 2u, 3u};
    stage.family = CudaFamilyId::Hill2;
    stage.matrix_begin = 0;
    stage.matrix_len = 4;
    host.stages.push_back(stage);

    ComposeStageParams grid_stage;
    grid_stage.family = CudaFamilyId::SpiralRead;
    grid_stage.grid.rows = 2;
    grid_stage.grid.cols = 2;
    grid_stage.grid.spiral_outward = 1;
    grid_stage.order_begin = 0;
    grid_stage.order_len = 4;
    host.stages.push_back(grid_stage);

    REQUIRE(host.stages.size() == 2);
    REQUIRE(host.matrix_arena.size() == 4);
    REQUIRE(host.order_arena.size() == 4);
    REQUIRE(host.stages[0].matrix_len == 4);
    REQUIRE(host.stages[1].order_len == 4);
    REQUIRE(host.stages[1].grid.spiral_outward == 1);
}

TEST_CASE("CUDA Dir matches TransformDirection", "[cuda][params]") {
    REQUIRE(CudaDirUtil::from_transform_direction(TransformDirection::Decrypt) == CudaDir::Decrypt);
    REQUIRE(CudaDirUtil::from_transform_direction(TransformDirection::Encrypt) == CudaDir::Encrypt);
    REQUIRE(CudaDirUtil::to_transform_direction(CudaDir::Encrypt) == TransformDirection::Encrypt);
}

TEST_CASE("CUDA FamilyId from TransformId", "[cuda][params]") {
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::caesar()).value() ==
            CudaFamilyId::Caesar);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::vigenere_key()).value() ==
            CudaFamilyId::VigenereKey);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::compose()).value() ==
            CudaFamilyId::Compose);

    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::hill_2()).value() ==
            CudaFamilyId::Hill2);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::hill_3()).value() ==
            CudaFamilyId::Hill3);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::ciphertext_autokey()).value() ==
            CudaFamilyId::CiphertextAutokey);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::plaintext_autokey()).value() ==
            CudaFamilyId::PlaintextAutokey);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::variable_delay_autokey()).value() ==
            CudaFamilyId::VariableDelayAutokey);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::spiral_read()).value() ==
            CudaFamilyId::SpiralRead);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::boustrophedon_read()).value() ==
            CudaFamilyId::BoustrophedonRead);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::diagonal_read()).value() ==
            CudaFamilyId::DiagonalRead);
    REQUIRE(CudaFamilyIdUtil::from_transform_id(TransformId::columnar_transposition()).value() ==
            CudaFamilyId::ColumnarTransposition);

    REQUIRE_FALSE(CudaFamilyIdUtil::from_transform_id(TransformId::unchecked("theory")).ok());
}

TEST_CASE("CUDA POD params hill2 / hill3 from JSON", "[cuda][params]") {
    SECTION("hill2 identity matrix") {
        StatusOr<CudaParamsJson::HillMatrixHost> m =
            CudaParamsJson::hill2_from_json(nlohmann::json{{"matrix", {1, 0, 0, 1}}});
        REQUIRE(m.ok());
        REQUIRE(m.value().entries == std::vector<std::uint8_t>{1, 0, 0, 1});
        REQUIRE(CudaParamsJson::hill_to_json(m.value()) ==
                nlohmann::json{{"matrix", {1, 0, 0, 1}}});
    }
    SECTION("hill2 singular rejected") {
        REQUIRE_FALSE(
            CudaParamsJson::hill2_from_json(nlohmann::json{{"matrix", {1, 0, 0, 0}}}).ok());
    }
    SECTION("hill3") {
        StatusOr<CudaParamsJson::HillMatrixHost> m = CudaParamsJson::hill3_from_json(
            nlohmann::json{{"matrix", {1, 0, 0, 0, 1, 0, 0, 0, 1}}});
        REQUIRE(m.ok());
        REQUIRE(m.value().entries.size() == 9);
        REQUIRE_FALSE(CudaParamsJson::hill3_from_json(
                          nlohmann::json{{"matrix", {1, 0, 0, 0, 0, 0, 0, 0, 0}}})
                          .ok());
    }
}

TEST_CASE("CUDA POD params autokey families from JSON", "[cuda][params]") {
    SECTION("ctak / ptak key") {
        const nlohmann::json json{{"key_indices", {3, 5}}};
        StatusOr<KeyParamsHost> ctak = CudaParamsJson::ciphertext_autokey_from_json(json);
        StatusOr<KeyParamsHost> ptak = CudaParamsJson::plaintext_autokey_from_json(json);
        REQUIRE(ctak.ok());
        REQUIRE(ptak.ok());
        REQUIRE(ctak.value().key == std::vector<std::uint8_t>{3, 5});
        REQUIRE_FALSE(
            CudaParamsJson::ciphertext_autokey_from_json(nlohmann::json{{"key_indices", {}}})
                .ok());
    }
    SECTION("vda lag and mode") {
        StatusOr<CudaParamsJson::VariableDelayAutokeyHost> vda =
            CudaParamsJson::variable_delay_autokey_from_json(
                nlohmann::json{{"key_indices", {1, 2}}, {"lag", 2}, {"mode", "plaintext"}});
        REQUIRE(vda.ok());
        REQUIRE(vda.value().vda.lag == 2);
        REQUIRE(vda.value().vda.mode == CudaAutokeyMode::Plaintext);
        REQUIRE_FALSE(CudaParamsJson::variable_delay_autokey_from_json(
                          nlohmann::json{{"key_indices", {1}}, {"lag", 4}})
                          .ok());
        StatusOr<CudaParamsJson::VariableDelayAutokeyHost> by_index =
            CudaParamsJson::variable_delay_autokey_from_json(
                nlohmann::json{{"key_indices", {7}}, {"lag_prime_index", 0}});
        REQUIRE(by_index.ok());
        REQUIRE(by_index.value().vda.lag == 2);
    }
}

TEST_CASE("CUDA POD params grid-read from JSON", "[cuda][params]") {
    SECTION("spiral 2x2 inward") {
        StatusOr<CudaParamsJson::GridReadHost> g =
            CudaParamsJson::spiral_read_from_json(nlohmann::json{{"rows", 2}, {"cols", 2}});
        REQUIRE(g.ok());
        REQUIRE(g.value().grid.rows == 2);
        REQUIRE(g.value().grid.cols == 2);
        REQUIRE(g.value().grid.spiral_outward == 0);
        REQUIRE(g.value().order.size() == 4);
        REQUIRE(g.value().order[0] == 0);
    }
    SECTION("boustrophedon rtl") {
        StatusOr<CudaParamsJson::GridReadHost> g = CudaParamsJson::boustrophedon_read_from_json(
            nlohmann::json{{"rows", 2}, {"cols", 2}, {"first_row", "rtl"}});
        REQUIRE(g.ok());
        REQUIRE(g.value().grid.boustrophedon_rtl == 1);
        REQUIRE(g.value().order.size() == 4);
    }
    SECTION("diagonal anti") {
        StatusOr<CudaParamsJson::GridReadHost> g = CudaParamsJson::diagonal_read_from_json(
            nlohmann::json{{"rows", 2}, {"cols", 2}, {"anti", true}});
        REQUIRE(g.ok());
        REQUIRE(g.value().grid.diagonal_anti == 1);
    }
    SECTION("columnar") {
        StatusOr<CudaParamsJson::GridReadHost> g = CudaParamsJson::columnar_transposition_from_json(
            nlohmann::json{{"rows", 2}, {"key_indices", {1, 0}}});
        REQUIRE(g.ok());
        REQUIRE(g.value().grid.cols == 2);
        REQUIRE(g.value().columnar_key.key == std::vector<std::uint8_t>{1, 0});
        REQUIRE(g.value().order.size() == 4);
    }
}

TEST_CASE("CUDA compose stages append hill autokey grid arenas", "[cuda][params]") {
    const nlohmann::json json{
        {"stages",
         {
             {{"transform_id", "hill_2"},
              {"direction", "decrypt"},
              {"params", {{"matrix", {1, 0, 0, 1}}}}},
             {{"transform_id", "ciphertext_autokey"},
              {"direction", "decrypt"},
              {"params", {{"key_indices", {3, 5}}}}},
             {{"transform_id", "spiral_read"},
              {"direction", "decrypt"},
              {"params", {{"rows", 2}, {"cols", 2}, {"spiral", "outward"}}}},
         }},
    };
    StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(json);
    REQUIRE(compose.ok());
    REQUIRE(compose.value().stages.size() == 3);
    REQUIRE(compose.value().stages[0].family == CudaFamilyId::Hill2);
    REQUIRE(compose.value().stages[0].matrix_len == 4);
    REQUIRE(compose.value().matrix_arena == std::vector<std::uint8_t>{1, 0, 0, 1});
    REQUIRE(compose.value().stages[1].family == CudaFamilyId::CiphertextAutokey);
    REQUIRE(compose.value().stages[1].key_len == 2);
    REQUIRE(compose.value().key_arena == std::vector<std::uint8_t>{3, 5});
    REQUIRE(compose.value().stages[2].family == CudaFamilyId::SpiralRead);
    REQUIRE(compose.value().stages[2].grid.spiral_outward == 1);
    REQUIRE(compose.value().stages[2].order_len == 4);
    REQUIRE(compose.value().order_arena.size() == 4);
    REQUIRE_FALSE(CudaParamsJson::compose_from_json(
                      nlohmann::json{{"stages",
                                      {{{"transform_id", "compose"},
                                        {"params", {{"stages", nlohmann::json::array()}}}}}}})
                      .ok());
}

TEST_CASE("CUDA catalog params JSON edge rejects", "[cuda][params][edge]") {
    SECTION("hill wrong length / range / unknown field") {
        REQUIRE_FALSE(
            CudaParamsJson::hill2_from_json(nlohmann::json{{"matrix", {1, 0, 0}}}).ok());
        REQUIRE_FALSE(
            CudaParamsJson::hill2_from_json(nlohmann::json{{"matrix", {1, 0, 0, 29}}}).ok());
        REQUIRE_FALSE(CudaParamsJson::hill2_from_json(
                          nlohmann::json{{"matrix", {1, 0, 0, 1}}, {"extra", 1}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::hill2_from_json(nlohmann::json::array()).ok());
        REQUIRE_FALSE(
            CudaParamsJson::hill3_from_json(nlohmann::json{{"matrix", {1, 0, 0, 0, 1, 0, 0, 0}}})
                .ok());
    }

    SECTION("autokey unknown field / OOB / vda lag rules") {
        REQUIRE_FALSE(CudaParamsJson::ciphertext_autokey_from_json(
                          nlohmann::json{{"key_indices", {1}}, {"shift", 1}})
                          .ok());
        REQUIRE_FALSE(
            CudaParamsJson::plaintext_autokey_from_json(nlohmann::json{{"key_indices", {29}}})
                .ok());
        REQUIRE_FALSE(CudaParamsJson::variable_delay_autokey_from_json(
                          nlohmann::json{{"key_indices", {1}}, {"lag", 2}, {"lag_prime_index", 0}})
                          .ok());
        REQUIRE_FALSE(
            CudaParamsJson::variable_delay_autokey_from_json(nlohmann::json{{"key_indices", {1}}})
                .ok());
        REQUIRE_FALSE(CudaParamsJson::variable_delay_autokey_from_json(
                          nlohmann::json{{"key_indices", {1}}, {"lag", 1}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::variable_delay_autokey_from_json(
                          nlohmann::json{{"key_indices", {1}}, {"lag", 2}, {"mode", "otp"}})
                          .ok());
        StatusOr<CudaParamsJson::VariableDelayAutokeyHost> ok_latin =
            CudaParamsJson::variable_delay_autokey_from_json(
                nlohmann::json{{"key_indices", {1}}, {"key_latin", "A"}, {"lag", 3}});
        REQUIRE(ok_latin.ok());
        REQUIRE(ok_latin.value().vda.lag == 3);
        REQUIRE(ok_latin.value().vda.mode == CudaAutokeyMode::Ciphertext);
    }

    SECTION("grid dims / flags / columnar") {
        REQUIRE_FALSE(
            CudaParamsJson::spiral_read_from_json(nlohmann::json{{"rows", 0}, {"cols", 2}}).ok());
        REQUIRE_FALSE(
            CudaParamsJson::spiral_read_from_json(nlohmann::json{{"rows", 2}}).ok());
        REQUIRE_FALSE(CudaParamsJson::spiral_read_from_json(
                          nlohmann::json{{"rows", 2}, {"cols", 2}, {"spiral", "sideways"}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::boustrophedon_read_from_json(
                          nlohmann::json{{"rows", 2}, {"cols", 2}, {"first_row", "up"}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::diagonal_read_from_json(
                          nlohmann::json{{"rows", 2}, {"cols", 2}, {"anti", 1}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::columnar_transposition_from_json(
                          nlohmann::json{{"rows", 2}, {"key_indices", nlohmann::json::array()}})
                          .ok());
        REQUIRE_FALSE(CudaParamsJson::columnar_transposition_from_json(
                          nlohmann::json{{"rows", 2}, {"key_indices", {1}}, {"foo", true}})
                          .ok());

        StatusOr<CudaParamsJson::GridReadHost> row =
            CudaParamsJson::spiral_read_from_json(nlohmann::json{{"rows", 1}, {"cols", 4}});
        REQUIRE(row.ok());
        REQUIRE(row.value().order.size() == 4);

        StatusOr<CudaParamsJson::GridReadHost> col =
            CudaParamsJson::spiral_read_from_json(nlohmann::json{{"rows", 4}, {"cols", 1}});
        REQUIRE(col.ok());
        REQUIRE(col.value().order.size() == 4);
    }
}

TEST_CASE("CUDA compose multi-family arena offsets and reject bad stage", "[cuda][params][edge]") {
    SECTION("accumulating arenas across hill3 ptak vda columnar") {
        const nlohmann::json json{
            {"stages",
             {
                 {{"transform_id", "hill_3"},
                  {"params", {{"matrix", {1, 0, 0, 0, 1, 0, 0, 0, 1}}}}},
                 {{"transform_id", "plaintext_autokey"}, {"params", {{"key_indices", {2, 4}}}}},
                 {{"transform_id", "variable_delay_autokey"},
                  {"params", {{"key_indices", {9}}, {"lag", 2}, {"mode", "ciphertext"}}}},
                 {{"transform_id", "columnar_transposition"},
                  {"params", {{"rows", 2}, {"key_indices", {0, 1}}}}},
                 {{"transform_id", "boustrophedon_read"},
                  {"params", {{"rows", 2}, {"cols", 2}, {"first_row", "ltr"}}}},
                 {{"transform_id", "diagonal_read"},
                  {"params", {{"rows", 2}, {"cols", 2}, {"anti", false}}}},
             }},
        };
        StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(json);
        REQUIRE(compose.ok());
        REQUIRE(compose.value().stages.size() == 6);

        REQUIRE(compose.value().stages[0].family == CudaFamilyId::Hill3);
        REQUIRE(compose.value().stages[0].matrix_begin == 0);
        REQUIRE(compose.value().stages[0].matrix_len == 9);
        REQUIRE(compose.value().matrix_arena.size() == 9);

        REQUIRE(compose.value().stages[1].family == CudaFamilyId::PlaintextAutokey);
        REQUIRE(compose.value().stages[1].key_begin == 0);
        REQUIRE(compose.value().stages[1].key_len == 2);

        REQUIRE(compose.value().stages[2].family == CudaFamilyId::VariableDelayAutokey);
        REQUIRE(compose.value().stages[2].key_begin == 2);
        REQUIRE(compose.value().stages[2].key_len == 1);
        REQUIRE(compose.value().stages[2].vda.lag == 2);
        REQUIRE(compose.value().key_arena == std::vector<std::uint8_t>{2, 4, 9, 0, 1});

        REQUIRE(compose.value().stages[3].family == CudaFamilyId::ColumnarTransposition);
        REQUIRE(compose.value().stages[3].key_begin == 3);
        REQUIRE(compose.value().stages[3].key_len == 2);
        REQUIRE(compose.value().stages[3].order_begin == 0);
        REQUIRE(compose.value().stages[3].order_len == 4);

        REQUIRE(compose.value().stages[4].family == CudaFamilyId::BoustrophedonRead);
        REQUIRE(compose.value().stages[4].order_begin == 4);
        REQUIRE(compose.value().stages[4].order_len == 4);
        REQUIRE(compose.value().stages[4].grid.boustrophedon_rtl == 0);

        REQUIRE(compose.value().stages[5].family == CudaFamilyId::DiagonalRead);
        REQUIRE(compose.value().stages[5].order_begin == 8);
        REQUIRE(compose.value().stages[5].order_len == 4);
        REQUIRE(compose.value().order_arena.size() == 12);
    }

    SECTION("bad stage params fail whole compose") {
        const nlohmann::json bad{
            {"stages",
             nlohmann::json::array(
                 {{{"transform_id", "atbash"}, {"params", nlohmann::json::object()}},
                  {{"transform_id", "hill_2"}, {"params", {{"matrix", {1, 0, 0, 0}}}}}})}};
        StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(bad);
        REQUIRE_FALSE(compose.ok());
    }
}
