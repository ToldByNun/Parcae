#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <parcae/core/status_or.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <string>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path data_root() {
    return std::filesystem::path(PARCAE_TEST_DATA_DIR);
}

[[nodiscard]] nlohmann::json valid_job_json() {
    return nlohmann::json{
        {"schema", "parcae.search_job.v0"},
        {"workspace_id", "_example"},
        {"family", "caesar"},
        {"score_id", "chi2_english_gp_v0"},
        {"score_version", "v0"},
        {"k", 16},
        {"seed", 1},
        {"backend", "cpu"},
        {"max_candidates", 4096},
        {"direction", "decrypt"},
        {"param_grid", nullptr},
        {"prior", nullptr},
    };
}

} // namespace

TEST_CASE("SearchJob make and round-trip JSON", "[search][job]") {
    StatusOr<SearchJob> job =
        SearchJob::make("_example", "caesar", "chi2_english_gp_v0", 16, 1, Backend::Cpu, 4096);
    REQUIRE(job.ok());
    REQUIRE(job.value().workspace_id() == "_example");
    REQUIRE(job.value().family() == "caesar");
    REQUIRE(job.value().k() == 16);
    REQUIRE(job.value().max_candidates() == 4096);
    REQUIRE(job.value().backend() == Backend::Cpu);
    REQUIRE(job.value().direction() == TransformDirection::Decrypt);
    REQUIRE(job.value().score_version() == "v0");
    REQUIRE_FALSE(job.value().prior().has_value());

    StatusOr<SearchJob> again = SearchJob::from_json(job.value().to_json());
    REQUIRE(again.ok());
    REQUIRE(again.value().to_json() == job.value().to_json());
    REQUIRE(again.value().job_digest_sha256() == job.value().job_digest_sha256());
}

TEST_CASE("SearchJob from_json accepts committed shape", "[search][job]") {
    StatusOr<SearchJob> job = SearchJob::from_json(valid_job_json());
    REQUIRE(job.ok());
    REQUIRE(job.value().require_workspace_dir(data_root()).ok());
}

TEST_CASE("SearchJob rejects bad schema and bounds", "[search][job]") {
    {
        nlohmann::json j = valid_job_json();
        j["schema"] = "wrong";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["k"] = 0;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["k"] = 32;
        j["max_candidates"] = 16;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "totient";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "totient";
        j["allow_extended_families"] = true;
        StatusOr<SearchJob> job = SearchJob::from_json(j);
        REQUIRE(job.ok());
        REQUIRE(job.value().family() == "totient");
        REQUIRE(job.value().allow_extended_families());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        REQUIRE_FALSE(SearchJob::from_json(j).ok()); // missing params_list
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        j["param_grid"] = {{"theory_uri", "parcae://theories/quadratic_polynomial_stream@1"},
                           {"params_list", nlohmann::json::array(
                                               {nlohmann::json{{"c2", 1}, {"c1", 0}, {"c0", 0}}})}};
        StatusOr<SearchJob> job = SearchJob::from_json(j);
        REQUIRE(job.ok());
        REQUIRE(job.value().family() == "theory");
        REQUIRE(job.value().allow_theory_uri());
        REQUIRE(job.value().to_json().at("allow_theory_uri") == true);
    }
    {
        nlohmann::json j = valid_job_json();
        j["backend"] = "metal";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["workspace_id"] = "BadId";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["prior"] = nlohmann::json{{"schema", "nope"}};
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
}

TEST_CASE("SearchJob digest is key-order stable", "[search][job]") {
    nlohmann::json a = valid_job_json();
    nlohmann::json b{
        {"prior", nullptr},
        {"max_candidates", 4096},
        {"backend", "cpu"},
        {"seed", 1},
        {"k", 16},
        {"score_version", "v0"},
        {"score_id", "chi2_english_gp_v0"},
        {"family", "caesar"},
        {"workspace_id", "_example"},
        {"schema", "parcae.search_job.v0"},
        {"direction", "decrypt"},
        {"param_grid", nullptr},
    };
    StatusOr<SearchJob> ja = SearchJob::from_json(a);
    StatusOr<SearchJob> jb = SearchJob::from_json(b);
    REQUIRE(ja.ok());
    REQUIRE(jb.ok());
    REQUIRE(ja.value().job_digest_sha256() == jb.value().job_digest_sha256());
}

TEST_CASE("SearchJob load_file round-trip", "[search][job]") {
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_search_job_b5";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);

    StatusOr<SearchJob> job = SearchJob::from_json(valid_job_json());
    REQUIRE(job.ok());
    const auto path = tmp / "job.json";
    {
        std::ofstream out(path);
        REQUIRE(out);
        out << job.value().to_json().dump(2);
    }
    StatusOr<SearchJob> loaded = SearchJob::load_file(path);
    REQUIRE(loaded.ok());
    REQUIRE(loaded.value().job_digest_sha256() == job.value().job_digest_sha256());

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("SearchJob require_workspace_dir fails for missing id", "[search][job]") {
    StatusOr<SearchJob> job =
        SearchJob::make("no_such_workspace_zzz", "atbash", "ic_mod29", 1, 42, Backend::Cpu, 1);
    REQUIRE(job.ok());
    REQUIRE_FALSE(job.value().require_workspace_dir(data_root()).ok());
}
