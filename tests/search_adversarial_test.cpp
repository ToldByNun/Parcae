#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <parcae/core/status_or.hpp>
#include <parcae/hypothesis/workspace_manifest.hpp>
#include <parcae/hypothesis/workspace_paths.hpp>
#include <parcae/search/cpu_candidate_export.hpp>
#include <parcae/search/search_job.hpp>
#include <parcae/search/search_scheduler.hpp>
#include <parcae/search/workspace_cipher.hpp>
#include <parcae/tool/context.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <parcae/transform/transform_direction.hpp>
#include <string>
#include <string_view>
#include <vector>

#ifndef PARCAE_TEST_DATA_DIR
#error "PARCAE_TEST_DATA_DIR must be defined"
#endif

namespace {

[[nodiscard]] std::filesystem::path repo_data() {
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

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    std::filesystem::create_directories(root / "profiles" / "scores", ec);
    std::filesystem::create_directories(root / "profiles" / "gematria", ec);
    std::filesystem::create_directories(root / "profiles" / "separators", ec);
    std::filesystem::create_directories(root / "fixtures" / "solved" / "a-warning", ec);

    std::filesystem::copy_file(repo_data() / "profiles" / "scores" / "english-gp-expected-v0.json",
                               root / "profiles" / "scores" / "english-gp-expected-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "gematria" / "gematria-primus-v0.json",
                               root / "profiles" / "gematria" / "gematria-primus-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(repo_data() / "profiles" / "separators" /
                                   "rtkd-separator-grammar-v0.json",
                               root / "profiles" / "separators" / "rtkd-separator-grammar-v0.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);

    const auto src_fix = repo_data() / "fixtures" / "solved" / "a-warning";
    const auto dst_fix = root / "fixtures" / "solved" / "a-warning";
    std::filesystem::copy_file(src_fix / "ciphertext.txt", dst_fix / "ciphertext.txt",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    std::filesystem::copy_file(src_fix / "manifest.json", dst_fix / "manifest.json",
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE(!ec);
    return root;
}

[[nodiscard]] StatusOr<WorkspaceManifest> make_fixture_workspace(std::string_view workspace_id,
                                                                 std::string_view utc) {
    nlohmann::json root{
        {"schema", "parcae.workspace.v0"},
        {"id", std::string(workspace_id)},
        {"created_utc", std::string(utc)},
        {"updated_utc", std::string(utc)},
        {"title", "l44"},
        {"notes", ""},
        {"input", {{"kind", "fixture_ciphertext"}, {"fixture_id", "a-warning"}, {"path", nullptr}}},
        {"default_score_id", "chi2_english_gp_v0"},
        {"default_score_version", "v0"},
    };
    return WorkspaceManifest::from_json(root);
}

} // namespace

TEST_CASE("L44 adversarial SearchJob JSON: types, missing fields, bounds",
          "[search][adversarial][job]") {
    REQUIRE_FALSE(SearchJob::from_json(nlohmann::json()).ok());
    REQUIRE_FALSE(SearchJob::from_json(nlohmann::json::array()).ok());
    REQUIRE_FALSE(SearchJob::from_json(nlohmann::json("string")).ok());
    REQUIRE_FALSE(SearchJob::parse("{not json").ok());
    REQUIRE_FALSE(SearchJob::parse("[]").ok());

    const std::vector<std::string> required = {
        "schema", "workspace_id", "family", "score_id", "k", "seed", "backend", "max_candidates",
    };
    for (const std::string& key : required) {
        nlohmann::json j = valid_job_json();
        j.erase(key);
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }

    {
        nlohmann::json j = valid_job_json();
        j["k"] = 1.5;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["k"] = "16";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["k"] = -1;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["max_candidates"] = 3.14;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["max_candidates"] = -5;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["seed"] = -1;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["seed"] = 0x100000000ll; // > uint32
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = 29;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "unknown_family_xyz";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["score_id"] = "";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["score_version"] = std::string(33, 'x');
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["param_grid"] = nlohmann::json::array({1, 2, 3});
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["direction"] = "sideways";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["allow_extended_families"] = "yes";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["allow_theory_uri"] = 1;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "beaufort";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "hill_3";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "plaintext_autokey";
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["k"] = 100;
        j["max_candidates"] = 50;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        // Soft-weight priors are rejected via inline prior validation.
        nlohmann::json j = valid_job_json();
        j["prior"] = {
            {"schema", "parcae.search_prior.v0"},
            {"workspace_id", "_example"},
            {"built_utc", "2026-09-23T00:00:00Z"},
            {"seeds", nlohmann::json::array()},
            {"exclusions", nlohmann::json::array()},
            {"soft_weights", nlohmann::json{{"caesar", 0.5}}},
        };
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
}

TEST_CASE("L44 adversarial workspace_id / path-like ids reject", "[search][adversarial][path]") {
    const std::vector<std::string> bad_ids = {
        "",    "..",    "../x",      "foo/bar", "foo\\bar",
        "Foo", "9bad",  "has space", "has.dot", std::string(65, 'a'),
        "c:",  "~home",
    };
    for (const std::string& id : bad_ids) {
        REQUIRE_FALSE(WorkspacePaths::validate_id(id).ok());
        nlohmann::json j = valid_job_json();
        j["workspace_id"] = id;
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }

    // Relative escapes must fail under workspace root resolution.
    const auto tmp = std::filesystem::temp_directory_path() / "parcae_l44_path";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp / "safe", ec);

    REQUIRE_FALSE(WorkspacePaths::resolve_under(tmp / "safe", "../outside.txt").ok());
    REQUIRE_FALSE(WorkspacePaths::resolve_under(tmp / "safe", "/etc/passwd").ok());
#if defined(_WIN32)
    REQUIRE_FALSE(WorkspacePaths::resolve_under(tmp / "safe", "C:\\Windows\\System32").ok());
#endif
    REQUIRE_FALSE(WorkspacePaths::resolve_under(tmp / "safe", "a/../../outside.txt").ok());

    // Absolute ciphertext path via workspace_file kind.
    StatusOr<WorkspaceManifest> base =
        WorkspaceManifest::make("l44_escape", "2026-09-23T06:00:00Z");
    REQUIRE(base.ok());
    nlohmann::json manifest = base.value().to_json();
    manifest["input"] = {
        {"kind", "workspace_file"},
        {"fixture_id", nullptr},
        {"path",
#if defined(_WIN32)
         "C:/Windows/System32/drivers/etc/hosts"
#else
         "/etc/passwd"
#endif
        },
    };
    StatusOr<WorkspaceManifest> with_abs = WorkspaceManifest::from_json(manifest);
    if (with_abs.ok()) {
        // Manifest may accept the string; load MUST still reject path escape.
        const auto data = make_sandbox("parcae_l44_abs_cipher");
        REQUIRE(with_abs.value().store(data).ok());
        REQUIRE_FALSE(WorkspaceCipher::load(data, "l44_escape").ok());
        std::filesystem::remove_all(data, ec);
    } else {
        // Or the manifest layer rejects absolute paths up front — also fine.
        REQUIRE_FALSE(with_abs.ok());
    }

    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("L44 caps: max_candidates enforced on export and scheduler",
          "[search][adversarial][caps]") {
    const auto data = make_sandbox("parcae_l44_caps");
    StatusOr<WorkspaceManifest> ws = make_fixture_workspace("l44_caps", "2026-09-23T06:00:00Z");
    REQUIRE(ws.ok());
    REQUIRE(ws.value().store(data).ok());

    // Caesar expands to 29 shifts — max_candidates=10 must fail loudly.
    StatusOr<SearchJob> tight =
        SearchJob::make("l44_caps", "caesar", "chi2_english_gp_v0", 5, 1, Backend::Cpu, 10);
    REQUIRE(tight.ok());

    const Context ctx(data);
    StatusOr<WorkspaceCipher> cipher = WorkspaceCipher::load(data, "l44_caps");
    REQUIRE(cipher.ok());

    StatusOr<CpuCandidateExport::Result> exported =
        CpuCandidateExport::from_job(cipher.value().indices(), tight.value(), ctx);
    REQUIRE_FALSE(exported.ok());
    REQUIRE(exported.status().message().find("max_candidates") != std::string::npos);

    SearchScheduler::Options opts;
    opts.omit_timing = true;
    opts.created_utc = "2026-09-23T06:00:00Z";
    opts.batch_id = "b_l44_caps_fail";
    StatusOr<SearchScheduler::CycleResult> cycle =
        SearchScheduler::run_once(data, ctx, tight.value(), opts);
    REQUIRE_FALSE(cycle.ok());
    REQUIRE(cycle.status().message().find("max_candidates") != std::string::npos);

    // Same job with a sufficient cap succeeds.
    StatusOr<SearchJob> ok_job =
        SearchJob::make("l44_caps", "caesar", "chi2_english_gp_v0", 5, 1, Backend::Cpu, 29);
    REQUIRE(ok_job.ok());
    opts.batch_id = "b_l44_caps_ok";
    StatusOr<SearchScheduler::CycleResult> ok_cycle =
        SearchScheduler::run_once(data, ctx, ok_job.value(), opts);
    REQUIRE(ok_cycle.ok());
    REQUIRE(ok_cycle.value().hypotheses_written() == 5);

    std::error_code ec;
    std::filesystem::remove_all(data, ec);
}

TEST_CASE("L44 theory param_grid adversarial shapes", "[search][adversarial][job][theory]") {
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        j["param_grid"] = {{"theory_uri", "parcae://theories/x@1"}};
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        j["param_grid"] = {
            {"theory_uri", ""},
            {"params_list", nlohmann::json::array({nlohmann::json::object()})},
        };
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        j["param_grid"] = {
            {"theory_uri", "parcae://theories/x@1"},
            {"params_list", nlohmann::json::array({1, 2, 3})},
        };
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
    {
        nlohmann::json j = valid_job_json();
        j["family"] = "theory";
        j["allow_theory_uri"] = true;
        j["param_grid"] = {
            {"theory_uri", "parcae://theories/x@1"},
            {"params_list", nlohmann::json::array()},
        };
        REQUIRE_FALSE(SearchJob::from_json(j).ok());
    }
}
