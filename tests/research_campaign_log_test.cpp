#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <parcae/hypothesis/research_campaign_log.hpp>
#include <parcae/hypothesis/workspace_scaffold.hpp>
#include <parcae/tool/tool_backend.hpp>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] std::filesystem::path make_sandbox(std::string_view name) {
    const auto root = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "workspaces", ec);
    return root;
}

[[nodiscard]] std::string read_text(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("ResearchCampaignLog: looks_host_absolute detects drive/UNC/unix",
          "[hypothesis][campaign][log]") {
    REQUIRE(ResearchCampaignLog::looks_host_absolute("/tmp/x"));
    REQUIRE(ResearchCampaignLog::looks_host_absolute("C:/Users/x"));
    REQUIRE(ResearchCampaignLog::looks_host_absolute("d:\\ws"));
    REQUIRE(ResearchCampaignLog::looks_host_absolute("\\\\server\\share"));
    REQUIRE_FALSE(ResearchCampaignLog::looks_host_absolute("inputs/ciphertext.txt"));
    REQUIRE_FALSE(ResearchCampaignLog::looks_host_absolute("research/run.log"));
    REQUIRE_FALSE(ResearchCampaignLog::looks_host_absolute(""));
}

TEST_CASE("ResearchCampaignLog: append no-op without research layout",
          "[hypothesis][campaign][log]") {
    const auto data = make_sandbox("parcae_campaign_log_noop");
    std::filesystem::create_directories(data / "workspaces" / "slim");

    ResearchCampaignLog::SearchCycleEntry entry;
    entry.family = "caesar";
    entry.score_id = "chi2_english_gp_v0";
    entry.candidates_retained = 16;
    entry.candidates_expanded = 29;
    entry.rune_count = 100;
    entry.wall_seconds = 0.5;

    Status st = ResearchCampaignLog::append_search_cycle(data, "slim", entry);
    REQUIRE(st.ok());
    REQUIRE_FALSE(std::filesystem::exists(data / "workspaces" / "slim" / "research" / "run.log"));
}

TEST_CASE("ResearchCampaignLog: append formats line and redacts abs paths",
          "[hypothesis][campaign][log]") {
    const auto data = make_sandbox("parcae_campaign_log_append");

    WorkspaceScaffold::Options opts;
    opts.created_utc = "2026-09-28T00:00:00Z";
    opts.title = "campaign log";
    StatusOr<WorkspaceScaffold::Result> sc =
        WorkspaceScaffold::ensure_research(data, "lp2-camp", opts);
    REQUIRE(sc.ok());

    ResearchCampaignLog::SearchCycleEntry entry;
    entry.family = "theory";
    entry.score_id = "chi2_english_gp_v0";
    entry.backend_requested = Backend::Cuda;
    entry.export_backend = Backend::Cpu;
    entry.candidates_retained = 8;
    entry.candidates_expanded = 25;
    entry.rune_count = 1000;
    entry.wall_seconds = 2.0;
    entry.batch_id = "batch_0001";
    entry.created_utc = "2026-09-28T00:00:00Z";
    entry.notes = "C:/Users/mikaj/secret"; // MUST be redacted

    Status st = ResearchCampaignLog::append_search_cycle(data, "lp2-camp", entry);
    REQUIRE(st.ok());

    const auto log_path = data / "workspaces" / "lp2-camp" / "research" / "run.log";
    REQUIRE(std::filesystem::is_regular_file(log_path));
    const std::string line = read_text(log_path);

    REQUIRE(line.find("event=search_cycle") != std::string::npos);
    REQUIRE(line.find("tool=search_cycle") != std::string::npos);
    REQUIRE(line.find("family=theory") != std::string::npos);
    REQUIRE(line.find("backend_requested=cuda") != std::string::npos);
    REQUIRE(line.find("export_backend=cpu") != std::string::npos);
    REQUIRE(line.find("candidates_retained=8") != std::string::npos);
    REQUIRE(line.find("candidates_expanded=25") != std::string::npos);
    REQUIRE(line.find("rune_count=1000") != std::string::npos);
    REQUIRE(line.find("runes_work=25000") != std::string::npos);
    REQUIRE(line.find("wall_seconds=2") != std::string::npos);
    REQUIRE(line.find("runes_per_s=12500") != std::string::npos);
    REQUIRE(line.find("engine=parcae") != std::string::npos);
    REQUIRE(line.find("notes=REDACTED_ABS_PATH") != std::string::npos);
    REQUIRE(line.find("C:/Users") == std::string::npos);
    REQUIRE((data.string().empty() || line.find(data.string()) == std::string::npos));

    // Second append grows the file (non-golden stream).
    entry.notes = "engine=parcae_search_cycle";
    entry.wall_seconds = 1.0;
    REQUIRE(ResearchCampaignLog::append_search_cycle(data, "lp2-camp", entry).ok());
    const std::string both = read_text(log_path);
    REQUIRE(both.find("notes=engine=parcae_search_cycle") != std::string::npos);
    const auto newlines = static_cast<std::size_t>(
        std::count(both.begin(), both.end(), '\n'));
    REQUIRE(newlines == 2);
}
