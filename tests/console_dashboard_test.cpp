#include <catch2/catch_test_macros.hpp>

#include "parcae/cli/console_ansi.hpp"
#include "parcae/cli/console_dashboard.hpp"
#include "parcae/cli/console_progress_sink.hpp"

#include <sstream>
#include <string>
#include <vector>

class ConsoleDashboardTestFixtures {
public:
    [[nodiscard]] static ConsoleProgressSnapshot sample_snapshot() {
        ConsoleProgressSnapshot snap;
        snap.set_tool("search_cycle");
        snap.set_workspace_id("ws-a");
        snap.set_family("caesar");
        snap.set_backend("cpu");
        snap.set_score_id("chi2_english_gp_v0");
        snap.set_stage("score");
        snap.set_candidates_done(25);
        snap.set_candidates_total(50);
        snap.set_rune_count(200);
        snap.set_elapsed_seconds(2.0);
        snap.set_best_score(580.5);
        snap.set_best_label("shift=11");
        snap.set_iteration(1, 3);
        snap.refresh_rates();
        return snap;
    }
};

class RecordingProgressSink : public ConsoleProgressSink {
public:
    void on_progress(const ConsoleProgressSnapshot& snapshot) override {
        progress_stages.push_back(snapshot.stage());
    }

    void on_stage(
        std::string_view stage,
        const ConsoleProgressSnapshot& /*snapshot*/) override {
        stages.emplace_back(stage);
    }

    std::vector<std::string> progress_stages;
    std::vector<std::string> stages;
};

TEST_CASE("ConsoleProgressNoOpSink accepts events", "[cli][dashboard]") {
    ConsoleProgressNoOpSink sink;
    const ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    sink.on_progress(snap);
    sink.on_stage("expand", snap);
}

TEST_CASE("RecordingProgressSink captures progress and stage", "[cli][dashboard]") {
    RecordingProgressSink sink;
    const ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    sink.on_progress(snap);
    sink.on_stage("bridge", snap);
    REQUIRE(sink.progress_stages.size() == 1);
    REQUIRE(sink.progress_stages[0] == "score");
    REQUIRE(sink.stages.size() == 1);
    REQUIRE(sink.stages[0] == "bridge");
}

TEST_CASE("ConsoleDashboard format_line golden", "[cli][dashboard]") {
    const ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    const std::string line = ConsoleDashboard::format_line(snap);
    REQUIRE(
        line ==
        "[search_cycle] score 25/50 (50.0%) 12.50c/s 2.50k runes/s eta=2.0s "
        "best=580.5000 shift=11 iter=1/3 t=2.00s");
}

TEST_CASE("ConsoleDashboard format_panel golden", "[cli][dashboard]") {
    const ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    REQUIRE(ConsoleDashboard::panel_line_count() == 6);
    const std::vector<std::string> lines =
        ConsoleDashboard::format_panel_lines(snap, 10);
    REQUIRE(lines.size() == 6);
    REQUIRE(
        lines[0] ==
        "PARCAE  search_cycle  ws=ws-a  family=caesar  backend=cpu  "
        "score=chi2_english_gp_v0");
    REQUIRE(lines[1] == "stage=score  iter=1/3");
    REQUIRE(lines[2] == "[#####-----] 50.0%  25/50");
    REQUIRE(lines[3] == "runes=200  2.50k runes/s  12.50 cand/s  eta=2.0s");
    REQUIRE(lines[4] == "best=580.5000  shift=11");
    REQUIRE(lines[5] == "elapsed=2.00s");

    const std::string panel = ConsoleDashboard::format_panel(snap, 10);
    REQUIRE(
        panel ==
        "PARCAE  search_cycle  ws=ws-a  family=caesar  backend=cpu  "
        "score=chi2_english_gp_v0\n"
        "stage=score  iter=1/3\n"
        "[#####-----] 50.0%  25/50\n"
        "runes=200  2.50k runes/s  12.50 cand/s  eta=2.0s\n"
        "best=580.5000  shift=11\n"
        "elapsed=2.00s");
}

TEST_CASE("ConsoleDashboard format_throughput scales B/M/k", "[cli][dashboard]") {
    REQUIRE(ConsoleDashboard::format_throughput(1500.0) == "1.50k runes/s");
    REQUIRE(ConsoleDashboard::format_throughput(2.5e6) == "2.50M runes/s");
    REQUIRE(ConsoleDashboard::format_throughput(3.25e9) == "3.25B runes/s");
    REQUIRE(ConsoleDashboard::format_throughput(42.0) == "42.00 runes/s");
}

TEST_CASE("ConsoleDashboard Off mode writes nothing", "[cli][dashboard]") {
    std::ostringstream out;
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Off};
    options.out = &out;
    ConsoleDashboard dash(options);
    const ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    dash.on_progress(snap);
    dash.on_stage("score", snap);
    dash.finish(snap);
    REQUIRE(out.str().empty());
}

TEST_CASE("ConsoleDashboard lines mode appends and finishes", "[cli][dashboard]") {
    std::ostringstream out;
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Lines};
    options.throttle_seconds = 0.0;
    options.out = &out;
    ConsoleDashboard dash(options);
    dash.clock().set_elapsed_override(0.0);

    ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    dash.on_stage("score", snap);
    snap.set_candidates_done(26);
    snap.refresh_rates();
    dash.clock().set_elapsed_override(1.0);
    dash.on_progress(snap);
    dash.finish(snap);

    const std::string text = out.str();
    REQUIRE(text.find("[search_cycle] score 25/50") != std::string::npos);
    REQUIRE(text.find("[done]") != std::string::npos);
    std::size_t newlines = 0;
    for (char c : text) {
        if (c == '\n') {
            ++newlines;
        }
    }
    REQUIRE(newlines == 3);
}

TEST_CASE("ConsoleDashboard throttles progress but not stage", "[cli][dashboard]") {
    std::ostringstream out;
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Lines};
    options.throttle_seconds = 1.0;
    options.out = &out;
    ConsoleDashboard dash(options);
    dash.clock().set_elapsed_override(0.0);

    ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    dash.on_progress(snap);
    dash.clock().set_elapsed_override(0.2);
    dash.on_progress(snap);
    dash.clock().set_elapsed_override(0.4);
    dash.on_stage("bridge", snap);
    dash.clock().set_elapsed_override(1.5);
    dash.on_progress(snap);

    const std::string text = out.str();
    std::size_t newlines = 0;
    for (char c : text) {
        if (c == '\n') {
            ++newlines;
        }
    }
    REQUIRE(newlines == 3);
    REQUIRE(text.find("] bridge ") != std::string::npos);
}

TEST_CASE("ConsoleDashboard panel mode emits ANSI cursor control", "[cli][dashboard]") {
    ConsoleAnsi::set_vt_enable_override(true);
    std::ostringstream out;
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Panel};
    options.throttle_seconds = 0.0;
    options.bar_width = 10;
    options.out = &out;
    ConsoleDashboard dash(options);
    dash.clock().set_elapsed_override(0.0);

    ConsoleProgressSnapshot snap = ConsoleDashboardTestFixtures::sample_snapshot();
    dash.on_stage("score", snap);
    dash.clock().set_elapsed_override(1.0);
    snap.set_candidates_done(30);
    snap.refresh_rates();
    dash.on_progress(snap);
    dash.finish(snap);
    ConsoleAnsi::clear_overrides();

    const std::string text = out.str();
    REQUIRE(text.find(ConsoleAnsi::hide_cursor()) != std::string::npos);
    REQUIRE(text.find(ConsoleAnsi::cursor_up(6)) != std::string::npos);
    REQUIRE(text.find(ConsoleAnsi::clear_to_eol()) != std::string::npos);
    REQUIRE(text.find(ConsoleAnsi::show_cursor()) != std::string::npos);
    REQUIRE(text.find("[#####-----]") != std::string::npos);
    REQUIRE(text.find("[done]") != std::string::npos);
}

TEST_CASE("ConsoleDashboard panel falls back to lines when VT fails", "[cli][dashboard]") {
    ConsoleAnsi::set_vt_enable_override(false);
    std::ostringstream out;
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Panel};
    options.throttle_seconds = 0.0;
    options.out = &out;
    ConsoleDashboard dash(options);
    dash.clock().set_elapsed_override(0.0);
    dash.on_stage("score", ConsoleDashboardTestFixtures::sample_snapshot());
    ConsoleAnsi::clear_overrides();

    REQUIRE(dash.mode().is_lines());
    const std::string text = out.str();
    REQUIRE(text.find(ConsoleAnsi::hide_cursor()) == std::string::npos);
    REQUIRE(text.find("[search_cycle] score") != std::string::npos);
}

TEST_CASE("ConsoleDashboard Auto mode normalizes to Lines", "[cli][dashboard]") {
    ConsoleDashboard::Options options;
    options.mode = ConsoleProgressMode{ConsoleProgressMode::Kind::Auto};
    ConsoleDashboard dash(options);
    REQUIRE(dash.mode().is_lines());
}
