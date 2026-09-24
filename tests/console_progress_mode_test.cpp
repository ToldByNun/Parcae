#include "parcae/cli/console_progress_clock.hpp"
#include "parcae/cli/console_progress_mode.hpp"
#include "parcae/cli/console_progress_snapshot.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <string>
#include <thread>

TEST_CASE("ConsoleProgressMode parse accepts auto panel lines off", "[cli][progress]") {
    REQUIRE(ConsoleProgressMode::parse("auto").value().kind() == ConsoleProgressMode::Kind::Auto);
    REQUIRE(ConsoleProgressMode::parse("PANEL").value().is_panel());
    REQUIRE(ConsoleProgressMode::parse("Lines").value().is_lines());
    REQUIRE(ConsoleProgressMode::parse("off").value().is_off());
    REQUIRE(ConsoleProgressMode::parse("auto").value().to_string() == "auto");
    REQUIRE(ConsoleProgressMode::parse("panel").value().to_string() == "panel");
    REQUIRE(ConsoleProgressMode::parse("lines").value().to_string() == "lines");
    REQUIRE(ConsoleProgressMode::parse("off").value().to_string() == "off");
}

TEST_CASE("ConsoleProgressMode parse rejects unknown tokens", "[cli][progress]") {
    const StatusOr<ConsoleProgressMode> bad = ConsoleProgressMode::parse("dashboard");
    REQUIRE_FALSE(bad.ok());
    REQUIRE(bad.status().message().find("auto|panel|lines|off") != std::string::npos);

    const StatusOr<ConsoleProgressMode> empty = ConsoleProgressMode::parse("");
    REQUIRE_FALSE(empty.ok());
}

TEST_CASE("ConsoleProgressMode from_flags precedence quiet > plain > progress", "[cli][progress]") {
    REQUIRE(ConsoleProgressMode::from_flags(true, true, "panel").value().is_off());
    REQUIRE(ConsoleProgressMode::from_flags(false, true, "panel").value().is_lines());
    REQUIRE(ConsoleProgressMode::from_flags(false, false, "panel").value().is_panel());
    REQUIRE(ConsoleProgressMode::from_flags(false, false, "").value().is_auto());
    REQUIRE_FALSE(ConsoleProgressMode::from_flags(false, false, "nope").ok());
}

TEST_CASE("ConsoleProgressMode resolve Auto against tty", "[cli][progress]") {
    const ConsoleProgressMode auto_mode{ConsoleProgressMode::Kind::Auto};
    REQUIRE(auto_mode.resolve(true).is_panel());
    REQUIRE(auto_mode.resolve(false).is_lines());

    const ConsoleProgressMode panel{ConsoleProgressMode::Kind::Panel};
    REQUIRE(panel.resolve(false).is_panel());

    const ConsoleProgressMode lines{ConsoleProgressMode::Kind::Lines};
    REQUIRE(lines.resolve(true).is_lines());

    const ConsoleProgressMode off{ConsoleProgressMode::Kind::Off};
    REQUIRE(off.resolve(true).is_off());
}

TEST_CASE("ConsoleProgressSnapshot defaults and fraction/eta/rates", "[cli][progress]") {
    ConsoleProgressSnapshot snap;
    REQUIRE(snap.tool().empty());
    REQUIRE(snap.workspace_id().empty());
    REQUIRE(snap.family().empty());
    REQUIRE(snap.backend().empty());
    REQUIRE(snap.score_id().empty());
    REQUIRE(snap.stage().empty());
    REQUIRE(snap.candidates_done() == 0);
    REQUIRE_FALSE(snap.candidates_total().has_value());
    REQUIRE(snap.rune_count() == 0);
    REQUIRE(snap.elapsed_seconds() == Catch::Approx(0.0));
    REQUIRE(snap.runes_per_sec() == Catch::Approx(0.0));
    REQUIRE(snap.candidates_per_sec() == Catch::Approx(0.0));
    REQUIRE_FALSE(snap.best_score().has_value());
    REQUIRE(snap.best_label().empty());
    REQUIRE(snap.iteration_index() == 0);
    REQUIRE(snap.iteration_total() == 0);
    REQUIRE_FALSE(snap.fraction_done().has_value());
    REQUIRE_FALSE(snap.eta_seconds().has_value());

    snap.set_tool("search_cycle");
    snap.set_workspace_id("ws");
    snap.set_family("caesar");
    snap.set_backend("cpu");
    snap.set_score_id("chi2_english_gp_v0");
    snap.set_stage("score");
    snap.set_candidates_done(25);
    snap.set_candidates_total(100);
    snap.set_rune_count(262);
    snap.set_elapsed_seconds(2.0);
    snap.set_best_score(580.5);
    snap.set_best_label("shift=11");
    snap.set_iteration(1, 3);
    snap.refresh_rates();

    REQUIRE(snap.tool() == "search_cycle");
    REQUIRE(snap.workspace_id() == "ws");
    REQUIRE(snap.family() == "caesar");
    REQUIRE(snap.backend() == "cpu");
    REQUIRE(snap.score_id() == "chi2_english_gp_v0");
    REQUIRE(snap.stage() == "score");
    REQUIRE(snap.candidates_done() == 25);
    REQUIRE(snap.candidates_total().value() == 100);
    REQUIRE(snap.rune_count() == 262);
    REQUIRE(snap.fraction_done().value() == Catch::Approx(0.25));
    REQUIRE(snap.candidates_per_sec() == Catch::Approx(12.5));
    REQUIRE(snap.runes_per_sec() == Catch::Approx(12.5 * 262.0));
    REQUIRE(snap.eta_seconds().value() == Catch::Approx(75.0 / 12.5));
    REQUIRE(snap.best_score().value() == Catch::Approx(580.5));
    REQUIRE(snap.best_label() == "shift=11");
    REQUIRE(snap.iteration_index() == 1);
    REQUIRE(snap.iteration_total() == 3);

    snap.set_candidates_done(100);
    snap.refresh_rates();
    REQUIRE(snap.eta_seconds().value() == Catch::Approx(0.0));

    snap.set_candidates_total(0);
    REQUIRE_FALSE(snap.fraction_done().has_value());

    snap.set_candidates_total(50);
    snap.set_candidates_done(10);
    snap.set_elapsed_seconds(0.0);
    snap.refresh_rates();
    REQUIRE(snap.candidates_per_sec() == Catch::Approx(0.0));
    REQUIRE(snap.runes_per_sec() == Catch::Approx(0.0));
    REQUIRE_FALSE(snap.eta_seconds().has_value());
}

TEST_CASE("ConsoleProgressClock override is deterministic", "[cli][progress]") {
    ConsoleProgressClock clock;
    clock.set_elapsed_override(1.25);
    REQUIRE(clock.elapsed_seconds() == Catch::Approx(1.25));
    REQUIRE(clock.elapsed_override().has_value());

    clock.set_elapsed_override(std::nullopt);
    REQUIRE_FALSE(clock.elapsed_override().has_value());

    clock.restart();
    clock.set_elapsed_override(0.0);
    REQUIRE(clock.elapsed_seconds() == Catch::Approx(0.0));
}

TEST_CASE("ConsoleProgressClock wall elapsed advances without override", "[cli][progress]") {
    ConsoleProgressClock clock;
    clock.restart();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    REQUIRE(clock.elapsed_seconds() > 0.0);
}
