#include <catch2/catch_test_macros.hpp>

#include "parcae/cli/console_ansi.hpp"

#include <string>

struct ConsoleAnsiOverrideGuard {
    ~ConsoleAnsiOverrideGuard() {
        ConsoleAnsi::clear_overrides();
    }
};

TEST_CASE("ConsoleAnsi cursor_up and clear helpers", "[cli][ansi]") {
    REQUIRE(ConsoleAnsi::cursor_up(0).empty());
    REQUIRE(ConsoleAnsi::cursor_up(1) == "\x1b[1A");
    REQUIRE(ConsoleAnsi::cursor_up(12) == "\x1b[12A");
    REQUIRE(ConsoleAnsi::clear_to_eol() == "\x1b[K");
    REQUIRE(ConsoleAnsi::carriage_return() == "\r");
    REQUIRE(ConsoleAnsi::hide_cursor() == "\x1b[?25l");
    REQUIRE(ConsoleAnsi::show_cursor() == "\x1b[?25h");
}

TEST_CASE("ConsoleAnsi ascii_bar is pure 7-bit", "[cli][ansi]") {
    REQUIRE(ConsoleAnsi::ascii_bar(0.0, 0).empty());
    REQUIRE(ConsoleAnsi::ascii_bar(0.0, 10) == "----------");
    REQUIRE(ConsoleAnsi::ascii_bar(1.0, 10) == "##########");
    REQUIRE(ConsoleAnsi::ascii_bar(0.5, 10) == "#####-----");
    REQUIRE(ConsoleAnsi::ascii_bar(-1.0, 4) == "----");
    REQUIRE(ConsoleAnsi::ascii_bar(2.0, 4) == "####");
    REQUIRE(ConsoleAnsi::ascii_bar(0.25, 8) == "##------");
}

TEST_CASE("ConsoleAnsi stderr_is_tty respects override", "[cli][ansi]") {
    ConsoleAnsiOverrideGuard guard;
    ConsoleAnsi::set_stderr_tty_override(true);
    REQUIRE(ConsoleAnsi::stderr_is_tty());
    ConsoleAnsi::set_stderr_tty_override(false);
    REQUIRE_FALSE(ConsoleAnsi::stderr_is_tty());
    ConsoleAnsi::clear_overrides();
}

TEST_CASE("ConsoleAnsi enable_virtual_terminal respects override", "[cli][ansi]") {
    ConsoleAnsiOverrideGuard guard;
    ConsoleAnsi::set_vt_enable_override(true);
    REQUIRE(ConsoleAnsi::enable_virtual_terminal_stderr());
    ConsoleAnsi::set_vt_enable_override(false);
    REQUIRE_FALSE(ConsoleAnsi::enable_virtual_terminal_stderr());
}

TEST_CASE("ConsoleAnsi real VT enable is best-effort no-throw", "[cli][ansi]") {
    ConsoleAnsiOverrideGuard guard;
    ConsoleAnsi::clear_overrides();
    // May be true or false depending on host console; must not throw/crash.
    (void)ConsoleAnsi::enable_virtual_terminal_stderr();
    (void)ConsoleAnsi::stderr_is_tty();
}
