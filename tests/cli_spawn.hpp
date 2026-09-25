#ifndef PARCAE_TESTS_CLI_SPAWN_HPP
#define PARCAE_TESTS_CLI_SPAWN_HPP

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

/// Captured CLI process result (exit code already normalized across Win/POSIX).
struct CliSpawnResult {
    int exit_code = 1;
    std::string stdout_text;
    std::string stderr_text;
};

namespace CliSpawnDetail {

[[nodiscard]] inline std::string quote_arg(const std::string& arg) {
#if defined(_WIN32)
    return std::string("\"") + arg + '"';
#else
    // POSIX single-quote; embed ' as '\'' .
    std::string out = "'";
    for (const char c : arg) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += '\'';
    return out;
#endif
}

[[nodiscard]] inline int normalize_system_status(int status) {
#if defined(_WIN32)
    return status;
#else
    if (status == -1) {
        return 127;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
#endif
}

[[nodiscard]] inline std::string read_file_if_exists(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        return {};
    }
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

} // namespace CliSpawnDetail

/// Run `exe` with `args`; capture stdout/stderr via a temp shell script.
/// Windows uses `cmd /C` + `.cmd`; POSIX uses `/bin/sh` + `.sh`.
[[nodiscard]] inline CliSpawnResult run_cli_capture(const std::filesystem::path& exe,
                                                    const std::vector<std::string>& args,
                                                    const std::string& tmp_tag) {
    using CliSpawnDetail::normalize_system_status;
    using CliSpawnDetail::quote_arg;
    using CliSpawnDetail::read_file_if_exists;

    const auto tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path out_path = tmp / ("parcae_cli_" + tmp_tag + "_out.txt");
    const std::filesystem::path err_path = tmp / ("parcae_cli_" + tmp_tag + "_err.txt");
#if defined(_WIN32)
    const std::filesystem::path script_path = tmp / ("parcae_cli_" + tmp_tag + "_run.cmd");
#else
    const std::filesystem::path script_path = tmp / ("parcae_cli_" + tmp_tag + "_run.sh");
#endif

    {
        std::ofstream script(script_path, std::ios::binary);
        REQUIRE(script);
#if defined(_WIN32)
        script << "@echo off\r\n";
        script << quote_arg(exe.string());
        for (const std::string& arg : args) {
            script << ' ' << quote_arg(arg);
        }
        script << " >" << quote_arg(out_path.string()) << " 2>" << quote_arg(err_path.string())
               << "\r\n";
        script << "exit /B %ERRORLEVEL%\r\n";
#else
        script << "#!/bin/sh\n";
        script << quote_arg(exe.string());
        for (const std::string& arg : args) {
            script << ' ' << quote_arg(arg);
        }
        script << " >" << quote_arg(out_path.string()) << " 2>" << quote_arg(err_path.string())
               << "\n";
        script << "exit $?\n";
#endif
    }

#if defined(_WIN32)
    const std::string launch = std::string("cmd /C ") + quote_arg(script_path.string());
#else
    const std::string launch = std::string("/bin/sh ") + quote_arg(script_path.string());
#endif

    const int raw = std::system(launch.c_str());

    CliSpawnResult result;
    result.exit_code = normalize_system_status(raw);
    result.stdout_text = read_file_if_exists(out_path);
    result.stderr_text = read_file_if_exists(err_path);

    std::error_code ec;
    std::filesystem::remove(out_path, ec);
    std::filesystem::remove(err_path, ec);
    std::filesystem::remove(script_path, ec);
    return result;
}

#endif // PARCAE_TESTS_CLI_SPAWN_HPP
