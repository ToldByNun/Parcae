#ifndef PARCAE_CLI_IO_HPP
#define PARCAE_CLI_IO_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/tool/context.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace parcae::cli {

inline constexpr int kExitOk = 0;
inline constexpr int kExitFail = 1;
inline constexpr int kExitUsage = 2;

[[nodiscard]] inline StatusOr<std::string> read_all_utf8(std::string_view path_or_dash) {
    if (path_or_dash == "-") {
        std::ostringstream buffer;
        buffer << std::cin.rdbuf();
        return buffer.str();
    }

    std::ifstream input(std::filesystem::path(path_or_dash), std::ios::binary);
    if (!input) {
        return Status::error("Failed to open input: " + std::string(path_or_dash));
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

[[nodiscard]] inline StatusOr<std::string> read_file_utf8(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Status::error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

/// Resolve data root: --data-dir > PARCAE_DATA_DIR > ./data > PARCAE_DEFAULT_DATA_DIR.
[[nodiscard]] inline StatusOr<parcae::tool::Context> make_context(
    const std::string& data_dir_flag,
    const char* default_data_dir) {
    std::filesystem::path root;
    if (!data_dir_flag.empty()) {
        root = data_dir_flag;
    } else {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
#endif
        const char* env = std::getenv("PARCAE_DATA_DIR");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        if (env != nullptr && env[0] != '\0') {
            root = env;
        } else if (std::filesystem::is_directory("data")) {
            root = "data";
        } else if (default_data_dir != nullptr && default_data_dir[0] != '\0' &&
                   std::filesystem::is_directory(default_data_dir)) {
            root = default_data_dir;
        } else {
            return Status::error(
                "Could not resolve data root (pass --data-dir or set PARCAE_DATA_DIR)");
        }
    }
    if (!std::filesystem::is_directory(root)) {
        return Status::error("Data root is not a directory: " + root.string());
    }
    return parcae::tool::Context{std::filesystem::weakly_canonical(root)};
}


[[nodiscard]] inline StatusOr<std::vector<std::size_t>> parse_size_list(std::string_view text) {
    std::vector<std::size_t> out;
    if (text.empty()) {
        return out;
    }
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == ',')) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        std::size_t j = i;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
            ++j;
        }
        if (j == i) {
            return Status::error("Invalid integer list (expected digits/commas)");
        }
        out.push_back(static_cast<std::size_t>(std::stoull(std::string(text.substr(i, j - i)))));
        i = j;
    }
    return out;
}

[[nodiscard]] inline StatusOr<std::vector<int>> parse_int_list(std::string_view text) {
    StatusOr<std::vector<std::size_t>> sizes = parse_size_list(text);
    if (!sizes.ok()) {
        return sizes.status();
    }
    std::vector<int> out;
    out.reserve(sizes.value().size());
    for (std::size_t v : sizes.value()) {
        out.push_back(static_cast<int>(v));
    }
    return out;
}

[[nodiscard]] inline bool has_flag(const std::vector<std::string>& args, std::string_view flag) {
    for (const std::string& arg : args) {
        if (arg == flag) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline StatusOr<std::string> require_option(
    const std::vector<std::string>& args,
    std::string_view flag) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == flag) {
            if (i + 1 >= args.size()) {
                return Status::error("Missing value for " + std::string(flag));
            }
            return args[i + 1];
        }
    }
    return Status::error("Missing required option " + std::string(flag));
}

[[nodiscard]] inline std::string optional_option(
    const std::vector<std::string>& args,
    std::string_view flag,
    std::string default_value = {}) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == flag) {
            if (i + 1 < args.size()) {
                return args[i + 1];
            }
            return default_value;
        }
    }
    return default_value;
}

[[nodiscard]] inline std::vector<std::string> argv_tail(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return args;
}

}  // namespace parcae::cli

#endif // PARCAE_CLI_IO_HPP
