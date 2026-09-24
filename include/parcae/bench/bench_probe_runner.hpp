#ifndef BENCH_PROBE_RUNNER_HPP
#define BENCH_PROBE_RUNNER_HPP

#include "parcae/bench/bench_config.hpp"
#include "parcae/bench/bench_probe_protocol.hpp"
#include "parcae/bench/bench_report.hpp"
#include "parcae/bench/bench_tier_spec.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/// Spawn external probe commands, enforce timeout, parse 1.0.0 JSON → report.
///
/// Normative contract: `docs/spec/bench-probe.md`.
class BenchProbeRunner {
public:
    class Capture {
    public:
        int exit_code = 1;
        bool timed_out = false;
        std::string stdout_text;
        std::string stderr_text;
    };

    /// Replace every `{tier}` occurrence in `cmd_template`.
    [[nodiscard]] static std::string substitute_tier(std::string_view cmd_template,
                                                     std::string_view tier) {
        std::string out;
        out.reserve(cmd_template.size() + tier.size());
        constexpr std::string_view kTok = "{tier}";
        std::size_t i = 0;
        while (i < cmd_template.size()) {
            if (i + kTok.size() <= cmd_template.size() &&
                cmd_template.substr(i, kTok.size()) == kTok) {
                out.append(tier);
                i += kTok.size();
            } else {
                out.push_back(cmd_template[i]);
                ++i;
            }
        }
        return out;
    }

    /// Spawn `command` via platform shell; kill on timeout. Captures stdout/stderr.
    [[nodiscard]] static StatusOr<Capture> spawn_with_timeout(const std::string& command,
                                                              std::uint32_t timeout_ms) {
        if (command.empty()) {
            return Status::error("BenchProbeRunner: empty command");
        }
        if (timeout_ms == 0) {
            return Status::error("BenchProbeRunner: timeout_ms must be >= 1");
        }
#ifdef _WIN32
        return spawn_win(command, timeout_ms);
#else
        return spawn_posix(command, timeout_ms);
#endif
    }

    /// Run all configured probe tiers into a `BenchReport::Document` (suite=probe).
    [[nodiscard]] static StatusOr<BenchReport::Document> run(const BenchConfig& config) {
        if (!config.has_probe_cmd()) {
            return Status::error("BenchProbeRunner: --probe-cmd is required");
        }
        if (config.probe_tiers().empty()) {
            return Status::error("BenchProbeRunner: probe_tiers is empty");
        }

        BenchReport::Document doc(BenchReport::Suite::Probe);
        for (const std::string& tier : config.probe_tiers()) {
            StatusOr<BenchReport::Row> row = run_one_tier(config, tier);
            if (!row.ok()) {
                // Hard orchestration errors (spawn plumbing) → fail the suite.
                return row.status();
            }
            doc.add_row(std::move(row.value()));
        }
        doc.recompute_all_pass();
        return doc;
    }

    /// Map a parsed probe (+ optional builtin compare) to a report row.
    [[nodiscard]] static BenchReport::Row to_row(const BenchProbeProtocol::Result& result,
                                                 bool compare_builtin, int child_exit,
                                                 bool timed_out,
                                                 std::string_view spawn_error = {}) {
        std::ostringstream detail;
        detail << "tool=" << result.tool_id;
        detail << " oracle=" << (result.accuracy.oracle_cracked ? "true" : "false");
        detail << " top_rank=" << result.accuracy.top_rank;
        if (!result.accuracy.notes.empty()) {
            detail << " notes=" << result.accuracy.notes;
        }
        if (timed_out) {
            detail << "; timed_out";
        }
        if (child_exit != 0) {
            detail << "; exit=" << child_exit;
        }
        if (!spawn_error.empty()) {
            detail << "; " << spawn_error;
        }

        BenchReport::RowStatus status = BenchReport::RowStatus::Pass;
        if (timed_out || !spawn_error.empty()) {
            status = BenchReport::RowStatus::Fail;
        }

        if (compare_builtin && status == BenchReport::RowStatus::Pass) {
            const BenchTierSpec::Tier* spec = BenchTierSpec::find_primary(result.tier);
            if (spec == nullptr) {
                status = BenchReport::RowStatus::Fail;
                detail << "; compare_builtin: unknown tier";
            } else if (result.config.candidates != spec->candidates ||
                       result.config.tokens != spec->tokens ||
                       result.config.repeats != spec->repeats) {
                status = BenchReport::RowStatus::Fail;
                detail << "; compare_builtin: config mismatch (expected C=" << spec->candidates
                       << " T=" << spec->tokens << " reps=" << spec->repeats << ")";
            } else {
                detail << "; compare_builtin: config_ok";
                if (spec->slo_min > 0.0 && result.runes_per_sec > 0.0) {
                    const double ratio = result.runes_per_sec / spec->slo_min;
                    detail << " ratio_vs_slo=" << std::fixed << std::setprecision(2) << ratio
                           << "x";
                }
            }
        }

        const double peak = compare_builtin ? BenchTierSpec::estimated_peak(result.tier) : 0.0;
        const double slo_min = compare_builtin
                                   ? (BenchTierSpec::find_primary(result.tier) != nullptr
                                          ? BenchTierSpec::find_primary(result.tier)->slo_min
                                          : 0.0)
                                   : 0.0;
        const double slo_max = compare_builtin
                                   ? (BenchTierSpec::find_primary(result.tier) != nullptr
                                          ? BenchTierSpec::find_primary(result.tier)->slo_max
                                          : 0.0)
                                   : 0.0;

        return BenchReport::Row::make(
            result.tier, result.tool_id, BenchReport::Suite::Probe, result.backend, status,
            result.runes_per_sec, result.keys_per_sec, result.wall_seconds, slo_min, slo_max, peak,
            result.config.candidates, result.config.tokens, result.config.repeats, detail.str());
    }

    [[nodiscard]] static BenchReport::Row make_fail_row(std::string_view tier, std::string detail) {
        return BenchReport::Row::make(std::string(tier), "probe", BenchReport::Suite::Probe,
                                      BenchReport::Backend::Unknown, BenchReport::RowStatus::Fail,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, std::move(detail));
    }

private:
    BenchProbeRunner() = delete;

    [[nodiscard]] static StatusOr<BenchReport::Row> run_one_tier(const BenchConfig& config,
                                                                 const std::string& tier) {
        const std::string cmd = substitute_tier(config.probe_cmd(), tier);
        StatusOr<Capture> cap = spawn_with_timeout(cmd, config.probe_timeout_ms());
        if (!cap.ok()) {
            return make_fail_row(tier, "spawn: " + cap.status().message());
        }
        if (cap.value().timed_out) {
            return make_fail_row(tier, "timeout after " +
                                           std::to_string(config.probe_timeout_ms()) + " ms");
        }

        StatusOr<BenchProbeProtocol::Result> parsed =
            BenchProbeProtocol::parse(cap.value().stdout_text);
        if (!parsed.ok()) {
            std::ostringstream detail;
            detail << "parse: " << parsed.status().message();
            if (!cap.value().stderr_text.empty()) {
                detail << "; stderr=" << truncate(cap.value().stderr_text, 200);
            }
            return make_fail_row(tier, detail.str());
        }
        if (parsed.value().tier != tier) {
            return make_fail_row(tier,
                                 "tier mismatch: probe reported '" + parsed.value().tier + "'");
        }
        return to_row(parsed.value(), config.compare_builtin(), cap.value().exit_code, false);
    }

    [[nodiscard]] static std::string truncate(std::string_view text, std::size_t max_n) {
        if (text.size() <= max_n) {
            return std::string(text);
        }
        return std::string(text.substr(0, max_n)) + "...";
    }

#ifdef _WIN32
    [[nodiscard]] static StatusOr<Capture> spawn_win(const std::string& command,
                                                     std::uint32_t timeout_ms) {
        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE out_read = nullptr;
        HANDLE out_write = nullptr;
        HANDLE err_read = nullptr;
        HANDLE err_write = nullptr;
        if (!CreatePipe(&out_read, &out_write, &sa, 0)) {
            return Status::error("BenchProbeRunner: CreatePipe stdout failed");
        }
        if (!SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0)) {
            CloseHandle(out_read);
            CloseHandle(out_write);
            return Status::error("BenchProbeRunner: SetHandleInformation stdout failed");
        }
        if (!CreatePipe(&err_read, &err_write, &sa, 0)) {
            CloseHandle(out_read);
            CloseHandle(out_write);
            return Status::error("BenchProbeRunner: CreatePipe stderr failed");
        }
        if (!SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0)) {
            CloseHandle(out_read);
            CloseHandle(out_write);
            CloseHandle(err_read);
            CloseHandle(err_write);
            return Status::error("BenchProbeRunner: SetHandleInformation stderr failed");
        }

        // Shell so operators can pass free-form --probe-cmd strings.
        std::string cmdline = "cmd.exe /C " + command;
        std::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
        mutable_cmd.push_back('\0');

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = out_write;
        si.hStdError = err_write;

        PROCESS_INFORMATION pi{};
        const BOOL ok = CreateProcessA(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE,
                                       CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        CloseHandle(out_write);
        CloseHandle(err_write);
        out_write = nullptr;
        err_write = nullptr;

        if (!ok) {
            CloseHandle(out_read);
            CloseHandle(err_read);
            return Status::error("BenchProbeRunner: CreateProcess failed; GetLastError=" +
                                 std::to_string(GetLastError()));
        }

        Capture cap;
        const DWORD wait = WaitForSingleObject(pi.hProcess, timeout_ms);
        if (wait == WAIT_TIMEOUT) {
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            cap.timed_out = true;
        }

        DWORD code = 1;
        GetExitCodeProcess(pi.hProcess, &code);
        cap.exit_code = static_cast<int>(code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);

        cap.stdout_text = read_pipe_win(out_read);
        cap.stderr_text = read_pipe_win(err_read);
        CloseHandle(out_read);
        CloseHandle(err_read);
        return cap;
    }

    [[nodiscard]] static std::string read_pipe_win(HANDLE pipe) {
        std::string out;
        char buf[4096];
        for (;;) {
            DWORD n = 0;
            const BOOL ok = ReadFile(pipe, buf, sizeof(buf), &n, nullptr);
            if (!ok || n == 0) {
                break;
            }
            out.append(buf, buf + n);
        }
        return out;
    }
#else
    [[nodiscard]] static StatusOr<Capture> spawn_posix(const std::string& command,
                                                       std::uint32_t timeout_ms) {
        int out_pipe[2] = {-1, -1};
        int err_pipe[2] = {-1, -1};
        if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
            return Status::error(std::string("BenchProbeRunner: pipe failed: ") +
                                 std::strerror(errno));
        }

        const pid_t pid = fork();
        if (pid < 0) {
            close(out_pipe[0]);
            close(out_pipe[1]);
            close(err_pipe[0]);
            close(err_pipe[1]);
            return Status::error(std::string("BenchProbeRunner: fork failed: ") +
                                 std::strerror(errno));
        }
        if (pid == 0) {
            close(out_pipe[0]);
            close(err_pipe[0]);
            dup2(out_pipe[1], STDOUT_FILENO);
            dup2(err_pipe[1], STDERR_FILENO);
            close(out_pipe[1]);
            close(err_pipe[1]);
            execl("/bin/sh", "sh", "-c", command.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }

        close(out_pipe[1]);
        close(err_pipe[1]);

        Capture cap;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        int status = 0;
        for (;;) {
            const pid_t waited = waitpid(pid, &status, WNOHANG);
            if (waited == pid) {
                break;
            }
            if (waited < 0 && errno != EINTR) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                close(out_pipe[0]);
                close(err_pipe[0]);
                return Status::error(std::string("BenchProbeRunner: waitpid failed: ") +
                                     std::strerror(errno));
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                cap.timed_out = true;
                break;
            }
            usleep(10000);
        }

        // Writers closed after child exit / kill — blocking drain is safe.
        drain_fd(out_pipe[0], cap.stdout_text);
        drain_fd(err_pipe[0], cap.stderr_text);
        close(out_pipe[0]);
        close(err_pipe[0]);

        if (WIFEXITED(status)) {
            cap.exit_code = WEXITSTATUS(status);
        } else {
            cap.exit_code = 1;
        }
        return cap;
    }

    static void drain_fd(int fd, std::string& out) {
        char buf[4096];
        for (;;) {
            const ssize_t n = read(fd, buf, sizeof(buf));
            if (n > 0) {
                out.append(buf, buf + n);
                continue;
            }
            break;
        }
    }
#endif
};

#endif // BENCH_PROBE_RUNNER_HPP
