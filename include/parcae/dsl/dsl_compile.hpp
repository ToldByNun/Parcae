#ifndef DSL_COMPILE_HPP
#define DSL_COMPILE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/core/version.hpp"
#include "parcae/dsl/dsl_ast_json_ingest.hpp"
#include "parcae/dsl/dsl_build_ir.hpp"
#include "parcae/dsl/dsl_catalog_builtins.hpp"
#include "parcae/dsl/dsl_divergence_gate.hpp"
#include "parcae/dsl/dsl_emit_cpu.hpp"
#include "parcae/dsl/dsl_emit_cuda.hpp"
#include "parcae/dsl/dsl_fuse.hpp"
#include "parcae/dsl/dsl_host_glue.hpp"
#include "parcae/dsl/dsl_semantic_gate.hpp"
#include "parcae/dsl/dsl_spec_version.hpp"
#include "parcae/dsl/dsl_verifier.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_envelope_bridge.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

/// End-to-end theory compile: ast_dump → ingest → gate → IR → verify → emit → artifact.
class DslCompile {
public:
    class Options {
    public:
        Options() = default;

        [[nodiscard]] Options& set_python_exe(std::string path) {
            python_exe_ = std::move(path);
            return *this;
        }

        [[nodiscard]] Options& set_python_path(std::string path) {
            python_path_ = std::move(path);
            return *this;
        }

        [[nodiscard]] Options& set_artifact_version(std::uint32_t version) {
            artifact_version_ = version;
            return *this;
        }

        [[nodiscard]] const std::string& python_exe() const noexcept {
            return python_exe_;
        }

        [[nodiscard]] const std::string& python_path() const noexcept {
            return python_path_;
        }

        [[nodiscard]] std::uint32_t artifact_version() const noexcept {
            return artifact_version_;
        }

    private:
        std::string python_exe_ = "python";
        std::string python_path_;
        std::uint32_t artifact_version_ = 1;
    };

    class Result {
    public:
        Result() = default;

        [[nodiscard]] const std::vector<TheoryArtifact>& artifacts() const noexcept {
            return artifacts_;
        }

        [[nodiscard]] const std::string& source_sha256() const noexcept {
            return source_sha256_;
        }

    private:
        friend class DslCompile;
        std::vector<TheoryArtifact> artifacts_;
        std::string source_sha256_;
    };

    [[nodiscard]] static bool pipeline_ready() {
        return pipeline_ready(Options{});
    }

    [[nodiscard]] static bool pipeline_ready(const Options& options) {
        if (options.python_path().empty()) {
            return false;
        }
        const std::filesystem::path pkg =
            std::filesystem::path(options.python_path()) / "parcae" / "dsl" / "ast_dump.py";
        std::error_code ec;
        return std::filesystem::is_regular_file(pkg, ec) && !ec;
    }

    /// Compile `theory_py` into `theories_root/<name>/<version>/`.
    [[nodiscard]] static StatusOr<Result> compile_file(
        const std::filesystem::path& theory_py,
        const std::filesystem::path& theories_root) {
        return compile_file(theory_py, theories_root, Options{});
    }

    [[nodiscard]] static StatusOr<Result> compile_file(
        const std::filesystem::path& theory_py,
        const std::filesystem::path& theories_root,
        const Options& options) {
        if (!std::filesystem::is_regular_file(theory_py)) {
            return Status::error("theory source is not a readable file: " + theory_py.string());
        }
        if (!pipeline_ready(options)) {
            return Status::error(
                "ast_dump frontend not found under PYTHONPATH=" + options.python_path() +
                " (expected parcae/dsl/ast_dump.py)");
        }

        StatusOr<std::string> json_text = spawn_ast_dump(theory_py, options);
        if (!json_text.ok()) {
            return json_text.status();
        }

        StatusOr<DslAstDocument> doc = DslAstJsonIngest::parse_text(json_text.value());
        if (!doc.ok()) {
            return doc.status();
        }
        Status gate = DslSemanticGate::check(doc.value());
        if (!gate.ok()) {
            return gate;
        }
        Status divergence = DslDivergenceGate::check_errors_only(doc.value());
        if (!divergence.ok()) {
            return divergence;
        }
        Status host = DslHostGlue::check_errors_only(doc.value());
        if (!host.ok()) {
            return host;
        }
        StatusOr<DslBuildIr::Unit> unit = DslBuildIr::build(doc.value());
        if (!unit.ok()) {
            return unit.status();
        }

        Result result;
        result.source_sha256_ = unit.value().source_sha256();

        for (const PrimitiveIr& prim : unit.value().primitives()) {
            StatusOr<DslVerifier::Report> vr = DslVerifier::verify_primitive(prim);
            if (!vr.ok()) {
                return vr.status();
            }
            if (!vr.value().passed()) {
                return Status::error(
                    "E050 verify failed for primitive '" + prim.name() + "': " + vr.value().detail());
            }
        }

        const std::string completed = utc_now();
        for (const TheoryIr& theory : unit.value().theories()) {
            // Re-verify primitives referenced by this theory are already done above.
            StatusOr<std::string> cpu = DslEmitCpu::emit_theory_header(theory);
            if (!cpu.ok()) {
                return cpu.status();
            }
            StatusOr<std::string> cuda_h = DslEmitCuda::emit_theory_header(theory);
            if (!cuda_h.ok()) {
                return cuda_h.status();
            }
            StatusOr<std::string> cuda_cu = DslEmitCuda::emit_theory_cu(theory);
            if (!cuda_cu.ok()) {
                return cuda_cu.status();
            }

            TheoryArtifact::Paths paths;
            paths.set_cpu_reference(std::string("cpu_reference.hpp"));
            const std::string class_stem = to_pascal_case(theory.name());
            paths.set_cuda_header(std::string("emitted/") + class_stem + "Kernel.hpp");
            paths.set_cuda_source(std::string("emitted/") + class_stem + "Kernel.cu");
            paths.set_envelope_template(std::string("envelope.json"));
            paths.set_apply_ir(std::string("apply_ir.json"));

            std::vector<TheoryArtifact::Param> params;
            for (const ParamIr& p : theory.params()) {
                params.emplace_back(
                    p.name(),
                    static_cast<std::int64_t>((p.min)()),
                    static_cast<std::int64_t>((p.max)()));
            }
            std::vector<std::string> prim_names;
            for (const PrimitiveIr& prim : unit.value().primitives()) {
                prim_names.push_back(prim.name());
            }

            DslVerifier::Mode mode = DslVerifier::Mode::Exhaustive;
            std::optional<std::uint32_t> seed;
            // Prefer exhaustive when all primitives ≤4 arity (already verified).
            for (const PrimitiveIr& prim : unit.value().primitives()) {
                if (prim.arity() > DslVerifier::max_exhaustive_arity) {
                    mode = DslVerifier::Mode::Fuzz;
                    seed = DslVerifier::default_fuzz_seed;
                    break;
                }
            }

            StatusOr<TheoryArtifact> artifact = TheoryArtifact::make(
                theory.name(),
                options.artifact_version(),
                theory.tier(),
                theory.family(),
                unit.value().source_sha256(),
                TheoryArtifact::Verification{mode, true, seed, completed},
                TheoryArtifact::FusionStatus::NotApplicable,
                TheoryArtifact::interrupt_mode_from_theory(theory.interrupt_mode()),
                std::move(params),
                std::move(prim_names),
                theory.structural_claim(),
                theory_py.generic_string(),
                std::move(paths));
            if (!artifact.ok()) {
                return artifact.status();
            }

            Status stored = artifact.value().store(theories_root);
            if (!stored.ok()) {
                return stored;
            }

            const std::filesystem::path dir = artifact.value().artifact_dir(theories_root);
            Status w = write_text(dir / "cpu_reference.hpp", cpu.value());
            if (!w.ok()) {
                return w;
            }
            std::error_code ec;
            std::filesystem::create_directories(dir / "emitted", ec);
            if (ec) {
                return Status::error("failed to create emitted/: " + ec.message());
            }
            w = write_text(dir / "emitted" / (class_stem + "Kernel.hpp"), cuda_h.value());
            if (!w.ok()) {
                return w;
            }
            w = write_text(dir / "emitted" / (class_stem + "Kernel.cu"), cuda_cu.value());
            if (!w.ok()) {
                return w;
            }

            StatusOr<TheoryEnvelopeBridge::Envelope> envelope =
                TheoryEnvelopeBridge::template_for(artifact.value());
            if (!envelope.ok()) {
                return envelope.status();
            }
            Status env_written =
                TheoryEnvelopeBridge::write(dir / "envelope.json", envelope.value());
            if (!env_written.ok()) {
                return env_written;
            }
            Status ir_written = TheoryApplyIr::write(dir / "apply_ir.json", theory);
            if (!ir_written.ok()) {
                return ir_written;
            }

            // Registry gate: freshly written artifacts must load.
            StatusOr<TheoryArtifact> reloaded =
                TheoryRegistry::load(theories_root, theory.name(), options.artifact_version());
            if (!reloaded.ok()) {
                return reloaded.status();
            }
            result.artifacts_.push_back(std::move(artifact.value()));
        }

        // Compose theories: fuse / bench / emit against catalog builtins + module theories.
        std::vector<TheoryIr> catalog = DslCatalogBuiltins::all();
        for (const TheoryIr& theory : unit.value().theories()) {
            catalog.push_back(theory);
        }
        for (const ComposeIr& compose : unit.value().composes()) {
            StatusOr<DslFuse::EmitBundle> bundle = DslFuse::emit_compose_auto(
                compose,
                catalog,
                unit.value().composes(),
                {},
                /*stream_len=*/256,
                /*reps=*/4);
            if (!bundle.ok()) {
                return bundle.status();
            }

            TheoryArtifact::Paths paths;
            paths.set_cpu_reference(std::string("cpu_reference.hpp"));
            const std::string class_stem = to_pascal_case(compose.name());
            paths.set_cuda_header(std::string("emitted/") + class_stem + "Kernel.hpp");
            if (bundle.value().status() == DslFuse::FusionStatus::Fused) {
                paths.set_cuda_source(std::string("emitted/") + class_stem + "Kernel.cu");
            }
            paths.set_envelope_template(std::string("envelope.json"));
            paths.set_apply_ir(std::string("apply_ir.json"));

            std::vector<TheoryArtifact::Param> params;
            for (const ParamIr& p : compose.params()) {
                params.emplace_back(
                    p.name(),
                    static_cast<std::int64_t>((p.min)()),
                    static_cast<std::int64_t>((p.max)()));
            }

            TheoryArtifact::FusionStatus fusion =
                bundle.value().status() == DslFuse::FusionStatus::Fused
                    ? TheoryArtifact::FusionStatus::Fused
                    : TheoryArtifact::FusionStatus::FallbackStaged;

            StatusOr<TheoryArtifact> artifact = TheoryArtifact::make(
                compose.name(),
                options.artifact_version(),
                compose.tier(),
                TheoryIr::Family::Compose,
                unit.value().source_sha256(),
                TheoryArtifact::Verification{
                    DslVerifier::Mode::Exhaustive, true, std::nullopt, completed},
                fusion,
                TheoryArtifact::InterruptMode::ElementwiseDefault,
                std::move(params),
                {},
                compose.structural_claim(),
                theory_py.generic_string(),
                std::move(paths));
            if (!artifact.ok()) {
                return artifact.status();
            }

            Status stored = artifact.value().store(theories_root);
            if (!stored.ok()) {
                return stored;
            }

            const std::filesystem::path dir = artifact.value().artifact_dir(theories_root);
            Status w = write_text(dir / "cpu_reference.hpp", bundle.value().selected_cpu_header());
            if (!w.ok()) {
                return w;
            }
            std::error_code ec;
            std::filesystem::create_directories(dir / "emitted", ec);
            if (ec) {
                return Status::error("failed to create emitted/: " + ec.message());
            }
            w = write_text(
                dir / "emitted" / (class_stem + "Kernel.hpp"),
                bundle.value().selected_cuda_header());
            if (!w.ok()) {
                return w;
            }
            if (bundle.value().status() == DslFuse::FusionStatus::Fused) {
                w = write_text(
                    dir / "emitted" / (class_stem + "Kernel.cu"),
                    bundle.value().fused_cuda_cu());
                if (!w.ok()) {
                    return w;
                }
            }

            StatusOr<TheoryEnvelopeBridge::Envelope> envelope =
                TheoryEnvelopeBridge::template_for(artifact.value());
            if (!envelope.ok()) {
                return envelope.status();
            }
            Status env_written =
                TheoryEnvelopeBridge::write(dir / "envelope.json", envelope.value());
            if (!env_written.ok()) {
                return env_written;
            }
            // Runtime apply uses fused IR even when emit selected staged (CPU IR path).
            Status ir_written = TheoryApplyIr::write(
                dir / "apply_ir.json", bundle.value().fused().theory());
            if (!ir_written.ok()) {
                return ir_written;
            }

            StatusOr<TheoryArtifact> reloaded =
                TheoryRegistry::load(theories_root, compose.name(), options.artifact_version());
            if (!reloaded.ok()) {
                return reloaded.status();
            }
            result.artifacts_.push_back(std::move(artifact.value()));
        }

        return result;
    }

private:
    DslCompile() = delete;

    [[nodiscard]] static std::string utc_now() {
        const auto now = std::chrono::system_clock::now();
        const std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &tt);
#else
        gmtime_r(&tt, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
        return std::string(buf);
    }

    [[nodiscard]] static std::string to_pascal_case(std::string_view name) {
        std::string out;
        bool cap = true;
        for (char ch : name) {
            if (ch == '_') {
                cap = true;
                continue;
            }
            if (cap) {
                out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
                cap = false;
            } else {
                out.push_back(ch);
            }
        }
        return out;
    }

    [[nodiscard]] static Status write_text(
        const std::filesystem::path& path,
        const std::string& text) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return Status::error("failed to write " + path.string());
        }
        out << text;
        if (!out) {
            return Status::error("failed while writing " + path.string());
        }
        return Status::success();
    }

    [[nodiscard]] static StatusOr<std::string> spawn_ast_dump(
        const std::filesystem::path& theory_py,
        const Options& options) {
        const auto tmp_dir = std::filesystem::temp_directory_path();
        const std::filesystem::path out_json =
            tmp_dir / ("parcae_ast_dump_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                       ".json");

#ifdef _WIN32
        StatusOr<int> rc = run_process_win(
            options.python_exe(),
            {"-m",
             "parcae.dsl.ast_dump",
             theory_py.string(),
             "-o",
             out_json.string()},
            options.python_path());
#else
        StatusOr<int> rc = run_process_posix(
            options.python_exe(),
            {"-m",
             "parcae.dsl.ast_dump",
             theory_py.string(),
             "-o",
             out_json.string()},
            options.python_path());
#endif
        if (!rc.ok()) {
            return rc.status();
        }
        std::ifstream in(out_json, std::ios::binary);
        if (!in) {
            return Status::error(
                "ast_dump produced no output file (exit " + std::to_string(rc.value()) + ")");
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        const std::string text = buf.str();
        std::error_code ec;
        std::filesystem::remove(out_json, ec);
        if (rc.value() != 0) {
            return Status::error(
                "ast_dump failed (exit " + std::to_string(rc.value()) + "): " + text);
        }
        return text;
    }

#ifdef _WIN32
    [[nodiscard]] static StatusOr<int> run_process_win(
        const std::string& exe,
        const std::vector<std::string>& args,
        const std::string& python_path) {
        std::ostringstream cmdline;
        cmdline << '"' << exe << '"';
        for (const std::string& a : args) {
            cmdline << " \"" << a << '"';
        }
        std::string cmd = cmdline.str();
        std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
        mutable_cmd.push_back('\0');

        char prev_pythonpath[32768];
        const DWORD prev_len =
            GetEnvironmentVariableA("PYTHONPATH", prev_pythonpath, sizeof(prev_pythonpath));
        const bool had_prev = prev_len > 0 && prev_len < sizeof(prev_pythonpath);
        SetEnvironmentVariableA("PYTHONPATH", python_path.c_str());

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        const BOOL ok = CreateProcessA(
            nullptr,
            mutable_cmd.data(),
            nullptr,
            nullptr,
            FALSE,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &si,
            &pi);
        if (had_prev) {
            SetEnvironmentVariableA("PYTHONPATH", prev_pythonpath);
        } else {
            SetEnvironmentVariableA("PYTHONPATH", nullptr);
        }
        if (!ok) {
            return Status::error(
                "failed to spawn ast_dump (" + exe + "); GetLastError=" +
                std::to_string(GetLastError()));
        }
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD code = 1;
        GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        return static_cast<int>(code);
    }
#else
    [[nodiscard]] static StatusOr<int> run_process_posix(
        const std::string& exe,
        const std::vector<std::string>& args,
        const std::string& python_path) {
        std::vector<std::string> storage;
        storage.push_back(exe);
        for (const std::string& a : args) {
            storage.push_back(a);
        }
        std::vector<char*> argv;
        for (std::string& s : storage) {
            argv.push_back(s.data());
        }
        argv.push_back(nullptr);

        const pid_t pid = fork();
        if (pid < 0) {
            return Status::error("fork failed for ast_dump");
        }
        if (pid == 0) {
            setenv("PYTHONPATH", python_path.c_str(), 1);
            execvp(exe.c_str(), argv.data());
            _exit(127);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            return Status::error("waitpid failed for ast_dump");
        }
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        return Status::error("ast_dump terminated abnormally");
    }
#endif
};

#endif // DSL_COMPILE_HPP
