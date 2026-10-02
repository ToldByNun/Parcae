#ifndef THEORY_EXPORT_CACHE_HPP
#define THEORY_EXPORT_CACHE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/dsl/theory_apply_ir.hpp"
#include "parcae/dsl/theory_artifact.hpp"
#include "parcae/dsl/theory_dispatch.hpp"
#include "parcae/dsl/theory_hist_chi2_emit.hpp"
#include "parcae/dsl/theory_ir.hpp"
#include "parcae/dsl/theory_registry.hpp"
#include "parcae/dsl/theory_uri.hpp"
#include "parcae/dsl/z29_bytecode.hpp"
#include "parcae/transform/transform_direction.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(PARCAE_HAS_CUDA)
#include "device_buffer.hpp"
#include "theory_chi2_batch.hpp"
#endif

/// Process-local cache for theory fused export: host bytecode once per URI, and
/// (when CUDA) reusable device `ops`/`imm` program buffers. Slots still upload
/// per `params_list` chunk. Class-only — no namespaces.
///
/// Pass the same instance across `GpuCandidateExport::theory_*` / scheduler
/// cycles to amortize prepare + H2D program traffic.
class TheoryExportCache {
public:
    /// Cached `TheoryHistChi2Emit` decision (sources discarded; plans kept).
    class HistPlan {
    public:
        HistPlan() = default;

        HistPlan(TheoryHistChi2Emit::Strategy emitted, bool specialized,
                 std::optional<TheoryHistChi2Emit::S1LutPlan> s1,
                 std::optional<TheoryHistChi2Emit::S2LinearPlan> s2, std::string cipher_var)
            : emitted_(emitted), specialized_(specialized), s1_(std::move(s1)),
              s2_(std::move(s2)), cipher_var_(std::move(cipher_var)) {}

        [[nodiscard]] TheoryHistChi2Emit::Strategy emitted_strategy() const noexcept {
            return emitted_;
        }

        [[nodiscard]] bool specialized() const noexcept { return specialized_; }

        [[nodiscard]] const std::optional<TheoryHistChi2Emit::S1LutPlan>& s1_lut() const noexcept {
            return s1_;
        }

        [[nodiscard]] const std::optional<TheoryHistChi2Emit::S2LinearPlan>&
        s2_linear() const noexcept {
            return s2_;
        }

        [[nodiscard]] const std::string& cipher_var() const noexcept { return cipher_var_; }

    private:
        TheoryHistChi2Emit::Strategy emitted_ = TheoryHistChi2Emit::Strategy::S0Bytecode;
        bool specialized_ = false;
        std::optional<TheoryHistChi2Emit::S1LutPlan> s1_;
        std::optional<TheoryHistChi2Emit::S2LinearPlan> s2_;
        std::string cipher_var_ = "x";
    };

    /// Prepared host theory + bytecode for one (root, uri, direction) key.
    class Entry {
    public:
        Entry(TheoryIr theory_in, Z29Bytecode::Program program_in, std::string uri,
              std::vector<std::uint8_t> ops_u8_in, HistPlan hist_plan_in)
            : theory_(std::move(theory_in)), program_(std::move(program_in)),
              uri_str_(std::move(uri)), ops_u8_(std::move(ops_u8_in)),
              hist_plan_(std::move(hist_plan_in)) {}

        [[nodiscard]] const TheoryIr& theory() const noexcept { return theory_; }

        [[nodiscard]] const Z29Bytecode::Program& program() const noexcept { return program_; }

        [[nodiscard]] const std::string& uri_str() const noexcept { return uri_str_; }

        [[nodiscard]] const std::vector<std::uint8_t>& ops_u8() const noexcept { return ops_u8_; }

        [[nodiscard]] const std::vector<std::uint8_t>& imm() const noexcept { return program_.imm; }

        [[nodiscard]] const HistPlan& hist_plan() const noexcept { return hist_plan_; }

    private:
        TheoryIr theory_;
        Z29Bytecode::Program program_;
        std::string uri_str_;
        std::vector<std::uint8_t> ops_u8_;
        HistPlan hist_plan_;
    };

    TheoryExportCache() = default;

    TheoryExportCache(const TheoryExportCache&) = delete;
    TheoryExportCache& operator=(const TheoryExportCache&) = delete;
    TheoryExportCache(TheoryExportCache&&) = default;
    TheoryExportCache& operator=(TheoryExportCache&&) = default;

    /// Load IR + compile bytecode when the cache key misses; otherwise return the
    /// cached entry. CUDA builds also enforce TheoryChi2Batch program caps.
    [[nodiscard]] StatusOr<const Entry*> ensure(const std::filesystem::path& theories_root,
                                                std::string_view theory_uri_text,
                                                TransformDirection direction) {
        const std::string root_key = theories_root.lexically_normal().string();
        const std::string uri_key(theory_uri_text);
        if (entry_.has_value() && root_key_ == root_key && uri_key_ == uri_key &&
            direction_ == direction) {
            ++host_hits_;
            return &*entry_;
        }

        StatusOr<Entry> built = compile_entry(theories_root, theory_uri_text, direction);
        if (!built.ok()) {
            return built.status();
        }

        entry_ = std::move(built.value());
        root_key_ = root_key;
        uri_key_ = uri_key;
        direction_ = direction;
#if defined(PARCAE_HAS_CUDA)
        device_ready_ = false;
        device_ops_.reset();
        device_imm_.reset();
#endif
        ++host_compiles_;
        return &*entry_;
    }

#if defined(PARCAE_HAS_CUDA)
    /// Upload `ops`/`imm` once for the current entry. No-op when already warm.
    [[nodiscard]] Status ensure_device_program() {
        if (!entry_.has_value()) {
            return Status::error("TheoryExportCache::ensure_device_program: empty cache");
        }
        if (device_ready_) {
            ++device_hits_;
            return Status::success();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> ops =
            DeviceBuffer<std::uint8_t>::from_host(entry_->ops_u8());
        if (!ops.ok()) {
            return ops.status();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> imm =
            DeviceBuffer<std::uint8_t>::from_host(entry_->imm());
        if (!imm.ok()) {
            return imm.status();
        }
        device_ops_ = std::move(ops.value());
        device_imm_ = std::move(imm.value());
        device_ready_ = true;
        ++device_uploads_;
        return Status::success();
    }

    [[nodiscard]] const DeviceBuffer<std::uint8_t>& device_ops() const { return device_ops_; }

    [[nodiscard]] const DeviceBuffer<std::uint8_t>& device_imm() const { return device_imm_; }

    [[nodiscard]] bool device_program_ready() const noexcept { return device_ready_; }
#endif

    void clear() {
        entry_.reset();
        root_key_.clear();
        uri_key_.clear();
        last_hist_launch_ = TheoryHistChi2Emit::Strategy::S0Bytecode;
#if defined(PARCAE_HAS_CUDA)
        device_ready_ = false;
        device_ops_.reset();
        device_imm_.reset();
#endif
    }

    [[nodiscard]] bool empty() const noexcept { return !entry_.has_value(); }

    [[nodiscard]] std::size_t host_compile_count() const noexcept { return host_compiles_; }

    [[nodiscard]] std::size_t host_hit_count() const noexcept { return host_hits_; }

    [[nodiscard]] std::size_t device_upload_count() const noexcept { return device_uploads_; }

    [[nodiscard]] std::size_t device_hit_count() const noexcept { return device_hits_; }

    /// Strategy used by the most recent `GpuCandidateExport` fused launch on this cache.
    [[nodiscard]] TheoryHistChi2Emit::Strategy last_hist_launch() const noexcept {
        return last_hist_launch_;
    }

    void note_hist_launch(TheoryHistChi2Emit::Strategy strategy) noexcept {
        last_hist_launch_ = strategy;
    }

private:
    [[nodiscard]] static StatusOr<Entry> compile_entry(const std::filesystem::path& theories_root,
                                                       std::string_view theory_uri_text,
                                                       TransformDirection direction) {
        StatusOr<TheoryUri> uri = TheoryUri::parse(theory_uri_text);
        if (!uri.ok()) {
            return uri.status();
        }
        StatusOr<TheoryIr> theory = TheoryDispatch::load_apply_ir(theories_root, uri.value());
        if (!theory.ok()) {
            return theory.status();
        }
        std::string cipher_var = "x";
        {
            StatusOr<TheoryArtifact> art =
                TheoryRegistry::load(theories_root, uri.value().name(), uri.value().version());
            if (art.ok() && art.value().paths().apply_ir().has_value()) {
                const std::filesystem::path path =
                    art.value().artifact_dir(theories_root) / *art.value().paths().apply_ir();
                StatusOr<std::string> cv = TheoryApplyIr::load_cipher_var(path);
                if (cv.ok()) {
                    cipher_var = std::move(cv.value());
                }
            }
        }
        StatusOr<Z29Bytecode::Program> prog =
            Z29Bytecode::compile_theory(theory.value(), direction, cipher_var);
        if (!prog.ok()) {
            return Status::error("TheoryExportCache: bytecode compile failed: " +
                                 prog.status().message());
        }
#if defined(PARCAE_HAS_CUDA)
        if (prog.value().max_stack > TheoryChi2Batch::kMaxDeviceStack) {
            return Status::error(
                "TheoryExportCache: program max_stack exceeds TheoryChi2Batch::kMaxDeviceStack");
        }
        if (prog.value().ops.size() > TheoryChi2Batch::kMaxProgramOps) {
            return Status::error(
                "TheoryExportCache: program exceeds TheoryChi2Batch::kMaxProgramOps");
        }
        if (prog.value().slot_names.size() > TheoryChi2Batch::kMaxSlots) {
            return Status::error(
                "TheoryExportCache: slot_count exceeds TheoryChi2Batch::kMaxSlots");
        }
#endif
        std::vector<std::uint8_t> ops_u8;
        ops_u8.reserve(prog.value().ops.size());
        for (Z29Bytecode::Op op : prog.value().ops) {
            ops_u8.push_back(Z29Bytecode::op_as_u8(op));
        }

        HistPlan hist_plan;
        if (direction == TransformDirection::Decrypt) {
            StatusOr<TheoryHistChi2Emit::EmitBundle> bundle =
                TheoryHistChi2Emit::emit_decrypt_hist(theory.value(), cipher_var);
            if (bundle.ok() && bundle.value().specialized()) {
                hist_plan = HistPlan{bundle.value().emitted_strategy(), true,
                                     bundle.value().s1_lut(), bundle.value().s2_linear(),
                                     cipher_var};
            } else {
                hist_plan = HistPlan{TheoryHistChi2Emit::Strategy::S0Bytecode, false, std::nullopt,
                                     std::nullopt, cipher_var};
            }
        } else {
            hist_plan = HistPlan{TheoryHistChi2Emit::Strategy::S0Bytecode, false, std::nullopt,
                                 std::nullopt, cipher_var};
        }

        return Entry{std::move(theory.value()), std::move(prog.value()), uri.value().to_string(),
                     std::move(ops_u8), std::move(hist_plan)};
    }

    std::optional<Entry> entry_;
    std::string root_key_;
    std::string uri_key_;
    TransformDirection direction_ = TransformDirection::Decrypt;

    std::size_t host_compiles_ = 0;
    std::size_t host_hits_ = 0;
    std::size_t device_uploads_ = 0;
    std::size_t device_hits_ = 0;
    TheoryHistChi2Emit::Strategy last_hist_launch_ = TheoryHistChi2Emit::Strategy::S0Bytecode;

#if defined(PARCAE_HAS_CUDA)
    bool device_ready_ = false;
    DeviceBuffer<std::uint8_t> device_ops_;
    DeviceBuffer<std::uint8_t> device_imm_;
#endif
};

#endif // THEORY_EXPORT_CACHE_HPP
