#ifndef THEORY_HIST_MODULE_HPP
#define THEORY_HIST_MODULE_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include "theory_chi2_batch.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/// Process-local cubin / NVRTC cache for fused theory hist modules.
///
/// Keys are `(theory_uri, content_digest)` where `content_digest` is lowercase
/// SHA-256 hex of the cubin bytes or CUDA source. Load failure leaves the URI
/// unspecialized. Launch failure is a soft error for search (caller → S0).
///
/// Launch ABI matches `TheoryChi2Batch` when a driver kernel is bound. Entries
/// registered via `ensure_proxy` (or cubin/NVRTC with `proxy_launch`) run that
/// ABI through the in-lib bytecode twin after a successful cache insert —
/// used when the artifact proves presence without a custom hist body.
///
/// No C++ namespaces. Test hooks: `set_inject_oom` / `set_inject_invalid_launch`.
class TheoryHistModule {
public:
    static constexpr std::string_view kDefaultKernelSymbol = "parcae_theory_hist_module";

    static constexpr std::size_t alphabet_size = TheoryChi2Batch::alphabet_size;
    static constexpr std::size_t kMaxCandidates = TheoryChi2Batch::kMaxCandidates;
    static constexpr std::size_t kMaxTokens = TheoryChi2Batch::kMaxTokens;

    enum class LaunchKind : std::uint8_t { ProxyS0 = 0, DriverKernel = 1 };

    class Entry {
    public:
        std::string uri;
        std::string digest;
        LaunchKind kind = LaunchKind::ProxyS0;
        void* module = nullptr;   // CUmodule
        void* function = nullptr; // CUfunction
        std::string symbol;
    };

    /// True when any ready module entry exists for `theory_uri`.
    [[nodiscard]] static bool has(std::string_view theory_uri) noexcept;

    /// True when a ready entry exists for the exact URI+digest pair.
    [[nodiscard]] static bool has(std::string_view theory_uri, std::string_view digest) noexcept;

    [[nodiscard]] static std::size_t cache_size() noexcept;

    /// Drop all modules and clear inject flags (tests).
    static void clear();

    /// Force launch to report CUDA OOM (tests).
    static void set_inject_oom(bool enabled) noexcept;

    /// Force launch to report invalid module (tests).
    static void set_inject_invalid_launch(bool enabled) noexcept;

    /// Register a presence-only entry (no driver load). Digest must be non-empty.
    [[nodiscard]] static Status ensure_proxy(std::string_view theory_uri, std::string_view digest);

    /// Load cubin/fatbin/PTX image; digest must match `Sha256` of `image`.
    /// When `proxy_launch` is true, cache the module but launch via TheoryChi2Batch.
    [[nodiscard]] static Status ensure_cubin(std::string_view theory_uri, std::string_view digest,
                                             std::span<const std::uint8_t> image,
                                             std::string_view kernel_symbol = kDefaultKernelSymbol,
                                             bool proxy_launch = false);

    /// NVRTC-compile `cuda_source` (digest = SHA-256 of source), then load.
    [[nodiscard]] static Status ensure_nvrtc(std::string_view theory_uri, std::string_view digest,
                                             std::string_view cuda_source,
                                             std::string_view kernel_symbol = kDefaultKernelSymbol,
                                             bool proxy_launch = false);

    /// Read file bytes, digest = SHA-256(file), then `ensure_cubin`.
    [[nodiscard]] static Status ensure_file(std::string_view theory_uri,
                                            const std::filesystem::path& path,
                                            std::string_view kernel_symbol = kDefaultKernelSymbol,
                                            bool proxy_launch = false);

    /// Launch cached module for `theory_uri` (any ready digest). Soft error on miss/fail.
    [[nodiscard]] static Status launch_async(
        std::string_view theory_uri, const std::uint8_t* device_in, const std::uint8_t* device_ops,
        const std::uint8_t* device_imm, std::uint32_t op_count, const std::uint8_t* device_slots,
        std::uint16_t slot_count, std::uint16_t cipher_slot, std::uint16_t index_slot,
        std::uint8_t binds_index_i, std::uint16_t max_stack, const double* device_probabilities,
        std::uint32_t* device_counts, double* device_scores, std::uint8_t* device_lane_err,
        std::size_t candidate_count, std::size_t token_count, cudaStream_t stream = nullptr);

private:
    TheoryHistModule() = delete;

    [[nodiscard]] static Status init_driver();

    [[nodiscard]] static StatusOr<std::string> read_file_bytes(const std::filesystem::path& path,
                                                               std::vector<std::uint8_t>* out);

    [[nodiscard]] static Status insert_loaded(std::string uri, std::string digest, LaunchKind kind,
                                              void* module, void* function, std::string symbol);

    [[nodiscard]] static Status load_module_image(std::string_view theory_uri,
                                                  std::string_view digest,
                                                  std::span<const std::uint8_t> image,
                                                  std::string_view kernel_symbol,
                                                  bool proxy_launch);

    [[nodiscard]] static Entry* find_ready(std::string_view theory_uri) noexcept;

    [[nodiscard]] static Status driver_to_status(int result, std::string_view context);

    [[nodiscard]] static Status nvrtc_to_status(int result, std::string_view context);
};

#endif // THEORY_HIST_MODULE_HPP
