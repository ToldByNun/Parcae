#ifndef NVTX_RANGE_HPP
#define NVTX_RANGE_HPP

// Prefer CUDA Toolkit nvtx3 (header-only). When absent, ranges are no-ops so
// CPU-only builds and hosts without the Toolkit still compile.

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    define PARCAE_HAS_NVTX 1
#    include <nvtx3/nvToolsExt.h>
// nvtx3 / Windows headers may define min/max macros that break std::min/max.
#    ifdef min
#      undef min
#    endif
#    ifdef max
#      undef max
#    endif
#  endif
#endif

/// RAII NVTX push/pop for Nsight Systems timelines (`nsys -t nvtx`).
///
/// Named ranges used by theory fused export / search cycle:
/// `prepare_theory`, `bind_slots`, `h2d`, `hist_kernel`, `finalize`, `d2h`,
/// `materialize`, `ingest`. Near-zero cost when no tool is attached.
class NvtxRange {
public:
    explicit NvtxRange(const char* name) noexcept {
#if defined(PARCAE_HAS_NVTX)
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange() {
#if defined(PARCAE_HAS_NVTX)
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

    /// True when `<nvtx3/nvToolsExt.h>` was found at compile time.
    [[nodiscard]] static constexpr bool available() noexcept {
#if defined(PARCAE_HAS_NVTX)
        return true;
#else
        return false;
#endif
    }
};

#endif // NVTX_RANGE_HPP
