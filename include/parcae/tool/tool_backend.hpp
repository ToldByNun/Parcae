#ifndef TOOL_BACKEND_HPP
#define TOOL_BACKEND_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstdint>
#include <string>
#include <string_view>

/// Transform / score execution backend for tool + CLI façades.
enum class Backend : std::uint8_t {
    Cpu = 0,
    Cuda = 1,
};

class BackendUtil {
public:
    [[nodiscard]] static StatusOr<Backend> from_string(std::string_view text) {
        if (text == "cpu") {
            return Backend::Cpu;
        }
        if (text == "cuda") {
            return Backend::Cuda;
        }
        return Status::error("Unknown backend (expected cpu|cuda)");
    }

    [[nodiscard]] static std::string_view to_string(Backend backend) noexcept {
        return backend == Backend::Cuda ? "cuda" : "cpu";
    }

    /// True when this translation unit was compiled with `PARCAE_HAS_CUDA`.
    [[nodiscard]] static bool cuda_built() noexcept {
#if defined(PARCAE_HAS_CUDA)
        return true;
#else
        return false;
#endif
    }

    /// Cpu always OK; Cuda requires a CUDA-linked build (`PARCAE_HAS_CUDA`).
    [[nodiscard]] static Status ensure_usable(Backend backend) {
        if (backend == Backend::Cpu) {
            return Status::success();
        }
        if (!cuda_built()) {
            return Status::error(
                "CUDA backend requested but Parcae was built without CUDA (PARCAE_BUILD_CUDA)");
        }
        return Status::success();
    }

private:
    BackendUtil() = delete;
};

#endif // TOOL_BACKEND_HPP
