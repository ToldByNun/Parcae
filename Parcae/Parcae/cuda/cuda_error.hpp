#ifndef PARCAE_CUDA_ERROR_HPP
#define PARCAE_CUDA_ERROR_HPP

#include "parcae/core/status.hpp"

#include <cuda_runtime_api.h>

#include <string>
#include <string_view>

namespace parcae::cuda {

/// Map a CUDA runtime status into Parcae `Status` (no raw `cudaError_t` in APIs).
[[nodiscard]] inline Status status_from_cuda(cudaError_t err, std::string_view context) {
    if (err == cudaSuccess) {
        return Status::success();
    }
    const char* name = cudaGetErrorName(err);
    const char* desc = cudaGetErrorString(err);
    std::string message;
    message.reserve(context.size() + 64);
    message.append(context);
    message.append(": ");
    message.append(name != nullptr ? name : "cudaError");
    message.append(" — ");
    message.append(desc != nullptr ? desc : "unknown");
    return Status::error(std::move(message));
}

}  // namespace parcae::cuda

#endif // PARCAE_CUDA_ERROR_HPP
