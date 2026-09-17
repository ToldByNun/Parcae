#ifndef PARCAE_CUDA_HPP
#define PARCAE_CUDA_HPP

/// Public façade for the CUDA twin library (sources under Parcae/Parcae/cuda/).
/// Built when the Visual Studio CUDA project or CMake `PARCAE_BUILD_CUDA=ON` is used.
namespace parcae::cuda {

/// Always true in a successfully linked `parcae_cuda` / VS CUDA build.
[[nodiscard]] bool available() noexcept;

}  // namespace parcae::cuda

#endif // PARCAE_CUDA_HPP
