#ifndef PARCAE_CUDA_HPP
#define PARCAE_CUDA_HPP

/// Façade for the CUDA twin library (sources under Parcae/Parcae/cuda/).
class ParcaeCuda {
public:
    /// Always true in a successfully linked `parcae_cuda` / VS CUDA build.
    [[nodiscard]] static bool available() noexcept;

private:
    ParcaeCuda() = delete;
};

#endif // PARCAE_CUDA_HPP
