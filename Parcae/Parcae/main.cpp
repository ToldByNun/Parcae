#include "parcae_cuda.hpp"

#include <iostream>

int main() {
#if defined(PARCAE_HAS_CUDA)
    std::cout << "Parcae CUDA: "
              << (parcae::cuda::available() ? "available" : "unavailable") << '\n';
#else
    std::cout << "Parcae (host-only build; enable x64 + CUDA Toolkit for twins)\n";
#endif
    return 0;
}
