#include "parcae_cuda.hpp"

#include "device_buffer.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

int main() {
#if defined(PARCAE_HAS_CUDA)
    std::cout << "Parcae CUDA: " << (ParcaeCuda::available() ? "available" : "unavailable")
              << '\n';

    const std::vector<std::uint8_t> host_in{3, 1, 4, 1, 5};
    StatusOr<DeviceBuffer<std::uint8_t>> device = DeviceBuffer<std::uint8_t>::from_host(host_in);
    if (!device.ok()) {
        std::cerr << device.status().message() << '\n';
        return 1;
    }
    std::vector<std::uint8_t> host_out(host_in.size());
    Status copied = device.value().copy_to_host(host_out);
    if (!copied.ok()) {
        std::cerr << copied.message() << '\n';
        return 1;
    }
    if (host_out != host_in) {
        std::cerr << "DeviceBuffer round-trip mismatch\n";
        return 1;
    }
    std::cout << "DeviceBuffer round-trip: ok (" << host_in.size() << " bytes)\n";
#else
    std::cout << "Parcae (host-only build; enable x64 + CUDA Toolkit for twins)\n";
#endif
    return 0;
}
