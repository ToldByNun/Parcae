#include "ic_mod29_score.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <array>
#include <cuda_runtime_api.h>

constexpr int kIcMod29Threads = 256;

static __global__ void ic_mod29_histogram_kernel(
    const std::uint8_t* indices,
    std::size_t count,
    unsigned long long* counts) {
    __shared__ unsigned long long shared[IcMod29Score::alphabet_size];

    if (threadIdx.x < static_cast<int>(IcMod29Score::alphabet_size)) {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();

    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
             static_cast<std::size_t>(threadIdx.x);
         i < count;
         i += stride) {
        atomicAdd(&shared[indices[i]], 1ull);
    }
    __syncthreads();

    if (threadIdx.x < static_cast<int>(IcMod29Score::alphabet_size)) {
        atomicAdd(&counts[threadIdx.x], shared[threadIdx.x]);
    }
}

Status IcMod29Score::histogram_device(
    const std::uint8_t* device_indices,
    std::size_t count,
    unsigned long long* device_counts) {
    if (count == 0) {
        return Status::success();
    }
    if (device_indices == nullptr || device_counts == nullptr) {
        return Status::error("IcMod29Score::histogram_device null device pointer");
    }

    const int blocks = static_cast<int>(
        (count + static_cast<std::size_t>(kIcMod29Threads) - 1u) /
        static_cast<std::size_t>(kIcMod29Threads));
    ic_mod29_histogram_kernel<<<blocks, kIcMod29Threads>>>(
        device_indices, count, device_counts);

    Status launch = CudaError::to_status(cudaGetLastError(), "IcMod29Score::histogram_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "IcMod29Score::histogram_device sync");
}

StatusOr<double> IcMod29Score::score_host(std::span<const std::uint8_t> indices) {
    const std::size_t n = indices.size();
    if (n < 2) {
        return Status::error("ic_mod29 requires at least 2 symbols");
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(indices);
    if (!device_in.ok()) {
        return device_in.status();
    }

    StatusOr<DeviceBuffer<unsigned long long>> device_counts =
        DeviceBuffer<unsigned long long>::allocate(alphabet_size);
    if (!device_counts.ok()) {
        return device_counts.status();
    }

    std::array<unsigned long long, alphabet_size> zeros{};
    Status cleared = device_counts.value().copy_from_host(
        std::span<const unsigned long long>(zeros.data(), zeros.size()));
    if (!cleared.ok()) {
        return cleared;
    }

    Status hist =
        histogram_device(device_in.value().data(), n, device_counts.value().data());
    if (!hist.ok()) {
        return hist;
    }

    std::array<unsigned long long, alphabet_size> counts{};
    Status copied = device_counts.value().copy_to_host(
        std::span<unsigned long long>(counts.data(), counts.size()));
    if (!copied.ok()) {
        return copied;
    }

    // Fixed order c = 0..28 (CPU twin); integer numerator then one FP divide.
    std::uint64_t numerator = 0;
    for (std::size_t c = 0; c < alphabet_size; ++c) {
        const std::uint64_t nc = counts[c];
        numerator += nc * (nc - 1);
    }
    const double denom = static_cast<double>(n) * static_cast<double>(n - 1);
    return static_cast<double>(numerator) / denom;
}
