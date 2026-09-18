#include "self_repeat_rate_score.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

constexpr int kSelfRepeatThreads = 256;

static __global__ void self_repeat_count_kernel(
    const std::uint8_t* indices,
    std::size_t count,
    unsigned long long* repeat_count) {
    __shared__ unsigned long long shared[kSelfRepeatThreads];

    unsigned long long local = 0;
    const std::size_t edge_count = count - 1;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
             static_cast<std::size_t>(threadIdx.x);
         i < edge_count;
         i += stride) {
        if (indices[i] == indices[i + 1]) {
            ++local;
        }
    }

    shared[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(repeat_count, shared[0]);
    }
}

Status SelfRepeatRateScore::count_repeats_device(
    const std::uint8_t* device_indices,
    std::size_t count,
    unsigned long long* device_repeat_count) {
    if (count < 2) {
        return Status::success();
    }
    if (device_indices == nullptr || device_repeat_count == nullptr) {
        return Status::error("SelfRepeatRateScore::count_repeats_device null device pointer");
    }

    const std::size_t edge_count = count - 1;
    const int blocks = static_cast<int>(
        (edge_count + static_cast<std::size_t>(kSelfRepeatThreads) - 1u) /
        static_cast<std::size_t>(kSelfRepeatThreads));
    self_repeat_count_kernel<<<blocks, kSelfRepeatThreads>>>(
        device_indices, count, device_repeat_count);

    Status launch =
        CudaError::to_status(cudaGetLastError(), "SelfRepeatRateScore::count_repeats_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(
        cudaDeviceSynchronize(), "SelfRepeatRateScore::count_repeats_device sync");
}

StatusOr<double> SelfRepeatRateScore::score_host(std::span<const std::uint8_t> indices) {
    const std::size_t n = indices.size();
    if (n < 2) {
        return Status::error("self_repeat_rate requires at least 2 symbols");
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(indices);
    if (!device_in.ok()) {
        return device_in.status();
    }

    StatusOr<DeviceBuffer<unsigned long long>> device_count =
        DeviceBuffer<unsigned long long>::allocate(1);
    if (!device_count.ok()) {
        return device_count.status();
    }
    const unsigned long long zero = 0;
    Status cleared =
        device_count.value().copy_from_host(std::span<const unsigned long long>(&zero, 1));
    if (!cleared.ok()) {
        return cleared;
    }

    Status counted =
        count_repeats_device(device_in.value().data(), n, device_count.value().data());
    if (!counted.ok()) {
        return counted;
    }

    unsigned long long repeats = 0;
    Status copied =
        device_count.value().copy_to_host(std::span<unsigned long long>(&repeats, 1));
    if (!copied.ok()) {
        return copied;
    }
    return static_cast<double>(repeats) / static_cast<double>(n - 1);
}
