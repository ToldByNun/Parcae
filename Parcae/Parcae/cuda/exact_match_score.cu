#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "exact_match_score.hpp"

#include <cuda_runtime_api.h>

constexpr int kExactMatchThreads = 256;

static __global__ void exact_match_mismatch_kernel(const std::uint8_t* candidate,
                                                   const std::uint8_t* reference, std::size_t count,
                                                   unsigned long long* mismatch_count) {
    __shared__ unsigned long long shared[kExactMatchThreads];

    unsigned long long local = 0;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
             static_cast<std::size_t>(threadIdx.x);
         i < count; i += stride) {
        if (candidate[i] != reference[i]) {
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
        atomicAdd(mismatch_count, shared[0]);
    }
}

Status ExactMatchScore::count_mismatches_device(const std::uint8_t* device_candidate,
                                                const std::uint8_t* device_reference,
                                                std::size_t count,
                                                unsigned long long* device_mismatch_count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_candidate == nullptr || device_reference == nullptr ||
        device_mismatch_count == nullptr) {
        return Status::error("ExactMatchScore::count_mismatches_device null device pointer");
    }

    const int blocks =
        static_cast<int>((count + static_cast<std::size_t>(kExactMatchThreads) - 1u) /
                         static_cast<std::size_t>(kExactMatchThreads));
    exact_match_mismatch_kernel<<<blocks, kExactMatchThreads>>>(device_candidate, device_reference,
                                                                count, device_mismatch_count);

    Status launch =
        CudaError::to_status(cudaGetLastError(), "ExactMatchScore::count_mismatches_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "ExactMatchScore::count_mismatches_device sync");
}

StatusOr<double> ExactMatchScore::score_host(std::span<const std::uint8_t> candidate,
                                             std::span<const std::uint8_t> reference) {
    if (candidate.size() != reference.size()) {
        return 0.0;
    }
    if (candidate.empty()) {
        return 1.0;
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_cand =
        DeviceBuffer<std::uint8_t>::from_host(candidate);
    if (!device_cand.ok()) {
        return device_cand.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_ref =
        DeviceBuffer<std::uint8_t>::from_host(reference);
    if (!device_ref.ok()) {
        return device_ref.status();
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

    Status counted = count_mismatches_device(device_cand.value().data(), device_ref.value().data(),
                                             candidate.size(), device_count.value().data());
    if (!counted.ok()) {
        return counted;
    }

    unsigned long long mismatches = 0;
    Status copied =
        device_count.value().copy_to_host(std::span<unsigned long long>(&mismatches, 1));
    if (!copied.ok()) {
        return copied;
    }
    return mismatches == 0 ? 1.0 : 0.0;
}
