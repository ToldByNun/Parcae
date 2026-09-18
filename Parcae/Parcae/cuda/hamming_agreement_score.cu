#include "hamming_agreement_score.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

constexpr int kHammingThreads = 256;

static __global__ void hamming_match_kernel(
    const std::uint8_t* candidate,
    const std::uint8_t* reference,
    std::size_t count,
    unsigned long long* match_count) {
    __shared__ unsigned long long shared[kHammingThreads];

    unsigned long long local = 0;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t i =
             static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
             static_cast<std::size_t>(threadIdx.x);
         i < count;
         i += stride) {
        if (candidate[i] == reference[i]) {
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
        atomicAdd(match_count, shared[0]);
    }
}

Status HammingAgreementScore::count_matches_device(
    const std::uint8_t* device_candidate,
    const std::uint8_t* device_reference,
    std::size_t count,
    unsigned long long* device_match_count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_candidate == nullptr || device_reference == nullptr ||
        device_match_count == nullptr) {
        return Status::error("HammingAgreementScore::count_matches_device null device pointer");
    }

    const int blocks = static_cast<int>(
        (count + static_cast<std::size_t>(kHammingThreads) - 1u) /
        static_cast<std::size_t>(kHammingThreads));
    hamming_match_kernel<<<blocks, kHammingThreads>>>(
        device_candidate, device_reference, count, device_match_count);

    Status launch =
        CudaError::to_status(cudaGetLastError(), "HammingAgreementScore::count_matches_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(
        cudaDeviceSynchronize(), "HammingAgreementScore::count_matches_device sync");
}

StatusOr<double> HammingAgreementScore::score_host(
    std::span<const std::uint8_t> candidate,
    std::span<const std::uint8_t> reference) {
    if (candidate.size() != reference.size()) {
        return Status::error("hamming_agreement requires equal lengths");
    }
    const std::size_t n = candidate.size();
    if (n == 0) {
        return Status::error("hamming_agreement requires a non-empty sequence");
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
    Status cleared = device_count.value().copy_from_host(std::span<const unsigned long long>(&zero, 1));
    if (!cleared.ok()) {
        return cleared;
    }

    Status counted = count_matches_device(
        device_cand.value().data(),
        device_ref.value().data(),
        n,
        device_count.value().data());
    if (!counted.ok()) {
        return counted;
    }

    unsigned long long matches = 0;
    Status copied =
        device_count.value().copy_to_host(std::span<unsigned long long>(&matches, 1));
    if (!copied.ok()) {
        return copied;
    }
    return static_cast<double>(matches) / static_cast<double>(n);
}
