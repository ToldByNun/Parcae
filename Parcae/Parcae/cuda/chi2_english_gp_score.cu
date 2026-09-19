#include "chi2_english_gp_score.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "ic_mod29_score.hpp"

#include <array>
#include <cuda_runtime_api.h>

StatusOr<double> Chi2EnglishGpScore::finalize(
    std::span<const unsigned long long> observed,
    std::size_t n,
    std::span<const double> probabilities) {
    if (n == 0) {
        return Status::error("chi2_english_gp_v0 requires a non-empty sequence");
    }
    if (observed.size() != alphabet_size) {
        return Status::error("chi2_english_gp_v0 observed histogram must have size 29");
    }
    if (probabilities.size() != alphabet_size) {
        return Status::error("chi2_english_gp_v0 probabilities must have size 29");
    }
    for (std::size_t c = 0; c < alphabet_size; ++c) {
        if (!(probabilities[c] > 0.0)) {
            return Status::error("chi2_english_gp_v0 expected frequency is zero");
        }
    }

    // Fixed order c = 0..28 — must match Chi2EnglishGp (no parallel FP tree).
    const double n_d = static_cast<double>(n);
    double chi2 = 0.0;
    for (std::size_t c = 0; c < alphabet_size; ++c) {
        const double e = probabilities[c] * n_d;
        if (!(e > 0.0)) {
            return Status::error("chi2_english_gp_v0 expected frequency is zero");
        }
        const double diff = static_cast<double>(observed[c]) - e;
        chi2 += (diff * diff) / e;
    }
    return chi2;
}

StatusOr<double> Chi2EnglishGpScore::score_device(
    const std::uint8_t* device_indices,
    std::size_t count,
    std::span<const double> probabilities,
    unsigned long long* device_counts) {
    if (count == 0) {
        return Status::error("chi2_english_gp_v0 requires a non-empty sequence");
    }
    if (device_indices == nullptr || device_counts == nullptr) {
        return Status::error("Chi2EnglishGpScore::score_device null device pointer");
    }
    if (probabilities.size() != alphabet_size) {
        return Status::error("chi2_english_gp_v0 probabilities must have size 29");
    }

    Status cleared = CudaError::to_status(
        cudaMemset(device_counts, 0, alphabet_size * sizeof(unsigned long long)),
        "Chi2EnglishGpScore::score_device clear histogram");
    if (!cleared.ok()) {
        return cleared;
    }

    Status hist = IcMod29Score::histogram_device(device_indices, count, device_counts);
    if (!hist.ok()) {
        return hist;
    }

    std::array<unsigned long long, alphabet_size> observed{};
    Status copied = CudaError::to_status(
        cudaMemcpy(
            observed.data(),
            device_counts,
            alphabet_size * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost),
        "Chi2EnglishGpScore::score_device D2H histogram");
    if (!copied.ok()) {
        return copied;
    }

    return finalize(observed, count, probabilities);
}

StatusOr<double> Chi2EnglishGpScore::score_host(
    std::span<const std::uint8_t> indices,
    std::span<const double> probabilities) {
    const std::size_t n = indices.size();
    if (n == 0) {
        return Status::error("chi2_english_gp_v0 requires a non-empty sequence");
    }
    if (probabilities.size() != alphabet_size) {
        return Status::error("chi2_english_gp_v0 probabilities must have size 29");
    }
    for (std::size_t c = 0; c < alphabet_size; ++c) {
        if (!(probabilities[c] > 0.0)) {
            return Status::error("chi2_english_gp_v0 expected frequency is zero");
        }
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

    return score_device(
        device_in.value().data(),
        n,
        probabilities,
        device_counts.value().data());
}
