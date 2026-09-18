#include "chi2_english_gp_score.hpp"

#include "device_buffer.hpp"
#include "ic_mod29_score.hpp"

#include <array>
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

    std::array<unsigned long long, alphabet_size> zeros{};
    Status cleared = device_counts.value().copy_from_host(
        std::span<const unsigned long long>(zeros.data(), zeros.size()));
    if (!cleared.ok()) {
        return cleared;
    }

    // Reuse IcMod29 integer histogram twin (associative uint64 reduce).
    Status hist =
        IcMod29Score::histogram_device(device_in.value().data(), n, device_counts.value().data());
    if (!hist.ok()) {
        return hist;
    }

    std::array<unsigned long long, alphabet_size> observed{};
    Status copied = device_counts.value().copy_to_host(
        std::span<unsigned long long>(observed.data(), observed.size()));
    if (!copied.ok()) {
        return copied;
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
