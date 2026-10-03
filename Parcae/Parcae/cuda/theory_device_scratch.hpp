#ifndef THEORY_DEVICE_SCRATCH_HPP
#define THEORY_DEVICE_SCRATCH_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "theory_chi2_batch.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <span>
#include <utility>

/// Persistent device scratch for theory fused export (cipher / probs / hist / params).
///
/// Grow-only capacities for `(C_max, T_max)`. Prefer one `cudaMalloc` per buffer
/// then reuse via `cudaMemcpyAsync` into reserved storage. Cipher / probs uploads
/// are skippable when the caller’s generation stamp matches. Param slabs
/// (slots / S1 LUT / S2 b0·b1) and `lane_err` are reserved for the live C.
/// Move-only. No C++ namespaces. Ops/imm stay on `TheoryExportCache`.
class TheoryDeviceScratch {
public:
    static constexpr std::size_t alphabet_size = TheoryChi2Batch::alphabet_size;
    static constexpr std::size_t kMaxCandidates = TheoryChi2Batch::kMaxCandidates;
    static constexpr std::size_t kMaxTokens = TheoryChi2Batch::kMaxTokens;
    static constexpr std::uint16_t kMaxSlots = TheoryChi2Batch::kMaxSlots;

    TheoryDeviceScratch() = default;

    TheoryDeviceScratch(const TheoryDeviceScratch&) = delete;
    TheoryDeviceScratch& operator=(const TheoryDeviceScratch&) = delete;

    TheoryDeviceScratch(TheoryDeviceScratch&& other) noexcept = default;
    TheoryDeviceScratch& operator=(TheoryDeviceScratch&& other) noexcept = default;

    ~TheoryDeviceScratch() = default;

    /// Grow-only reserve for up to `C` candidates and `T` cipher tokens.
    /// Does not shrink. Rejects `C==0`, `T==0`, or values above device caps.
    [[nodiscard]] Status ensure_capacity(std::size_t C, std::size_t T) {
        if (C == 0 || T == 0) {
            return Status::error("TheoryDeviceScratch::ensure_capacity: C and T must be > 0");
        }
        if (C > kMaxCandidates) {
            return Status::error(
                "TheoryDeviceScratch::ensure_capacity: C exceeds TheoryChi2Batch::kMaxCandidates");
        }
        if (T > kMaxTokens) {
            return Status::error(
                "TheoryDeviceScratch::ensure_capacity: T exceeds TheoryChi2Batch::kMaxTokens");
        }
        if (C <= capacity_C_ && T <= capacity_T_ && probs_.size() == alphabet_size) {
            return Status::success();
        }

        const std::size_t need_C = C > capacity_C_ ? C : capacity_C_;
        const std::size_t need_T = T > capacity_T_ ? T : capacity_T_;
        const bool grow_T = need_T > capacity_T_;
        const bool grow_C = need_C > capacity_C_;
        const bool need_probs = probs_.size() != alphabet_size;

        // Stage new buffers first; commit only if every needed alloc succeeds
        // (OOM leaves the previous scratch intact).
        DeviceBuffer<std::uint8_t> new_cipher;
        DeviceBuffer<double> new_probs;
        DeviceBuffer<std::uint32_t> new_counts;
        DeviceBuffer<double> new_scores;
        DeviceBuffer<std::uint8_t> new_slots;
        DeviceBuffer<std::uint8_t> new_luts;
        DeviceBuffer<std::uint8_t> new_b0;
        DeviceBuffer<std::uint8_t> new_b1;
        DeviceBuffer<std::uint8_t> new_lane_err;

        if (grow_T) {
            StatusOr<DeviceBuffer<std::uint8_t>> cipher =
                DeviceBuffer<std::uint8_t>::allocate(need_T);
            if (!cipher.ok()) {
                return cipher.status();
            }
            new_cipher = std::move(cipher.value());
        }
        if (need_probs) {
            StatusOr<DeviceBuffer<double>> probs = DeviceBuffer<double>::allocate(alphabet_size);
            if (!probs.ok()) {
                return probs.status();
            }
            new_probs = std::move(probs.value());
        }
        if (grow_C) {
            StatusOr<DeviceBuffer<std::uint32_t>> counts =
                DeviceBuffer<std::uint32_t>::allocate(need_C * alphabet_size);
            if (!counts.ok()) {
                return counts.status();
            }
            StatusOr<DeviceBuffer<double>> scores = DeviceBuffer<double>::allocate(need_C);
            if (!scores.ok()) {
                return scores.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> slots =
                DeviceBuffer<std::uint8_t>::allocate(need_C * kMaxSlots);
            if (!slots.ok()) {
                return slots.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> luts =
                DeviceBuffer<std::uint8_t>::allocate(need_C * alphabet_size);
            if (!luts.ok()) {
                return luts.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> b0 = DeviceBuffer<std::uint8_t>::allocate(need_C);
            if (!b0.ok()) {
                return b0.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> b1 = DeviceBuffer<std::uint8_t>::allocate(need_C);
            if (!b1.ok()) {
                return b1.status();
            }
            StatusOr<DeviceBuffer<std::uint8_t>> lane_err =
                DeviceBuffer<std::uint8_t>::allocate(need_C);
            if (!lane_err.ok()) {
                return lane_err.status();
            }
            new_counts = std::move(counts.value());
            new_scores = std::move(scores.value());
            new_slots = std::move(slots.value());
            new_luts = std::move(luts.value());
            new_b0 = std::move(b0.value());
            new_b1 = std::move(b1.value());
            new_lane_err = std::move(lane_err.value());
        }

        if (grow_T) {
            cipher_ = std::move(new_cipher);
            cipher_generation_ = 0;
            live_T_ = 0;
            capacity_T_ = need_T;
        }
        if (need_probs) {
            probs_ = std::move(new_probs);
            probs_generation_ = 0;
        }
        if (grow_C) {
            counts_ = std::move(new_counts);
            scores_ = std::move(new_scores);
            slots_ = std::move(new_slots);
            luts_ = std::move(new_luts);
            b0_ = std::move(new_b0);
            b1_ = std::move(new_b1);
            lane_err_ = std::move(new_lane_err);
            capacity_C_ = need_C;
            live_C_ = 0;
            live_slot_count_ = 0;
        }
        return Status::success();
    }

    /// H2D cipher; skipped when `generation` matches the last upload and size matches.
    /// Returns `true` if a copy was issued. Requires prior `ensure_capacity`.
    [[nodiscard]] StatusOr<bool> upload_cipher_async(std::span<const std::uint8_t> cipher,
                                                     std::uint64_t generation,
                                                     cudaStream_t stream = nullptr) {
        if (cipher.size() > capacity_T_ || cipher_.size() < cipher.size()) {
            return Status::error("TheoryDeviceScratch::upload_cipher_async exceeds capacity");
        }
        if (generation != 0 && generation == cipher_generation_ && cipher.size() == live_T_) {
            return false;
        }
        Status copied =
            h2d_async(cipher_.data(), cipher, "TheoryDeviceScratch::upload_cipher_async", stream);
        if (!copied.ok()) {
            return copied;
        }
        cipher_generation_ = generation;
        live_T_ = cipher.size();
        ++cipher_upload_count_;
        return true;
    }

    /// H2D expected-frequency probs (29 bins); skippable via `generation`.
    [[nodiscard]] StatusOr<bool> upload_probs_async(std::span<const double> probs,
                                                    std::uint64_t generation,
                                                    cudaStream_t stream = nullptr) {
        if (probs.size() != alphabet_size || probs_.size() != alphabet_size) {
            return Status::error("TheoryDeviceScratch::upload_probs_async requires 29 bins");
        }
        if (generation != 0 && generation == probs_generation_) {
            return false;
        }
        Status copied =
            h2d_async(probs_.data(), probs, "TheoryDeviceScratch::upload_probs_async", stream);
        if (!copied.ok()) {
            return copied;
        }
        probs_generation_ = generation;
        ++probs_upload_count_;
        return true;
    }

    /// H2D S0 slot matrix (`C * slot_count` row-major). Always copies.
    [[nodiscard]] Status upload_slots_async(std::span<const std::uint8_t> slots,
                                            std::size_t candidate_count, std::uint16_t slot_count,
                                            cudaStream_t stream = nullptr) {
        if (slot_count > kMaxSlots) {
            return Status::error("TheoryDeviceScratch::upload_slots_async slot_count exceeds cap");
        }
        if (candidate_count > capacity_C_) {
            return Status::error("TheoryDeviceScratch::upload_slots_async C exceeds capacity");
        }
        const std::size_t need = candidate_count * static_cast<std::size_t>(slot_count);
        if (slots.size() != need) {
            return Status::error("TheoryDeviceScratch::upload_slots_async size mismatch");
        }
        Status copied =
            h2d_async(slots_.data(), slots, "TheoryDeviceScratch::upload_slots_async", stream);
        if (!copied.ok()) {
            return copied;
        }
        live_C_ = candidate_count;
        live_slot_count_ = slot_count;
        return Status::success();
    }

    /// H2D S1 LUT table (`C * alphabet_size`). Always copies.
    [[nodiscard]] Status upload_luts_async(std::span<const std::uint8_t> luts,
                                           std::size_t candidate_count,
                                           cudaStream_t stream = nullptr) {
        if (candidate_count > capacity_C_) {
            return Status::error("TheoryDeviceScratch::upload_luts_async C exceeds capacity");
        }
        const std::size_t need = candidate_count * alphabet_size;
        if (luts.size() != need) {
            return Status::error("TheoryDeviceScratch::upload_luts_async size mismatch");
        }
        Status copied =
            h2d_async(luts_.data(), luts, "TheoryDeviceScratch::upload_luts_async", stream);
        if (!copied.ok()) {
            return copied;
        }
        live_C_ = candidate_count;
        return Status::success();
    }

    /// H2D S2 linear coeffs. Always copies.
    [[nodiscard]] Status upload_b0_b1_async(std::span<const std::uint8_t> b0,
                                            std::span<const std::uint8_t> b1,
                                            std::size_t candidate_count,
                                            cudaStream_t stream = nullptr) {
        if (candidate_count > capacity_C_) {
            return Status::error("TheoryDeviceScratch::upload_b0_b1_async C exceeds capacity");
        }
        if (b0.size() != candidate_count || b1.size() != candidate_count) {
            return Status::error("TheoryDeviceScratch::upload_b0_b1_async size mismatch");
        }
        Status c0 = h2d_async(b0_.data(), b0, "TheoryDeviceScratch::upload_b0_b1_async b0", stream);
        if (!c0.ok()) {
            return c0;
        }
        Status c1 = h2d_async(b1_.data(), b1, "TheoryDeviceScratch::upload_b0_b1_async b1", stream);
        if (!c1.ok()) {
            return c1;
        }
        live_C_ = candidate_count;
        return Status::success();
    }

    /// D2H scores prefix (`host.size()` elements; must be ≤ capacity).
    [[nodiscard]] Status download_scores_async(std::span<double> host,
                                               cudaStream_t stream = nullptr) {
        if (host.size() > scores_.size()) {
            return Status::error("TheoryDeviceScratch::download_scores_async exceeds capacity");
        }
        if (host.empty()) {
            return Status::success();
        }
        return CudaError::to_status(
            cudaMemcpyAsync(host.data(), scores_.data(), host.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream),
            "TheoryDeviceScratch::download_scores_async");
    }

    [[nodiscard]] std::uint8_t* cipher() noexcept { return cipher_.data(); }
    [[nodiscard]] const std::uint8_t* cipher() const noexcept { return cipher_.data(); }

    [[nodiscard]] double* probs() noexcept { return probs_.data(); }
    [[nodiscard]] const double* probs() const noexcept { return probs_.data(); }

    [[nodiscard]] std::uint32_t* counts() noexcept { return counts_.data(); }
    [[nodiscard]] const std::uint32_t* counts() const noexcept { return counts_.data(); }

    [[nodiscard]] double* scores() noexcept { return scores_.data(); }
    [[nodiscard]] const double* scores() const noexcept { return scores_.data(); }

    [[nodiscard]] std::uint8_t* slots() noexcept { return slots_.data(); }
    [[nodiscard]] const std::uint8_t* slots() const noexcept { return slots_.data(); }

    [[nodiscard]] std::uint8_t* luts() noexcept { return luts_.data(); }
    [[nodiscard]] const std::uint8_t* luts() const noexcept { return luts_.data(); }

    [[nodiscard]] std::uint8_t* b0() noexcept { return b0_.data(); }
    [[nodiscard]] const std::uint8_t* b0() const noexcept { return b0_.data(); }

    [[nodiscard]] std::uint8_t* b1() noexcept { return b1_.data(); }
    [[nodiscard]] const std::uint8_t* b1() const noexcept { return b1_.data(); }

    [[nodiscard]] std::uint8_t* lane_err() noexcept { return lane_err_.data(); }
    [[nodiscard]] const std::uint8_t* lane_err() const noexcept { return lane_err_.data(); }

    [[nodiscard]] std::size_t capacity_C() const noexcept { return capacity_C_; }
    [[nodiscard]] std::size_t capacity_T() const noexcept { return capacity_T_; }
    [[nodiscard]] std::size_t live_C() const noexcept { return live_C_; }
    [[nodiscard]] std::size_t live_T() const noexcept { return live_T_; }
    [[nodiscard]] std::uint16_t live_slot_count() const noexcept { return live_slot_count_; }

    [[nodiscard]] std::uint64_t cipher_generation() const noexcept { return cipher_generation_; }
    [[nodiscard]] std::uint64_t probs_generation() const noexcept { return probs_generation_; }

    [[nodiscard]] std::size_t cipher_upload_count() const noexcept { return cipher_upload_count_; }
    [[nodiscard]] std::size_t probs_upload_count() const noexcept { return probs_upload_count_; }

    [[nodiscard]] bool empty() const noexcept { return capacity_C_ == 0 && capacity_T_ == 0; }

    void reset() noexcept {
        cipher_.reset();
        probs_.reset();
        counts_.reset();
        scores_.reset();
        slots_.reset();
        luts_.reset();
        b0_.reset();
        b1_.reset();
        lane_err_.reset();
        capacity_C_ = 0;
        capacity_T_ = 0;
        live_C_ = 0;
        live_T_ = 0;
        live_slot_count_ = 0;
        cipher_generation_ = 0;
        probs_generation_ = 0;
        cipher_upload_count_ = 0;
        probs_upload_count_ = 0;
    }

private:
    template <typename T>
    [[nodiscard]] static Status h2d_async(T* dst, std::span<const T> host, const char* context,
                                          cudaStream_t stream) {
        if (host.empty()) {
            return Status::success();
        }
        if (dst == nullptr) {
            (void)context;
            return Status::error("TheoryDeviceScratch::h2d_async: null device pointer");
        }
        return CudaError::to_status(
            cudaMemcpyAsync(dst, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice,
                            stream),
            context);
    }

    DeviceBuffer<std::uint8_t> cipher_;
    DeviceBuffer<double> probs_;
    DeviceBuffer<std::uint32_t> counts_;
    DeviceBuffer<double> scores_;
    DeviceBuffer<std::uint8_t> slots_;
    DeviceBuffer<std::uint8_t> luts_;
    DeviceBuffer<std::uint8_t> b0_;
    DeviceBuffer<std::uint8_t> b1_;
    DeviceBuffer<std::uint8_t> lane_err_;

    std::size_t capacity_C_ = 0;
    std::size_t capacity_T_ = 0;
    std::size_t live_C_ = 0;
    std::size_t live_T_ = 0;
    std::uint16_t live_slot_count_ = 0;
    std::uint64_t cipher_generation_ = 0;
    std::uint64_t probs_generation_ = 0;
    std::size_t cipher_upload_count_ = 0;
    std::size_t probs_upload_count_ = 0;
};

#endif // THEORY_DEVICE_SCRATCH_HPP
