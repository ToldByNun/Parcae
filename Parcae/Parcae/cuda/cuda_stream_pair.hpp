#ifndef CUDA_STREAM_PAIR_HPP
#define CUDA_STREAM_PAIR_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include "cuda_error.hpp"

#include <cuda_runtime_api.h>
#include <utility>

/// Copy + compute CUDA streams with events for H2D → hist → D2H ordering.
///
/// Prefer `create()` (two non-blocking streams). On failure use
/// `create_or_legacy()` / `legacy_default()` so callers stay correct on the
/// default stream with full-device sync. Move-only. No C++ namespaces.
class CudaStreamPair {
public:
    CudaStreamPair() = default;

    CudaStreamPair(const CudaStreamPair&) = delete;
    CudaStreamPair& operator=(const CudaStreamPair&) = delete;

    CudaStreamPair(CudaStreamPair&& other) noexcept { steal_from(other); }

    CudaStreamPair& operator=(CudaStreamPair&& other) noexcept {
        if (this != &other) {
            reset();
            steal_from(other);
        }
        return *this;
    }

    ~CudaStreamPair() { reset(); }

    /// Two non-blocking streams + disable-timing events.
    [[nodiscard]] static StatusOr<CudaStreamPair> create() {
        CudaStreamPair pair;
        cudaError_t err = cudaStreamCreateWithFlags(&pair.copy_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            return CudaError::to_status(err, "CudaStreamPair::create copy");
        }
        err = cudaStreamCreateWithFlags(&pair.compute_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            (void)cudaStreamDestroy(pair.copy_);
            pair.copy_ = nullptr;
            return CudaError::to_status(err, "CudaStreamPair::create compute");
        }
        err = cudaEventCreateWithFlags(&pair.h2d_done_, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            (void)cudaStreamDestroy(pair.copy_);
            (void)cudaStreamDestroy(pair.compute_);
            pair.copy_ = nullptr;
            pair.compute_ = nullptr;
            return CudaError::to_status(err, "CudaStreamPair::create h2d_done");
        }
        err = cudaEventCreateWithFlags(&pair.compute_done_, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            (void)cudaEventDestroy(pair.h2d_done_);
            (void)cudaStreamDestroy(pair.copy_);
            (void)cudaStreamDestroy(pair.compute_);
            pair.copy_ = nullptr;
            pair.compute_ = nullptr;
            pair.h2d_done_ = nullptr;
            return CudaError::to_status(err, "CudaStreamPair::create compute_done");
        }
        pair.owns_ = true;
        return pair;
    }

    /// Dedicated streams when possible; otherwise legacy default-stream pair.
    [[nodiscard]] static CudaStreamPair create_or_legacy() {
        StatusOr<CudaStreamPair> created = create();
        if (created.ok()) {
            return std::move(created.value());
        }
        return legacy_default();
    }

    /// Both streams null → `cudaMemcpyAsync` / launches use the default stream;
    /// `synchronize_copy` falls back to `cudaDeviceSynchronize`.
    [[nodiscard]] static CudaStreamPair legacy_default() { return CudaStreamPair{}; }

    [[nodiscard]] bool uses_dedicated_streams() const noexcept { return owns_; }

    [[nodiscard]] cudaStream_t copy() const noexcept { return copy_; }

    [[nodiscard]] cudaStream_t compute() const noexcept { return compute_; }

    /// Record on the copy stream after H2D work has been enqueued.
    [[nodiscard]] Status record_h2d_done() {
        if (!owns_) {
            return Status::success();
        }
        return CudaError::to_status(cudaEventRecord(h2d_done_, copy_),
                                    "CudaStreamPair::record_h2d_done");
    }

    /// Compute stream waits until H2D event is complete.
    [[nodiscard]] Status wait_h2d_on_compute() {
        if (!owns_) {
            return Status::success();
        }
        return CudaError::to_status(cudaStreamWaitEvent(compute_, h2d_done_, 0),
                                    "CudaStreamPair::wait_h2d_on_compute");
    }

    /// Record on the compute stream after hist/finalize have been enqueued.
    [[nodiscard]] Status record_compute_done() {
        if (!owns_) {
            return Status::success();
        }
        return CudaError::to_status(cudaEventRecord(compute_done_, compute_),
                                    "CudaStreamPair::record_compute_done");
    }

    /// Copy stream waits until compute event is complete (before D2H).
    [[nodiscard]] Status wait_compute_on_copy() {
        if (!owns_) {
            return Status::success();
        }
        return CudaError::to_status(cudaStreamWaitEvent(copy_, compute_done_, 0),
                                    "CudaStreamPair::wait_compute_on_copy");
    }

    [[nodiscard]] Status synchronize_copy() {
        if (!owns_) {
            return CudaError::to_status(cudaDeviceSynchronize(),
                                        "CudaStreamPair::legacy synchronize");
        }
        return CudaError::to_status(cudaStreamSynchronize(copy_),
                                    "CudaStreamPair::synchronize_copy");
    }

    [[nodiscard]] Status synchronize_compute() {
        if (!owns_) {
            return CudaError::to_status(cudaDeviceSynchronize(),
                                        "CudaStreamPair::legacy synchronize");
        }
        return CudaError::to_status(cudaStreamSynchronize(compute_),
                                    "CudaStreamPair::synchronize_compute");
    }

    void reset() noexcept {
        if (owns_) {
            if (compute_done_ != nullptr) {
                (void)cudaEventDestroy(compute_done_);
            }
            if (h2d_done_ != nullptr) {
                (void)cudaEventDestroy(h2d_done_);
            }
            if (compute_ != nullptr) {
                (void)cudaStreamDestroy(compute_);
            }
            if (copy_ != nullptr) {
                (void)cudaStreamDestroy(copy_);
            }
        }
        copy_ = nullptr;
        compute_ = nullptr;
        h2d_done_ = nullptr;
        compute_done_ = nullptr;
        owns_ = false;
    }

private:
    void steal_from(CudaStreamPair& other) noexcept {
        copy_ = other.copy_;
        compute_ = other.compute_;
        h2d_done_ = other.h2d_done_;
        compute_done_ = other.compute_done_;
        owns_ = other.owns_;
        other.copy_ = nullptr;
        other.compute_ = nullptr;
        other.h2d_done_ = nullptr;
        other.compute_done_ = nullptr;
        other.owns_ = false;
    }

    cudaStream_t copy_ = nullptr;
    cudaStream_t compute_ = nullptr;
    cudaEvent_t h2d_done_ = nullptr;
    cudaEvent_t compute_done_ = nullptr;
    bool owns_ = false;
};

#endif // CUDA_STREAM_PAIR_HPP
