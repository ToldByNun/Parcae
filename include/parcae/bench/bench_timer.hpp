#ifndef BENCH_TIMER_HPP
#define BENCH_TIMER_HPP

#include "parcae/bench/bench_metric.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <chrono>
#include <cstddef>
#include <utility>

#if defined(PARCAE_HAS_CUDA)
#include "cuda_error.hpp"

#include <cuda_runtime_api.h>
#endif

/// Shared timing protocol for Parcae bench / SLO diagnostics.
///
/// Contract (matches historical `ThroughputTiers` CUDA path):
/// - 4 warmup launches (setup / clocks settle; not timed)
/// - then **median of 3** timed samples
/// - each sample: `repeats` launches inside one wall / cudaEvent window
/// - rates via `BenchMetric::Sample::from_elapsed` (setup excluded)
///
/// CPU uses `std::chrono::steady_clock`. CUDA uses `cudaEvent` elapsed time
/// (only when `PARCAE_HAS_CUDA`).
///
/// `LaunchFn` must be callable as `Status()` (success or error).
class BenchTimer {
public:
    static constexpr int warmup_launches = 4;
    static constexpr int timed_samples = 3;

    /// One CPU timed window: `repeats` launches, no warmup.
    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<BenchMetric::Sample>
    time_cpu_once(std::size_t repeats, std::size_t candidates, std::size_t tokens,
                  LaunchFn&& launch) {
        using clock = std::chrono::steady_clock;
        const clock::time_point t0 = clock::now();
        for (std::size_t r = 0; r < repeats; ++r) {
            Status launched = launch();
            if (!launched.ok()) {
                return launched;
            }
        }
        const clock::time_point t1 = clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        return BenchMetric::Sample::from_elapsed(repeats, candidates, tokens, seconds);
    }

    /// CPU protocol: `warmup_launches` + median-of-`timed_samples`.
    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<BenchMetric::Sample>
    time_cpu(std::size_t repeats, std::size_t candidates, std::size_t tokens, LaunchFn&& launch) {
        for (int w = 0; w < warmup_launches; ++w) {
            Status warm = launch();
            if (!warm.ok()) {
                return warm;
            }
        }

        BenchMetric::Sample samples[timed_samples]{};
        for (int i = 0; i < timed_samples; ++i) {
            StatusOr<BenchMetric::Sample> one = time_cpu_once(repeats, candidates, tokens, launch);
            if (!one.ok()) {
                return one.status();
            }
            samples[i] = std::move(one.value());
        }
        return BenchMetric::median3_sample(samples[0], samples[1], samples[2]);
    }

#if defined(PARCAE_HAS_CUDA)
    /// One CUDA timed window via `cudaEvent` (stream 0). No warmup.
    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<BenchMetric::Sample>
    time_cuda_once(std::size_t repeats, std::size_t candidates, std::size_t tokens,
                   LaunchFn&& launch) {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        Status ev0 = CudaError::to_status(cudaEventCreate(&start), "BenchTimer event create start");
        if (!ev0.ok()) {
            return ev0;
        }
        Status ev1 = CudaError::to_status(cudaEventCreate(&stop), "BenchTimer event create stop");
        if (!ev1.ok()) {
            cudaEventDestroy(start);
            return ev1;
        }

        Status rec0 =
            CudaError::to_status(cudaEventRecord(start, 0), "BenchTimer event record start");
        if (!rec0.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return rec0;
        }
        for (std::size_t r = 0; r < repeats; ++r) {
            Status launched = launch();
            if (!launched.ok()) {
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                return launched;
            }
        }
        Status rec1 =
            CudaError::to_status(cudaEventRecord(stop, 0), "BenchTimer event record stop");
        if (!rec1.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return rec1;
        }
        Status synced = CudaError::to_status(cudaEventSynchronize(stop), "BenchTimer event sync");
        if (!synced.ok()) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return synced;
        }
        float ms = 0.0f;
        Status elapsed = CudaError::to_status(cudaEventElapsedTime(&ms, start, stop),
                                              "BenchTimer event elapsed");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (!elapsed.ok()) {
            return elapsed;
        }
        const double seconds = static_cast<double>(ms) * 1.0e-3;
        return BenchMetric::Sample::from_elapsed(repeats, candidates, tokens, seconds);
    }

    /// CUDA protocol: 4 warmups + device sync + median-of-3 cudaEvent samples.
    template <typename LaunchFn>
    [[nodiscard]] static StatusOr<BenchMetric::Sample>
    time_cuda(std::size_t repeats, std::size_t candidates, std::size_t tokens, LaunchFn&& launch) {
        for (int w = 0; w < warmup_launches; ++w) {
            Status warm = launch();
            if (!warm.ok()) {
                return warm;
            }
        }
        Status warm_sync = CudaError::to_status(cudaDeviceSynchronize(), "BenchTimer warmup sync");
        if (!warm_sync.ok()) {
            return warm_sync;
        }

        BenchMetric::Sample samples[timed_samples]{};
        for (int i = 0; i < timed_samples; ++i) {
            StatusOr<BenchMetric::Sample> one = time_cuda_once(repeats, candidates, tokens, launch);
            if (!one.ok()) {
                return one.status();
            }
            samples[i] = std::move(one.value());
        }
        return BenchMetric::median3_sample(samples[0], samples[1], samples[2]);
    }
#endif // PARCAE_HAS_CUDA

private:
    BenchTimer() = delete;
};

#endif // BENCH_TIMER_HPP
