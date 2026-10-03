#ifndef PINNED_HOST_ARENA_HPP
#define PINNED_HOST_ARENA_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include "cuda_error.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
#include <span>
#include <utility>

/// Grow-only pinned host memory (`cudaHostAlloc` / `cudaFreeHost`) for async H2D/D2H.
///
/// Byte arena used to stage cipher / slots / LUT / b0·b1 / scores before
/// `cudaMemcpyAsync`. Move-only. Destructor swallows free errors (same pattern
/// as `DeviceBuffer`). No C++ namespaces.
class PinnedHostArena {
public:
    PinnedHostArena() = default;

    PinnedHostArena(const PinnedHostArena&) = delete;
    PinnedHostArena& operator=(const PinnedHostArena&) = delete;

    PinnedHostArena(PinnedHostArena&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }

    PinnedHostArena& operator=(PinnedHostArena&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~PinnedHostArena() { reset(); }

    /// Allocate exactly `bytes` of pinned host memory (0 → empty success).
    [[nodiscard]] static StatusOr<PinnedHostArena> allocate(std::size_t bytes) {
        PinnedHostArena arena;
        if (bytes == 0) {
            return arena;
        }
        void* raw = nullptr;
        const cudaError_t err =
            cudaHostAlloc(&raw, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            return CudaError::to_status(err, "PinnedHostArena::allocate");
        }
        arena.data_ = static_cast<std::uint8_t*>(raw);
        arena.capacity_ = bytes;
        return arena;
    }

    /// Grow-only: if `bytes > capacity()`, reallocate (copies old prefix).
    /// Never shrinks. `bytes == 0` is a no-op success.
    [[nodiscard]] Status ensure_capacity(std::size_t bytes) {
        if (bytes <= capacity_) {
            return Status::success();
        }
        StatusOr<PinnedHostArena> grown = allocate(bytes);
        if (!grown.ok()) {
            return grown.status();
        }
        PinnedHostArena next = std::move(grown.value());
        if (data_ != nullptr && capacity_ > 0) {
            // Host-to-host copy of the live prefix (pinned ↔ pinned is fine).
            std::memcpy(next.data(), data_, capacity_);
        }
        reset();
        *this = std::move(next);
        return Status::success();
    }

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }

    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] bool empty() const noexcept { return capacity_ == 0; }

    [[nodiscard]] std::span<std::uint8_t> span() noexcept {
        return std::span<std::uint8_t>(data_, capacity_);
    }

    [[nodiscard]] std::span<const std::uint8_t> span() const noexcept {
        return std::span<const std::uint8_t>(data_, capacity_);
    }

    /// First `n` bytes; error if `n > capacity()`.
    [[nodiscard]] StatusOr<std::span<std::uint8_t>> span_prefix(std::size_t n) noexcept {
        if (n > capacity_) {
            return Status::error("PinnedHostArena::span_prefix exceeds capacity");
        }
        return std::span<std::uint8_t>(data_, n);
    }

    [[nodiscard]] StatusOr<std::span<const std::uint8_t>>
    span_prefix(std::size_t n) const noexcept {
        if (n > capacity_) {
            return Status::error("PinnedHostArena::span_prefix exceeds capacity");
        }
        return std::span<const std::uint8_t>(data_, n);
    }

    /// Fill first `n` bytes from host; requires `n ≤ capacity()`.
    [[nodiscard]] Status fill_from(std::span<const std::uint8_t> host) {
        if (host.size() > capacity_) {
            return Status::error("PinnedHostArena::fill_from exceeds capacity");
        }
        if (host.empty()) {
            return Status::success();
        }
        std::memcpy(data_, host.data(), host.size());
        return Status::success();
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            (void)cudaFreeHost(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr || capacity_ == 0;
    }

private:
    std::uint8_t* data_ = nullptr;
    std::size_t capacity_ = 0;
};

#endif // PINNED_HOST_ARENA_HPP
