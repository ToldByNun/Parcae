#ifndef DEVICE_BUFFER_HPP
#define DEVICE_BUFFER_HPP

#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include "cuda_error.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <span>
#include <utility>

/// Owning device allocation (`cudaMalloc` / `cudaFree`) with host↔device copies.
///
/// Move-only. Destructors swallow free errors (same pattern as unique_ptr); check
/// `Status` on allocate / copy. Element type `T` must be trivially copyable for
/// `cudaMemcpy` (Index29 streams use `std::uint8_t`).
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    /// Allocate `count` elements on the current CUDA device.
    [[nodiscard]] static StatusOr<DeviceBuffer> allocate(std::size_t count) {
        DeviceBuffer buffer;
        if (count == 0) {
            return buffer;
        }
        void* raw = nullptr;
        const cudaError_t err = cudaMalloc(&raw, count * sizeof(T));
        if (err != cudaSuccess) {
            return CudaError::to_status(err, "DeviceBuffer::allocate");
        }
        buffer.data_ = static_cast<T*>(raw);
        buffer.count_ = count;
        return buffer;
    }

    [[nodiscard]] Status copy_from_host(std::span<const T> host) {
        if (host.size() != count_) {
            return Status::error("DeviceBuffer::copy_from_host size mismatch");
        }
        if (count_ == 0) {
            return Status::success();
        }
        return CudaError::to_status(
            cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice),
            "DeviceBuffer::copy_from_host");
    }

    [[nodiscard]] Status copy_to_host(std::span<T> host) const {
        if (host.size() != count_) {
            return Status::error("DeviceBuffer::copy_to_host size mismatch");
        }
        if (count_ == 0) {
            return Status::success();
        }
        return CudaError::to_status(
            cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost),
            "DeviceBuffer::copy_to_host");
    }

    /// Allocate then upload in one step.
    [[nodiscard]] static StatusOr<DeviceBuffer> from_host(std::span<const T> host) {
        StatusOr<DeviceBuffer> buffer = allocate(host.size());
        if (!buffer.ok()) {
            return buffer.status();
        }
        Status copied = buffer.value().copy_from_host(host);
        if (!copied.ok()) {
            return copied;
        }
        return buffer;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    [[nodiscard]] T* data() noexcept { return data_; }

    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr || count_ == 0;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

#endif // DEVICE_BUFFER_HPP
