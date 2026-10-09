#include "ciphertext_autokey_kernel.hpp"

#include "autokey_ctak_device.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "interrupt_device_ops.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__device__ inline std::uint8_t ctak_dense_decrypt_byte(const std::uint8_t* __restrict__ in,
                                                       const std::uint8_t* __restrict__ key,
                                                       std::uint32_t key_len, std::size_t t) {
    const std::uint8_t key_symbol = AutokeyCtakDevice::decrypt_key(in, key, key_len, t);
    return AutokeyCtakDevice::decrypt_symbol(__ldg(in + t), key_symbol);
}

__global__ void ctak_dense_decrypt_uchar4_kernel(const std::uint8_t* __restrict__ in,
                                                 std::uint8_t* out, std::size_t count,
                                                 const std::uint8_t* __restrict__ key,
                                                 std::uint32_t key_len) {
    const std::size_t n4 = count / 4u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    uchar4* out4 = reinterpret_cast<uchar4*>(out);

    for (std::size_t i = tid; i < n4; i += stride) {
        const std::size_t t0 = i * 4u;
        uchar4 w;
        w.x = ctak_dense_decrypt_byte(in, key, key_len, t0);
        w.y = ctak_dense_decrypt_byte(in, key, key_len, t0 + 1u);
        w.z = ctak_dense_decrypt_byte(in, key, key_len, t0 + 2u);
        w.w = ctak_dense_decrypt_byte(in, key, key_len, t0 + 3u);
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        out[t] = ctak_dense_decrypt_byte(in, key, key_len, t);
    }
}

__global__ void ctak_dense_decrypt_scalar_kernel(const std::uint8_t* __restrict__ in,
                                                 std::uint8_t* out, std::size_t count,
                                                 const std::uint8_t* __restrict__ key,
                                                 std::uint32_t key_len) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        out[t] = ctak_dense_decrypt_byte(in, key, key_len, t);
    }
}

/// Serial CTAK for encrypt and/or interrupt skips (feedback over consumed runes).
__global__ void ctak_serial_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                   std::size_t count, const std::uint8_t* __restrict__ key,
                                   std::uint32_t key_len, const std::uint32_t* interrupt_data,
                                   std::uint32_t skip_count, std::uint8_t use_bitmask,
                                   std::uint8_t encrypt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    extern __shared__ std::uint8_t ring[];
    std::uint32_t consumed = 0;

    for (std::size_t i = 0; i < count; ++i) {
        bool skip = false;
        if (use_bitmask != 0u) {
            skip = InterruptDeviceOps::bitmask_should_skip(interrupt_data, i);
        } else if (skip_count != 0u) {
            skip = InterruptDeviceOps::sorted_should_skip(interrupt_data, skip_count,
                                                          static_cast<std::uint32_t>(i));
        }

        if (skip) {
            out[i] = __ldg(in + i);
            continue;
        }

        std::uint8_t key_symbol;
        if (consumed < key_len) {
            key_symbol = __ldg(key + consumed);
        } else {
            key_symbol = ring[(consumed - key_len) % key_len];
        }

        const std::uint8_t x = __ldg(in + i);
        if (encrypt != 0u) {
            out[i] = AutokeyCtakDevice::encrypt_symbol(x, key_symbol);
            ring[consumed % key_len] = out[i];
        } else {
            out[i] = AutokeyCtakDevice::decrypt_symbol(x, key_symbol);
            ring[consumed % key_len] = x;
        }
        ++consumed;
    }
}

Status CiphertextAutokeyKernel::launch_device_async(
    const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
    const std::uint8_t* device_key, std::uint32_t key_len, const InterruptDeviceView& interrupts,
    const std::uint32_t* device_bitmask_or_skips, CudaDir direction) {
    if (key_len == 0) {
        return Status::error("CiphertextAutokeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != count) {
        return Status::error("CiphertextAutokeyKernel: interrupt view length mismatch");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr || device_key == nullptr) {
        return Status::error("CiphertextAutokeyKernel::launch_device_async null device pointer");
    }

    const bool use_bitmask_encoding =
        interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask;
    if (use_bitmask_encoding) {
        if (device_bitmask_or_skips == nullptr) {
            return Status::error("CiphertextAutokeyKernel: bitmask encoding requires device words");
        }
    } else if (!interrupts.sorted_skips().empty() && device_bitmask_or_skips == nullptr) {
        return Status::error("CiphertextAutokeyKernel: sorted skips require device buffer");
    }

    const bool dense_no_skips = interrupts.sorted_skips().empty();

    if (dense_no_skips && direction == CudaDir::Decrypt) {
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
            (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
        const std::size_t work = aligned ? (count + 3u) / 4u : count;
        const int blocks =
            static_cast<int>((work + static_cast<std::size_t>(HistFast::threads) - 1u) /
                             static_cast<std::size_t>(HistFast::threads));
        if (aligned) {
            ctak_dense_decrypt_uchar4_kernel<<<blocks, HistFast::threads>>>(
                device_in, device_out, count, device_key, key_len);
        } else {
            ctak_dense_decrypt_scalar_kernel<<<blocks, HistFast::threads>>>(
                device_in, device_out, count, device_key, key_len);
        }
        return CudaError::to_status(cudaGetLastError(),
                                    "CiphertextAutokeyKernel::launch_device_async dense");
    }

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());
    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);

    const std::size_t shared_bytes = static_cast<std::size_t>(key_len);
    ctak_serial_kernel<<<1, 1, shared_bytes>>>(device_in, device_out, count, device_key, key_len,
                                               device_bitmask_or_skips, skip_count, use_bitmask,
                                               encrypt);
    return CudaError::to_status(cudaGetLastError(),
                                "CiphertextAutokeyKernel::launch_device_async serial");
}

Status CiphertextAutokeyKernel::launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction) {
    Status launched =
        launch_device_async(device_in, device_out, count, device_key, key_len, interrupts,
                            device_bitmask_or_skips, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "CiphertextAutokeyKernel::launch_device sync");
}

Status CiphertextAutokeyKernel::apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out,
                                           std::span<const std::uint8_t> host_key,
                                           const InterruptDeviceView& interrupts,
                                           CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("CiphertextAutokeyKernel::apply_host size mismatch");
    }
    if (host_key.empty()) {
        return Status::error("CiphertextAutokeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != host_in.size()) {
        return Status::error("CiphertextAutokeyKernel: interrupt view length mismatch");
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_key =
        DeviceBuffer<std::uint8_t>::from_host(host_key);
    if (!device_key.ok()) {
        return device_key.status();
    }

    StatusOr<DeviceBuffer<std::uint32_t>> device_interrupt =
        [&]() -> StatusOr<DeviceBuffer<std::uint32_t>> {
        if (interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask) {
            return DeviceBuffer<std::uint32_t>::from_host(interrupts.bitmask_words());
        }
        if (!interrupts.sorted_skips().empty()) {
            return DeviceBuffer<std::uint32_t>::from_host(interrupts.sorted_skips());
        }
        return DeviceBuffer<std::uint32_t>::allocate(0);
    }();
    if (!device_interrupt.ok()) {
        return device_interrupt.status();
    }

    const std::uint32_t* interrupt_ptr =
        device_interrupt.value().empty() ? nullptr : device_interrupt.value().data();

    auto run = [&](DeviceBuffer<std::uint8_t>& device_in,
                   DeviceBuffer<std::uint8_t>& device_out) -> Status {
        Status launched = launch_device(
            device_in.data(), device_out.data(), host_in.size(), device_key.value().data(),
            static_cast<std::uint32_t>(host_key.size()), interrupts, interrupt_ptr, direction);
        if (!launched.ok()) {
            return launched;
        }
        return device_out.copy_to_host(host_out);
    };

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        return run(device.value(), device.value());
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in = DeviceBuffer<std::uint8_t>::from_host(host_in);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_out =
        DeviceBuffer<std::uint8_t>::allocate(host_out.size());
    if (!device_out.ok()) {
        return device_out.status();
    }
    return run(device_in.value(), device_out.value());
}
