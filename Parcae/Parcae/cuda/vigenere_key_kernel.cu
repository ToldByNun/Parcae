#include "vigenere_key_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "interrupt_device_ops.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__global__ void vigenere_dense_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                             std::size_t count, const std::uint8_t* __restrict__ key,
                                             std::uint32_t key_len, std::uint8_t encrypt) {
    const std::size_t n4 = count / 4u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);
    uchar4* out4 = reinterpret_cast<uchar4*>(out);

    for (std::size_t i = tid; i < n4; i += stride) {
        const std::size_t t0 = i * 4u;
        const uchar4 v = __ldg(in4 + i);
        uchar4 w;
        const std::uint8_t k0 = __ldg(key + (t0 % key_len));
        const std::uint8_t k1 = __ldg(key + ((t0 + 1u) % key_len));
        const std::uint8_t k2 = __ldg(key + ((t0 + 2u) % key_len));
        const std::uint8_t k3 = __ldg(key + ((t0 + 3u) % key_len));
        if (encrypt != 0u) {
            w.x = Z29Device::add(v.x, k0);
            w.y = Z29Device::add(v.y, k1);
            w.z = Z29Device::add(v.z, k2);
            w.w = Z29Device::add(v.w, k3);
        } else {
            w.x = Z29Device::sub(v.x, k0);
            w.y = Z29Device::sub(v.y, k1);
            w.z = Z29Device::sub(v.z, k2);
            w.w = Z29Device::sub(v.w, k3);
        }
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        const std::uint8_t k = __ldg(key + (t % key_len));
        const std::uint8_t x = __ldg(in + t);
        out[t] = encrypt != 0u ? Z29Device::add(x, k) : Z29Device::sub(x, k);
    }
}

__global__ void vigenere_dense_scalar_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                             std::size_t count, const std::uint8_t* __restrict__ key,
                                             std::uint32_t key_len, std::uint8_t encrypt) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        const std::uint8_t k = __ldg(key + (t % key_len));
        const std::uint8_t x = __ldg(in + t);
        out[t] = encrypt != 0u ? Z29Device::add(x, k) : Z29Device::sub(x, k);
    }
}

__global__ void vigenere_skip_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count, const std::uint8_t* __restrict__ key,
                                     std::uint32_t key_len, const std::uint32_t* interrupt_data,
                                     std::uint32_t skip_count, std::uint8_t use_bitmask,
                                     std::uint8_t encrypt) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t i = tid; i < count; i += stride) {
        bool skip = false;
        std::uint32_t key_cursor = 0;
        if (use_bitmask != 0u) {
            skip = InterruptDeviceOps::bitmask_should_skip(interrupt_data, i);
            key_cursor = InterruptDeviceOps::bitmask_consumed_before(interrupt_data, i);
        } else {
            const std::uint32_t index = static_cast<std::uint32_t>(i);
            skip = InterruptDeviceOps::sorted_should_skip(interrupt_data, skip_count, index);
            key_cursor = index - InterruptDeviceOps::sorted_skips_before(interrupt_data, skip_count,
                                                                        index);
        }

        const std::uint8_t x = __ldg(in + i);
        if (skip) {
            out[i] = x;
            continue;
        }
        const std::uint8_t key_symbol = __ldg(key + (key_cursor % key_len));
        out[i] = encrypt != 0u ? Z29Device::add(x, key_symbol) : Z29Device::sub(x, key_symbol);
    }
}

Status VigenereKeyKernel::launch_device_async(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction) {
    if (key_len == 0) {
        return Status::error("VigenereKeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != count) {
        return Status::error("VigenereKeyKernel: interrupt view length mismatch");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr || device_key == nullptr) {
        return Status::error("VigenereKeyKernel::launch_device_async null device pointer");
    }

    const bool use_bitmask_encoding =
        interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask;
    if (use_bitmask_encoding) {
        if (device_bitmask_or_skips == nullptr) {
            return Status::error("VigenereKeyKernel: bitmask encoding requires device words");
        }
    } else if (!interrupts.sorted_skips().empty() && device_bitmask_or_skips == nullptr) {
        return Status::error("VigenereKeyKernel: sorted skips require device buffer");
    }

    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const bool dense_no_skips = interrupts.sorted_skips().empty();

    if (dense_no_skips) {
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
            (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
        const std::size_t work = aligned ? (count + 3u) / 4u : count;
        const int blocks =
            static_cast<int>((work + static_cast<std::size_t>(HistFast::threads) - 1u) /
                             static_cast<std::size_t>(HistFast::threads));
        if (aligned) {
            vigenere_dense_uchar4_kernel<<<blocks, HistFast::threads>>>(
                device_in, device_out, count, device_key, key_len, encrypt);
        } else {
            vigenere_dense_scalar_kernel<<<blocks, HistFast::threads>>>(
                device_in, device_out, count, device_key, key_len, encrypt);
        }
        return CudaError::to_status(cudaGetLastError(), "VigenereKeyKernel::launch_device_async");
    }

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());
    const int blocks =
        static_cast<int>((count + static_cast<std::size_t>(HistFast::threads) - 1u) /
                         static_cast<std::size_t>(HistFast::threads));
    vigenere_skip_kernel<<<blocks, HistFast::threads>>>(
        device_in, device_out, count, device_key, key_len, device_bitmask_or_skips, skip_count,
        use_bitmask, encrypt);
    return CudaError::to_status(cudaGetLastError(), "VigenereKeyKernel::launch_device_async");
}

Status VigenereKeyKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t count, const std::uint8_t* device_key,
                                        std::uint32_t key_len,
                                        const InterruptDeviceView& interrupts,
                                        const std::uint32_t* device_bitmask_or_skips,
                                        CudaDir direction) {
    Status launched = launch_device_async(device_in, device_out, count, device_key, key_len,
                                          interrupts, device_bitmask_or_skips, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "VigenereKeyKernel::launch_device sync");
}

Status VigenereKeyKernel::apply_host(std::span<const std::uint8_t> host_in,
                                     std::span<std::uint8_t> host_out,
                                     std::span<const std::uint8_t> host_key,
                                     const InterruptDeviceView& interrupts, CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("VigenereKeyKernel::apply_host size mismatch");
    }
    if (host_key.empty()) {
        return Status::error("VigenereKeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != host_in.size()) {
        return Status::error("VigenereKeyKernel: interrupt view length mismatch");
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
