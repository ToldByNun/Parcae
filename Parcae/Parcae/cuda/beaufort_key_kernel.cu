#include "beaufort_key_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "launch_geom.hpp"
#include "interrupt_device_ops.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__global__ void beaufort_dense_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                             std::size_t count, const std::uint8_t* __restrict__ key,
                                             std::uint32_t key_len) {
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
        w.x = Z29Device::sub(__ldg(key + (t0 % key_len)), v.x);
        w.y = Z29Device::sub(__ldg(key + ((t0 + 1u) % key_len)), v.y);
        w.z = Z29Device::sub(__ldg(key + ((t0 + 2u) % key_len)), v.z);
        w.w = Z29Device::sub(__ldg(key + ((t0 + 3u) % key_len)), v.w);
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        out[t] = Z29Device::sub(__ldg(key + (t % key_len)), __ldg(in + t));
    }
}

__global__ void beaufort_dense_scalar_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                             std::size_t count, const std::uint8_t* __restrict__ key,
                                             std::uint32_t key_len) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        out[t] = Z29Device::sub(__ldg(key + (t % key_len)), __ldg(in + t));
    }
}

__global__ void beaufort_skip_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count, const std::uint8_t* __restrict__ key,
                                     std::uint32_t key_len, const std::uint32_t* interrupt_data,
                                     std::uint32_t skip_count, std::uint8_t use_bitmask) {
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
        out[i] = Z29Device::sub(__ldg(key + (key_cursor % key_len)), x);
    }
}

Status BeaufortKeyKernel::launch_device_async(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips) {
    if (key_len == 0) {
        return Status::error("BeaufortKeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != count) {
        return Status::error("BeaufortKeyKernel: interrupt view length mismatch");
    }
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr || device_key == nullptr) {
        return Status::error("BeaufortKeyKernel::launch_device_async null device pointer");
    }

    const bool use_bitmask_encoding =
        interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask;
    if (use_bitmask_encoding) {
        if (device_bitmask_or_skips == nullptr) {
            return Status::error("BeaufortKeyKernel: bitmask encoding requires device words");
        }
    } else if (!interrupts.sorted_skips().empty() && device_bitmask_or_skips == nullptr) {
        return Status::error("BeaufortKeyKernel: sorted skips require device buffer");
    }

    const bool dense_no_skips = interrupts.sorted_skips().empty();
    if (dense_no_skips) {
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
            (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
        const std::size_t work = aligned ? (count + 3u) / 4u : count;
        const int blocks =
            LaunchGeom::blocks_for(work);
        if (aligned) {
            beaufort_dense_uchar4_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out,
                                                                        count, device_key, key_len);
        } else {
            beaufort_dense_scalar_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out,
                                                                        count, device_key, key_len);
        }
        return CudaError::to_status(cudaGetLastError(), "BeaufortKeyKernel::launch_device_async");
    }

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());
    const int blocks =
        LaunchGeom::blocks_for(count);
    beaufort_skip_kernel<<<blocks, LaunchGeom::threads()>>>(device_in, device_out, count, device_key,
                                                        key_len, device_bitmask_or_skips, skip_count,
                                                        use_bitmask);
    return CudaError::to_status(cudaGetLastError(), "BeaufortKeyKernel::launch_device_async");
}

Status BeaufortKeyKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t count, const std::uint8_t* device_key,
                                        std::uint32_t key_len,
                                        const InterruptDeviceView& interrupts,
                                        const std::uint32_t* device_bitmask_or_skips) {
    Status launched = launch_device_async(device_in, device_out, count, device_key, key_len,
                                          interrupts, device_bitmask_or_skips);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "BeaufortKeyKernel::launch_device sync");
}

Status BeaufortKeyKernel::apply_host(std::span<const std::uint8_t> host_in,
                                     std::span<std::uint8_t> host_out,
                                     std::span<const std::uint8_t> host_key,
                                     const InterruptDeviceView& interrupts) {
    if (host_in.size() != host_out.size()) {
        return Status::error("BeaufortKeyKernel::apply_host size mismatch");
    }
    if (host_key.empty()) {
        return Status::error("BeaufortKeyKernel: key must be non-empty");
    }
    if (interrupts.consumable_length() != host_in.size()) {
        return Status::error("BeaufortKeyKernel: interrupt view length mismatch");
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
            static_cast<std::uint32_t>(host_key.size()), interrupts, interrupt_ptr);
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
