#include "totient_prime_stream_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"
#include "launch_geom.hpp"
#include "interrupt_device_ops.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__global__ void totient_dense_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                            std::size_t count,
                                            const std::uint8_t* __restrict__ shifts,
                                            std::uint32_t shift_len, std::uint8_t encrypt) {
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
        // Dense host path requires shift_len == count; defensive copy if short.
        if (t0 + 3u >= shift_len) {
            w.x = v.x;
            w.y = v.y;
            w.z = v.z;
            w.w = v.w;
            if (t0 < shift_len) {
                const std::uint8_t s0 = __ldg(shifts + t0);
                w.x = encrypt != 0u ? Z29Device::add(v.x, s0) : Z29Device::sub(v.x, s0);
            }
            if (t0 + 1u < shift_len) {
                const std::uint8_t s1 = __ldg(shifts + (t0 + 1u));
                w.y = encrypt != 0u ? Z29Device::add(v.y, s1) : Z29Device::sub(v.y, s1);
            }
            if (t0 + 2u < shift_len) {
                const std::uint8_t s2 = __ldg(shifts + (t0 + 2u));
                w.z = encrypt != 0u ? Z29Device::add(v.z, s2) : Z29Device::sub(v.z, s2);
            }
            out4[i] = w;
            continue;
        }
        const std::uint8_t s0 = __ldg(shifts + t0);
        const std::uint8_t s1 = __ldg(shifts + (t0 + 1u));
        const std::uint8_t s2 = __ldg(shifts + (t0 + 2u));
        const std::uint8_t s3 = __ldg(shifts + (t0 + 3u));
        if (encrypt != 0u) {
            w.x = Z29Device::add(v.x, s0);
            w.y = Z29Device::add(v.y, s1);
            w.z = Z29Device::add(v.z, s2);
            w.w = Z29Device::add(v.w, s3);
        } else {
            w.x = Z29Device::sub(v.x, s0);
            w.y = Z29Device::sub(v.y, s1);
            w.z = Z29Device::sub(v.z, s2);
            w.w = Z29Device::sub(v.w, s3);
        }
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        const std::uint8_t x = __ldg(in + t);
        if (t >= shift_len) {
            out[t] = x;
            continue;
        }
        const std::uint8_t s = __ldg(shifts + t);
        out[t] = encrypt != 0u ? Z29Device::add(x, s) : Z29Device::sub(x, s);
    }
}

__global__ void totient_dense_scalar_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                            std::size_t count,
                                            const std::uint8_t* __restrict__ shifts,
                                            std::uint32_t shift_len, std::uint8_t encrypt) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        const std::uint8_t x = __ldg(in + t);
        if (t >= shift_len) {
            out[t] = x;
            continue;
        }
        const std::uint8_t s = __ldg(shifts + t);
        out[t] = encrypt != 0u ? Z29Device::add(x, s) : Z29Device::sub(x, s);
    }
}

__global__ void totient_skip_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                    std::size_t count, const std::uint8_t* __restrict__ shifts,
                                    std::uint32_t shift_len, const std::uint32_t* interrupt_data,
                                    std::uint32_t skip_count, std::uint8_t use_bitmask,
                                    std::uint8_t encrypt) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t i = tid; i < count; i += stride) {
        bool skip = false;
        std::uint32_t stream_cursor = 0;
        if (use_bitmask != 0u) {
            skip = InterruptDeviceOps::bitmask_should_skip(interrupt_data, i);
            stream_cursor = InterruptDeviceOps::bitmask_consumed_before(interrupt_data, i);
        } else {
            const std::uint32_t index = static_cast<std::uint32_t>(i);
            skip = InterruptDeviceOps::sorted_should_skip(interrupt_data, skip_count, index);
            stream_cursor =
                index - InterruptDeviceOps::sorted_skips_before(interrupt_data, skip_count, index);
        }

        const std::uint8_t x = __ldg(in + i);
        if (skip) {
            out[i] = x;
            continue;
        }
        if (stream_cursor >= shift_len) {
            out[i] = x;
            continue;
        }
        const std::uint8_t shift = __ldg(shifts + stream_cursor);
        out[i] = encrypt != 0u ? Z29Device::add(x, shift) : Z29Device::sub(x, shift);
    }
}

std::size_t TotientPrimeStreamKernel::consumable_count(const InterruptDeviceView& interrupts) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < interrupts.consumable_length(); ++i) {
        if (!interrupts.should_skip(i)) {
            ++n;
        }
    }
    return n;
}

Status TotientPrimeStreamKernel::launch_device_async(
    const std::uint8_t* device_in, std::uint8_t* device_out, std::size_t count,
    const std::uint8_t* device_shifts, std::uint32_t shift_len,
    const InterruptDeviceView& interrupts, const std::uint32_t* device_bitmask_or_skips,
    CudaDir direction) {
    if (interrupts.consumable_length() != count) {
        return Status::error("TotientPrimeStreamKernel: interrupt view length mismatch");
    }
    if (count == 0) {
        if (shift_len != 0) {
            return Status::error("TotientPrimeStreamKernel: shifts longer than consumable count");
        }
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("TotientPrimeStreamKernel::launch_device_async null device pointer");
    }
    if (shift_len > 0 && device_shifts == nullptr) {
        return Status::error("TotientPrimeStreamKernel: null shifts pointer");
    }

    const bool use_bitmask_encoding =
        interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask;
    if (use_bitmask_encoding) {
        if (device_bitmask_or_skips == nullptr) {
            return Status::error(
                "TotientPrimeStreamKernel: bitmask encoding requires device words");
        }
    } else if (!interrupts.sorted_skips().empty() && device_bitmask_or_skips == nullptr) {
        return Status::error("TotientPrimeStreamKernel: sorted skips require device buffer");
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
            LaunchGeom::blocks_for(work);
        if (aligned) {
            totient_dense_uchar4_kernel<<<blocks, LaunchGeom::threads()>>>(
                device_in, device_out, count, device_shifts, shift_len, encrypt);
        } else {
            totient_dense_scalar_kernel<<<blocks, LaunchGeom::threads()>>>(
                device_in, device_out, count, device_shifts, shift_len, encrypt);
        }
        return CudaError::to_status(cudaGetLastError(),
                                    "TotientPrimeStreamKernel::launch_device_async");
    }

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());
    const int blocks =
        LaunchGeom::blocks_for(count);
    totient_skip_kernel<<<blocks, LaunchGeom::threads()>>>(
        device_in, device_out, count, device_shifts, shift_len, device_bitmask_or_skips, skip_count,
        use_bitmask, encrypt);
    return CudaError::to_status(cudaGetLastError(), "TotientPrimeStreamKernel::launch_device_async");
}

Status TotientPrimeStreamKernel::launch_device(const std::uint8_t* device_in,
                                               std::uint8_t* device_out, std::size_t count,
                                               const std::uint8_t* device_shifts,
                                               std::uint32_t shift_len,
                                               const InterruptDeviceView& interrupts,
                                               const std::uint32_t* device_bitmask_or_skips,
                                               CudaDir direction) {
    Status launched =
        launch_device_async(device_in, device_out, count, device_shifts, shift_len, interrupts,
                            device_bitmask_or_skips, direction);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "TotientPrimeStreamKernel::launch_device sync");
}

Status TotientPrimeStreamKernel::apply_host(std::span<const std::uint8_t> host_in,
                                            std::span<std::uint8_t> host_out,
                                            std::span<const std::uint8_t> host_shifts,
                                            const InterruptDeviceView& interrupts,
                                            CudaDir direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("TotientPrimeStreamKernel::apply_host size mismatch");
    }
    if (interrupts.consumable_length() != host_in.size()) {
        return Status::error("TotientPrimeStreamKernel: interrupt view length mismatch");
    }

    const std::size_t expected = consumable_count(interrupts);
    if (host_shifts.size() != expected) {
        return Status::error("TotientPrimeStreamKernel: shifts length must equal consumable count");
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(host_shifts);
    if (!device_shifts.ok()) {
        return device_shifts.status();
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
            device_in.data(), device_out.data(), host_in.size(), device_shifts.value().data(),
            static_cast<std::uint32_t>(host_shifts.size()), interrupts, interrupt_ptr, direction);
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
