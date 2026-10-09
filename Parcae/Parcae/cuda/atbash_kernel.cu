#include "atbash_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "hist_fast.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

__global__ void atbash_uchar4_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count) {
    const std::size_t n4 = count / 4u;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    const uchar4* __restrict__ in4 = reinterpret_cast<const uchar4*>(in);
    uchar4* out4 = reinterpret_cast<uchar4*>(out);

    for (std::size_t i = tid; i < n4; i += stride) {
        const uchar4 v = __ldg(in4 + i);
        uchar4 w;
        w.x = HistFast::dec_atbash(v.x);
        w.y = HistFast::dec_atbash(v.y);
        w.z = HistFast::dec_atbash(v.z);
        w.w = HistFast::dec_atbash(v.w);
        out4[i] = w;
    }
    for (std::size_t t = n4 * 4u + tid; t < count; t += stride) {
        out[t] = HistFast::dec_atbash(__ldg(in + t));
    }
}

__global__ void atbash_scalar_kernel(const std::uint8_t* __restrict__ in, std::uint8_t* out,
                                     std::size_t count) {
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t t = tid; t < count; t += stride) {
        out[t] = HistFast::dec_atbash(__ldg(in + t));
    }
}

Status AtbashKernel::launch_device_async(const std::uint8_t* device_in, std::uint8_t* device_out,
                                         std::size_t count) {
    if (count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("AtbashKernel::launch_device_async null device pointer");
    }

    const bool aligned = (reinterpret_cast<std::uintptr_t>(device_in) % alignof(uchar4)) == 0u &&
                         (reinterpret_cast<std::uintptr_t>(device_out) % alignof(uchar4)) == 0u;
    const std::size_t work = aligned ? (count + 3u) / 4u : count;
    const int blocks =
        static_cast<int>((work + static_cast<std::size_t>(HistFast::threads) - 1u) /
                         static_cast<std::size_t>(HistFast::threads));

    if (aligned) {
        atbash_uchar4_kernel<<<blocks, HistFast::threads>>>(device_in, device_out, count);
    } else {
        atbash_scalar_kernel<<<blocks, HistFast::threads>>>(device_in, device_out, count);
    }
    return CudaError::to_status(cudaGetLastError(), "AtbashKernel::launch_device_async");
}

Status AtbashKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                   std::size_t count) {
    Status launched = launch_device_async(device_in, device_out, count);
    if (!launched.ok()) {
        return launched;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AtbashKernel::launch_device sync");
}

Status AtbashKernel::apply_host(std::span<const std::uint8_t> host_in,
                                std::span<std::uint8_t> host_out) {
    if (host_in.size() != host_out.size()) {
        return Status::error("AtbashKernel::apply_host size mismatch");
    }

    if (host_in.data() == host_out.data()) {
        StatusOr<DeviceBuffer<std::uint8_t>> device =
            DeviceBuffer<std::uint8_t>::from_host(host_in);
        if (!device.ok()) {
            return device.status();
        }
        Status launched =
            launch_device(device.value().data(), device.value().data(), host_in.size());
        if (!launched.ok()) {
            return launched;
        }
        return device.value().copy_to_host(host_out);
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

    Status launched =
        launch_device(device_in.value().data(), device_out.value().data(), host_in.size());
    if (!launched.ok()) {
        return launched;
    }

    return device_out.value().copy_to_host(host_out);
}
