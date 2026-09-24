#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "vigenere_key_kernel.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

[[nodiscard]] __device__ bool bitmask_should_skip(const std::uint32_t* words, std::size_t index) {
    return (words[index >> 5] & (1u << (index & 31u))) != 0u;
}

/// Non-skip count in `[0, index)` for bitmask encoding.
[[nodiscard]] __device__ std::uint32_t bitmask_consumed_before(const std::uint32_t* words,
                                                               std::size_t index) {
    std::uint32_t consumed = 0;
    const std::size_t full_words = index >> 5;
    for (std::size_t w = 0; w < full_words; ++w) {
        consumed += 32u - static_cast<std::uint32_t>(__popc(words[w]));
    }
    const std::uint32_t rem = static_cast<std::uint32_t>(index & 31u);
    if (rem != 0u) {
        const std::uint32_t mask = (1u << rem) - 1u;
        consumed += rem - static_cast<std::uint32_t>(__popc(words[full_words] & mask));
    }
    return consumed;
}

[[nodiscard]] __device__ std::uint32_t
sorted_skips_before(const std::uint32_t* skips, std::uint32_t skip_count, std::uint32_t index) {
    std::uint32_t lo = 0;
    std::uint32_t hi = skip_count;
    while (lo < hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2u;
        if (skips[mid] < index) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

[[nodiscard]] __device__ bool sorted_should_skip(const std::uint32_t* skips,
                                                 std::uint32_t skip_count, std::uint32_t index) {
    std::uint32_t lo = 0;
    std::uint32_t hi = skip_count;
    while (lo < hi) {
        const std::uint32_t mid = lo + (hi - lo) / 2u;
        if (skips[mid] < index) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < skip_count && skips[lo] == index;
}

__global__ void vigenere_key_kernel(const std::uint8_t* in, std::uint8_t* out, std::size_t count,
                                    const std::uint8_t* key, std::uint32_t key_len,
                                    const std::uint32_t* interrupt_data, std::uint32_t skip_count,
                                    std::uint8_t use_bitmask, std::uint8_t encrypt) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }

    bool skip = false;
    std::uint32_t key_cursor = 0;
    if (use_bitmask != 0u) {
        skip = bitmask_should_skip(interrupt_data, i);
        key_cursor = bitmask_consumed_before(interrupt_data, i);
    } else {
        const std::uint32_t index = static_cast<std::uint32_t>(i);
        skip = sorted_should_skip(interrupt_data, skip_count, index);
        key_cursor = index - sorted_skips_before(interrupt_data, skip_count, index);
    }

    if (skip) {
        out[i] = in[i];
        return;
    }

    const std::uint8_t key_symbol = key[key_cursor % key_len];
    if (encrypt != 0u) {
        out[i] = Z29Device::add(in[i], key_symbol);
    } else {
        out[i] = Z29Device::sub(in[i], key_symbol);
    }
}

} // namespace

Status VigenereKeyKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t count, const std::uint8_t* device_key,
                                        std::uint32_t key_len,
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
        return Status::error("VigenereKeyKernel::launch_device null device pointer");
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

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());
    const std::uint8_t encrypt =
        direction == CudaDir::Encrypt ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);

    const int blocks = static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    vigenere_key_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, count, device_key,
                                                      key_len, device_bitmask_or_skips, skip_count,
                                                      use_bitmask, encrypt);

    Status launch = CudaError::to_status(cudaGetLastError(), "VigenereKeyKernel::launch_device");
    if (!launch.ok()) {
        return launch;
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

    // Upload interrupt encoding once; keep buffer alive across launch + D2H.
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
