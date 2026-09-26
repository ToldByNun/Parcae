#include "autokey_ctak_device.hpp"
#include "ciphertext_autokey_kernel.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

[[nodiscard]] __device__ bool bitmask_should_skip(const std::uint32_t* words, std::size_t index) {
    return (words[index >> 5] & (1u << (index & 31u))) != 0u;
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

/// Dense CTAK decrypt — parallel; matches DeepScoreBatch / AutokeyCtakDevice.
__global__ void ctak_dense_decrypt_kernel(const std::uint8_t* in, std::uint8_t* out,
                                          std::size_t count, const std::uint8_t* key,
                                          std::uint32_t key_len) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    if (i >= count) {
        return;
    }
    const std::uint8_t key_symbol = AutokeyCtakDevice::decrypt_key(in, key, key_len, i);
    out[i] = AutokeyCtakDevice::decrypt_symbol(in[i], key_symbol);
}

/// Serial CTAK for encrypt and/or interrupt skips (feedback over consumed runes).
__global__ void ctak_serial_kernel(const std::uint8_t* in, std::uint8_t* out, std::size_t count,
                                   const std::uint8_t* key, std::uint32_t key_len,
                                   const std::uint32_t* interrupt_data, std::uint32_t skip_count,
                                   std::uint8_t use_bitmask, std::uint8_t encrypt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Feedback of consumed ciphertext (encrypt: written outs; decrypt: inputs).
    // Bounded stack buffer for typical parity / twin sizes; fall back to in-place scan
    // via storing feedback in `out` temporarily is unsafe for decrypt — use dynamic
    // device heap only when needed. Cap matches DeepScoreBatch::kMaxTokens use cases
    // in tests; for large streams allocate via a scratch pointer passed from host.
    // Here we stream feedback through a rolling ring of key_len symbols.
    //
    // Ring of last `key_len` ciphertext feedback symbols among consumed positions.
    // Primer phase: consumed < key_len → use key[consumed].
    extern __shared__ std::uint8_t ring[];
    std::uint32_t consumed = 0;

    for (std::size_t i = 0; i < count; ++i) {
        bool skip = false;
        if (use_bitmask != 0u) {
            skip = bitmask_should_skip(interrupt_data, i);
        } else if (skip_count != 0u) {
            skip = sorted_should_skip(interrupt_data, skip_count, static_cast<std::uint32_t>(i));
        }

        if (skip) {
            out[i] = in[i];
            continue;
        }

        std::uint8_t key_symbol;
        if (consumed < key_len) {
            key_symbol = key[consumed];
        } else {
            key_symbol = ring[(consumed - key_len) % key_len];
        }

        if (encrypt != 0u) {
            out[i] = AutokeyCtakDevice::encrypt_symbol(in[i], key_symbol);
            ring[consumed % key_len] = out[i];
        } else {
            out[i] = AutokeyCtakDevice::decrypt_symbol(in[i], key_symbol);
            ring[consumed % key_len] = in[i];
        }
        ++consumed;
    }
}

} // namespace

Status CiphertextAutokeyKernel::launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              const std::uint8_t* device_key, std::uint32_t key_len,
                                              const InterruptDeviceView& interrupts,
                                              const std::uint32_t* device_bitmask_or_skips,
                                              CudaDir direction) {
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
        return Status::error("CiphertextAutokeyKernel::launch_device null device pointer");
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

    // `sorted_skips()` is always filled from the policy (even under bitmask encoding).
    const bool dense_no_skips = interrupts.sorted_skips().empty();

    // Dense decrypt: parallel CTAK matching DeepScoreBatch.
    if (dense_no_skips && direction == CudaDir::Decrypt) {
        const int blocks =
            static_cast<int>((count + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                             static_cast<std::size_t>(kThreadsPerBlock));
        ctak_dense_decrypt_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, count,
                                                                device_key, key_len);
        Status launch =
            CudaError::to_status(cudaGetLastError(), "CiphertextAutokeyKernel::dense decrypt");
        if (!launch.ok()) {
            return launch;
        }
        return CudaError::to_status(cudaDeviceSynchronize(),
                                    "CiphertextAutokeyKernel::dense decrypt sync");
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

    Status launch = CudaError::to_status(cudaGetLastError(), "CiphertextAutokeyKernel::serial");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "CiphertextAutokeyKernel::serial sync");
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
