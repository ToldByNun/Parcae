#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "vigenere_batch_kernel.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

[[nodiscard]] __device__ bool bitmask_should_skip(const std::uint32_t* words, std::size_t index) {
    return (words[index >> 5] & (1u << (index & 31u))) != 0u;
}

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

__global__ void vigenere_batch_shared_kernel(const std::uint8_t* in, const std::uint8_t* key_bytes,
                                             const std::uint32_t* key_begin,
                                             const std::uint32_t* key_len,
                                             const std::uint8_t* directions, std::uint8_t* out,
                                             std::size_t candidate_count, std::size_t token_count,
                                             const std::uint32_t* interrupt_data,
                                             std::uint32_t skip_count, std::uint8_t use_bitmask) {
    const std::size_t flat =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = candidate_count * token_count;
    if (flat >= total) {
        return;
    }

    const std::size_t candidate = flat / token_count;
    const std::size_t token = flat % token_count;

    bool skip = false;
    std::uint32_t key_cursor = 0;
    if (use_bitmask != 0u) {
        skip = bitmask_should_skip(interrupt_data, token);
        key_cursor = bitmask_consumed_before(interrupt_data, token);
    } else {
        const std::uint32_t index = static_cast<std::uint32_t>(token);
        skip = sorted_should_skip(interrupt_data, skip_count, index);
        key_cursor = index - sorted_skips_before(interrupt_data, skip_count, index);
    }

    if (skip) {
        out[flat] = in[token];
        return;
    }

    const std::uint32_t begin = key_begin[candidate];
    const std::uint32_t len = key_len[candidate];
    const std::uint8_t key_symbol = key_bytes[begin + (key_cursor % len)];
    if (directions[candidate] != 0u) {
        out[flat] = Z29Device::add(in[token], key_symbol);
    } else {
        out[flat] = Z29Device::sub(in[token], key_symbol);
    }
}

[[nodiscard]] Status validate_batch(std::size_t candidate_count, std::size_t token_count,
                                    std::span<const std::uint8_t> key_bytes,
                                    std::span<const std::uint32_t> key_begin,
                                    std::span<const std::uint32_t> key_len,
                                    std::span<const std::uint8_t> directions,
                                    const InterruptDeviceView& interrupts) {
    if (candidate_count == 0) {
        return Status::error("VigenereBatchKernel: C must be >= 1");
    }
    if (candidate_count > CandidateBatchBuffers::kMaxC) {
        return Status::error("VigenereBatchKernel: C exceeds kMaxC");
    }
    if (token_count > CandidateBatchBuffers::kMaxT) {
        return Status::error("VigenereBatchKernel: T exceeds kMaxT");
    }
    if (key_begin.size() != candidate_count || key_len.size() != candidate_count) {
        return Status::error("VigenereBatchKernel: key_begin/key_len size must equal C");
    }
    if (directions.size() != candidate_count) {
        return Status::error("VigenereBatchKernel: directions size must equal C");
    }
    if (interrupts.consumable_length() != token_count) {
        return Status::error("VigenereBatchKernel: interrupt view length mismatch");
    }
    for (std::size_t c = 0; c < candidate_count; ++c) {
        if (key_len[c] == 0) {
            return Status::error("VigenereBatchKernel: key must be non-empty");
        }
        if (static_cast<std::size_t>(key_begin[c]) + static_cast<std::size_t>(key_len[c]) >
            key_bytes.size()) {
            return Status::error("VigenereBatchKernel: key slice out of arena");
        }
        if (directions[c] > 1u) {
            return Status::error("VigenereBatchKernel: direction must be 0 or 1");
        }
    }
    return Status::success();
}

} // namespace

Status VigenereBatchKernel::launch_device(
    const std::uint8_t* device_in, const std::uint8_t* device_key_bytes,
    const std::uint32_t* device_key_begin, const std::uint32_t* device_key_len,
    const std::uint8_t* device_directions, std::uint8_t* device_out, std::size_t candidate_count,
    std::size_t token_count, const InterruptDeviceView& interrupts,
    const std::uint32_t* device_bitmask_or_skips) {
    if (candidate_count == 0) {
        return Status::error("VigenereBatchKernel::launch_device C must be >= 1");
    }
    if (token_count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_key_bytes == nullptr || device_key_begin == nullptr ||
        device_key_len == nullptr || device_directions == nullptr || device_out == nullptr) {
        return Status::error("VigenereBatchKernel::launch_device null device pointer");
    }
    if (interrupts.consumable_length() != token_count) {
        return Status::error("VigenereBatchKernel: interrupt view length mismatch");
    }

    const bool use_bitmask_encoding =
        interrupts.encoding() == InterruptDeviceView::Encoding::Bitmask;
    if (use_bitmask_encoding) {
        if (device_bitmask_or_skips == nullptr) {
            return Status::error("VigenereBatchKernel: bitmask encoding requires device words");
        }
    } else if (!interrupts.sorted_skips().empty() && device_bitmask_or_skips == nullptr) {
        return Status::error("VigenereBatchKernel: sorted skips require device buffer");
    }

    const std::uint8_t use_bitmask =
        use_bitmask_encoding ? static_cast<std::uint8_t>(1) : static_cast<std::uint8_t>(0);
    const std::uint32_t skip_count =
        use_bitmask_encoding ? 0u : static_cast<std::uint32_t>(interrupts.sorted_skips().size());

    const std::size_t total = candidate_count * token_count;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    vigenere_batch_shared_kernel<<<blocks, kThreadsPerBlock>>>(
        device_in, device_key_bytes, device_key_begin, device_key_len, device_directions,
        device_out, candidate_count, token_count, device_bitmask_or_skips, skip_count, use_bitmask);

    Status launch = CudaError::to_status(cudaGetLastError(), "VigenereBatchKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "VigenereBatchKernel::launch_device sync");
}

Status VigenereBatchKernel::apply_host(std::span<const std::uint8_t> shared_in,
                                       std::span<const std::uint8_t> key_bytes,
                                       std::span<const std::uint32_t> key_begin,
                                       std::span<const std::uint32_t> key_len,
                                       std::span<const std::uint8_t> directions,
                                       std::span<std::uint8_t> out,
                                       const InterruptDeviceView& interrupts) {
    const std::size_t candidate_count = key_begin.size();
    const std::size_t token_count = shared_in.size();

    Status valid = validate_batch(candidate_count, token_count, key_bytes, key_begin, key_len,
                                  directions, interrupts);
    if (!valid.ok()) {
        return valid;
    }
    if (out.size() != candidate_count * token_count) {
        return Status::error("VigenereBatchKernel::apply_host out size must be C * T");
    }
    if (token_count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(shared_in);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        DeviceBuffer<std::uint8_t>::from_host(key_bytes);
    if (!device_keys.ok()) {
        return device_keys.status();
    }
    StatusOr<DeviceBuffer<std::uint32_t>> device_begin =
        DeviceBuffer<std::uint32_t>::from_host(key_begin);
    if (!device_begin.ok()) {
        return device_begin.status();
    }
    StatusOr<DeviceBuffer<std::uint32_t>> device_len =
        DeviceBuffer<std::uint32_t>::from_host(key_len);
    if (!device_len.ok()) {
        return device_len.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_directions =
        DeviceBuffer<std::uint8_t>::from_host(directions);
    if (!device_directions.ok()) {
        return device_directions.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_out =
        DeviceBuffer<std::uint8_t>::allocate(out.size());
    if (!device_out.ok()) {
        return device_out.status();
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

    Status launched = launch_device(device_in.value().data(), device_keys.value().data(),
                                    device_begin.value().data(), device_len.value().data(),
                                    device_directions.value().data(), device_out.value().data(),
                                    candidate_count, token_count, interrupts, interrupt_ptr);
    if (!launched.ok()) {
        return launched;
    }
    return device_out.value().copy_to_host(out);
}

Status VigenereBatchKernel::apply_host(CandidateBatchBuffers& buffers,
                                       const InterruptDeviceView& interrupts) {
    if (buffers.token_layout() != CandidateBatchBuffers::TokenLayout::Shared) {
        return Status::error("VigenereBatchKernel requires Shared token layout");
    }
    if (buffers.family() != CudaFamilyId::VigenereKey) {
        return Status::error("VigenereBatchKernel requires VigenereKey family buffers");
    }
    return apply_host(buffers.token_index29(), buffers.key_bytes(), buffers.key_begin(),
                      buffers.key_len(), buffers.directions(), buffers.out_index29(), interrupts);
}
