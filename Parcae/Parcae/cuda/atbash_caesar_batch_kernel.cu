#include "atbash_caesar_batch_kernel.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::uint8_t kModulus = 29;

__global__ void atbash_caesar_batch_shared_kernel(const std::uint8_t* in,
                                                  const std::uint8_t* shifts,
                                                  const std::uint8_t* directions, std::uint8_t* out,
                                                  std::size_t candidate_count,
                                                  std::size_t token_count) {
    const std::size_t flat =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = candidate_count * token_count;
    if (flat >= total) {
        return;
    }

    const std::size_t candidate = flat / token_count;
    const std::size_t token = flat % token_count;
    const std::uint8_t x = in[token];
    const std::uint8_t shift = shifts[candidate];
    if (directions[candidate] != 0u) {
        // Outer encrypt: −shift then Atbash.
        const std::uint8_t y = static_cast<std::uint8_t>((x + kModulus - shift) % kModulus);
        out[flat] = static_cast<std::uint8_t>(28u - y);
    } else {
        // Outer decrypt: Atbash then +shift (stage caesar direction=encrypt).
        out[flat] = static_cast<std::uint8_t>((28u - x + shift) % kModulus);
    }
}

[[nodiscard]] Status validate_batch(std::size_t candidate_count, std::size_t token_count,
                                    std::span<const std::uint8_t> shifts,
                                    std::span<const std::uint8_t> directions) {
    if (candidate_count == 0) {
        return Status::error("AtbashCaesarBatchKernel: C must be >= 1");
    }
    if (candidate_count > CandidateBatchBuffers::kMaxC) {
        return Status::error("AtbashCaesarBatchKernel: C exceeds kMaxC");
    }
    if (token_count > CandidateBatchBuffers::kMaxT) {
        return Status::error("AtbashCaesarBatchKernel: T exceeds kMaxT");
    }
    if (shifts.size() != candidate_count) {
        return Status::error("AtbashCaesarBatchKernel: shifts size must equal C");
    }
    if (directions.size() != candidate_count) {
        return Status::error("AtbashCaesarBatchKernel: directions size must equal C");
    }
    for (std::size_t c = 0; c < candidate_count; ++c) {
        if (shifts[c] >= kModulus) {
            return Status::error("AtbashCaesarBatchKernel: shift must be in 0..28");
        }
        if (directions[c] > 1u) {
            return Status::error("AtbashCaesarBatchKernel: direction must be 0 or 1");
        }
    }
    return Status::success();
}

} // namespace

Status AtbashCaesarBatchKernel::launch_device(const std::uint8_t* device_in,
                                              const std::uint8_t* device_shifts,
                                              const std::uint8_t* device_directions,
                                              std::uint8_t* device_out, std::size_t candidate_count,
                                              std::size_t token_count) {
    if (candidate_count == 0) {
        return Status::error("AtbashCaesarBatchKernel::launch_device C must be >= 1");
    }
    if (token_count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_shifts == nullptr || device_directions == nullptr ||
        device_out == nullptr) {
        return Status::error("AtbashCaesarBatchKernel::launch_device null device pointer");
    }

    const std::size_t total = candidate_count * token_count;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    atbash_caesar_batch_shared_kernel<<<blocks, kThreadsPerBlock>>>(
        device_in, device_shifts, device_directions, device_out, candidate_count, token_count);

    Status launch =
        CudaError::to_status(cudaGetLastError(), "AtbashCaesarBatchKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(),
                                "AtbashCaesarBatchKernel::launch_device sync");
}

Status AtbashCaesarBatchKernel::apply_host(std::span<const std::uint8_t> shared_in,
                                           std::span<const std::uint8_t> shifts,
                                           std::span<const std::uint8_t> directions,
                                           std::span<std::uint8_t> out) {
    const std::size_t candidate_count = shifts.size();
    const std::size_t token_count = shared_in.size();

    Status valid = validate_batch(candidate_count, token_count, shifts, directions);
    if (!valid.ok()) {
        return valid;
    }
    if (out.size() != candidate_count * token_count) {
        return Status::error("AtbashCaesarBatchKernel::apply_host out size must be C * T");
    }
    if (token_count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(shared_in);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
        DeviceBuffer<std::uint8_t>::from_host(shifts);
    if (!device_shifts.ok()) {
        return device_shifts.status();
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

    Status launched = launch_device(device_in.value().data(), device_shifts.value().data(),
                                    device_directions.value().data(), device_out.value().data(),
                                    candidate_count, token_count);
    if (!launched.ok()) {
        return launched;
    }
    return device_out.value().copy_to_host(out);
}

Status AtbashCaesarBatchKernel::apply_host(CandidateBatchBuffers& buffers) {
    if (buffers.token_layout() != CandidateBatchBuffers::TokenLayout::Shared) {
        return Status::error("AtbashCaesarBatchKernel requires Shared token layout");
    }
    if (buffers.family() != CudaFamilyId::Compose) {
        return Status::error("AtbashCaesarBatchKernel requires Compose family buffers");
    }
    return apply_host(buffers.token_index29(), buffers.caesar_shifts(), buffers.directions(),
                      buffers.out_index29());
}
