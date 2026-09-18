#include "affine_batch_kernel.hpp"

#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "z29_device.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void affine_batch_shared_kernel(
    const std::uint8_t* in,
    const std::uint8_t* affine_a,
    const std::uint8_t* affine_b,
    const std::uint8_t* directions,
    std::uint8_t* out,
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
    const std::uint8_t a = affine_a[candidate];
    const std::uint8_t b = affine_b[candidate];
    if (directions[candidate] != 0u) {
        out[flat] = Z29Device::add(Z29Device::mul(a, x), b);
    } else {
        const std::uint8_t inv_a = Z29Device::inv(a);
        out[flat] = Z29Device::mul(inv_a, Z29Device::sub(x, b));
    }
}

[[nodiscard]] Status validate_batch(
    std::size_t candidate_count,
    std::size_t token_count,
    std::span<const std::uint8_t> affine_a,
    std::span<const std::uint8_t> affine_b,
    std::span<const std::uint8_t> directions) {
    if (candidate_count == 0) {
        return Status::error("AffineBatchKernel: C must be >= 1");
    }
    if (candidate_count > CandidateBatchBuffers::kMaxC) {
        return Status::error("AffineBatchKernel: C exceeds kMaxC");
    }
    if (token_count > CandidateBatchBuffers::kMaxT) {
        return Status::error("AffineBatchKernel: T exceeds kMaxT");
    }
    if (affine_a.size() != candidate_count || affine_b.size() != candidate_count) {
        return Status::error("AffineBatchKernel: a/b size must equal C");
    }
    if (directions.size() != candidate_count) {
        return Status::error("AffineBatchKernel: directions size must equal C");
    }
    for (std::size_t c = 0; c < candidate_count; ++c) {
        if (affine_a[c] < 1u || affine_a[c] > 28u) {
            return Status::error("AffineBatchKernel: a must be in 1..28");
        }
        if (affine_b[c] > 28u) {
            return Status::error("AffineBatchKernel: b must be in 0..28");
        }
        if (directions[c] > 1u) {
            return Status::error("AffineBatchKernel: direction must be 0 or 1");
        }
    }
    return Status::success();
}

}  // namespace

Status AffineBatchKernel::launch_device(
    const std::uint8_t* device_in,
    const std::uint8_t* device_a,
    const std::uint8_t* device_b,
    const std::uint8_t* device_directions,
    std::uint8_t* device_out,
    std::size_t candidate_count,
    std::size_t token_count) {
    if (candidate_count == 0) {
        return Status::error("AffineBatchKernel::launch_device C must be >= 1");
    }
    if (token_count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_a == nullptr || device_b == nullptr ||
        device_directions == nullptr || device_out == nullptr) {
        return Status::error("AffineBatchKernel::launch_device null device pointer");
    }

    const std::size_t total = candidate_count * token_count;
    const int blocks = static_cast<int>(
        (total + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
        static_cast<std::size_t>(kThreadsPerBlock));
    affine_batch_shared_kernel<<<blocks, kThreadsPerBlock>>>(
        device_in,
        device_a,
        device_b,
        device_directions,
        device_out,
        candidate_count,
        token_count);

    Status launch = CudaError::to_status(cudaGetLastError(), "AffineBatchKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AffineBatchKernel::launch_device sync");
}

Status AffineBatchKernel::apply_host(
    std::span<const std::uint8_t> shared_in,
    std::span<const std::uint8_t> affine_a,
    std::span<const std::uint8_t> affine_b,
    std::span<const std::uint8_t> directions,
    std::span<std::uint8_t> out) {
    const std::size_t candidate_count = affine_a.size();
    const std::size_t token_count = shared_in.size();

    Status valid = validate_batch(candidate_count, token_count, affine_a, affine_b, directions);
    if (!valid.ok()) {
        return valid;
    }
    if (out.size() != candidate_count * token_count) {
        return Status::error("AffineBatchKernel::apply_host out size must be C * T");
    }
    if (token_count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(shared_in);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_a = DeviceBuffer<std::uint8_t>::from_host(affine_a);
    if (!device_a.ok()) {
        return device_a.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_b = DeviceBuffer<std::uint8_t>::from_host(affine_b);
    if (!device_b.ok()) {
        return device_b.status();
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

    Status launched = launch_device(
        device_in.value().data(),
        device_a.value().data(),
        device_b.value().data(),
        device_directions.value().data(),
        device_out.value().data(),
        candidate_count,
        token_count);
    if (!launched.ok()) {
        return launched;
    }
    return device_out.value().copy_to_host(out);
}

Status AffineBatchKernel::apply_host(CandidateBatchBuffers& buffers) {
    if (buffers.token_layout() != CandidateBatchBuffers::TokenLayout::Shared) {
        return Status::error("AffineBatchKernel requires Shared token layout");
    }
    if (buffers.family() != CudaFamilyId::Affine) {
        return Status::error("AffineBatchKernel requires Affine family buffers");
    }
    return apply_host(
        buffers.token_index29(),
        buffers.affine_a(),
        buffers.affine_b(),
        buffers.directions(),
        buffers.out_index29());
}
