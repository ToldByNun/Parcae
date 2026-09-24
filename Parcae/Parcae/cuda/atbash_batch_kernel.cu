#include "atbash_batch_kernel.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"

#include <cuda_runtime_api.h>

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void atbash_batch_shared_kernel(const std::uint8_t* in, std::uint8_t* out,
                                           std::size_t candidate_count, std::size_t token_count) {
    const std::size_t flat =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t total = candidate_count * token_count;
    if (flat >= total) {
        return;
    }
    const std::size_t token = flat % token_count;
    out[flat] = static_cast<std::uint8_t>(28u - in[token]);
}

} // namespace

Status AtbashBatchKernel::launch_device(const std::uint8_t* device_in, std::uint8_t* device_out,
                                        std::size_t candidate_count, std::size_t token_count) {
    if (candidate_count == 0) {
        return Status::error("AtbashBatchKernel::launch_device C must be >= 1");
    }
    if (token_count == 0) {
        return Status::success();
    }
    if (device_in == nullptr || device_out == nullptr) {
        return Status::error("AtbashBatchKernel::launch_device null device pointer");
    }

    const std::size_t total = candidate_count * token_count;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(kThreadsPerBlock) - 1u) /
                                        static_cast<std::size_t>(kThreadsPerBlock));
    atbash_batch_shared_kernel<<<blocks, kThreadsPerBlock>>>(device_in, device_out, candidate_count,
                                                             token_count);

    Status launch = CudaError::to_status(cudaGetLastError(), "AtbashBatchKernel::launch_device");
    if (!launch.ok()) {
        return launch;
    }
    return CudaError::to_status(cudaDeviceSynchronize(), "AtbashBatchKernel::launch_device sync");
}

Status AtbashBatchKernel::apply_host(std::span<const std::uint8_t> shared_in,
                                     std::span<std::uint8_t> out, std::size_t candidate_count) {
    if (candidate_count == 0) {
        return Status::error("AtbashBatchKernel::apply_host C must be >= 1");
    }
    if (candidate_count > CandidateBatchBuffers::kMaxC) {
        return Status::error("AtbashBatchKernel: C exceeds kMaxC");
    }
    const std::size_t token_count = shared_in.size();
    if (token_count > CandidateBatchBuffers::kMaxT) {
        return Status::error("AtbashBatchKernel: T exceeds kMaxT");
    }
    if (out.size() != candidate_count * token_count) {
        return Status::error("AtbashBatchKernel::apply_host out size must be C * T");
    }
    if (token_count == 0) {
        return Status::success();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_in =
        DeviceBuffer<std::uint8_t>::from_host(shared_in);
    if (!device_in.ok()) {
        return device_in.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> device_out =
        DeviceBuffer<std::uint8_t>::allocate(out.size());
    if (!device_out.ok()) {
        return device_out.status();
    }

    Status launched = launch_device(device_in.value().data(), device_out.value().data(),
                                    candidate_count, token_count);
    if (!launched.ok()) {
        return launched;
    }
    return device_out.value().copy_to_host(out);
}

Status AtbashBatchKernel::apply_host(CandidateBatchBuffers& buffers) {
    if (buffers.token_layout() != CandidateBatchBuffers::TokenLayout::Shared) {
        return Status::error("AtbashBatchKernel requires Shared token layout");
    }
    if (buffers.family() != CudaFamilyId::Atbash) {
        return Status::error("AtbashBatchKernel requires Atbash family buffers");
    }
    return apply_host(buffers.token_index29(), buffers.out_index29(), buffers.candidate_count());
}
