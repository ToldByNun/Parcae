#ifndef IDENTITY_COPY_HPP
#define IDENTITY_COPY_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// Smoke / identity twin: device `uint8_t` stream copy (`out[i] = in[i]`).
///
/// Host API; kernel in `identity_copy.cu`. Used by `CudaBackend` for
/// `TransformId::identity()`. In-place OK (`host_in.data() == host_out.data()`).
///
/// Device path: uchar4 packs when pointers are aligned; scalar `__ldg`
/// otherwise. `launch_device_async` does not synchronize; `launch_device` /
/// `apply_host` sync (host / compose API, not SLO).
class IdentityCopy {
public:
    /// Launch copy on device pointers (`count` elements). No-op success if `count == 0`.
    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count);

    /// Same as `launch_device` but does not `cudaDeviceSynchronize`.
    [[nodiscard]] static Status launch_device_async(const std::uint8_t* device_in,
                                                    std::uint8_t* device_out, std::size_t count);

    /// Allocate device buffers, H2D → kernel → D2H. Requires equal span sizes.
    /// Host convenience only — not a Kernel SLO path.
    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out);

private:
    IdentityCopy() = delete;
};

#endif // IDENTITY_COPY_HPP
