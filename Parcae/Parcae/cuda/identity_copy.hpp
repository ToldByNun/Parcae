#ifndef IDENTITY_COPY_HPP
#define IDENTITY_COPY_HPP

#include "parcae/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

/// Smoke / identity twin helper: device `uint8_t` stream copy (`out[i] = in[i]`).
///
/// Host API only; kernel lives in `identity_copy.cu`. Wired into `CudaBackend`
/// for `TransformId::identity()` in a later commit.
class IdentityCopy {
public:
    /// Launch copy on device pointers (`count` elements). No-op success if `count == 0`.
    [[nodiscard]] static Status launch_device(
        const std::uint8_t* device_in,
        std::uint8_t* device_out,
        std::size_t count);

    /// Allocate device buffers, H2D → kernel → D2H. Requires equal span sizes.
    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out);

private:
    IdentityCopy() = delete;
};

#endif // IDENTITY_COPY_HPP
