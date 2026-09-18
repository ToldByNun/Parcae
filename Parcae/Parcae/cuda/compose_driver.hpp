#ifndef COMPOSE_DRIVER_HPP
#define COMPOSE_DRIVER_HPP

#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "params_json.hpp"

#include "parcae/core/status.hpp"

#include <cstdint>
#include <span>

/// Host-orchestrated CUDA compose: stage kernel launches + device ping-pong buffers.
/// Matches `ComposeTransform` order / direction rules (incl. Koan-1 Caesar encrypt stage).
/// Nested compose is not supported (`ComposeParamsHost` already rejects nesting).
class ComposeDriver {
public:
    [[nodiscard]] static Status apply_host(
        std::span<const std::uint8_t> host_in,
        std::span<std::uint8_t> host_out,
        const ComposeParamsHost& recipe,
        const InterruptDeviceView& interrupts,
        CudaDir outer_direction);

private:
    ComposeDriver() = delete;
};

#endif // COMPOSE_DRIVER_HPP
