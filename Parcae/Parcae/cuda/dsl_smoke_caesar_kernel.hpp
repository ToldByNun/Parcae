#ifndef DSL_SMOKE_CAESAR_KERNEL_HPP
#define DSL_SMOKE_CAESAR_KERNEL_HPP

#include "parcae/core/status.hpp"

#include "params.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

/// Golden twin for `DslEmitCuda` smoke (`theory_id=dsl_smoke_caesar`).
/// Shape matches emitted Kernel façades (Z29Device in the .cu). Not a catalog id.
class DslSmokeCaesarKernel {
public:
    static constexpr std::string_view theory_id = "dsl_smoke_caesar";
    static constexpr std::string_view interrupt_mode = "elementwise_default";

    [[nodiscard]] static Status launch_device(const std::uint8_t* device_in,
                                              std::uint8_t* device_out, std::size_t count,
                                              std::uint8_t shift, CudaDir direction);

    [[nodiscard]] static Status apply_host(std::span<const std::uint8_t> host_in,
                                           std::span<std::uint8_t> host_out, std::uint8_t shift,
                                           CudaDir direction);

private:
    DslSmokeCaesarKernel() = delete;
};

#endif // DSL_SMOKE_CAESAR_KERNEL_HPP
