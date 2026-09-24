#include "parcae/core/index29.hpp"
#include "parcae/math/totient_keystream.hpp"

#include "affine_kernel.hpp"
#include "atbash_kernel.hpp"
#include "beaufort_key_kernel.hpp"
#include "caesar_kernel.hpp"
#include "compose_driver.hpp"
#include "cuda_error.hpp"
#include "device_buffer.hpp"
#include "identity_copy.hpp"
#include "totient_prime_stream_kernel.hpp"
#include "vigenere_key_kernel.hpp"

#include <cstddef>
#include <vector>

namespace {

[[nodiscard]] CudaDir invert_dir(CudaDir direction) noexcept {
    return direction == CudaDir::Decrypt ? CudaDir::Encrypt : CudaDir::Decrypt;
}

[[nodiscard]] std::size_t consumable_count(const InterruptDeviceView& interrupts) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < interrupts.consumable_length(); ++i) {
        if (!interrupts.should_skip(i)) {
            ++n;
        }
    }
    return n;
}

[[nodiscard]] Status run_stage(const std::uint8_t* device_in, std::uint8_t* device_out,
                               std::size_t count, const ComposeStageParams& stage,
                               CudaDir effective_dir, const ComposeParamsHost& recipe,
                               const InterruptDeviceView& interrupts,
                               const std::uint32_t* device_interrupt,
                               const std::uint8_t* device_key_arena) {
    switch (stage.family) {
    case CudaFamilyId::Identity:
        return IdentityCopy::launch_device(device_in, device_out, count);
    case CudaFamilyId::Atbash:
        return AtbashKernel::launch_device(device_in, device_out, count);
    case CudaFamilyId::Caesar:
        return CaesarKernel::launch_device(device_in, device_out, count, stage.caesar.shift,
                                           effective_dir);
    case CudaFamilyId::Affine:
        return AffineKernel::launch_device(device_in, device_out, count, stage.affine.a,
                                           stage.affine.b, effective_dir);
    case CudaFamilyId::VigenereKey: {
        if (stage.key_len == 0) {
            return Status::error("ComposeDriver: vigenere stage key must be non-empty");
        }
        if (static_cast<std::size_t>(stage.key_begin) + stage.key_len > recipe.key_arena.size()) {
            return Status::error("ComposeDriver: vigenere key slice out of arena");
        }
        if (device_key_arena == nullptr) {
            return Status::error("ComposeDriver: missing device key arena");
        }
        return VigenereKeyKernel::launch_device(device_in, device_out, count,
                                                device_key_arena + stage.key_begin, stage.key_len,
                                                interrupts, device_interrupt, effective_dir);
    }
    case CudaFamilyId::BeaufortKey: {
        if (stage.key_len == 0) {
            return Status::error("ComposeDriver: beaufort stage key must be non-empty");
        }
        if (static_cast<std::size_t>(stage.key_begin) + stage.key_len > recipe.key_arena.size()) {
            return Status::error("ComposeDriver: beaufort key slice out of arena");
        }
        if (device_key_arena == nullptr) {
            return Status::error("ComposeDriver: missing device key arena");
        }
        return BeaufortKeyKernel::launch_device(device_in, device_out, count,
                                                device_key_arena + stage.key_begin, stage.key_len,
                                                interrupts, device_interrupt);
    }
    case CudaFamilyId::TotientPrimeStream: {
        const std::size_t shifts_needed = consumable_count(interrupts);
        std::vector<Index29> shifts(shifts_needed);
        Status filled = TotientKeystream::shifts_into(
            shifts, static_cast<std::size_t>(stage.totient.prime_start_index));
        if (!filled.ok()) {
            return filled;
        }
        std::vector<std::uint8_t> shift_bytes(shifts_needed);
        for (std::size_t i = 0; i < shifts_needed; ++i) {
            shift_bytes[i] = shifts[i].value();
        }
        StatusOr<DeviceBuffer<std::uint8_t>> device_shifts =
            DeviceBuffer<std::uint8_t>::from_host(shift_bytes);
        if (!device_shifts.ok()) {
            return device_shifts.status();
        }
        return TotientPrimeStreamKernel::launch_device(
            device_in, device_out, count, device_shifts.value().data(),
            static_cast<std::uint32_t>(shift_bytes.size()), interrupts, device_interrupt,
            effective_dir);
    }
    case CudaFamilyId::Compose:
        return Status::error("ComposeDriver: nested compose is not supported");
    }
    return Status::error("ComposeDriver: unknown stage family");
}

} // namespace

Status ComposeDriver::apply_host(std::span<const std::uint8_t> host_in,
                                 std::span<std::uint8_t> host_out, const ComposeParamsHost& recipe,
                                 const InterruptDeviceView& interrupts, CudaDir outer_direction) {
    if (host_in.size() != host_out.size()) {
        return Status::error("ComposeDriver::apply_host size mismatch");
    }
    if (interrupts.consumable_length() != host_in.size()) {
        return Status::error("ComposeDriver: interrupt view length mismatch");
    }
    if (recipe.stages.empty()) {
        return Status::error("ComposeDriver: stages must be non-empty");
    }

    const std::size_t count = host_in.size();
    const std::size_t stage_count = recipe.stages.size();

    StatusOr<DeviceBuffer<std::uint8_t>> buf_a = DeviceBuffer<std::uint8_t>::from_host(host_in);
    if (!buf_a.ok()) {
        return buf_a.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> buf_b = DeviceBuffer<std::uint8_t>::allocate(count);
    if (!buf_b.ok()) {
        return buf_b.status();
    }
    StatusOr<DeviceBuffer<std::uint8_t>> buf_final = DeviceBuffer<std::uint8_t>::allocate(count);
    if (!buf_final.ok()) {
        return buf_final.status();
    }

    StatusOr<DeviceBuffer<std::uint8_t>> device_keys =
        [&]() -> StatusOr<DeviceBuffer<std::uint8_t>> {
        if (recipe.key_arena.empty()) {
            return DeviceBuffer<std::uint8_t>::allocate(0);
        }
        return DeviceBuffer<std::uint8_t>::from_host(recipe.key_arena);
    }();
    if (!device_keys.ok()) {
        return device_keys.status();
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
    const std::uint8_t* key_ptr =
        device_keys.value().empty() ? nullptr : device_keys.value().data();

    auto stage_at = [&](std::size_t logical_index) -> const ComposeStageParams& {
        if (outer_direction == CudaDir::Decrypt) {
            return recipe.stages[logical_index];
        }
        return recipe.stages[stage_count - 1 - logical_index];
    };

    auto effective_dir_for = [&](const ComposeStageParams& stage) -> CudaDir {
        if (outer_direction == CudaDir::Decrypt) {
            return stage.direction;
        }
        return invert_dir(stage.direction);
    };

    for (std::size_t s = 0; s < stage_count; ++s) {
        const bool last = (s + 1 == stage_count);
        const bool even = (s % 2 == 0);
        const std::uint8_t* in_ptr = even ? buf_a.value().data() : buf_b.value().data();
        std::uint8_t* out_ptr =
            last ? buf_final.value().data() : (even ? buf_b.value().data() : buf_a.value().data());

        const ComposeStageParams& stage = stage_at(s);
        Status status = run_stage(in_ptr, out_ptr, count, stage, effective_dir_for(stage), recipe,
                                  interrupts, interrupt_ptr, key_ptr);
        if (!status.ok()) {
            return status;
        }
    }

    return buf_final.value().copy_to_host(host_out);
}
