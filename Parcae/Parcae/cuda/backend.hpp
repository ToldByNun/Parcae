#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "params_json.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/parity/parity_record.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "affine_kernel.hpp"
#include "atbash_kernel.hpp"
#include "beaufort_key_kernel.hpp"
#include "caesar_kernel.hpp"
#include "compose_driver.hpp"
#include "identity_copy.hpp"
#include "parcae_cuda.hpp"
#include "totient_prime_stream_kernel.hpp"
#include "vigenere_key_kernel.hpp"

#include "parcae/math/totient_keystream.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

/// CUDA twin entry mirroring CPU `ApplyTransform` (`docs/architecture/cuda-handoff.md`).
///
/// Full catalog dispatch: identity / atbash / caesar / affine / vigenere_key /
/// beaufort_key / totient_prime_stream / compose (via `ComposeDriver`).
class CudaBackend {
public:
    [[nodiscard]] static bool available() noexcept {
#if defined(PARCAE_HAS_CUDA)
        return ParcaeCuda::available();
#else
        return false;
#endif
    }

    [[nodiscard]] static Status apply_into(
        const TransformId& id,
        std::span<const Index29> input,
        std::span<Index29> output,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        if (input.size() != output.size()) {
            return Status::error("CudaBackend::apply_into size mismatch");
        }

        StatusOr<CudaFamilyId> family = CudaFamilyIdUtil::from_transform_id(id);
        if (!family.ok()) {
            return family.status();
        }

        StatusOr<InterruptDeviceView> interrupts =
            InterruptDeviceView::from_policy(interrupt, input.size());
        if (!interrupts.ok()) {
            return interrupts.status();
        }

        Status prepared = prepare_params(family.value(), params);
        if (!prepared.ok()) {
            return prepared;
        }

        if (!available()) {
            return Status::error("CudaBackend: CUDA not available");
        }

        const CudaDir dir = CudaDirUtil::from_transform_direction(direction);

#if defined(PARCAE_HAS_CUDA)
        switch (family.value()) {
            case CudaFamilyId::Identity:
                return apply_identity(input, output);
            case CudaFamilyId::Atbash:
                return apply_atbash(input, output);
            case CudaFamilyId::Caesar: {
                StatusOr<CaesarParams> caesar = CudaParamsJson::caesar_from_json(params);
                return apply_caesar(input, output, caesar.value(), dir);
            }
            case CudaFamilyId::Affine: {
                StatusOr<AffineParams> affine = CudaParamsJson::affine_from_json(params);
                return apply_affine(input, output, affine.value(), dir);
            }
            case CudaFamilyId::VigenereKey: {
                StatusOr<KeyParamsHost> key = CudaParamsJson::key_from_json(params);
                return apply_vigenere(input, output, key.value(), interrupts.value(), dir);
            }
            case CudaFamilyId::BeaufortKey: {
                StatusOr<KeyParamsHost> key = CudaParamsJson::key_from_json(params);
                return apply_beaufort(input, output, key.value(), interrupts.value());
            }
            case CudaFamilyId::TotientPrimeStream: {
                StatusOr<TotientParams> totient = CudaParamsJson::totient_from_json(params);
                return apply_totient(input, output, totient.value(), interrupts.value(), dir);
            }
            case CudaFamilyId::Compose: {
                StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(params);
                return apply_compose(input, output, compose.value(), interrupts.value(), dir);
            }
        }
        return Status::error("CudaBackend: unknown CudaFamilyId");
#else
        (void)dir;
        (void)interrupts;
        (void)params;
        (void)output;
        return Status::error("CudaBackend: CUDA not available");
#endif
    }

    [[nodiscard]] static StatusOr<std::vector<Index29>> apply(
        const TransformId& id,
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        std::vector<Index29> out(input.size());
        Status status = apply_into(id, input, out, params, direction, interrupt);
        if (!status.ok()) {
            return status;
        }
        return out;
    }

    /// CUDA twin of `ParityRecord::apply_and_capture` (`backend` = `"cuda"`).
    [[nodiscard]] static StatusOr<std::pair<std::vector<Index29>, ParityRecord>> apply_and_capture(
        const TransformId& id,
        std::span<const Index29> input,
        const nlohmann::json& params,
        TransformDirection direction,
        const InterruptPolicy& interrupt = InterruptPolicy::none()) {
        StatusOr<std::vector<Index29>> output = apply(id, input, params, direction, interrupt);
        if (!output.ok()) {
            return output.status();
        }
        ParityRecord record = ParityRecord::capture(
            id.str(), params, interrupt, input, output.value(), "cuda");
        return std::make_pair(std::move(output.value()), std::move(record));
    }

private:
    CudaBackend() = delete;

#if defined(PARCAE_HAS_CUDA)
    [[nodiscard]] static std::vector<std::uint8_t> to_bytes(std::span<const Index29> input) {
        std::vector<std::uint8_t> bytes(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            bytes[i] = input[i].value();
        }
        return bytes;
    }

    [[nodiscard]] static Status write_indices(
        std::span<const std::uint8_t> bytes,
        std::span<Index29> output) {
        if (bytes.size() != output.size()) {
            return Status::error("CudaBackend: output size mismatch after kernel");
        }
        for (std::size_t i = 0; i < output.size(); ++i) {
            output[i] = Index29{bytes[i]};
        }
        return Status::success();
    }

    [[nodiscard]] static Status apply_identity(
        std::span<const Index29> input,
        std::span<Index29> output) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status copied = IdentityCopy::apply_host(bytes, bytes);
        if (!copied.ok()) {
            return copied;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_atbash(
        std::span<const Index29> input,
        std::span<Index29> output) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status = AtbashKernel::apply_host(bytes, bytes);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_caesar(
        std::span<const Index29> input,
        std::span<Index29> output,
        const CaesarParams& params,
        CudaDir direction) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status = CaesarKernel::apply_host(bytes, bytes, params.shift, direction);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_affine(
        std::span<const Index29> input,
        std::span<Index29> output,
        const AffineParams& params,
        CudaDir direction) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status =
            AffineKernel::apply_host(bytes, bytes, params.a, params.b, direction);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_vigenere(
        std::span<const Index29> input,
        std::span<Index29> output,
        const KeyParamsHost& key,
        const InterruptDeviceView& interrupts,
        CudaDir direction) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status =
            VigenereKeyKernel::apply_host(bytes, bytes, key.key, interrupts, direction);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_beaufort(
        std::span<const Index29> input,
        std::span<Index29> output,
        const KeyParamsHost& key,
        const InterruptDeviceView& interrupts) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status = BeaufortKeyKernel::apply_host(bytes, bytes, key.key, interrupts);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_totient(
        std::span<const Index29> input,
        std::span<Index29> output,
        const TotientParams& params,
        const InterruptDeviceView& interrupts,
        CudaDir direction) {
        std::size_t consumable = 0;
        for (std::size_t i = 0; i < interrupts.consumable_length(); ++i) {
            if (!interrupts.should_skip(i)) {
                ++consumable;
            }
        }
        std::vector<Index29> shifts(consumable);
        Status filled = TotientKeystream::shifts_into(
            shifts, static_cast<std::size_t>(params.prime_start_index));
        if (!filled.ok()) {
            return filled;
        }
        std::vector<std::uint8_t> shift_bytes(consumable);
        for (std::size_t i = 0; i < consumable; ++i) {
            shift_bytes[i] = shifts[i].value();
        }

        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status = TotientPrimeStreamKernel::apply_host(
            bytes, bytes, shift_bytes, interrupts, direction);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }

    [[nodiscard]] static Status apply_compose(
        std::span<const Index29> input,
        std::span<Index29> output,
        const ComposeParamsHost& recipe,
        const InterruptDeviceView& interrupts,
        CudaDir direction) {
        std::vector<std::uint8_t> bytes = to_bytes(input);
        Status status =
            ComposeDriver::apply_host(bytes, bytes, recipe, interrupts, direction);
        if (!status.ok()) {
            return status;
        }
        return write_indices(bytes, output);
    }
#endif

    [[nodiscard]] static Status prepare_params(
        CudaFamilyId family,
        const nlohmann::json& params) {
        switch (family) {
            case CudaFamilyId::Identity:
            case CudaFamilyId::Atbash: {
                StatusOr<EmptyParams> empty = CudaParamsJson::empty_from_json(params);
                if (!empty.ok()) {
                    return empty.status();
                }
                return Status::success();
            }
            case CudaFamilyId::Caesar: {
                StatusOr<CaesarParams> caesar = CudaParamsJson::caesar_from_json(params);
                if (!caesar.ok()) {
                    return caesar.status();
                }
                return Status::success();
            }
            case CudaFamilyId::Affine: {
                StatusOr<AffineParams> affine = CudaParamsJson::affine_from_json(params);
                if (!affine.ok()) {
                    return affine.status();
                }
                return Status::success();
            }
            case CudaFamilyId::VigenereKey:
            case CudaFamilyId::BeaufortKey: {
                StatusOr<KeyParamsHost> key = CudaParamsJson::key_from_json(params);
                if (!key.ok()) {
                    return key.status();
                }
                return Status::success();
            }
            case CudaFamilyId::TotientPrimeStream: {
                StatusOr<TotientParams> totient = CudaParamsJson::totient_from_json(params);
                if (!totient.ok()) {
                    return totient.status();
                }
                return Status::success();
            }
            case CudaFamilyId::Compose: {
                StatusOr<ComposeParamsHost> compose = CudaParamsJson::compose_from_json(params);
                if (!compose.ok()) {
                    return compose.status();
                }
                return Status::success();
            }
        }
        return Status::error("CudaBackend: unknown CudaFamilyId");
    }
};

#endif // BACKEND_HPP
