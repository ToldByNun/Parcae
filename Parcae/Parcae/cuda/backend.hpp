#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "interrupt_device_view.hpp"
#include "params.hpp"
#include "params_json.hpp"

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"
#include "parcae/interrupt/policy.hpp"
#include "parcae/transform/transform_direction.hpp"
#include "parcae/transform/transform_id.hpp"

#if defined(PARCAE_HAS_CUDA)
#include "parcae_cuda.hpp"
#endif

#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

/// CUDA twin entry mirroring CPU `ApplyTransform` (`docs/architecture/cuda-handoff.md`).
///
/// Host-side: validates sizes, resolves `CudaFamilyId`, builds `InterruptDeviceView`,
/// and parses JSON → POD params. Device kernels are wired in later commits; until then
/// each family returns a clear "not implemented" `Status` when CUDA is linked.
class CudaBackend {
public:
    /// True when this binary was built with the CUDA twin library.
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

        const CudaDir dir = CudaDirUtil::from_transform_direction(direction);
        Status prepared = prepare_params(family.value(), params);
        if (!prepared.ok()) {
            return prepared;
        }

        if (!available()) {
            return Status::error("CudaBackend: CUDA not available");
        }

        (void)dir;
        (void)interrupts;
        return not_implemented(family.value());
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

private:
    CudaBackend() = delete;

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

    [[nodiscard]] static Status not_implemented(CudaFamilyId family) {
        std::string message = "CudaBackend: not implemented (";
        message.append(family_name(family));
        message.append(")");
        return Status::error(std::move(message));
    }

    [[nodiscard]] static const char* family_name(CudaFamilyId family) noexcept {
        switch (family) {
            case CudaFamilyId::Identity:
                return "identity";
            case CudaFamilyId::Atbash:
                return "atbash";
            case CudaFamilyId::Caesar:
                return "caesar";
            case CudaFamilyId::Affine:
                return "affine";
            case CudaFamilyId::VigenereKey:
                return "vigenere_key";
            case CudaFamilyId::BeaufortKey:
                return "beaufort_key";
            case CudaFamilyId::TotientPrimeStream:
                return "totient_prime_stream";
            case CudaFamilyId::Compose:
                return "compose";
        }
        return "unknown";
    }
};

#endif // BACKEND_HPP
