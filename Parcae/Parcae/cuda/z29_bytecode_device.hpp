#ifndef Z29_BYTECODE_DEVICE_HPP
#define Z29_BYTECODE_DEVICE_HPP

#include "autokey_ring_device.hpp"
#include "z29_device.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PARCAE_BC_HD __host__ __device__
#else
#define PARCAE_BC_HD
#endif

/// Device/host twin of `Z29Bytecode::eval_at` for HotLoop stack programs.
///
/// Op numerics MUST match `Z29Bytecode::Op` (`op_as_u8`). No heap: caller
/// supplies a scratch stack of length `stack_cap` (`<= kMaxDeviceStack`).
/// Domain errors (div0 / inv0 / …) and stack faults set `*err_flag != 0` and
/// leave `*out_byte` undefined.
///
/// `eval_at` is the safe oracle (full checks). `eval_at_trusted` skips
/// structural bounds checks after host validation (`TheoryChi2Batch` launch);
/// domain errors (div0/inv0/…) remain.
class Z29BytecodeDevice {
public:
    static constexpr std::uint16_t kMaxDeviceStack = 64;

    // Keep in lockstep with Z29Bytecode::Op (tests assert equality).
    static constexpr std::uint8_t kOpConst = 0;
    static constexpr std::uint8_t kOpLoad = 1;
    static constexpr std::uint8_t kOpAdd = 2;
    static constexpr std::uint8_t kOpSub = 3;
    static constexpr std::uint8_t kOpMul = 4;
    static constexpr std::uint8_t kOpDiv = 5;
    static constexpr std::uint8_t kOpFloorDiv = 6;
    static constexpr std::uint8_t kOpMod = 7;
    static constexpr std::uint8_t kOpPow = 8;
    static constexpr std::uint8_t kOpBitAnd = 9;
    static constexpr std::uint8_t kOpBitOr = 10;
    static constexpr std::uint8_t kOpBitXor = 11;
    static constexpr std::uint8_t kOpLShift = 12;
    static constexpr std::uint8_t kOpRShift = 13;
    static constexpr std::uint8_t kOpEq = 14;
    static constexpr std::uint8_t kOpNe = 15;
    static constexpr std::uint8_t kOpLt = 16;
    static constexpr std::uint8_t kOpLe = 17;
    static constexpr std::uint8_t kOpGt = 18;
    static constexpr std::uint8_t kOpGe = 19;
    static constexpr std::uint8_t kOpBoolAnd = 20;
    static constexpr std::uint8_t kOpBoolOr = 21;
    static constexpr std::uint8_t kOpNeg = 22;
    static constexpr std::uint8_t kOpInv = 23;
    static constexpr std::uint8_t kOpAtbash = 24;
    static constexpr std::uint8_t kOpBitNot = 25;
    static constexpr std::uint8_t kOpBoolNot = 26;
    static constexpr std::uint8_t kOpSelect = 27;
    static constexpr std::uint8_t kOpAutokeyShift = 28;

    static constexpr std::uint8_t kErrOk = 0;
    static constexpr std::uint8_t kErrDomain = 1;
    static constexpr std::uint8_t kErrStack = 2;
    static constexpr std::uint8_t kErrProgram = 3;
    static constexpr std::uint8_t kErrIndex = 4;

    /// Evaluate one stream index. Overwrites `slots[cipher_slot]` (and index
    /// slot when `binds_index_i != 0`). Returns true on success.
    [[nodiscard]] PARCAE_BC_HD static bool
    eval_at(const std::uint8_t* ops, const std::uint8_t* imm, std::uint32_t op_count,
            std::uint8_t* slots, std::uint16_t slot_count, std::uint16_t cipher_slot,
            std::uint16_t index_slot, std::uint8_t binds_index_i, const std::uint8_t* stream,
            std::size_t stream_len, std::size_t i, std::uint8_t* stack, std::uint16_t stack_cap,
            std::uint8_t* out_byte, std::uint8_t* err_flag) noexcept {
        return eval_impl</*Trusted=*/false>(ops, imm, op_count, slots, slot_count, cipher_slot,
                                             index_slot, binds_index_i, stream, stream_len, i, stack,
                                             stack_cap, out_byte, err_flag);
    }

    /// Hot-path twin: assumes host-validated program/slots/stack and `i < stream_len`.
    /// Still reports domain errors. Prefer for `TheoryChi2Batch` after launch checks.
    [[nodiscard]] PARCAE_BC_HD static bool
    eval_at_trusted(const std::uint8_t* ops, const std::uint8_t* imm, std::uint32_t op_count,
                    std::uint8_t* slots, std::uint16_t slot_count, std::uint16_t cipher_slot,
                    std::uint16_t index_slot, std::uint8_t binds_index_i, const std::uint8_t* stream,
                    std::size_t /*stream_len*/, std::size_t i, std::uint8_t* stack,
                    std::uint16_t stack_cap, std::uint8_t* out_byte,
                    std::uint8_t* err_flag) noexcept {
        return eval_impl</*Trusted=*/true>(ops, imm, op_count, slots, slot_count, cipher_slot,
                                            index_slot, binds_index_i, stream, /*stream_len=*/0, i,
                                            stack, stack_cap, out_byte, err_flag);
    }

private:
    Z29BytecodeDevice() = delete;

    template <bool Trusted>
    [[nodiscard]] PARCAE_BC_HD static bool
    eval_impl(const std::uint8_t* ops, const std::uint8_t* imm, std::uint32_t op_count,
              std::uint8_t* slots, std::uint16_t slot_count, std::uint16_t cipher_slot,
              std::uint16_t index_slot, std::uint8_t binds_index_i, const std::uint8_t* stream,
              std::size_t stream_len, std::size_t i, std::uint8_t* stack, std::uint16_t stack_cap,
              std::uint8_t* out_byte, std::uint8_t* err_flag) noexcept {
        if constexpr (!Trusted) {
            if (err_flag == nullptr || out_byte == nullptr) {
                return false;
            }
            *err_flag = kErrOk;
            if (ops == nullptr || imm == nullptr || slots == nullptr || stack == nullptr ||
                stream == nullptr || op_count == 0u || slot_count == 0u || stack_cap == 0u ||
                stack_cap > kMaxDeviceStack || cipher_slot >= slot_count) {
                *err_flag = kErrProgram;
                return false;
            }
            if (binds_index_i != 0u && index_slot >= slot_count) {
                *err_flag = kErrProgram;
                return false;
            }
            if (i >= stream_len) {
                *err_flag = kErrIndex;
                return false;
            }
        } else {
            *err_flag = kErrOk;
            (void)slot_count;
            (void)stream_len;
        }

        slots[cipher_slot] = stream[i];
        if (binds_index_i != 0u) {
            slots[index_slot] = static_cast<std::uint8_t>(i % Z29Device::modulus);
        }

        std::uint16_t sp = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#endif
        for (std::uint32_t pc = 0; pc < op_count; ++pc) {
            const std::uint8_t op = ops[pc];
            const std::uint8_t imm_b = imm[pc];
            switch (op) {
            case kOpConst: {
                if constexpr (!Trusted) {
                    if (imm_b >= Z29Device::modulus) {
                        *err_flag = kErrProgram;
                        return false;
                    }
                }
                if (!push<Trusted>(stack, stack_cap, &sp, imm_b, err_flag)) {
                    return false;
                }
                break;
            }
            case kOpLoad: {
                if constexpr (!Trusted) {
                    if (imm_b >= slot_count) {
                        *err_flag = kErrProgram;
                        return false;
                    }
                }
                if (!push<Trusted>(stack, stack_cap, &sp, slots[imm_b], err_flag)) {
                    return false;
                }
                break;
            }
            case kOpAutokeyShift: {
                std::uint8_t lag = 0;
                if (!pop<Trusted>(stack, &sp, &lag, err_flag)) {
                    return false;
                }
                if (!push<Trusted>(stack, stack_cap, &sp, AutokeyRingDevice::shift(stream, i, lag),
                                   err_flag)) {
                    return false;
                }
                break;
            }
            case kOpSelect: {
                std::uint8_t f = 0;
                std::uint8_t t = 0;
                std::uint8_t c = 0;
                if (!pop<Trusted>(stack, &sp, &f, err_flag) ||
                    !pop<Trusted>(stack, &sp, &t, err_flag) ||
                    !pop<Trusted>(stack, &sp, &c, err_flag)) {
                    return false;
                }
                if (!push<Trusted>(stack, stack_cap, &sp, Z29Device::select(c, t, f), err_flag)) {
                    return false;
                }
                break;
            }
            default: {
                if (!eval_op<Trusted>(op, stack, stack_cap, &sp, err_flag)) {
                    return false;
                }
                break;
            }
            }
        }
        if (sp != 1u) {
            *err_flag = kErrProgram;
            return false;
        }
        *out_byte = stack[0];
        return true;
    }

    template <bool Trusted>
    [[nodiscard]] PARCAE_BC_HD static bool push(std::uint8_t* stack, std::uint16_t stack_cap,
                                                std::uint16_t* sp, std::uint8_t value,
                                                std::uint8_t* err_flag) noexcept {
        if constexpr (!Trusted) {
            if (*sp >= stack_cap) {
                *err_flag = kErrStack;
                return false;
            }
        } else {
            (void)stack_cap;
            (void)err_flag;
        }
        stack[*sp] = value;
        ++(*sp);
        return true;
    }

    template <bool Trusted>
    [[nodiscard]] PARCAE_BC_HD static bool pop(std::uint8_t* stack, std::uint16_t* sp,
                                               std::uint8_t* out, std::uint8_t* err_flag) noexcept {
        if constexpr (!Trusted) {
            if (*sp == 0u) {
                *err_flag = kErrStack;
                return false;
            }
        } else {
            (void)err_flag;
        }
        --(*sp);
        *out = stack[*sp];
        return true;
    }

    template <bool Trusted>
    [[nodiscard]] PARCAE_BC_HD static bool eval_op(std::uint8_t op, std::uint8_t* stack,
                                                   std::uint16_t stack_cap, std::uint16_t* sp,
                                                   std::uint8_t* err_flag) noexcept {
        switch (op) {
        case kOpAdd:
        case kOpSub:
        case kOpMul:
        case kOpDiv:
        case kOpFloorDiv:
        case kOpMod:
        case kOpPow:
        case kOpBitAnd:
        case kOpBitOr:
        case kOpBitXor:
        case kOpLShift:
        case kOpRShift:
        case kOpEq:
        case kOpNe:
        case kOpLt:
        case kOpLe:
        case kOpGt:
        case kOpGe:
        case kOpBoolAnd:
        case kOpBoolOr: {
            std::uint8_t b = 0;
            std::uint8_t a = 0;
            if (!pop<Trusted>(stack, sp, &b, err_flag) || !pop<Trusted>(stack, sp, &a, err_flag)) {
                return false;
            }
            std::uint8_t r = 0;
            if (op == kOpAdd) {
                r = Z29Device::add(a, b);
            } else if (op == kOpSub) {
                r = Z29Device::sub(a, b);
            } else if (op == kOpMul) {
                r = Z29Device::mul(a, b);
            } else if (op == kOpDiv) {
                if (b == 0u) {
                    *err_flag = kErrDomain;
                    return false;
                }
                r = Z29Device::mul(a, Z29Device::inv(b));
            } else if (op == kOpFloorDiv) {
                if (b == 0u) {
                    *err_flag = kErrDomain;
                    return false;
                }
                r = Z29Device::floor_div(a, b);
            } else if (op == kOpMod) {
                if (b == 0u) {
                    *err_flag = kErrDomain;
                    return false;
                }
                r = static_cast<std::uint8_t>(a % b);
            } else if (op == kOpPow) {
                r = Z29Device::pow(a, b);
            } else if (op == kOpBitAnd) {
                r = Z29Device::bit_and(a, b);
            } else if (op == kOpBitOr) {
                r = Z29Device::bit_or(a, b);
            } else if (op == kOpBitXor) {
                r = Z29Device::bit_xor(a, b);
            } else if (op == kOpLShift) {
                r = Z29Device::lshift(a, b);
            } else if (op == kOpRShift) {
                r = Z29Device::rshift(a, b);
            } else if (op == kOpEq) {
                r = Z29Device::eq(a, b);
            } else if (op == kOpNe) {
                r = Z29Device::ne(a, b);
            } else if (op == kOpLt) {
                r = Z29Device::lt(a, b);
            } else if (op == kOpLe) {
                r = Z29Device::le(a, b);
            } else if (op == kOpGt) {
                r = Z29Device::gt(a, b);
            } else if (op == kOpGe) {
                r = Z29Device::ge(a, b);
            } else if (op == kOpBoolAnd) {
                r = Z29Device::bool_and(a, b);
            } else {
                r = Z29Device::bool_or(a, b);
            }
            return push<Trusted>(stack, stack_cap, sp, r, err_flag);
        }
        case kOpNeg:
        case kOpInv:
        case kOpAtbash:
        case kOpBitNot:
        case kOpBoolNot: {
            std::uint8_t a = 0;
            if (!pop<Trusted>(stack, sp, &a, err_flag)) {
                return false;
            }
            std::uint8_t r = 0;
            if (op == kOpNeg) {
                r = Z29Device::neg(a);
            } else if (op == kOpInv) {
                if (a == 0u) {
                    *err_flag = kErrDomain;
                    return false;
                }
                r = Z29Device::inv(a);
            } else if (op == kOpAtbash || op == kOpBitNot) {
                r = Z29Device::bit_not(a);
            } else {
                r = Z29Device::bool_not(a);
            }
            return push<Trusted>(stack, stack_cap, sp, r, err_flag);
        }
        default:
            *err_flag = kErrProgram;
            return false;
        }
    }
};

#undef PARCAE_BC_HD

#endif // Z29_BYTECODE_DEVICE_HPP
