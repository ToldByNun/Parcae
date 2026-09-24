#ifndef TRANSFORM_BUFFER_HPP
#define TRANSFORM_BUFFER_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/interrupt/policy.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>

/// Shared buffer / interrupt checks for CUDA-ready transform kernels.
class TransformBuffer {
public:
    [[nodiscard]] static Status require_same_length(std::span<const Index29> input,
                                                    std::span<Index29> output) {
        if (input.size() != output.size()) {
            return Status::error("transform output span length must equal input length");
        }
        return Status::success();
    }

    [[nodiscard]] static Status validate_interrupt_range(const InterruptPolicy& interrupt,
                                                         std::size_t input_size,
                                                         const char* family) {
        for (std::size_t skip : interrupt.skip_indices()) {
            if (skip >= input_size) {
                return Status::error(std::string(family) + " skip_indices out of range for input");
            }
        }
        return Status::success();
    }

    [[nodiscard]] static std::span<const std::size_t>
    skip_span(const InterruptPolicy& interrupt) noexcept {
        return std::span<const std::size_t>(interrupt.skip_indices().data(),
                                            interrupt.skip_indices().size());
    }

    [[nodiscard]] static bool should_skip(std::span<const std::size_t> skip_indices_sorted,
                                          std::size_t consumable_index) noexcept {
        return std::binary_search(skip_indices_sorted.begin(), skip_indices_sorted.end(),
                                  consumable_index);
    }

private:
    TransformBuffer() = delete;
};

#endif // TRANSFORM_BUFFER_HPP
