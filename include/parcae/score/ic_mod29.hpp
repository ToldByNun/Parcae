#ifndef IC_MOD29_HPP
#define IC_MOD29_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/// Index of coincidence over Z/29Z symbols (score_id `ic_mod29`, version v0).
///
/// IC = sum_c n_c(n_c-1) / (N(N-1)) for N >= 2. Higher is typically more
/// monoalphabetic / language-like (ScoreOrder::Desc).
class IcMod29 {
public:
    static constexpr std::string_view score_id = "ic_mod29";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(const std::vector<Index29>& indices) {
        const std::size_t n = indices.size();
        if (n < 2) {
            return Status::error("ic_mod29 requires at least 2 symbols");
        }

        std::array<std::uint64_t, Index29::modulus> counts{};
        for (const Index29 idx : indices) {
            ++counts[idx.value()];
        }

        std::uint64_t numerator = 0;
        for (std::uint64_t c : counts) {
            numerator += c * (c - 1);
        }

        const double denom = static_cast<double>(n) * static_cast<double>(n - 1);
        return static_cast<double>(numerator) / denom;
    }
};

#endif // IC_MOD29_HPP
