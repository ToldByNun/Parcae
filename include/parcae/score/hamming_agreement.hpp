#ifndef HAMMING_AGREEMENT_HPP
#define HAMMING_AGREEMENT_HPP

#include "parcae/core/index29.hpp"
#include "parcae/core/status.hpp"
#include "parcae/core/status_or.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

/// Position-wise agreement rate — score_id `hamming_agreement`, v0.
///
/// matches / N in [0,1]. Equal non-empty lengths required; mismatched lengths
/// or N == 0 are hard errors. ScoreOrder::Desc.
class HammingAgreement {
public:
    static constexpr std::string_view score_id = "hamming_agreement";
    static constexpr std::string_view score_version = "v0";

    [[nodiscard]] static StatusOr<double> score(
        const std::vector<Index29>& candidate,
        const std::vector<Index29>& reference) {
        if (candidate.size() != reference.size()) {
            return Status::error("hamming_agreement requires equal lengths");
        }
        const std::size_t n = candidate.size();
        if (n == 0) {
            return Status::error("hamming_agreement requires a non-empty sequence");
        }

        std::uint64_t matches = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (candidate[i] == reference[i]) {
                ++matches;
            }
        }
        return static_cast<double>(matches) / static_cast<double>(n);
    }
};

#endif // HAMMING_AGREEMENT_HPP
