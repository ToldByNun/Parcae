#ifndef BENCH_TIER_SPEC_HPP
#define BENCH_TIER_SPEC_HPP

#include "parcae/core/index29.hpp"

#include <cstddef>
#include <string_view>

/// Canonical SLO tier constants for Parcae bench / throughput diagnostics.
///
/// **Single source of truth** for T1–T3 config (C, T, reps) and for practical
/// peak / SLO floor tables. `DslPeakSanity` and `ThroughputTiers` delegate here
/// (Catch2 `[bench][spec]` / `[dsl][peak]`).
///
/// Metric: `repeats × C × T / elapsed` (setup excluded). Peak ceilings are
/// calibrated on RTX 5070 Ti; see `docs/architecture/cuda-throughput.md`.
///
/// Pass rule (same as historical `ThroughputTiers::pass_tier`):
///   rps >= slo_min  AND  (peak<=0 OR 100*rps/peak + 0.5 >= 90).
/// Display bands (`slo_max`) are expectations only — faster than max still passes.
class BenchTierSpec {
public:
    /// ≥90% of practical ceiling (89.5% raw so rounded display of 90% matches).
    static constexpr double peak_band_pct = 90.0;
    static constexpr double peak_band_raw = 89.5;

    /// One timed SLO tier (T1 / T2 / T3). Aggregate for MSVC `constexpr` init.
    class Tier {
    public:
        const char* id;
        const char* workload;
        /// Candidate / key lanes (C).
        std::size_t candidates;
        /// Stream length in Index29 runes (T).
        std::size_t tokens;
        /// Timed inner-loop repetitions (reps).
        std::size_t repeats;
        /// SLO floor (runes/s).
        double slo_min;
        /// Display band upper bound (runes/s); 0 = no upper bound (e.g. T3).
        double slo_max;
        /// Practical ceiling on the calibration GPU (runes/s).
        double estimated_peak;
    };

    // --- Primary SLO tiers (canonical config) --------------------------------

    /// Caesar fused χ² (simple substitution). C=29, T=2^20, reps=64.
    static constexpr Tier t1{"T1",
                             "Caesar fused chi2 (simple sub)",
                             static_cast<std::size_t>(Index29::modulus),
                             1048576u, // 1 << 20
                             64u,
                             15.0e9,
                             35.0e9,
                             392.0e9};

    /// Filtered multi-key / autokey / dynamic-shift (worst of three).
    /// C=4096, T=2^18, reps=8.
    static constexpr Tier t2{"T2",    "Filtered multi-key/autokey/dyn (worst)",
                             4096u,
                             262144u, // 1 << 18
                             8u,      3.0e9,
                             10.0e9,  402.0e9};

    /// Caesar bigram + synthetic dictionary validation.
    /// C=512, T=2^18, reps=8. slo_max=0 → display ">=".
    static constexpr Tier t3{"T3",  "Caesar bigram+dict validation", 512u, 262144u, 8u, 1.0e9, 0.0,
                             55.0e9};

    // --- Theory fused-χ² shapes (RTX 5070 Ti provisional; see cuda-profile-theory.md) ---

    /// S0 interpreter: Caesar-as-bytecode. Peak from post–shared/trusted cudaEvent (~69B)
    /// rounded up. Fair gate: T≥2^20.
    static constexpr Tier theory_s0_caesar{"T.theory.caesar_bytecode",
                                          "TheoryChi2Batch Caesar bytecode (S0)",
                                          static_cast<std::size_t>(Index29::modulus),
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          75.0e9};

    /// S1 LUT-29 (caesar/affine-shaped). Affine-class ceiling until specialized remesaure.
    static constexpr Tier theory_s1_lut29{"T.theory.s1_lut29",
                                         "TheoryHistChi2S1 LUT-29 (f(x)-only)",
                                         static_cast<std::size_t>(Index29::modulus),
                                         1048576u,
                                         8u,
                                         15.0e9,
                                         0.0,
                                         473.0e9};

    /// S2 linear uchar4 (progressive / bitmask_blend-shaped). Plan provisional 350B.
    /// Default C=841 (=29²) matches full (b0,b1) grid; suite may use smaller C for Catch2.
    static constexpr Tier theory_s2_linear{"T.theory.s2_linear",
                                          "TheoryHistChi2S2 progressive/bitmask linear",
                                          841u,
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          350.0e9};

    /// Alias name used by microbench progressive row (same peak as S2 linear).
    static constexpr const char* theory_progressive_id = "T.theory.progressive";

    static constexpr std::size_t primary_tier_count = 3;

    /// Optional interim checkpoint when `estimated_peak ≫ 50B` (never replaces 90% gate).
    static constexpr double checkpoint_50B_rps = 50.0e9;
    /// Treat peak as ≫ 50B when at least 2× the checkpoint (plan: Peak ≫ 50B).
    static constexpr double checkpoint_50B_peak_min = 100.0e9;

    [[nodiscard]] static constexpr bool checkpoint_50B_applicable(double peak) noexcept {
        return peak >= checkpoint_50B_peak_min;
    }

    [[nodiscard]] static constexpr bool checkpoint_50B_hit(double rps, double peak) noexcept {
        return checkpoint_50B_applicable(peak) && rps >= checkpoint_50B_rps;
    }

    /// Fair Kernel-SLO token floor (T≥2^20). Shorter T → underfill, not a fail gate.
    [[nodiscard]] static constexpr std::size_t fair_gate_tokens() noexcept { return t1.tokens; }

    [[nodiscard]] static constexpr bool is_fair_gate_tokens(std::size_t tokens) noexcept {
        return tokens >= fair_gate_tokens();
    }

    /// Ordered T1, T2, T3 for suite iteration. Out-of-range → T1.
    [[nodiscard]] static constexpr const Tier& tier_at(std::size_t index) noexcept {
        switch (index) {
        case 0:
            return t1;
        case 1:
            return t2;
        case 2:
            return t3;
        default:
            return t1;
        }
    }

    [[nodiscard]] static constexpr const Tier* find_primary(std::string_view id) noexcept {
        if (id == std::string_view{t1.id}) {
            return &t1;
        }
        if (id == std::string_view{t2.id}) {
            return &t2;
        }
        if (id == std::string_view{t3.id}) {
            return &t3;
        }
        return nullptr;
    }

    // --- Peak / SLO tables (incl. F.* / C.* extended rows) --------------------

    /// Practical ceilings (runes/s). Must match `DslPeakSanity` / throughput docs.
    [[nodiscard]] static constexpr double estimated_peak(std::string_view tier) noexcept {
        if (tier == "T1") {
            return t1.estimated_peak;
        }
        if (tier == "T2") {
            return t2.estimated_peak;
        }
        if (tier == "T3") {
            return t3.estimated_peak;
        }
        if (tier == "F.atbash") {
            return 550.0e9;
        }
        if (tier == "F.affine") {
            return 473.0e9;
        }
        if (tier == "F.vigenere") {
            return 398.0e9;
        }
        if (tier == "F.beaufort") {
            return 402.0e9;
        }
        if (tier == "F.totient") {
            return 460.0e9;
        }
        if (tier == "C.koan1_fused") {
            return 372.0e9;
        }
        if (tier == "C.koan1_stages") {
            return 322.0e9;
        }
        if (tier == "T.theory.caesar_bytecode" || tier == "T.theory.s0") {
            return theory_s0_caesar.estimated_peak;
        }
        if (tier == "T.theory.s1_lut29" || tier == "T.theory.s1") {
            return theory_s1_lut29.estimated_peak;
        }
        if (tier == "T.theory.s2_linear" || tier == "T.theory.progressive" ||
            tier == "T.theory.s2") {
            return theory_s2_linear.estimated_peak;
        }
        return 0.0;
    }

    /// SLO floors (runes/s). Extended F/C/T.theory rows keep historical floors.
    [[nodiscard]] static constexpr double slo_floor(std::string_view tier) noexcept {
        if (tier == "T1") {
            return t1.slo_min;
        }
        if (tier == "T2") {
            return t2.slo_min;
        }
        if (tier == "T3") {
            return t3.slo_min;
        }
        if (tier == "F.atbash" || tier == "F.affine" || tier == "C.koan1_fused" ||
            tier == "C.koan1_stages" || tier == "T.theory.caesar_bytecode" ||
            tier == "T.theory.s0" || tier == "T.theory.s1_lut29" || tier == "T.theory.s1" ||
            tier == "T.theory.s2_linear" || tier == "T.theory.progressive" ||
            tier == "T.theory.s2") {
            return 15.0e9;
        }
        if (tier == "F.vigenere" || tier == "F.beaufort" || tier == "F.totient") {
            return 3.0e9;
        }
        return 0.0;
    }

    /// Look up a theory-shape Tier by id (nullptr if unknown).
    [[nodiscard]] static constexpr const Tier* find_theory(std::string_view id) noexcept {
        if (id == std::string_view{theory_s0_caesar.id} || id == "T.theory.s0") {
            return &theory_s0_caesar;
        }
        if (id == std::string_view{theory_s1_lut29.id} || id == "T.theory.s1") {
            return &theory_s1_lut29;
        }
        if (id == std::string_view{theory_s2_linear.id} || id == "T.theory.s2" ||
            id == std::string_view{theory_progressive_id}) {
            return &theory_s2_linear;
        }
        return nullptr;
    }

    [[nodiscard]] static constexpr bool known_tier(std::string_view tier) noexcept {
        return estimated_peak(tier) > 0.0;
    }

    /// Identical rule to historical `ThroughputTiers::pass_tier`.
    [[nodiscard]] static constexpr bool pass_tier(double rps, double slo_min,
                                                  double peak) noexcept {
        if (rps < slo_min) {
            return false;
        }
        if (peak <= 0.0) {
            return true;
        }
        const double pct = 100.0 * rps / peak;
        return pct + 0.5 >= peak_band_pct;
    }

    [[nodiscard]] static constexpr double percent_peak(double rps, double peak) noexcept {
        if (peak <= 0.0) {
            return 0.0;
        }
        return 100.0 * rps / peak;
    }

private:
    BenchTierSpec() = delete;
};

#endif // BENCH_TIER_SPEC_HPP
