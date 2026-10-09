#ifndef BENCH_TIER_SPEC_HPP
#define BENCH_TIER_SPEC_HPP

#include "parcae/core/index29.hpp"

#include <cstddef>
#include <string_view>

/// Canonical SLO tier constants for Parcae bench / throughput diagnostics.
///
/// **Single source of truth** for T1–T3 config (C, T, reps) and for peak / SLO
/// floor tables. `DslPeakSanity` and `ThroughputTiers` delegate here
/// (Catch2 `[bench][spec]` / `[dsl][peak]`).
///
/// Metric: `repeats × C × T / elapsed` (setup excluded).
///
/// **`estimated_peak`** for fused decrypt+χ² hist on the calibration GPU
/// (RTX 5070 Ti):
///   - **Alphabet / column / lag / bigram remap** (production CipherHistOnce /
///     ColumnHistOnce / LagDiffHistOnce / BigramCountOnce + bin remap):
///     **Remap roofs** (`kAlphabetRemapHistRoofRps`, …) — Done gates on
///     **logical** `C·T` runes/s, not 1 B/rune DRAM. Cipher traffic is ~O(T).
///   - **Legacy unique-key decode→hist** (hard-S0 bytecode, koan stages):
///     physical **DRAM roofline** = GDDR7 BW / 1 B cipher/rune (**896B**).
///   - **Shared-cipher occupancy** (F.atbash / F.totient / dsl_smart Atbash):
///     **SM/atomic compute roof** (`kSharedCipherComputeRoofRps`), same numeric
///     freeze as alphabet remap until a dedicated CipherHistOnce plate lands.
/// `%peak` cannot exceed 100 by construction; if it does, the roof model is wrong.
/// Dual rates: `BenchMetric::logical_runes_per_sec` vs `cipher_bytes_per_sec`.
/// See `docs/architecture/cuda-throughput.md`.
///
/// Pass rule (same as historical `ThroughputTiers::pass_tier`):
///   rps >= slo_min  AND  (peak<=0 OR 100*rps/peak + 0.5 >= 90).
/// Display bands (`slo_max`) are expectations only — faster than max still passes.
class BenchTierSpec {
public:
    /// ≥90% of shape `estimated_peak` (89.5% raw so rounded display of 90% matches).
    static constexpr double peak_band_pct = 90.0;
    static constexpr double peak_band_raw = 89.5;

    /// RTX 5070 Ti published GDDR7 peak bandwidth (bytes/s).
    static constexpr double kDramBandwidthBytesPerSec = 896.0e9;
    /// Minimum DRAM traffic model for fused hist: one cipher byte per (c,t) rune.
    static constexpr double kHistCipherBytesPerRune = 1.0;
    /// Absolute physical ceiling (runes/s) = BW / bytes_per_rune. Same for every
    /// fused hist shape that streams ≥1 B cipher/rune (S0/S1/S2/T1/F.* unique-key).
    /// Named constant (not a call) so MSVC can use it in `static constexpr Tier` inits.
    static constexpr double kDramRooflineHistPeak =
        kDramBandwidthBytesPerSec / kHistCipherBytesPerRune;

    /// Shared-cipher occupancy-padded hist (F.atbash C=512 identical lanes;
    /// F.totient C=512 starts). ncu 2026-10-06 RTX 5070 Ti (`profiles/traffic_model/`):
    ///   `atbash_chi2_hist_kernel` @ C=512 T=262144: `dram__bytes.sum`=1.490688 MB
    ///     → bytes/rune = 1.490688e6/(512·262144) ≈ **0.01111**
    ///   `totient_chi2_hist_kernel` same grid: 1.47 MB → ≈ **0.0110**
    /// DRAM SoL ~0.4–1% (compute/L2-bound). Peak = BW/bytes ≈ **80.7 TB** runes/s.
    /// **Diary only** — not `estimated_peak` for Atbash/totient Done (see compute roof).
    static constexpr double kHistBytesPerRuneSharedCipherOccupancy = 0.01111;
    static constexpr double kDramRooflineSharedCipherOccupancyPeak =
        kDramBandwidthBytesPerSec / kHistBytesPerRuneSharedCipherOccupancy;

    /// Shared-cipher **compute** roof (runes/s) — Done / Stretch gate for
    /// `F.atbash` / `F.totient` / dsl_smart Atbash. Calibrated 2026-10-06 on
    /// RTX 5070 Ti via `HistOccupancyRoof` identity hist+finalize @ C=512 T=2^20
    /// fat-64 (`[cuda][hist][compute_roof]`): quiet identity plate ~1.2–2.2 TB
    /// (median ~1.6 TB). Spec freezes **2.0e12** with plate-noise headroom so
    /// quiet Atbash (~1.1–1.7 TB) stays `%peak≤100`. Not Atbash quiet max; not
    /// the DRAM occupancy diary (~80.7 TB). DRAM-bound is **not** required for
    /// Done on this traffic class (cipher is L2-resident; hist is atomic-bound).
    static constexpr double kSharedCipherComputeRoofRps = 2.0e12;

    /// Alphabet mono remap Done roof (logical runes/s): CipherHistOnce + C×29
    /// bin remap + finalize (Caesar / S1 / ShapeInline mono / Affine remap).
    /// Interim freeze = identity-hist plate class (**2.0e12**); re-calibrate on
    /// a quiet CipherHistOnce+finalize plate when available. Not 896B DRAM.
    static constexpr double kAlphabetRemapHistRoofRps = kSharedCipherComputeRoofRps;

    /// Column period remap Done roof (logical runes/s): ColumnHistOnce(L) +
    /// keystream/key remap (S2 / S5 / Vigenère / Beaufort). Same interim freeze.
    static constexpr double kColumnRemapHistRoofRps = kSharedCipherComputeRoofRps;

    /// Lag-diff / AutokeyRing remap Done roof (logical runes/s): LagDiffHistOnce
    /// + prefix merge (S4 / CTAK). Same interim freeze.
    static constexpr double kLagRemapHistRoofRps = kSharedCipherComputeRoofRps;

    /// Bigram-LL remap Done roof (logical runes/s): BigramCountOnce + C×841 Dot
    /// (T3 / DeepScore bigram-LL). Slightly lower interim freeze (more once work).
    static constexpr double kBigramRemapHistRoofRps = 1.0e12;

    /// Affine fused hist — **diary** ncu bytes/rune for legacy decode→hist
    /// (F.affine / dsl_smart Affine before remap). ncu 2026-10-06 RTX 5070 Ti:
    ///   `affine_chi2_hist_kernel` @ C=812 T=262144: ≈ **0.01906** B/rune →
    ///   Peak ≈ **47.0 TB**. Production Affine remap Done uses
    ///   `kAlphabetRemapHistRoofRps` instead.
    static constexpr double kHistBytesPerRuneAffineSharedCipher = 0.01906;
    static constexpr double kDramRooflineAffineSharedCipherPeak =
        kDramBandwidthBytesPerSec / kHistBytesPerRuneAffineSharedCipher;

    [[nodiscard]] static constexpr double dram_roofline_hist_peak() noexcept {
        return kDramRooflineHistPeak;
    }

    /// Diary accessor — traffic-model DRAM ceiling; not the Done gate.
    [[nodiscard]] static constexpr double dram_roofline_shared_cipher_occupancy_peak() noexcept {
        return kDramRooflineSharedCipherOccupancyPeak;
    }

    /// Done / Stretch gate for Atbash / totient / dsl_smart Atbash.
    [[nodiscard]] static constexpr double shared_cipher_compute_roof_rps() noexcept {
        return kSharedCipherComputeRoofRps;
    }

    [[nodiscard]] static constexpr double alphabet_remap_hist_roof_rps() noexcept {
        return kAlphabetRemapHistRoofRps;
    }

    [[nodiscard]] static constexpr double column_remap_hist_roof_rps() noexcept {
        return kColumnRemapHistRoofRps;
    }

    [[nodiscard]] static constexpr double lag_remap_hist_roof_rps() noexcept {
        return kLagRemapHistRoofRps;
    }

    [[nodiscard]] static constexpr double bigram_remap_hist_roof_rps() noexcept {
        return kBigramRemapHistRoofRps;
    }

    /// Diary only — legacy Affine decode→hist DRAM class (not production Done).
    [[nodiscard]] static constexpr double dram_roofline_affine_shared_cipher_peak() noexcept {
        return kDramRooflineAffineSharedCipherPeak;
    }

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
        /// Shape ceiling for Done/Stretch (runes/s) — DRAM roof or compute roof.
        double estimated_peak;
    };

    // --- Primary SLO tiers (canonical config) --------------------------------

    /// Caesar fused χ² (simple substitution). C=29, T=2^20, reps=64.
    /// Peak = alphabet remap roof (CipherHistOnce + rotate), not 896B DRAM.
    static constexpr Tier t1{"T1",
                             "Caesar fused chi2 (simple sub)",
                             static_cast<std::size_t>(Index29::modulus),
                             1048576u, // 1 << 20
                             64u,
                             15.0e9,
                             35.0e9,
                             kAlphabetRemapHistRoofRps};

    /// Filtered multi-key / autokey / dynamic-shift (worst of three).
    /// C=4096, T=2^18, reps=8. Peak = lag remap roof (CTAK production).
    static constexpr Tier t2{"T2",    "Filtered multi-key/autokey/dyn (worst)",
                             4096u,
                             262144u, // 1 << 18
                             8u,      3.0e9,
                             10.0e9,  kLagRemapHistRoofRps};

    /// Caesar bigram + synthetic dictionary validation.
    /// C=512, T=2^18, reps=8. slo_max=0 → display ">=".
    /// Peak = bigram remap roof (Once-B + C×841 Dot); Dict path separate.
    static constexpr Tier t3{"T3",  "Caesar bigram+dict validation", 512u, 262144u, 8u, 1.0e9, 0.0,
                             kBigramRemapHistRoofRps};

    // --- Theory fused-χ² shapes (remap roofs; hard-S0 stays on 896B diary) -----

    /// Fair Caesar-as-bytecode Spec row. Peak = alphabet remap (specialize → S1).
    /// Suite prefers S1 when emit classifies f(x)-only (`specialize_S1`); hard-S0
    /// (autokey / prefer_branch / caps) stays on the interpreter (fails Remap gate).
    static constexpr Tier theory_s0_caesar{"T.theory.caesar_bytecode",
                                          "Theory Caesar fair (specialize S1 when eligible)",
                                          static_cast<std::size_t>(Index29::modulus),
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          kAlphabetRemapHistRoofRps};

    /// S1 mono-LUT. Peak = alphabet remap roof.
    static constexpr Tier theory_s1_lut29{"T.theory.s1_lut29",
                                         "TheoryHistChi2S1 LUT-29 (f(x)-only)",
                                         static_cast<std::size_t>(Index29::modulus),
                                         1048576u,
                                         8u,
                                         15.0e9,
                                         0.0,
                                         kAlphabetRemapHistRoofRps};

    /// S2 linear. Peak = column period-29 remap roof.
    /// Default C=841 (=29²) matches full (b0,b1) grid; suite may use smaller C for Catch2.
    static constexpr Tier theory_s2_linear{"T.theory.s2_linear",
                                          "TheoryHistChi2S2 progressive/bitmask linear",
                                          841u,
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          kColumnRemapHistRoofRps};

    /// Alias name used by microbench progressive row (same peak as S2 linear).
    static constexpr const char* theory_progressive_id = "T.theory.progressive";

    /// S4 AutokeyRing fair Spec. C=28 (lag 1..28). Peak = lag remap roof.
    static constexpr Tier theory_s4_autokey{"T.theory.s4_autokey",
                                           "TheoryHistChi2S4 AutokeyRing (vigenere_lag)",
                                           28u,
                                           1048576u,
                                           8u,
                                           15.0e9,
                                           0.0,
                                           kLagRemapHistRoofRps};

    /// S5 poly keystream fair Spec. C=841 (b0,b1 grid; b2 cycles). Column remap.
    static constexpr Tier theory_s5_poly{"T.theory.s5_poly",
                                        "TheoryHistChi2S5 poly2 (b0+b1·i+b2·i·i)",
                                        841u,
                                        1048576u,
                                        4u,
                                        15.0e9,
                                        0.0,
                                        kColumnRemapHistRoofRps};

    // --- DSL smart customs (hand-written HotLoop → ShapeInline twin) ----------
    // Fair Kernel SLO only (T≥2^20). Campaign wall is never PRIMARY.

    /// Custom Atbash arith (`28-x`) vs catalog `F.atbash` occupancy (C=512).
    /// Peak = shared-cipher compute roof (SM/atomic); DRAM occupancy is diary.
    static constexpr Tier dsl_smart_atbash{"T.dsl_smart.custom_atbash",
                                          "Self-written Atbash arith → ShapeInline twin",
                                          512u,
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          kSharedCipherComputeRoofRps};

    /// Catalog `FamilyChi2Batch` atbash twin (same C/T as custom_atbash).
    static constexpr Tier dsl_smart_compare_atbash{"T.dsl_smart.compare_Fatbash",
                                                  "Catalog F.atbash twin (same C/T)",
                                                  512u,
                                                  1048576u,
                                                  8u,
                                                  15.0e9,
                                                  0.0,
                                                  kSharedCipherComputeRoofRps};

    /// Custom Caesar → ShapeInline remap.
    static constexpr Tier dsl_smart_caesar{"T.dsl_smart.custom_caesar",
                                          "Self-written Caesar → ShapeInline twin",
                                          static_cast<std::size_t>(Index29::modulus),
                                          1048576u,
                                          8u,
                                          15.0e9,
                                          0.0,
                                          kAlphabetRemapHistRoofRps};

    /// Catalog CaesarChi2Batch twin (same C/T as custom_caesar).
    static constexpr Tier dsl_smart_compare_caesar{"T.dsl_smart.compare_caesar",
                                                  "Catalog CaesarChi2Batch twin (same C/T)",
                                                  static_cast<std::size_t>(Index29::modulus),
                                                  1048576u,
                                                  8u,
                                                  15.0e9,
                                                  0.0,
                                                  kAlphabetRemapHistRoofRps};

    /// Custom Affine decrypt → ShapeInline remap; fair C=812 (a=1..28 × b=0..28).
    /// Peak = alphabet remap roof (CipherHistOnce); Affine DRAM class is diary.
    static constexpr Tier dsl_smart_affine{"T.dsl_smart.custom_affine",
                                          "Self-written Affine decrypt → ShapeInline twin",
                                          812u,
                                          1048576u,
                                          4u,
                                          15.0e9,
                                          0.0,
                                          kAlphabetRemapHistRoofRps};

    /// Catalog `FamilyChi2Batch` affine twin (same C/T as custom_affine).
    static constexpr Tier dsl_smart_compare_affine{"T.dsl_smart.compare_Faffine",
                                                  "Catalog F.affine twin (same C/T)",
                                                  812u,
                                                  1048576u,
                                                  4u,
                                                  15.0e9,
                                                  0.0,
                                                  kAlphabetRemapHistRoofRps};

    /// Custom linear `x±(b0+b1·i)` → S2 column remap; fair C=841 (=29²).
    static constexpr Tier dsl_smart_linear{"T.dsl_smart.custom_linear",
                                          "Self-written linear → S2 twin",
                                          841u,
                                          1048576u,
                                          4u,
                                          15.0e9,
                                          0.0,
                                          kColumnRemapHistRoofRps};

    /// TheoryHistChi2S2 twin (same C/T as custom_linear).
    static constexpr Tier dsl_smart_compare_linear{"T.dsl_smart.compare_S2",
                                                  "Catalog S2 linear twin (same C/T)",
                                                  841u,
                                                  1048576u,
                                                  4u,
                                                  15.0e9,
                                                  0.0,
                                                  kColumnRemapHistRoofRps};

    /// Custom autokey vigenere_lag → S4 lag remap; fair C=28 (lag 1..28).
    static constexpr Tier dsl_smart_autokey{"T.dsl_smart.custom_autokey",
                                           "Self-written autokey → S4 AutokeyRing twin",
                                           28u,
                                           1048576u,
                                           8u,
                                           15.0e9,
                                           0.0,
                                           kLagRemapHistRoofRps};

    /// S4 AutokeyRing twin (same C/T as custom_autokey).
    static constexpr Tier dsl_smart_compare_autokey{"T.dsl_smart.compare_S4",
                                                   "Catalog S4 AutokeyRing twin (same C/T)",
                                                   28u,
                                                   1048576u,
                                                   8u,
                                                   15.0e9,
                                                   0.0,
                                                   kLagRemapHistRoofRps};

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

    /// Shape ceilings (logical runes/s). Remap shapes → Remap roofs; hard-S0 /
    /// koan stages → 896B DRAM diary; Atbash/totient → shared-cipher compute roof.
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
        // Shared-cipher occupancy pad (C=512): Done gated on SM/atomic compute roof.
        if (tier == "F.atbash" || tier == "F.totient" || tier == "T.dsl_smart.custom_atbash" ||
            tier == "T.dsl_smart.compare_Fatbash") {
            return shared_cipher_compute_roof_rps();
        }
        // Affine production remap (CipherHistOnce + permute).
        if (tier == "F.affine" || tier == "T.dsl_smart.custom_affine" ||
            tier == "T.dsl_smart.compare_Faffine") {
            return alphabet_remap_hist_roof_rps();
        }
        // Column remap (Vigenère / Beaufort interrupt-free).
        if (tier == "F.vigenere" || tier == "F.beaufort") {
            return column_remap_hist_roof_rps();
        }
        // Legacy decode→hist / multi-stage — physical 1 B/rune DRAM diary.
        if (tier == "C.koan1_fused" || tier == "C.koan1_stages") {
            return dram_roofline_hist_peak();
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
        if (tier == "T.theory.s4_autokey" || tier == "T.theory.s4") {
            return theory_s4_autokey.estimated_peak;
        }
        if (tier == "T.theory.s5_poly" || tier == "T.theory.s5") {
            return theory_s5_poly.estimated_peak;
        }
        if (tier == "T.dsl_smart.custom_caesar" || tier == "T.dsl_smart.compare_caesar") {
            return alphabet_remap_hist_roof_rps();
        }
        if (tier == "T.dsl_smart.custom_linear" || tier == "T.dsl_smart.compare_S2") {
            return column_remap_hist_roof_rps();
        }
        if (tier == "T.dsl_smart.custom_autokey" || tier == "T.dsl_smart.compare_S4") {
            return lag_remap_hist_roof_rps();
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
            tier == "T.theory.s2" || tier == "T.theory.s4_autokey" || tier == "T.theory.s4" ||
            tier == "T.theory.s5_poly" || tier == "T.theory.s5" ||
            tier == "T.dsl_smart.custom_atbash" || tier == "T.dsl_smart.compare_Fatbash" ||
            tier == "T.dsl_smart.custom_caesar" || tier == "T.dsl_smart.compare_caesar" ||
            tier == "T.dsl_smart.custom_affine" || tier == "T.dsl_smart.compare_Faffine" ||
            tier == "T.dsl_smart.custom_linear" || tier == "T.dsl_smart.compare_S2" ||
            tier == "T.dsl_smart.custom_autokey" || tier == "T.dsl_smart.compare_S4") {
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
        if (id == std::string_view{theory_s4_autokey.id} || id == "T.theory.s4") {
            return &theory_s4_autokey;
        }
        if (id == std::string_view{theory_s5_poly.id} || id == "T.theory.s5") {
            return &theory_s5_poly;
        }
        return nullptr;
    }

    /// Look up a dsl_smart Tier by id (nullptr if unknown).
    [[nodiscard]] static constexpr const Tier* find_dsl_smart(std::string_view id) noexcept {
        if (id == std::string_view{dsl_smart_atbash.id}) {
            return &dsl_smart_atbash;
        }
        if (id == std::string_view{dsl_smart_compare_atbash.id}) {
            return &dsl_smart_compare_atbash;
        }
        if (id == std::string_view{dsl_smart_caesar.id}) {
            return &dsl_smart_caesar;
        }
        if (id == std::string_view{dsl_smart_compare_caesar.id}) {
            return &dsl_smart_compare_caesar;
        }
        if (id == std::string_view{dsl_smart_affine.id}) {
            return &dsl_smart_affine;
        }
        if (id == std::string_view{dsl_smart_compare_affine.id}) {
            return &dsl_smart_compare_affine;
        }
        if (id == std::string_view{dsl_smart_linear.id}) {
            return &dsl_smart_linear;
        }
        if (id == std::string_view{dsl_smart_compare_linear.id}) {
            return &dsl_smart_compare_linear;
        }
        if (id == std::string_view{dsl_smart_autokey.id}) {
            return &dsl_smart_autokey;
        }
        if (id == std::string_view{dsl_smart_compare_autokey.id}) {
            return &dsl_smart_compare_autokey;
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
