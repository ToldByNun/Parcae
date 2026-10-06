#ifndef HIST_TILE_CAP_HPP
#define HIST_TILE_CAP_HPP

#include "hist_fast.hpp"

#include <cstddef>

/// Optional `grid.y` clamp overrides for fused-hist façades.
///
/// `0` = production default (`HistFast::production_tile_cap`). Positive values
/// clamp via `HistFast::tiles_for_capped` (A/B roof sweeps). Process-wide
/// mutable state — same host-thread contract as the former Caesar anon-namespace
/// override (tests mutate a slot, then restore `0`).
///
/// Slot ids are stable integers so later façades (S1/S2/Atbash/Affine) can wire
/// without reshuffling. No C++ namespaces.
class HistTileCap {
public:
    /// Caesar catalog / ShapeInline twin (`CaesarChi2Batch`).
    static constexpr int kCaesar = 0;
    /// Reserved: `TheoryHistChi2S1` / fair specialize.
    static constexpr int kS1 = 1;
    /// Reserved: `TheoryHistChi2S2` linear keystream.
    static constexpr int kS2 = 2;
    /// Reserved: catalog / ShapeInline Atbash.
    static constexpr int kAtbash = 3;
    /// Reserved: catalog / ShapeInline Affine.
    static constexpr int kAffine = 4;
    static constexpr int kSlotCount = 5;

    /// `cap < 0` treated as `0` (production). Unknown `slot` is a no-op.
    static void set(int slot, int cap) noexcept {
        if (!valid_slot(slot)) {
            return;
        }
        caps_[slot] = cap < 0 ? 0 : cap;
    }

    /// Override value (`0` = production). Unknown slot → `0`.
    [[nodiscard]] static int get(int slot) noexcept {
        if (!valid_slot(slot)) {
            return 0;
        }
        return caps_[slot];
    }

    /// Effective clamp passed to `HistFast::tiles_for_capped`.
    [[nodiscard]] static int effective(int slot) noexcept {
        const int override_cap = get(slot);
        return override_cap > 0 ? override_cap : HistFast::production_tile_cap;
    }

    /// `grid.y` for `token_count` under this slot's effective clamp.
    [[nodiscard]] static int tiles_for(int slot, std::size_t token_count) {
        return HistFast::tiles_for_capped(token_count, effective(slot));
    }

private:
    HistTileCap() = delete;

    [[nodiscard]] static bool valid_slot(int slot) noexcept {
        return slot >= 0 && slot < kSlotCount;
    }

    /// C++17 inline static — one definition across TUs (replaces anon-namespace).
    static inline int caps_[kSlotCount] = {};
};

#endif // HIST_TILE_CAP_HPP
