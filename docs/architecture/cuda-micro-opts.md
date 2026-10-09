# CUDA micro-opts (LaunchGeom / shuffle / cp.async)

**Status:** production defaults + A/B policy after Remap / Naive cleanup  
**Hardware reference:** RTX 5070 Ti (sm_120) — same plate as [`cuda-throughput.md`](cuda-throughput.md)

Agents implement and test only; they do **not** create git commits or tags.

---

## 1. `__restrict__` / `__ldg`

Production hotpaths (Once-count, Remap, Theory S1/S2/S4/S5/Shape, Family /
DeepScore legacy decode-hist, single-stream twins) use `__restrict__` on distinct
device pointers and `__ldg` on read-only global loads.

In-place stream twins (`device_in == device_out`) keep `out` **without**
`__restrict__` so aliasing stays well-defined.

---

## 2. Block size + `LaunchGeom`

| Symbol | Value | Role |
|--------|-------|------|
| `LaunchGeom::kDefaultThreads` | **256** | Stream single-kernels; equals `HistFast::threads` |
| `LaunchGeom::AbThreads::{k128,k256,k512}` | A/B only | Quiet-plate experiments |
| `HistFast::threads` | **256** | Fused-hist shared layouts (warps / priv / local) |

Fused-hist **must not** change block size without resizing `HistFast` shared
arrays. Stream kernels call `LaunchGeom::blocks_for` / `threads()`.

`__launch_bounds__` — **not** shipped; add only after ncu occupancy evidence.

---

## 3. Warp-shuffle / `flush_*` reduce — **not shipped**

Prior A/B on this plate:

| Path | Result |
|------|--------|
| `HistFast::add_private_match` (`__match_any_sync`) | LOSE vs fat-64 warp-private |
| Warp-shuffle flush experiments | LOSE / no win vs `flush_private` |

Production remains `HistFast::flush_private` (sum warp rows → one `atomicAdd`
per bin). Revisit only with a quiet-plate win ≥ baseline.

---

## 4. `cp.async` — **Doc-Skip** (fair path)

Fair Kernel SLO uses `T≈2^20` (~1 MiB cipher). ncu `cache_bound` shows cipher
is **L2-resident** (DRAM SoL ~1–3%). Staging via `cp.async` does not help that
regime.

Large-T Once kernels (`T→2^22`) remain candidates for a future A/B; no
production wire until a quiet plate shows ≥ baseline absolute RPS.

---

## 5. Related

- Remap traffic / roofs: [`hist-alphabet-remap.md`](hist-alphabet-remap.md)
- Header: [`launch_geom.hpp`](../../Parcae/Parcae/cuda/launch_geom.hpp)
- Hist primitives: [`hist_fast.hpp`](../../Parcae/Parcae/cuda/hist_fast.hpp)
