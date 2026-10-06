#!/usr/bin/env python3
"""Derive DRAM GB/s / L2 / bytes-per-rune tables from cache_bound *_metrics.csv.

Reads docs/architecture/profiles/cache_bound/{tag}_metrics.csv written by
scripts/cuda/capture_cache_bound.ps1 (ncu --csv --page raw).

Writes ncu_derived.json and fair_derived.json next to the CSVs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs" / "architecture" / "profiles" / "cache_bound"

# Token length used by capture_cache_bound.ps1 -NcuTokens (default).
FALLBACK_CT = {
    "caesar_hist": (29, 262144),
    "atbash_hist": (512, 262144),
    "s1_lut": (29, 262144),
    "s2_linear": (841, 262144),
    "affine_hist": (812, 262144),
}


def unit_to_bytes(val: str, unit: str) -> float:
    v = float(val)
    u = (unit or "").strip().lower()
    if u in ("byte", "bytes", ""):
        return v
    if u == "kbyte":
        return v * 1024.0
    if u == "mbyte":
        return v * (1024.0**2)
    if u == "gbyte":
        return v * (1024.0**3)
    raise ValueError(f"unknown byte unit: {unit!r}")


def unit_to_s(val: str, unit: str) -> float:
    v = float(val)
    u = (unit or "").strip().lower()
    if u == "ns":
        return v * 1e-9
    if u == "us":
        return v * 1e-6
    if u == "ms":
        return v * 1e-3
    if u == "s":
        return v
    raise ValueError(f"unknown time unit: {unit!r}")


def load_csv(path: Path) -> tuple[dict[str, int], list[str], list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise RuntimeError(f"{path}: expected header+units+values, got {len(rows)} rows")
    hdr, units, vals = rows[0], rows[1], rows[2]
    idx = {h: i for i, h in enumerate(hdr)}
    return idx, units, vals


def main() -> int:
    if not ROOT.is_dir():
        raise SystemExit(f"missing {ROOT} — run capture_cache_bound.ps1 first")

    out: list[dict[str, float | int | str]] = []
    for tag, (_c0, tokens) in FALLBACK_CT.items():
        path = ROOT / f"{tag}_metrics.csv"
        if not path.is_file():
            print(f"skip missing {path.name}")
            continue
        idx, units, vals = load_csv(path)

        def get(name: str) -> tuple[str, str]:
            i = idx[name]
            return vals[i], units[i]

        b_s, b_u = get("dram__bytes.sum")
        d_s, d_u = get("gpu__time_duration.sum")
        bytes_ = unit_to_bytes(b_s, b_u)
        dur = unit_to_s(d_s, d_u)
        cand = int(float(vals[idx["launch__grid_dim_x"]]))
        tiles = int(float(vals[idx["launch__grid_dim_y"]]))
        dram_gbs = bytes_ / dur / 1e9
        row = {
            "tag": tag,
            "C": cand,
            "grid_y": tiles,
            "T": tokens,
            "dur_us": dur * 1e6,
            "dram_bytes": bytes_,
            "dram_gbs": dram_gbs,
            "pct_896": 100.0 * dram_gbs / 896.0,
            "sm_pct": float(vals[idx["sm__throughput.avg.pct_of_peak_sustained_elapsed"]]),
            "dram_pct": float(vals[idx["dram__throughput.avg.pct_of_peak_sustained_elapsed"]]),
            "l2_hit_pct": float(vals[idx["lts__t_sector_hit_rate.pct"]]),
            "bytes_rune": bytes_ / (cand * tokens),
            "rps": (cand * tokens) / dur,
        }
        out.append(row)
        print(
            f"{tag}: DRAM={dram_gbs:.2f} GB/s  L2hit={row['l2_hit_pct']:.1f}%  "
            f"bytes/rune={row['bytes_rune']:.5f}  SM={row['sm_pct']:.1f}%"
        )

    (ROOT / "ncu_derived.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    fair: dict[str, dict[str, float]] = {}
    for name in ("theory_fair.json", "catalog_slo_extended.json"):
        p = ROOT / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        for r in data.get("result", {}).get("rows", []):
            fair[r["name"]] = {
                "rps": float(r["runes_per_sec"]),
                "pct": float(r["percent_peak"]),
            }
    (ROOT / "fair_derived.json").write_text(json.dumps(fair, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'ncu_derived.json'}")
    print(f"wrote {ROOT / 'fair_derived.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
