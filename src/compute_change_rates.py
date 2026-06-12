r"""
Compute the per-interval land-cover CHANGE RATE directly from the fused maps,
so the values shown in Figure 3b (the change-rate bar panel) are reproducible
from data rather than hardcoded.

Change rate for interval (t -> t+5) is defined as:
    (number of pixels whose fused class differs between year t and t+5)
    / (number of pixels valid — class > 0 — in BOTH years)
expressed as a percentage. This is the standard "fraction of the landscape
that changed class" metric.

Run locally (needs all five fused maps):
    python -m src.compute_change_rates

Reads (from OUT_FUSED = D:\paper2_outputs\02_fused_lulc):
    fused_2000.tif, fused_2005.tif, fused_2010.tif, fused_2015.tif, fused_2020.tif
Writes (to OUT_FIGURES/"Figure 3"):
    change_rate_per_interval.csv     interval, n_valid_both, n_changed, change_rate_pct

The script prints the four percentages so they can be checked directly against
the values currently drawn in Figure 3b (34.5 / 27.9 / 30.5 / 21.9).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_FUSED, OUT_FIGURES
from src.raster_utils import log

ANCHOR_YEARS = [2000, 2005, 2010, 2015, 2020]
OUT3 = OUT_FIGURES / "Figure 3"


def _read(year):
    p = OUT_FUSED / f"fused_{year}.tif"
    if not p.exists():
        raise SystemExit(f"missing fused map: {p}")
    with rasterio.open(p) as s:
        return s.read(1)


def main():
    OUT3.mkdir(parents=True, exist_ok=True)
    log.info("Computing per-interval change rates from fused maps")

    maps = {y: _read(y) for y in ANCHOR_YEARS}
    # sanity: all same shape
    shapes = {m.shape for m in maps.values()}
    if len(shapes) != 1:
        raise SystemExit(f"fused maps have different shapes: {shapes}")

    rows = []
    for t0, t1 in zip(ANCHOR_YEARS[:-1], ANCHOR_YEARS[1:]):
        a = maps[t0]; b = maps[t1]
        valid = (a > 0) & (b > 0)          # class present in both years
        n_valid = int(valid.sum())
        changed = valid & (a != b)
        n_changed = int(changed.sum())
        pct = 100.0 * n_changed / n_valid if n_valid else float("nan")
        rows.append({
            "interval": f"{t0}-{t1}",
            "n_valid_both": n_valid,
            "n_changed": n_changed,
            "change_rate_pct": round(pct, 2),
        })
        log.info(f"  {t0}-{t1}: {pct:6.2f}%  ({n_changed:,} of {n_valid:,} valid px changed)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT3 / "change_rate_per_interval.csv", index=False)
    log.info(f"\n  ✓ change_rate_per_interval.csv written to {OUT3}")

    # Print a direct comparison against the values currently in Figure 3b
    drawn = {"2000-2005": 34.5, "2005-2010": 27.9, "2010-2015": 30.5, "2015-2020": 21.9}
    log.info("\n  Comparison with values currently drawn in Figure 3b:")
    all_match = True
    for _, r in df.iterrows():
        d = drawn.get(r["interval"])
        if d is None:
            continue
        match = abs(r["change_rate_pct"] - d) < 0.1
        all_match &= match
        flag = "✓" if match else "✗ MISMATCH"
        log.info(f"    {r['interval']}: computed {r['change_rate_pct']:.2f}%  vs  drawn {d}%   {flag}")
    if all_match:
        log.info("\n  All four match — Figure 3b values are confirmed from data.")
    else:
        log.info("\n  Some values differ — update RATES in make_fig3_variants.py "
                 "(and the §3.2 text) to the computed values above.")
    log.info("\nUpload change_rate_per_interval.csv to verify.")


if __name__ == "__main__":
    main()
