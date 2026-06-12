r"""
make_transition_matrix.py — full 14x14 land-cover transition matrix (2000 -> 2020)
for Supplementary Table S8.

The main text reports the principal transition flows; this produces the complete
origin x destination pixel-count matrix from the fused maps, exactly like
compute_change_rates.py but keeping every from->to pair.

Run locally:
    python -m src.make_transition_matrix

Reads OUT_FUSED/fused_2000.tif and fused_2020.tif.
Writes OUT_FIGURES/"Figure 3"/:
    transition_matrix_2000_2020.csv         (rows=from class, cols=to class, pixel counts)
    transition_matrix_2000_2020_pct.csv     (row-normalised %: of each origin class,
                                             where did it go)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_FUSED, OUT_FIGURES
from src.raster_utils import log

OUT3 = OUT_FIGURES / "Figure 3"
FROM_YEAR, TO_YEAR = 2000, 2020

CLASS_NAMES = {1: "Forest", 2: "Shrub", 3: "Meadow steppe", 4: "Real steppe",
               5: "Dry steppe", 6: "Desert steppe", 7: "Wetland", 8: "Water",
               9: "Cropland", 10: "Built-up", 11: "Bare", 12: "Desert",
               13: "Sand", 14: "Ice"}


def _read(year):
    p = OUT_FUSED / f"fused_{year}.tif"
    if not p.exists():
        raise SystemExit(f"missing fused map: {p}")
    with rasterio.open(p) as s:
        return s.read(1)


def main():
    OUT3.mkdir(parents=True, exist_ok=True)
    log.info(f"Computing full transition matrix {FROM_YEAR} -> {TO_YEAR}")

    a = _read(FROM_YEAR)
    b = _read(TO_YEAR)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    valid = (a > 0) & (b > 0)
    af = a[valid].astype(int)
    bf = b[valid].astype(int)
    log.info(f"  valid pixels in both years: {valid.sum():,}")

    classes = list(range(1, 15))
    # counts matrix
    cm = pd.crosstab(pd.Series(af, name="from"),
                     pd.Series(bf, name="to"))
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    cm.index = [CLASS_NAMES[c] for c in classes]
    cm.columns = [CLASS_NAMES[c] for c in classes]
    # drop all-zero rows/cols (classes absent in both) for readability
    nz_rows = cm.sum(axis=1) > 0
    nz_cols = cm.sum(axis=0) > 0
    keep = nz_rows | nz_cols
    cm = cm.loc[keep[keep].index.intersection(cm.index),
                keep[keep].index.intersection(cm.columns)]
    cm["Total_from"] = cm.sum(axis=1)
    cm.to_csv(OUT3 / "transition_matrix_2000_2020.csv")
    log.info(f"  ✓ transition_matrix_2000_2020.csv written to {OUT3}")

    # row-normalised percentages (of each origin class, fate)
    counts = cm.drop(columns=["Total_from"])
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct = pct.round(2)
    pct.to_csv(OUT3 / "transition_matrix_2000_2020_pct.csv")
    log.info(f"  ✓ transition_matrix_2000_2020_pct.csv written to {OUT3}")

    # quick on-screen check: diagonal (persistence) and top off-diagonal flows
    diag = {c: int(counts.loc[c, c]) if c in counts.columns else 0 for c in counts.index}
    log.info("  persistence (stayed same class):")
    for c, n in sorted(diag.items(), key=lambda x: -x[1])[:6]:
        log.info(f"     {c:16s} {n:,}")
    log.info("\nUpload transition_matrix_2000_2020.csv so it can be inserted as "
             "Supplementary Table S8.")


if __name__ == "__main__":
    main()
