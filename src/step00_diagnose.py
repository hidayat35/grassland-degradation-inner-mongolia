"""
================================================================================
STEP 00 — DIAGNOSTIC: inspect remapped LULC products
================================================================================
Several of your LULC files (CCI, MODIS, WorldCover) appear to have been
pre-remapped to 0-9 codes. We do not know what mapping was used.

This script:
  1. Opens ONE sample file from each product (latest year that overlaps Wang_MP).
  2. Prints unique values + frequencies.
  3. CROSS-TABULATES each product's codes against the known Wang2024_MP classes
     for the same year on the same target grid. This reveals which native code
     corresponds to which conceptual class, so we can build the correct
     harmonization table.

After this you can either:
  - Send me the printed cross-tabulations, or
  - Read them yourself and update PRODUCT_REGISTRY in class_harmonization.py.

Output (to OUT_LOGS):
  - diagnostic_<product>_<year>.csv  (cross-tab as table)
  - diagnostic_report.md              (human-readable summary)

Run
---
$ python -m src.step00_diagnose
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import LULC_PRODUCTS, OUT_LOGS, ensure_dirs
from configs.class_harmonization import UNIFIED_LEGEND, WANG_MP_MAP
from src.raster_utils import log, reproject_to_target, get_aoi_mask


# Products to diagnose + the year to use (must also have Wang_MP for cross-tab)
# MODIS and WorldCover excluded from paper 2 (see class_harmonization.py).
DIAGNOSTIC_TARGETS = [
    ("CCI",        2020),
    ("CLCD",       2020),   # also include CLCD as a sanity check (known mapping)
    ("GLC_FCS30D", 2020),   # and GLC (verify band mapping works)
]


def load_raw_to_target(product: str, year: int) -> np.ndarray | None:
    """Load native (un-remapped) values reprojected to target grid."""
    cfg = LULC_PRODUCTS[product]
    src_nodata = cfg.get("src_nodata")  # CCI uses 32767

    if product == "GLC_FCS30D":
        # 2020 -> second file, band 21 (since file starts at 2000 = band 1)
        src_path = cfg["files"][1]
        band = year - 2000 + 1
    elif product == "CCI":
        # Try each filename pattern in turn
        src_path = None
        if year in cfg["years"]:
            for tmpl in cfg["year_templates"]:
                cand = cfg["folder"] / tmpl.format(year=year)
                if cand.exists():
                    src_path = cand
                    break
        band = 1
    else:
        if year not in cfg["years"]:
            return None
        src_path = cfg["folder"] / cfg["year_template"].format(year=year)
        band = 1

    if src_path is None or not src_path.exists():
        log.warning(f"Missing file for {product} {year}")
        return None
    arr = reproject_to_target(src_path, band=band,
                              resampling=Resampling.nearest,
                              src_nodata=src_nodata,
                              target_dtype="int32")
    return arr


def cross_tab(product_arr: np.ndarray, wang_arr: np.ndarray,
              aoi_mask: np.ndarray) -> pd.DataFrame:
    """Build a confusion-style table: rows = product codes, cols = Wang_MP classes."""
    valid = aoi_mask & (product_arr > 0) & (wang_arr > 0)
    pa = product_arr[valid]
    wa = wang_arr[valid]
    df = pd.DataFrame({"product_code": pa, "wang_code": wa})
    tab = (df.groupby(["product_code", "wang_code"]).size()
             .unstack(fill_value=0))
    # Rename columns with class names. Unknown codes (not in UNIFIED_LEGEND)
    # get a generic label so the cross-tab doesn't crash.
    new_cols = []
    for c in tab.columns:
        if c in UNIFIED_LEGEND:
            new_cols.append(f"Wang_{c}_{UNIFIED_LEGEND[c][0]}")
        else:
            new_cols.append(f"Wang_{c}_UNKNOWN")
    tab.columns = new_cols
    # Add row totals + dominant Wang class per row
    tab["row_total"] = tab.sum(axis=1)
    only_cls = tab.drop(columns=["row_total"])
    tab["dominant_Wang"] = only_cls.idxmax(axis=1)
    tab["dominant_pct"] = (only_cls.max(axis=1) / tab["row_total"] * 100).round(1)
    return tab


def main():
    ensure_dirs()
    log.info("=" * 80)
    log.info("STEP 00 — Diagnostic inspection of remapped LULC products")
    log.info("=" * 80)

    aoi_mask = get_aoi_mask()

    # Reference: Wang2024_MP 2020 in native codes (already in target legend)
    wang_native = load_raw_to_target("Wang2024_MP", 2020)
    if wang_native is None:
        log.error("Wang2024_MP 2020 not found — cannot run cross-tabulations")
        return
    # Wang native = unified codes already (codes 1-13)
    log.info(f"Wang2024_MP 2020 unique values: "
             f"{sorted(set(wang_native[aoi_mask].tolist()))}")

    report_lines = [
        "# Diagnostic Report — Remapped LULC products\n",
        "Generated by `src/step00_diagnose.py`.\n",
        "For each non-Wang product we list the unique native codes inside the\n"
        "Inner Mongolia AOI and cross-tabulate them against the Wang2024_MP\n"
        "classes for the same year (2020), to reveal what each code means.\n",
    ]

    for product, year in DIAGNOSTIC_TARGETS:
        log.info(f"\n── {product} {year} ──")
        arr = load_raw_to_target(product, year)
        if arr is None:
            log.warning(f"  ⚠ skipped {product} {year}")
            report_lines.append(f"\n## {product} {year}\n*Skipped (file missing).*\n")
            continue

        # 1) Unique values + counts
        u, c = np.unique(arr[aoi_mask], return_counts=True)
        total = c.sum()
        freq = pd.DataFrame({
            "native_code": u.tolist(),
            "n_pixels": c.tolist(),
            "pct_of_AOI": (c / total * 100).round(2).tolist(),
        }).sort_values("n_pixels", ascending=False)
        freq_csv = OUT_LOGS / f"diagnostic_{product}_{year}_freq.csv"
        freq.to_csv(freq_csv, index=False)
        log.info(f"  Native code frequencies:\n{freq.to_string(index=False)}")
        log.info(f"  Saved: {freq_csv}")

        # 2) Cross-tab vs Wang_MP
        try:
            ct = cross_tab(arr, wang_native, aoi_mask)
            ct_csv = OUT_LOGS / f"diagnostic_{product}_{year}_vs_Wang.csv"
            ct.to_csv(ct_csv)
            log.info(f"  Cross-tab vs Wang_MP saved: {ct_csv}")
            log.info(f"  Inferred meanings (dominant Wang class per native code):")
            for code, row in ct.iterrows():
                log.info(f"    native {code:>3} -> {row['dominant_Wang']} "
                         f"({row['dominant_pct']}% of {int(row['row_total']):,} px)")
        except Exception as e:
            log.error(f"  cross-tab failed: {e}")
            ct = None

        # Append to markdown report
        report_lines.append(f"\n## {product} {year}\n")
        report_lines.append("**Native code frequencies inside Inner Mongolia AOI:**\n")
        report_lines.append(freq.to_markdown(index=False))
        report_lines.append("\n")
        if ct is not None:
            report_lines.append("**Cross-tabulation vs Wang2024_MP (dominant Wang class for each native code):**\n")
            small = ct[["row_total", "dominant_Wang", "dominant_pct"]].copy()
            small["row_total"] = small["row_total"].astype(int)
            report_lines.append(small.to_markdown())
            report_lines.append("\n")

    # Final report
    report_path = OUT_LOGS / "diagnostic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log.info(f"\n✓ Full report: {report_path}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
