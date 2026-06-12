r"""
================================================================================
TRANSITION DIAGNOSTIC — Step 04 features → step 05 transition design
================================================================================
Reads all 5 yearly parquet files from step 04 and:

  1. Builds the cross-tabulation of (from_class, to_class) for every consecutive
     anchor pair (2000→2005, 2005→2010, 2010→2015, 2015→2020).
  2. Reports per-window counts AND a combined "all-windows" matrix.
  3. Filters transitions by minimum sample size (≥5,000 pixels combined) and
     significance (≥0.5% of inside-AOI pixels) — these are the candidates that
     are worth modeling in step 05.
  4. Identifies transitions concentrated in specific time windows (could be
     candidates for time-window-specific attribution).
  5. Writes outputs (CSV + Markdown + console summary) to:
        D:\paper2_outputs\05_attribution\transitions_diagnostic.md
        D:\paper2_outputs\05_attribution\transitions_*.csv

Run from project root:
    python transitions_diagnostic.py
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow this script to be run from anywhere
HERE = Path(__file__).resolve().parent
ROOT_CANDIDATES = [HERE, HERE / "paper2" / "files", HERE.parent]
for cand in ROOT_CANDIDATES:
    if (cand / "configs" / "paths.py").exists():
        sys.path.insert(0, str(cand))
        break

from configs.paths import OUT_FEATURES, ANCHOR_YEARS, OUT_ROOT, ensure_dirs
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import log


OUT_ATTRIB = OUT_ROOT / "05_attribution"
OUT_ATTRIB.mkdir(parents=True, exist_ok=True)


def class_name(code: int) -> str:
    """Return class name from unified legend, with fallback."""
    return UNIFIED_LEGEND.get(int(code), (f"class{code}",))[0]


def load_year(year: int) -> pd.DataFrame:
    """Load only the columns we need for transition counting."""
    p = OUT_FEATURES / f"features_{year}.parquet"
    if not p.exists():
        return None
    cols = ["row", "col", "lulc_class", "conf_joint", "conf_l1", "conf_l2",
            "agreement_count"]
    return pd.read_parquet(p, columns=cols)


def cross_tab_year_pair(df_from: pd.DataFrame, df_to: pd.DataFrame,
                        min_confidence: int = 0) -> pd.DataFrame:
    """Merge two yearly dataframes on (row, col) and compute the transition
    contingency table.

    Optionally filter to high-confidence pixels (default: keep all)."""
    merged = df_from.merge(
        df_to, on=["row", "col"], suffixes=("_from", "_to")
    )
    if min_confidence > 0:
        merged = merged[
            (merged["conf_joint_from"] >= min_confidence) &
            (merged["conf_joint_to"]   >= min_confidence)
        ]
    # Both must be valid (>0)
    merged = merged[(merged["lulc_class_from"] > 0) & (merged["lulc_class_to"] > 0)]
    ct = (merged.groupby(["lulc_class_from", "lulc_class_to"])
                .size()
                .reset_index(name="n_pixels"))
    ct["from"] = ct["lulc_class_from"].astype(int)
    ct["to"]   = ct["lulc_class_to"].astype(int)
    ct["from_name"] = ct["from"].map(class_name)
    ct["to_name"]   = ct["to"].map(class_name)
    ct = ct[["from", "from_name", "to", "to_name", "n_pixels"]]
    return ct, len(merged)


def main():
    ensure_dirs()
    log.info("=" * 80)
    log.info("TRANSITION DIAGNOSTIC")
    log.info("=" * 80)

    # Load all parquets
    by_year = {}
    for y in ANCHOR_YEARS:
        df = load_year(y)
        if df is None:
            log.warning(f"  missing parquet for {y}")
            continue
        by_year[y] = df
        log.info(f"  {y}: {len(df):,} rows")
    if len(by_year) < 2:
        log.error("Need at least 2 anchor years")
        return

    years = sorted(by_year.keys())
    pairs = [(years[i], years[i+1]) for i in range(len(years) - 1)]

    # ----- Per-window cross-tabs -----
    all_tabs = {}
    pair_totals = {}
    for y0, y1 in pairs:
        ct, n_total = cross_tab_year_pair(by_year[y0], by_year[y1])
        ct["window"] = f"{y0}_{y1}"
        ct["pct_of_window"] = 100 * ct["n_pixels"] / n_total
        all_tabs[(y0, y1)] = ct
        pair_totals[(y0, y1)] = n_total
        out_csv = OUT_ATTRIB / f"transitions_{y0}_{y1}.csv"
        ct.to_csv(out_csv, index=False)
        log.info(f"  {y0} → {y1}: {n_total:,} aligned pixels; "
                 f"{(ct[ct['from'] != ct['to']]['n_pixels'].sum()):,} changed "
                 f"({100*(ct[ct['from'] != ct['to']]['n_pixels'].sum())/n_total:.2f}%)")

    # ----- Combined cross-tab (all 4 windows stacked) -----
    combined = pd.concat(list(all_tabs.values()), ignore_index=True)
    agg = (combined.groupby(["from", "from_name", "to", "to_name"])
                   .agg(n_pixels_total=("n_pixels", "sum"))
                   .reset_index())
    total_pixel_windows = combined["n_pixels"].sum()
    agg["pct_of_combined"] = 100 * agg["n_pixels_total"] / total_pixel_windows
    agg = agg.sort_values("n_pixels_total", ascending=False)

    out_csv = OUT_ATTRIB / "transitions_combined.csv"
    agg.to_csv(out_csv, index=False)

    # ----- Stable (self) vs. changed -----
    stable = agg[agg["from"] == agg["to"]]
    changed = agg[agg["from"] != agg["to"]]
    log.info(f"\n  Stable (no-change) pixel-windows: {stable['n_pixels_total'].sum():,} "
             f"({100*stable['n_pixels_total'].sum()/total_pixel_windows:.2f}%)")
    log.info(f"  Changed pixel-windows:            {changed['n_pixels_total'].sum():,} "
             f"({100*changed['n_pixels_total'].sum()/total_pixel_windows:.2f}%)")
    log.info(f"  Total pixel-windows:              {total_pixel_windows:,}")

    # ----- Modellable transitions: ≥5,000 pixels AND ≥0.5% of combined ----
    candidates = changed[(changed["n_pixels_total"] >= 5_000)
                         & (changed["pct_of_combined"] >= 0.05)].copy()
    candidates = candidates.sort_values("n_pixels_total", ascending=False)

    log.info(f"\n  ── Modellable transitions (≥5,000 pixels, ≥0.05% combined) ──")
    log.info(f"  {len(candidates)} transitions meet both thresholds:\n")
    for _, row in candidates.iterrows():
        log.info(f"    {row['from']:>2} {row['from_name']:<13} → "
                 f"{row['to']:>2} {row['to_name']:<13}  "
                 f"{row['n_pixels_total']:>10,}  ({row['pct_of_combined']:5.2f}%)")

    # ----- Per-window breakdown of the top transitions -----
    log.info(f"\n  ── Per-window breakdown of top 10 candidate transitions ──")
    top10 = candidates.head(10)
    rows_md = []
    for _, t in top10.iterrows():
        fc, tc = int(t["from"]), int(t["to"])
        per_window = []
        for (y0, y1) in pairs:
            ct = all_tabs[(y0, y1)]
            row = ct[(ct["from"] == fc) & (ct["to"] == tc)]
            n = int(row["n_pixels"].values[0]) if len(row) else 0
            per_window.append(n)
        rows_md.append({
            "from": fc, "from_name": class_name(fc),
            "to":   tc, "to_name":   class_name(tc),
            "total": int(t["n_pixels_total"]),
            **{f"{y0}→{y1}": per_window[i] for i, (y0, y1) in enumerate(pairs)},
        })
    per_window_df = pd.DataFrame(rows_md)
    per_window_df.to_csv(OUT_ATTRIB / "transitions_top10_per_window.csv", index=False)
    log.info("\n" + per_window_df.to_string(index=False))

    # Identify time-concentrated transitions
    log.info("\n  ── Time-concentration analysis ──")
    log.info("    A transition is 'concentrated' if >60% of its pixel-windows")
    log.info("    occurred in ONE of the 4 anchor windows.")
    for r in rows_md:
        windows = {k: v for k, v in r.items() if "→" in k}
        max_w = max(windows.values())
        max_w_name = max(windows, key=windows.get)
        if max_w > 0.6 * r["total"]:
            log.info(f"    ⚡ {r['from']:>2} {r['from_name']:<12} → "
                     f"{r['to']:>2} {r['to_name']:<12}  "
                     f"is {100*max_w/r['total']:.0f}% concentrated in {max_w_name}")

    # ----- Confidence filter sensitivity ----
    log.info(f"\n  ── Sensitivity: how do counts change if we require conf≥80? ──")
    for y0, y1 in pairs:
        ct, n_total = cross_tab_year_pair(by_year[y0], by_year[y1], min_confidence=80)
        n_changed = ct[ct["from"] != ct["to"]]["n_pixels"].sum()
        log.info(f"  {y0}→{y1}: confidence≥80 keeps {n_total:,} aligned pixels "
                 f"({100*n_total/pair_totals[(y0,y1)]:.1f}% of all), "
                 f"{n_changed:,} changed ({100*n_changed/n_total:.2f}%)")

    # ----- Compact Markdown report ----
    md = []
    md.append("# Transition diagnostic — IM 2000-2020\n")
    md.append("## 1. Stable vs changed pixel-windows\n")
    md.append(f"- Stable (no class change): {stable['n_pixels_total'].sum():,} "
              f"({100*stable['n_pixels_total'].sum()/total_pixel_windows:.2f}%)")
    md.append(f"- Changed:                  {changed['n_pixels_total'].sum():,} "
              f"({100*changed['n_pixels_total'].sum()/total_pixel_windows:.2f}%)")
    md.append(f"- Total pixel-windows:      {total_pixel_windows:,}\n")

    md.append("## 2. Per-window change rates\n")
    md.append("| Window | Aligned pixels | Changed | % changed |")
    md.append("|--------|---------------|---------|-----------|")
    for (y0, y1) in pairs:
        ct = all_tabs[(y0, y1)]
        n_total = pair_totals[(y0, y1)]
        n_chg = ct[ct["from"] != ct["to"]]["n_pixels"].sum()
        md.append(f"| {y0}→{y1} | {n_total:,} | {n_chg:,} | {100*n_chg/n_total:.2f}% |")
    md.append("")

    md.append(f"## 3. Modellable transitions (≥5,000 pixels, ≥0.05% combined)\n")
    md.append("These are the candidates for step 05 XGBoost+SHAP attribution.\n")
    md.append("| from | name | to | name | total pixels | % |")
    md.append("|-----:|------|---:|------|-------------:|--:|")
    for _, r in candidates.iterrows():
        md.append(f"| {int(r['from'])} | {r['from_name']} | {int(r['to'])} | "
                  f"{r['to_name']} | {int(r['n_pixels_total']):,} | "
                  f"{r['pct_of_combined']:.2f}% |")
    md.append("")

    md.append("## 4. Per-window breakdown of top transitions\n")
    md.append(per_window_df.to_markdown(index=False))
    md.append("")

    out_md = OUT_ATTRIB / "transitions_diagnostic.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"\n✓ Markdown report: {out_md}")
    log.info(f"✓ Per-window CSVs: {OUT_ATTRIB}/transitions_*.csv")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
