r"""
Figure 4 — SHAP driver-attribution companion CSV + regenerated figure.

Reads the step-05 SHAP summaries (mean|SHAP| per feature per class, with an
`overall` column), for both model versions A (high-confidence) and B (all
pixels), and produces:

  1. fig4_shap_attribution.{png,pdf,svg}   clean horizontal bar chart of the
     top drivers, A vs B side by side, EXCLUDING the autocorrelation features
     (from_class one-hots and nbr3_* neighbour features) so the figure shows
     the environmental/anthropogenic drivers that carry the science.
  2. fig4_shap_companion.csv               the data-availability table: every
     feature's mean|SHAP| (overall) in A and B, its rank in each, a driver
     group, and whether it was excluded from the figure as autocorrelation.

Run locally:
    python -m src.render_fig4_shap

Reads (from OUT_ATTRIBUTION/"multinomial"):
    shap_summary_A.csv , shap_summary_B.csv      (index = feature, cols = class_* + overall)
Writes (to OUT_FIGURES/"fig4"):
    fig4_shap_attribution.{png,pdf,svg}
    fig4_shap_companion.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ATTRIBUTION, OUT_FIGURES
from src.raster_utils import log

MN = OUT_ATTRIBUTION / "multinomial"
OUT4 = OUT_FIGURES / "fig4"

# How many top (non-autocorrelation) drivers to show
TOP_N = 15

# Human-readable labels + driver groups for known features.
PRETTY = {
    "ndvi_gs_mean": "NDVI (growing-season mean)",
    "ndvi_annual_mean": "NDVI (annual mean)",
    "ndvi_gs_max": "NDVI (growing-season max)",
    "popdens": "Population density",
    "slope": "Slope",
    "elevation": "Elevation",
    "glw_density": "Livestock density (GLW)",
    "phen_sos": "Phenology: start of season",
    "phen_eos": "Phenology: end of season",
    "phen_los": "Phenology: length of season",
    "clim_tn90p_anomaly": "Warm-night frequency anomaly",
    "clim_txn": "Min of daily-max temperature",
    "clim_tx90p": "Warm-day frequency",
    "graz_cattle": "Grazing: cattle",
    "graz_sheep": "Grazing: sheep",
    "graz_total": "Grazing: total",
    "spei12_annual_mean": "SPEI-12 (annual mean)",
    "spei06_annual_mean": "SPEI-6 (annual mean)",
    "aspect": "Aspect",
    "aspect_cos": "Aspect (cos)",
    "aspect_sin": "Aspect (sin)",
    "ndvi_gs_mean_slope5y": "NDVI 5-yr trend",
    "ndvi_annual_mean_slope5y": "NDVI annual 5-yr trend",
    "roads_dist": "Distance to roads",
}

def group_of(feat):
    f = feat.lower()
    if f.startswith("ndvi"): return "Vegetation productivity"
    if f.startswith("phen"): return "Phenology"
    if f.startswith("spei"): return "Drought (SPEI)"
    if f.startswith("clim"): return "Climate extremes"
    if f.startswith("graz") or "glw" in f or "livestock" in f: return "Livestock/grazing"
    if "pop" in f: return "Population"
    if f in ("elevation", "slope", "aspect") or "terrain" in f: return "Terrain"
    if "road" in f: return "Infrastructure"
    if f.startswith("from_") or f.startswith("from_class"): return "Prior class (autocorr.)"
    if f.startswith("nbr"): return "Neighbour class (autocorr.)"
    return "Other"

def is_autocorr(feat):
    f = feat.lower()
    return f.startswith("from_") or f.startswith("from_class") or f.startswith("nbr")

def pretty(feat):
    return PRETTY.get(feat, feat.replace("_", " "))


def _load(label):
    p = MN / f"shap_summary_{label}.csv"
    df = pd.read_csv(p, index_col=0)
    if "overall" not in df.columns:
        df["overall"] = df.mean(axis=1)
    return df["overall"].rename(label)


def main():
    OUT4.mkdir(parents=True, exist_ok=True)
    log.info("Figure 4 — SHAP attribution companion + figure")

    a = _load("A"); b = _load("B")
    allf = pd.concat([a, b], axis=1).fillna(0.0)
    allf["group"] = [group_of(f) for f in allf.index]
    allf["autocorr"] = [is_autocorr(f) for f in allf.index]
    allf["rank_A"] = allf["A"].rank(ascending=False).astype(int)
    allf["rank_B"] = allf["B"].rank(ascending=False).astype(int)

    # ---- companion CSV (all features) ----
    comp = allf.reset_index().rename(columns={"index": "feature"})
    comp["feature_label"] = comp["feature"].map(pretty)
    comp = comp[["feature", "feature_label", "group", "autocorr",
                 "A", "B", "rank_A", "rank_B"]]
    comp = comp.sort_values("A", ascending=False)
    comp.to_csv(OUT4 / "fig4_shap_companion.csv", index=False)
    log.info(f"  ✓ fig4_shap_companion.csv ({len(comp)} features)")

    # ---- figure: top-N non-autocorrelation drivers, A vs B ----
    drivers = allf[~allf["autocorr"]].copy().sort_values("A", ascending=False).head(TOP_N)
    drivers = drivers.iloc[::-1]   # so largest is at top of horizontal bar
    labels = [pretty(f) for f in drivers.index]
    y = np.arange(len(drivers))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y + h/2, drivers["A"], height=h, color="#2c7fb8",
            label="Model A (high-confidence pixels)")
    ax.barh(y - h/2, drivers["B"], height=h, color="#7fcdbb",
            label="Model B (all pixels)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("mean |SHAP| (impact on class prediction)", fontsize=11)
    ax.set_title("Driver attribution of land-cover transitions (SHAP)\n"
                 "environmental & anthropogenic drivers; autocorrelation features excluded",
                 fontsize=12, loc="left")
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT4 / f"fig4_shap_attribution.{ext}",
                    dpi=200 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close()
    log.info("  ✓ fig4_shap_attribution.[png/pdf/svg]")

    # console: the ranking for Results text
    log.info("\n  Top environmental/anthropogenic drivers (mean|SHAP|, model A):")
    for f, row in allf[~allf["autocorr"]].sort_values("A", ascending=False).head(TOP_N).iterrows():
        log.info(f"    {pretty(f):<34} A={row['A']:.3f}  B={row['B']:.3f}  [{row['group']}]")
    log.info("\nUpload fig4_shap_attribution.png + fig4_shap_companion.csv for Results text.")


if __name__ == "__main__":
    main()
