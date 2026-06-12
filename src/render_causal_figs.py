r"""
Re-render the step 06 CAUSAL maps (per-pixel GCCM) WITH the Inner Mongolia
boundary overlaid, write GeoTIFFs, and emit zone-level CSV companions for the
manuscript Results text and data availability.

Run locally (needs the shapefile at AOI_SHP + the step06 outputs + fused map):
    python -m src.render_causal_figs

Reads (from OUT_CAUSAL = D:\paper2_outputs\06_causal):
    gccm_causal_map_grazing_total.tif
    gccm_causal_map_spei12.tif
    gccm_causal_map_popdens.tif
    gccm_causal_map_clim_tx90p.tif
Reads (from OUT_FUSED): fused_2020.tif   (for AOI mask + ecological zonation)

Writes (to OUT_CAUSAL):
    fig_causal_dominance.{png,pdf,svg,tif}        dominant-driver map (smoothed)
    fig_causal_maps_4panel.{png,pdf,svg}          per-driver strength panels
    causal_map_zone_summary.csv                   per-zone mean rho per driver +
                                                  dominance fractions (Results data)
    causal_dominance.tif                          dominance class raster (data avail)

Method note (for the caption): per-pixel CCM rho is gap-filled within a local
7-km neighbourhood where >=5 valid estimates exist and lightly median-smoothed
FOR VISUALIZATION ONLY; quantitative causal inference uses the regional Granger
+ CCM analysis (Fig. causal heatmap). Maps are exploratory spatial illustration.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ROOT, OUT_FUSED, TARGET_CRS
from src.raster_utils import log
from src.mapfig import render_map, render_categorical_map, _write_geotiff

OUT_CAUSAL = OUT_ROOT / "06_causal"

# driver -> display title; order defines dominance-map class codes (1..4)
DRIVERS = [
    ("spei12",        "SPEI-12 drought → NDVI",     "#2c7fb8"),
    ("clim_tx90p",    "Warm-day extremes → NDVI",   "#d7301f"),
    ("grazing_total", "Grazing intensity → NDVI",   "#762a83"),
    ("popdens",       "Population density → NDVI",  "#1a9850"),
]
DOM_LABELS = {1: "SPEI-12 drought", 2: "Warm-day extremes",
              3: "Grazing intensity", 4: "Population density"}

# ecological zones from fused 2020 (same mapping as step06 build_zones)
ZONE_MAP = {1: [1, 3], 2: [4], 3: [5], 4: [6], 5: [9],
            6: [13], 7: [11, 12]}
ZONE_NAMES = {1: "Forest/Meadow", 2: "Real steppe", 3: "Dry steppe",
              4: "Desert steppe", 5: "Cropland", 6: "Sandy land",
              7: "Desert/Barren"}


def _read(p):
    with rasterio.open(p) as s:
        return s.read(1), s.transform


def neighborhood_fill(arr, radius=3, min_valid=5):
    """Fill NaNs with local mean of valid values within `radius`, only where
    >= min_valid valid neighbours exist (never fabricates in empty regions)."""
    valid = np.isfinite(arr)
    a0 = np.where(valid, arr, 0.0).astype(np.float32)
    vm = valid.astype(np.float32)
    size = 2 * radius + 1
    ssum = uniform_filter(a0, size=size, mode="constant") * (size * size)
    scnt = uniform_filter(vm, size=size, mode="constant") * (size * size)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_mean = ssum / scnt
    filled = arr.copy()
    fill = (~valid) & (scnt >= min_valid)
    filled[fill] = local_mean[fill]
    return filled


def smooth(arr, aoi):
    a = arr.copy(); a[~aoi] = np.nan
    f = neighborhood_fill(a, radius=3, min_valid=5)
    fm = np.isfinite(f)
    med = median_filter(np.where(fm, f, 0.0), size=3, mode="constant")
    out = np.where(fm, med, np.nan)
    out[~aoi] = np.nan
    return out


def build_zones(fused):
    zones = np.zeros_like(fused, dtype=np.uint8)
    for z, codes in ZONE_MAP.items():
        zones[np.isin(fused, codes)] = z
    return zones


def main():
    log.info("Re-rendering causal maps with boundary + zone CSVs")
    fused, _ = _read(OUT_FUSED / "fused_2020.tif")
    aoi = fused > 0
    zones = build_zones(fused)

    # load + smooth each driver map
    smoothed = {}
    tf = None
    for name, _, _ in DRIVERS:
        arr, tf = _read(OUT_CAUSAL / f"gccm_causal_map_{name}.tif")
        smoothed[name] = smooth(arr, aoi)

    # ---- 4-panel per-driver strength (single figure, multi-format) ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    from src.mapfig import _overlay_boundary
    for ax, (name, title, _) in zip(axes.flat, DRIVERS):
        masked = np.ma.masked_invalid(smoothed[name])
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       interpolation="bilinear")
        _overlay_boundary(ax, tf, smoothed[name].shape, lw=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        cb = plt.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label("CCM ρ (causal strength)", fontsize=9)
    plt.suptitle("Per-pixel causal strength of drivers on NDVI "
                 "(gap-filled & smoothed for visualization)", fontsize=15, y=0.99)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT_CAUSAL / f"fig_causal_maps_4panel.{ext}",
                    dpi=180 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    # GeoTIFFs for each driver panel (data availability)
    for name, _, _ in DRIVERS:
        _write_geotiff(smoothed[name], tf, OUT_CAUSAL / f"gccm_causal_map_{name}_smoothed.tif")
    log.info("  ✓ fig_causal_maps_4panel.[png/pdf/svg] + per-driver smoothed tifs")

    # ---- dominance map: argmax over smoothed driver maps (positive only) ----
    stack = np.stack([smoothed[n] for n, _, _ in DRIVERS], axis=0)  # (4,H,W)
    valid_any = np.isfinite(stack).any(axis=0)
    with np.errstate(invalid="ignore"):
        argmax = np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
    winner = np.take_along_axis(stack, argmax[None], axis=0)[0]
    dom = np.zeros(stack.shape[1:], dtype=np.uint8)
    good = valid_any & (winner > 0.1) & aoi
    dom[good] = (argmax[good] + 1).astype(np.uint8)

    legend = {c: (DOM_LABELS[c], col)
              for c, (_, _, col) in zip([1, 2, 3, 4], DRIVERS)}
    n_assigned = int((dom > 0).sum())
    render_categorical_map(
        array=dom.astype(np.float32), transform=tf,
        title="Dominant causal driver of vegetation productivity (per pixel)",
        out_stem=OUT_CAUSAL / "fig_causal_dominance",
        legend=legend,
        subtitle=f"smoothed per-pixel CCM; {n_assigned:,} pixels with a positive "
                 f"causal driver (ρ>0.1) — exploratory spatial illustration",
        write_tif=True,
    )
    log.info("  ✓ fig_causal_dominance.[png/pdf/svg/tif]")

    # ---- zone-level CSV: mean rho per driver per zone + dominance fractions ----
    rows = []
    for z in sorted(ZONE_NAMES):
        zmask = (zones == z) & aoi
        npix = int(zmask.sum())
        if npix == 0:
            continue
        row = {"zone": ZONE_NAMES[z], "n_pixels": npix}
        for name, _, _ in DRIVERS:
            vals = smoothed[name][zmask]
            vals = vals[np.isfinite(vals)]
            row[f"mean_rho_{name}"] = round(float(vals.mean()), 4) if vals.size else np.nan
        # dominance fractions within the zone
        domz = dom[zmask]
        assigned = domz[domz > 0]
        for c in [1, 2, 3, 4]:
            frac = 100.0 * (assigned == c).sum() / max(assigned.size, 1)
            row[f"dom_pct_{DOM_LABELS[c].split()[0].lower()}"] = round(frac, 1)
        row["n_dominant_assigned"] = int(assigned.size)
        rows.append(row)
    zdf = pd.DataFrame(rows)
    zdf.to_csv(OUT_CAUSAL / "causal_map_zone_summary.csv", index=False)
    log.info(f"  ✓ causal_map_zone_summary.csv ({len(zdf)} zones)")
    log.info("\nUpload causal_map_zone_summary.csv + the two figure PNGs for Results text.")


if __name__ == "__main__":
    main()
