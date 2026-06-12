r"""
Render Figure 2 — hierarchical Bayesian fused land cover + uncertainty (2020),
WITH the Inner Mongolia boundary overlaid, GeoTIFF exports, and a CSV companion.

Run locally (needs AOI_SHP + the step02 fusion outputs):
    python -m src.render_fig2

Reads (from OUT_FUSED = D:\paper2_outputs\02_fused):
    fused_2020.tif            uint8 14-class fused map (0 = nodata)
    confidence_2020.tif       uint8 0-100 super-class (level-1) posterior conf.
    confidence_l2_2020.tif    uint8 0-100 sub-class (level-2) posterior conf.
    agreement_2020.tif        uint8 0-8 count of products agreeing

Writes (to OUT_FIGURES / "fig2"):
    fig2_fused_uncertainty.{png,pdf,svg}   the 4-panel figure (boundary overlaid)
    fig2_fused_landcover.tif               coloured? no -> raw class codes GeoTIFF
    (panels b/c/d GeoTIFFs are the input tifs themselves; copied for convenience)
    fig2_class_composition.csv             per-class area% + mean conf + agreement

Caption note (for the manuscript):
  Panels (b) and (c) use different confidence definitions: (b) is the posterior
  probability of the assigned SUPER-class (Level 1), (c) of the assigned
  SUB-class (Level 2). Classes without a sub-division (Forest, Cropland, Water,
  Built-up, Wetland) take sub-class confidence = 100% by construction; the
  informative sub-class uncertainty is therefore concentrated in the grassland
  and bare/desert/sand families, where it is genuinely lower (e.g. desert 70%,
  bare land 73%). Year shown is 2020; the fusion is produced for all anchor
  years 2000-2020.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_FUSED, OUT_FIGURES, TARGET_CRS
from src.raster_utils import log
from src.mapfig import boundary_in_albers, _overlay_boundary, _write_geotiff

OUT2 = OUT_FIGURES / "fig2"

# Palette 2 (locked): ecological aridity gradient
CLASS = {
    1:  ("Forest",         "#0b4d2c"),
    2:  ("Shrub",          "#3f9b5e"),
    3:  ("Meadow steppe",  "#5cab5a"),
    4:  ("Typical steppe", "#94c465"),
    5:  ("Dry steppe",     "#c2a23a"),
    6:  ("Desert steppe",  "#9c7a30"),
    7:  ("Wetland",        "#4eb3d3"),
    8:  ("Water",          "#2b6cb0"),
    9:  ("Cropland",       "#e8722a"),
    10: ("Built-up",       "#d11141"),
    11: ("Bare land",      "#b0a08f"),
    12: ("Desert",         "#cda15a"),
    13: ("Sand",           "#efe0a8"),
    14: ("Ice/snow",       "#e8f4f8"),
}


def _read(name):
    with rasterio.open(OUT_FUSED / name) as s:
        return s.read(1).astype(float), s.transform


def main():
    OUT2.mkdir(parents=True, exist_ok=True)
    log.info("Rendering Figure 2 (fused land cover + uncertainty) with boundary")

    fused, tf = _read("fused_2020.tif")
    conf, _ = _read("confidence_2020.tif")
    confl2, _ = _read("confidence_l2_2020.tif")
    agree, _ = _read("agreement_2020.tif")

    nod = fused == 0
    fused_m = np.ma.masked_where(nod, fused)
    conf_m = np.ma.masked_where(nod, conf)
    confl2_m = np.ma.masked_where(nod, confl2)
    agree_m = np.ma.masked_where(nod, agree)
    present = sorted(int(c) for c in np.unique(fused[fused > 0]))

    codes = list(range(1, 15))
    cmap_lc = ListedColormap([CLASS[c][1] for c in codes])
    norm_lc = BoundaryNorm([c - 0.5 for c in codes] + [14.5], cmap_lc.N)

    # load boundary once (cached)
    boundary_in_albers()

    fig = plt.figure(figsize=(21, 15.5))
    gs = fig.add_gridspec(2, 2, hspace=0.10, wspace=0.28)

    # (a) land cover
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(fused_m, cmap=cmap_lc, norm=norm_lc, interpolation="nearest")
    _overlay_boundary(axA, tf, fused.shape, lw=1.0)
    axA.set_title("(a) Fused land cover, 2020", fontsize=15, loc="left")
    axA.set_xticks([]); axA.set_yticks([])
    pat = [Patch(color=CLASS[c][1], label=CLASS[c][0]) for c in present]
    axA.legend(handles=pat, loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=10.5, frameon=False, title="Land-cover class",
               title_fontsize=11)

    # (b) super-class confidence
    axB = fig.add_subplot(gs[0, 1])
    imB = axB.imshow(conf_m, cmap="viridis", vmin=50, vmax=100, interpolation="nearest")
    _overlay_boundary(axB, tf, fused.shape, lw=1.0)
    axB.set_title("(b) Super-class posterior confidence", fontsize=15, loc="left")
    axB.set_xticks([]); axB.set_yticks([])
    cbB = plt.colorbar(imB, ax=axB, shrink=0.72, pad=0.02)
    cbB.set_label("confidence (%)", fontsize=11)

    # (c) sub-class confidence
    axC = fig.add_subplot(gs[1, 0])
    imC = axC.imshow(confl2_m, cmap="viridis", vmin=40, vmax=100, interpolation="nearest")
    _overlay_boundary(axC, tf, fused.shape, lw=1.0)
    axC.set_title("(c) Sub-class posterior confidence", fontsize=15, loc="left")
    axC.set_xticks([]); axC.set_yticks([])
    cbC = plt.colorbar(imC, ax=axC, shrink=0.72, pad=0.02)
    cbC.set_label("confidence (%)", fontsize=11)

    # (d) agreement
    axD = fig.add_subplot(gs[1, 1])
    cmap_ag = plt.colormaps.get_cmap("RdYlGn").resampled(8)
    imD = axD.imshow(agree_m, cmap=cmap_ag, vmin=-0.5, vmax=7.5, interpolation="nearest")
    _overlay_boundary(axD, tf, fused.shape, lw=1.0)
    axD.set_title("(d) Product agreement (of 8)", fontsize=15, loc="left")
    axD.set_xticks([]); axD.set_yticks([])
    cbD = plt.colorbar(imD, ax=axD, shrink=0.72, pad=0.02, ticks=range(0, 8))
    cbD.set_label("products agreeing", fontsize=11)

    plt.suptitle("Hierarchical Bayesian land-cover fusion and its uncertainty (2020)",
                 fontsize=18, y=0.95)
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT2 / f"fig2_fused_uncertainty.{ext}",
                    dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  ✓ fig2_fused_uncertainty.[png/pdf/svg]")

    # GeoTIFF of the class map (raw codes) for data availability
    _write_geotiff(np.where(nod, np.nan, fused), tf,
                   OUT2 / "fig2_fused_landcover.tif")

    # CSV companion
    rows = []
    total = int((fused > 0).sum())
    for c in present:
        m = fused == c
        rows.append({
            "class_code": c, "class_name": CLASS[c][0],
            "n_pixels": int(m.sum()),
            "area_pct": round(100 * m.sum() / total, 2),
            "mean_superclass_conf": round(float(conf[m].mean()), 1),
            "mean_subclass_conf": round(float(confl2[m].mean()), 1),
            "mean_agreement": round(float(agree[m].mean()), 2),
        })
    df = pd.DataFrame(rows).sort_values("area_pct", ascending=False)
    df.to_csv(OUT2 / "fig2_class_composition.csv", index=False)
    log.info(f"  ✓ fig2_class_composition.csv")
    log.info(f"  Overall: super-class conf {conf[fused>0].mean():.1f}%, "
             f"sub-class conf {confl2[fused>0].mean():.1f}%, "
             f"agreement {agree[fused>0].mean():.2f}/8")
    log.info("\nUpload fig2_fused_uncertainty.png + fig2_class_composition.csv for Results text.")


if __name__ == "__main__":
    main()
