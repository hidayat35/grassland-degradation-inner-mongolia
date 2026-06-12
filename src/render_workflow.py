r"""
Graphical methodological overview (swimlane) for Paper 2.

Doubles as the graphical abstract: inputs → two-tier fusion (Novelty 1) →
feature matrix → three parallel analyses (Novelties 2–3) → two headline
findings. Pure matplotlib (boxes + arrows), no external dependencies.

Run:
    python -m src.render_workflow      # writes to OUT_FIGURES/"workflow"
Outputs:
    fig_workflow_overview.{png,pdf,svg}
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_FIGURES
from src.raster_utils import log

OUTW = OUT_FIGURES / "workflow"

BAND = {
    "inputs":   ("#eef3f8", "#3a6ea5"),
    "fusion":   ("#eaf4ec", "#2e7d4f"),
    "features": ("#fdf3e7", "#c46a1b"),
    "analyses": ("#f3eef7", "#6a3d9a"),
    "synth":    ("#fbecec", "#b03030"),
}


def main():
    OUTW.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 11.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    bands = [
        ("inputs",   83, 100, "INPUTS"),
        ("fusion",   63, 81,  "NOVELTY 1 · FUSION"),
        ("features", 51, 61,  "DRIVERS & FEATURES"),
        ("analyses", 21, 49,  "NOVELTIES 2–3 · ANALYSES"),
        ("synth",     2, 19,  "SYNTHESIS"),
    ]
    for key, y0, y1, lab in bands:
        fill, edge = BAND[key]
        ax.add_patch(mp.Rectangle((2, y0), 96, y1 - y0, facecolor=fill,
                                  edgecolor="none", zorder=0))
        ax.text(3.0, y1 - 1.4, lab, fontsize=10.5, fontweight="bold",
                color=edge, va="top", ha="left")

    def box(x, y, w, h, text, key, fs=10, bold=False):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                     boxstyle="round,pad=0.3,rounding_size=0.8",
                     facecolor="white", edgecolor=BAND[key][1], lw=1.8, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4, color="#222")

    def arrow(p1, p2, key, lw=1.7):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16,
                     lw=lw, color=BAND[key][1], zorder=2, shrinkA=3, shrinkB=3))

    box(30, 91.5, 40, 9, "8 land-cover products\nWang2024 · GLC-FCS30D · CLCD · CCI\n"
        "MODIS · WorldCover · MPSDSL · Grassland-IM", "inputs", fs=9)
    box(75, 91.5, 40, 9, "Driver datasets\nMODIS NDVI · SPEI · ETCCDI climate\n"
        "grazing · population · SRTM terrain", "features", fs=9)

    box(30, 75.5, 30, 6.5, "Harmonisation → 14-class\nunified legend", "fusion", fs=9.5)
    box(30, 67, 34, 7.5, "Two-tier hierarchical\nBayesian fusion\n(super-class + sub-class)",
        "fusion", fs=9.5, bold=True)
    box(75, 70, 30, 7.5, "Fused land cover\n+ per-pixel uncertainty\n(1990–2020, 1 km)",
        "fusion", fs=9)

    box(50, 56, 42, 6.5, "Per-pixel feature matrix (128 features)\n"
        "fused state + drivers + temporal context + terrain", "features", fs=9)

    box(20, 37, 26, 9, "Attribution\nXGBoost + SHAP\n(which drivers covary\nwith transitions)",
        "analyses", fs=9, bold=True)
    box(50, 37, 26, 9, "Causal inference\nGranger + CCM\n(which drivers cause\nNDVI change, per zone)",
        "analyses", fs=9, bold=True)
    box(80, 37, 26, 9, "Spatial spillover\nST-GNN (A3T-GCN)\n(do neighbours add\npredictive signal?)",
        "analyses", fs=9, bold=True)

    box(20, 25, 25, 6.5, "Local drivers dominate:\nNDVI · population · terrain\n(82% CV accuracy)",
        "analyses", fs=8.3)
    box(50, 25, 25, 6.5, "Grazing · drought · pop.\n5 consensus causal links\n(per ecological zone)",
        "analyses", fs=8.3)
    box(80, 25, 25, 6.5, "Neighbours add nothing\nspillover gap −0.31 R²\n(local control)",
        "analyses", fs=8.3)

    box(28, 10, 42, 9, "Degradation is LOCALLY governed,\nnot spatially contagious\n"
        "(corroborated by 3 independent methods)", "synth", fs=9.5, bold=True)
    box(72, 10, 42, 9, "Dynamics are BIDIRECTIONAL\nrecovery ≈ 1.8 × degradation\n(2000–2020)",
        "synth", fs=9.5, bold=True)

    arrow((30, 87), (30, 79), "fusion")
    arrow((30, 71.5), (30, 70.75), "fusion")
    arrow((30, 63.25), (45, 59.5), "features")
    arrow((75, 87), (75, 73.75), "features")
    arrow((75, 66.25), (58, 59.5), "features")
    arrow((50, 52.75), (20, 41.5), "analyses")
    arrow((50, 52.75), (50, 41.5), "analyses")
    arrow((50, 52.75), (80, 41.5), "analyses")
    arrow((20, 32.5), (20, 28.25), "analyses")
    arrow((50, 32.5), (50, 28.25), "analyses")
    arrow((80, 32.5), (80, 28.25), "analyses")
    arrow((20, 21.75), (26, 14.5), "synth")
    arrow((50, 21.75), (30, 14.5), "synth")
    arrow((80, 21.75), (74, 14.5), "synth")

    ax.set_title("Methodological overview: multi-product fusion, causal attribution, "
                 "and spatial-spillover testing", fontsize=13.5, fontweight="bold", pad=12)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUTW / f"fig_workflow_overview.{ext}",
                    dpi=200 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close()
    log.info(f"  ✓ fig_workflow_overview.[png/pdf/svg] in {OUTW}")


if __name__ == "__main__":
    main()
