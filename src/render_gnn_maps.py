from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

U = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/gnn_fig"); OUT.mkdir(exist_ok=True)

# AOI footprint for masking (from fused map if available; else use finite mask)
def load(f):
    with rasterio.open(U / f) as s:
        return s.read(1)

spill = load("spillover_influence_map.tif")
risk = load("degradation_risk_map.tif")

# ---- Spillover influence map ----
# values are |Δprediction| when neighbours ablated; tiny => low spatial dependence
masked = np.ma.masked_invalid(spill)
fig, ax = plt.subplots(figsize=(12, 9))
cmap = LinearSegmentedColormap.from_list("sp", ["#f7fbff", "#9ecae1", "#08519c"])
# cap at p99 for display so the color range is meaningful
vmax = np.nanpercentile(spill[np.isfinite(spill)], 99)
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Spatial-spillover influence on NDVI-change prediction\n"
             "(|Δ| when neighbour signal is removed; 3 km nodes)", fontsize=13)
cb = plt.colorbar(im, ax=ax, shrink=0.7)
cb.set_label("|Δ prediction| (standardized NDVI change)", fontsize=10)
ax.text(0.5, -0.06, f"mean {np.nanmean(spill):.4f} · median {np.nanmedian(spill):.4f} · "
        f"near-zero everywhere → vegetation dynamics are locally governed",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic")
plt.tight_layout()
for ext in ("png", "pdf", "svg"):
    plt.savefig(OUT / f"fig_spillover_influence.{ext}",
                dpi=180 if ext == "png" else None, bbox_inches="tight", facecolor="white")
plt.close()

# ---- Degradation-risk map ----
masked_r = np.ma.masked_where(~np.isfinite(risk) | (risk <= 0), risk)
fig, ax = plt.subplots(figsize=(12, 9))
cmap_r = LinearSegmentedColormap.from_list("risk", ["#ffffcc", "#fd8d3c", "#bd0026"])
im = ax.imshow(masked_r, cmap=cmap_r, vmin=0, vmax=1, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Projected near-future degradation-risk index (2027)\n"
             "relative NDVI decline from A3T-GCN forecast; grey = no projected decline",
             fontsize=13)
cb = plt.colorbar(im, ax=ax, shrink=0.7)
cb.set_label("relative degradation-risk (0–1)", fontsize=10)
n_risk = int((risk > 0.1).sum())
ax.text(0.5, -0.06, f"{n_risk:,} of {np.isfinite(risk).sum():,} nodes show projected "
        f"decline (risk > 0.1); most of the region is stable or greening",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic")
plt.tight_layout()
for ext in ("png", "pdf", "svg"):
    plt.savefig(OUT / f"fig_degradation_risk.{ext}",
                dpi=180 if ext == "png" else None, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved fig_spillover_influence + fig_degradation_risk (png/pdf/svg)")
print(f"spillover display vmax (p99): {vmax:.4f}")
print(f"risk nodes >0.1: {n_risk:,}")
