"""
Smooth the sparse per-pixel CCM causal maps for DISPLAY ONLY.

Method (honest, gap-filling that never invents signal in empty regions):
  1. For each driver map, take the raw per-pixel CCM rho (7-10% coverage).
  2. Restrict to the AOI.
  3. Gap-fill using a DISTANCE-LIMITED nearest-valid approach: a NaN pixel is
     filled with the local mean of valid pixels within radius R ONLY IF at
     least `min_valid` valid pixels exist in that window. Pixels with no nearby
     evidence stay NaN (we never fabricate values in data-empty regions).
  4. Light median filter (3x3) to reduce speckle.
  5. Re-mask to the AOI footprint.

This produces a smooth, readable map that is faithful to where evidence
actually exists. Caption must say: "Per-pixel CCM rho, gap-filled within a
local 7-km neighborhood where >=5 valid estimates exist and lightly median-
smoothed for visualization; analysis statistics use the regional aggregation."
"""
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import generic_filter, median_filter, uniform_filter

U = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/causal_fig"); OUT.mkdir(exist_ok=True)

# AOI footprint from the fused map
with rasterio.open(U / "fused_2020.tif") as src:
    fused = src.read(1)
aoi = fused > 0

def neighborhood_fill(arr, radius=3, min_valid=5):
    """Fill NaNs with local mean of valid values within `radius`, only where
    at least `min_valid` valid neighbours exist. Vectorised via uniform_filter
    on the valid mask and the zero-filled array."""
    valid = np.isfinite(arr)
    a0 = np.where(valid, arr, 0.0).astype(np.float32)
    vm = valid.astype(np.float32)
    size = 2 * radius + 1
    # local sum of values and counts
    ssum = uniform_filter(a0, size=size, mode="constant") * (size * size)
    scnt = uniform_filter(vm, size=size, mode="constant") * (size * size)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_mean = ssum / scnt
    filled = arr.copy()
    fill_mask = (~valid) & (scnt >= min_valid)
    filled[fill_mask] = local_mean[fill_mask]
    return filled

drivers = [("grazing_total", "Grazing intensity → NDVI"),
           ("spei12", "SPEI-12 drought → NDVI"),
           ("popdens", "Population density → NDVI"),
           ("clim_tx90p", "Warm-day extremes → NDVI")]

smoothed = {}
for d, _ in drivers:
    with rasterio.open(U / f"gccm_causal_map_{d}.tif") as src:
        a = src.read(1)
    a[~aoi] = np.nan
    f = neighborhood_fill(a, radius=3, min_valid=5)   # ~7km window
    # light median filter on the finite values
    finite_mask = np.isfinite(f)
    tmp = np.where(finite_mask, f, 0.0)
    med = median_filter(tmp, size=3, mode="constant")
    f2 = np.where(finite_mask, med, np.nan)
    f2[~aoi] = np.nan
    smoothed[d] = f2

# ---- 4-panel smoothed ----
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
for ax, (d, title) in zip(axes.flat, drivers):
    masked = np.ma.masked_invalid(smoothed[d])
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="bilinear")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("CCM ρ (causal strength)", fontsize=9)
plt.suptitle("Per-pixel causal strength of drivers on NDVI (gap-filled & smoothed for visualization)",
             fontsize=15, y=0.99)
plt.tight_layout()
for ext in ("png", "pdf", "svg"):
    plt.savefig(OUT / f"fig_causal_maps_4panel_smoothed.{ext}",
                dpi=180 if ext=="png" else None, bbox_inches="tight", facecolor="white")
plt.close()

# ---- Smoothed dominance map (argmax over smoothed driver maps) ----
legend = {1: ("SPEI-12 drought", "#2c7fb8"),
          2: ("Warm-day extremes", "#d7301f"),
          3: ("Grazing intensity", "#762a83"),
          4: ("Population density", "#1a9850")}
order = ["spei12", "clim_tx90p", "grazing_total", "popdens"]
stack = np.stack([smoothed[d] for d in order], axis=0)
valid_any = np.isfinite(stack).any(axis=0)
with np.errstate(invalid="ignore"):
    argmax = np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf), axis=0)
# require the winning rho to be positive (causal), else leave blank
winner_val = np.take_along_axis(stack, argmax[None], axis=0)[0]
dom = np.zeros(stack.shape[1:], dtype=np.uint8)
good = valid_any & (winner_val > 0.1) & aoi
dom[good] = (argmax[good] + 1).astype(np.uint8)

codes = [1, 2, 3, 4]
cmap = ListedColormap([legend[c][1] for c in codes])
masked = np.ma.masked_where(dom == 0, dom)
fig, ax = plt.subplots(figsize=(13, 9))
im = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=4.5, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Dominant causal driver of NDVI (smoothed; ρ>0.1)", fontsize=14)
patches = [Patch(color=legend[c][1], label=legend[c][0]) for c in codes]
ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
          fontsize=11, frameon=False, title="Dominant driver")
plt.tight_layout()
for ext in ("png", "pdf", "svg"):
    plt.savefig(OUT / f"fig_causal_dominance_smoothed.{ext}",
                dpi=180 if ext=="png" else None, bbox_inches="tight", facecolor="white")
plt.close()

# Coverage report
for d, _ in drivers:
    cov = 100*np.isfinite(smoothed[d]).sum()/aoi.sum()
    print(f"{d}: smoothed coverage {cov:.1f}% of AOI")
print(f"dominance assigned: {100*(dom>0).sum()/aoi.sum():.1f}% of AOI")
print("Saved smoothed 4-panel + dominance figures")
