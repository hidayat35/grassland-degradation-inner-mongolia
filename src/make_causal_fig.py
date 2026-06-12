"""
Regional causal-structure figure for paper 2.
Combines Granger (linear) + CCM (nonlinear) into one publication-grade panel.

Reads:
    granger_results.csv, ccm_results.csv  (from step 06)
Writes (to OUT_CAUSAL):
    fig_causal_heatmap.{png,pdf,svg}
    causal_consensus_table.csv   (the combined evidence table)

Design:
  - Two-panel heatmap: left = Granger -log10(p), right = CCM rho (convergent only)
  - Rows = ecological zones (aridity gradient order), cols = drivers
  - A "consensus" star marks cells where BOTH methods agree (driver→NDVI causal)
  - Driver names and zone names are human-readable
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Allow running standalone (paths) or with project import
UPLOAD = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/causal_fig")
OUT.mkdir(exist_ok=True)

g = pd.read_csv(UPLOAD / "granger_results.csv")
c = pd.read_csv(UPLOAD / "ccm_results.csv")

# Human-readable driver labels, ordered: drought, climate, human
DRIVER_ORDER = ["spei12", "spei03", "clim_cdd", "clim_gsl",
                "clim_tx90p", "clim_tn90p", "grazing_total", "popdens"]
DRIVER_LABELS = {
    "spei12": "SPEI-12\n(drought)", "spei03": "SPEI-3\n(drought)",
    "clim_cdd": "CDD\n(dry days)", "clim_gsl": "GSL\n(season len)",
    "clim_tx90p": "TX90p\n(warm days)", "clim_tn90p": "TN90p\n(warm nights)",
    "grazing_total": "Grazing\nintensity", "popdens": "Population\ndensity",
}
# Zone order along aridity gradient (mesic -> arid)
ZONE_ORDER = [1, 2, 3, 4, 5, 6, 7]
ZONE_LABELS = {1: "Forest/Meadow", 2: "Real steppe", 3: "Dry steppe",
               4: "Desert steppe", 5: "Cropland", 6: "Sandy land",
               7: "Desert/Barren"}

# Build matrices
nЗ, nD = len(ZONE_ORDER), len(DRIVER_ORDER)
granger_mat = np.full((nЗ, nD), np.nan)
ccm_mat = np.full((nЗ, nD), np.nan)
consensus = np.zeros((nЗ, nD), dtype=bool)

GRANGER_SIG = 0.05
CCM_RHO_MIN = 0.30
CCM_CONV_MIN = 0.05

for i, z in enumerate(ZONE_ORDER):
    for j, drv in enumerate(DRIVER_ORDER):
        gr = g[(g.zone == z) & (g.driver == drv)]
        cc = c[(c.zone == z) & (c.driver == drv)]
        if len(gr):
            p = float(gr.p_value.iloc[0])
            granger_mat[i, j] = -np.log10(max(p, 1e-6))
        if len(cc):
            rho = float(cc.rho_max_lib.iloc[0])
            conv = float(cc.convergence.iloc[0])
            # show convergent positive coupling only
            ccm_mat[i, j] = rho if (conv > CCM_CONV_MIN and rho > 0) else np.nan
        # Consensus: Granger sig AND CCM causal
        gsig = len(gr) and float(gr.p_value.iloc[0]) < GRANGER_SIG
        csig = len(cc) and bool(cc.causal.iloc[0])
        consensus[i, j] = bool(gsig and csig)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# Panel 1: Granger
ax = axes[0]
cmap_g = LinearSegmentedColormap.from_list("g", ["#f7f7f7", "#fdbb84", "#d7301f"])
im = ax.imshow(granger_mat, cmap=cmap_g, vmin=0, vmax=3.5, aspect="auto")
ax.set_xticks(range(nD)); ax.set_xticklabels([DRIVER_LABELS[d] for d in DRIVER_ORDER], fontsize=8)
ax.set_yticks(range(nЗ)); ax.set_yticklabels([ZONE_LABELS[z] for z in ZONE_ORDER], fontsize=9)
ax.set_title("Granger causality (driver → NDVI)\n$-\\log_{10}(p)$, linear", fontsize=11)
cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.set_label("$-\\log_{10}(p)$", fontsize=9)
# mark significance threshold
for i in range(nЗ):
    for j in range(nD):
        v = granger_mat[i, j]
        if np.isfinite(v) and v > -np.log10(0.05):
            ax.text(j, i, "•", ha="center", va="center", color="black", fontsize=14)
# contour at p=0.05
ax.text(0.5, -0.13, "• = p < 0.05", transform=ax.transAxes, ha="center", fontsize=8)

# Panel 2: CCM
ax = axes[1]
cmap_c = LinearSegmentedColormap.from_list("c", ["#f7f7f7", "#a6bddb", "#045a8d"])
im = ax.imshow(ccm_mat, cmap=cmap_c, vmin=0, vmax=0.9, aspect="auto")
ax.set_xticks(range(nD)); ax.set_xticklabels([DRIVER_LABELS[d] for d in DRIVER_ORDER], fontsize=8)
ax.set_yticks(range(nЗ)); ax.set_yticklabels([ZONE_LABELS[z] for z in ZONE_ORDER], fontsize=9)
ax.set_title("Convergent cross mapping (driver → NDVI)\ncross-map $\\rho$, nonlinear", fontsize=11)
cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.set_label("CCM $\\rho$", fontsize=9)

# Consensus stars on BOTH panels
for panel in axes:
    for i in range(nЗ):
        for j in range(nD):
            if consensus[i, j]:
                panel.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                              edgecolor="#00a000", lw=2.5))
axes[1].text(0.5, -0.13, "green box = causal in BOTH methods",
             transform=axes[1].transAxes, ha="center", fontsize=8, color="#00a000")

plt.suptitle("Causal drivers of vegetation productivity across Inner Mongolia ecological zones",
             fontsize=13, y=1.02)
plt.tight_layout()
for ext in ("png", "pdf", "svg"):
    plt.savefig(OUT / f"fig_causal_heatmap.{ext}", dpi=200 if ext=="png" else None,
                bbox_inches="tight", facecolor="white")
plt.close()

# ---- Consensus evidence table ----
rows = []
for i, z in enumerate(ZONE_ORDER):
    for j, drv in enumerate(DRIVER_ORDER):
        gr = g[(g.zone == z) & (g.driver == drv)]
        cc = c[(c.zone == z) & (c.driver == drv)]
        rows.append({
            "zone": ZONE_LABELS[z], "driver": drv,
            "granger_p": round(float(gr.p_value.iloc[0]), 4) if len(gr) else np.nan,
            "granger_lag": int(gr.best_lag.iloc[0]) if len(gr) else None,
            "ccm_rho": round(float(cc.rho_max_lib.iloc[0]), 3) if len(cc) else np.nan,
            "ccm_convergence": round(float(cc.convergence.iloc[0]), 3) if len(cc) else np.nan,
            "ccm_lag": int(cc.best_lag.iloc[0]) if len(cc) else None,
            "granger_sig": bool(len(gr) and gr.p_value.iloc[0] < 0.05),
            "ccm_causal": bool(len(cc) and cc.causal.iloc[0]),
            "consensus": bool(consensus[i, j]),
        })
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "causal_consensus_table.csv", index=False)

# Summary stats
print("Consensus (both methods) causal links:")
con = tab[tab.consensus]
for _, r in con.iterrows():
    print(f"  {r['zone']:<16} <- {r['driver']:<14} "
          f"Granger p={r['granger_p']:.3f}(lag{r['granger_lag']:+d})  "
          f"CCM rho={r['ccm_rho']:.2f}(lag{r['ccm_lag']:+d})")
print(f"\nGranger-significant links: {tab.granger_sig.sum()}")
print(f"CCM-causal links:          {tab.ccm_causal.sum()}")
print(f"Consensus links:           {tab.consensus.sum()}")
print("\nSaved fig_causal_heatmap.[png/pdf/svg] + causal_consensus_table.csv")
