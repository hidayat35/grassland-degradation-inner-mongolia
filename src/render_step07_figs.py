r"""
Re-render the step 07 maps (spillover influence + degradation risk) WITH the
Inner Mongolia boundary overlaid, and emit CSV companions for the manuscript.

Run locally (needs the shapefile at AOI_SHP and the .tif outputs from step 07):
    python -m src.render_step07_figs

Reads:
    D:\paper2_outputs\07_stgnn\spillover_influence_map.tif
    D:\paper2_outputs\07_stgnn\degradation_risk_map.tif
    D:\paper2_outputs\07_stgnn\node_table.parquet      (for lon/lat in the CSV)
Writes (same folder):
    fig_spillover_influence.{png,pdf,svg,tif,csv}
    fig_degradation_risk.{png,pdf,svg,tif,csv}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ROOT
from src.raster_utils import log
from src.mapfig import render_map

OUT_GNN = OUT_ROOT / "07_stgnn"


def _read(p):
    with rasterio.open(p) as s:
        return s.read(1), s.transform


def _node_csv(values_grid, node_table_path, value_name):
    """Build a per-node CSV (node, row_c, col_c, x, y, lon, lat, value) by
    sampling the grid at each node's coarse (row,col)."""
    if not Path(node_table_path).exists():
        # fall back: just dump finite grid cells with their row/col
        rows, cols = np.where(np.isfinite(values_grid))
        return pd.DataFrame({"row_c": rows, "col_c": cols,
                             value_name: values_grid[rows, cols]})
    nt = pd.read_parquet(node_table_path)
    vals = values_grid[nt["row_c"].values, nt["col_c"].values]
    out = nt.copy()
    out[value_name] = vals
    return out


def main():
    log.info("Re-rendering step 07 maps with boundary overlay + CSVs")
    node_table = OUT_GNN / "node_table.parquet"

    # WATER (8), WETLAND (7), ICE (14): NDVI is near-zero over these, so any
    # NDVI-derived per-node metric is noise there. We mask them from BOTH maps
    # for a clean figure, but keep them (flagged) in the CSVs for transparency.
    WATER_CLASSES = {7, 8, 14}
    water_grid = None
    nt_df = None
    if Path(node_table).exists():
        nt_df = pd.read_parquet(node_table)
        is_water = nt_df["dom_class"].isin(WATER_CLASSES).values
        # water_grid built per-array below (needs the array shape); store flags
        _water_rc = (nt_df["row_c"].values[is_water], nt_df["col_c"].values[is_water])
        n_water = int(is_water.sum())
        log.info(f"  will mask {n_water:,} water/wetland/ice nodes from both maps")
    else:
        _water_rc = None
        log.warning("  node_table missing — cannot mask water; maps may show water artifacts")

    def _mask_water(grid):
        if _water_rc is None:
            return grid
        g = grid.copy()
        g[_water_rc[0], _water_rc[1]] = np.nan
        return g

    # ---- Spillover influence ----
    spill_raw, tf = _read(OUT_GNN / "spillover_influence_map.tif")
    spill = _mask_water(spill_raw)                      # water masked for display
    vmax = float(np.nanpercentile(spill[np.isfinite(spill)], 99))
    csv = _node_csv(spill_raw, node_table, "spillover_influence_raw")
    if "dom_class" in csv.columns:
        csv["is_water_or_wetland"] = csv["dom_class"].isin(WATER_CLASSES)
        csv["spillover_influence"] = np.where(csv["is_water_or_wetland"], np.nan,
                                              csv["spillover_influence_raw"])
    veg_sp = spill[np.isfinite(spill)]
    render_map(
        array=spill, transform=tf,
        title="Spatial-spillover influence on NDVI-change prediction\n"
              "(|Δ| when neighbour signal removed; 3 km nodes, water masked)",
        cbar_label="|Δ prediction| (standardized NDVI change)",
        out_stem=OUT_GNN / "fig_spillover_influence",
        cmap="Blues", vmin=0, vmax=vmax,
        subtitle=f"mean {np.nanmean(veg_sp):.4f} · median {np.nanmedian(veg_sp):.4f} "
                 f"· near-zero throughout → vegetation dynamics are locally governed",
        csv_values={c: csv[c].values for c in csv.columns},
    )


    # ---- Degradation risk ----
    risk_raw, tf2 = _read(OUT_GNN / "degradation_risk_map.tif")
    risk_masked = _mask_water(risk_raw)   # water/wetland/ice -> NaN

    # display: only positive risk on vegetated land
    disp = np.where(np.isfinite(risk_masked) & (risk_masked > 0), risk_masked, np.nan)

    # CSV keeps ALL nodes but flags water and gives both raw + masked values
    csv2 = _node_csv(risk_raw, node_table, "degradation_risk_raw")
    if "dom_class" in csv2.columns:
        csv2["is_water_or_wetland"] = csv2["dom_class"].isin(WATER_CLASSES)
        csv2["degradation_risk"] = np.where(csv2["is_water_or_wetland"], np.nan,
                                            csv2["degradation_risk_raw"])

    veg = risk_masked[np.isfinite(risk_masked)]
    n_risk = int((veg > 0.1).sum())
    n_valid = int(veg.size)

    # land base = all vegetated (non-water) nodes, so stable low-risk land is
    # visibly grey rather than blank against the page (Style A).
    land_base = None
    if nt_df is not None:
        land_base = np.zeros_like(risk_raw, dtype=bool)
        not_water = ~nt_df["dom_class"].isin(WATER_CLASSES).values
        land_base[nt_df["row_c"].values[not_water],
                  nt_df["col_c"].values[not_water]] = True

    render_map(
        array=disp, transform=tf2,
        title="Projected near-future degradation-risk index (2027)\n"
              "relative NDVI decline from A3T-GCN forecast (water bodies masked)",
        cbar_label="relative degradation-risk (≥0.4 saturated)",
        out_stem=OUT_GNN / "fig_degradation_risk",
        cmap="YlOrRd", vmin=0, vmax=0.4, cbar_extend="max",
        land_base=land_base,
        subtitle=f"{n_risk:,} of {n_valid:,} vegetated nodes show projected decline "
                 f"(risk > 0.1); most of the region is stable or greening",
        csv_values={c: csv2[c].values for c in csv2.columns},
    )
    log.info("Done. Upload the two .csv files for Results text.")


if __name__ == "__main__":
    main()
