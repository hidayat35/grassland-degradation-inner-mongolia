"""
================================================================================
COMPREHENSIVE DIAGNOSTIC — inspect ALL datasets in the paper 2 project
================================================================================
Single-shot inspector that:
  1. Verifies the AOI loads + has a valid CRS
  2. Inspects ONE representative file from each LULC product:
       - native code distribution inside the IM AOI (downsampled for speed)
       - flags codes that don't match the documented legend for each product
       - cross-tabulates against Wang2024_MP 2020 to validate harmonization
  3. Inspects ONE representative file from each driver group (climate, grazing,
     phenology, popdens, cropping, sandstorm, desertification, etc.)
  4. Writes a single Markdown report and per-file CSVs to D:\\paper2_outputs\\logs\\

Run from project root:
    python comprehensive_diagnose.py

Output:
    D:\\paper2_outputs\\logs\\comprehensive_diagnostic.md
    D:\\paper2_outputs\\logs\\diag_<product>_<year>_freq.csv
    D:\\paper2_outputs\\logs\\diag_<product>_<year>_vs_Wang.csv
================================================================================
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.paths import LULC_PRODUCTS, DRIVERS, AOI_SHP, OUT_LOGS, ensure_dirs
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import (
    log, get_aoi_mask, compute_target_grid, reproject_to_target,
    load_aoi_in_target_crs
)


# ============================================================================
# OFFICIAL LEGENDS (for code-validation checking)
# ============================================================================

# MCD12Q1 IGBP Type 1 — 17 classes + nodata
MODIS_IGBP_OFFICIAL = {
    0: "Water bodies (filled)",
    1: "Evergreen Needleleaf Forests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up Lands",
    14: "Cropland/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
    255: "Unclassified",
}

# ESA WorldCover v100/v200 — 11 classes
WORLDCOVER_OFFICIAL = {
    0:   "No Data",
    10:  "Tree cover",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    50:  "Built-up",
    60:  "Bare / sparse vegetation",
    70:  "Snow and ice",
    80:  "Permanent water bodies",
    90:  "Herbaceous wetland",
    95:  "Mangroves",
    100: "Moss and lichen",
}

# CCI / C3S LCCS — 22+ codes
CCI_OFFICIAL = {
    0: "No Data", 10: "Cropland, rainfed", 11: "Herbaceous cover",
    12: "Tree/shrub cover (cropland)", 20: "Cropland, irrigated",
    30: "Mosaic cropland (>50%)", 40: "Mosaic natural veg (>50%)",
    50: "Tree cover, broadleaved evergreen", 60: "Tree cover, broadleaved deciduous",
    61: "Tree cover, broadleaved deciduous (>40%)", 62: "Tree cover, broadleaved deciduous (15-40%)",
    70: "Tree cover, needleleaved evergreen", 71: "Tree cover, needleleaved evergreen (>40%)",
    72: "Tree cover, needleleaved evergreen (15-40%)", 80: "Tree cover, needleleaved deciduous",
    81: "Tree cover, needleleaved deciduous (>40%)", 82: "Tree cover, needleleaved deciduous (15-40%)",
    90: "Tree cover, mixed leaf", 100: "Mosaic tree/shrub (>50%)",
    110: "Mosaic herbaceous (>50%)", 120: "Shrubland", 121: "Shrubland evergreen",
    122: "Shrubland deciduous", 130: "Grassland", 140: "Lichens and mosses",
    150: "Sparse vegetation", 151: "Sparse tree", 152: "Sparse shrub",
    153: "Sparse herbaceous", 160: "Tree, flooded, fresh/brakish",
    170: "Tree, flooded, saline", 180: "Shrub/herbaceous, flooded",
    190: "Urban areas", 200: "Bare areas", 201: "Consolidated bare",
    202: "Unconsolidated bare", 210: "Water bodies", 220: "Permanent snow and ice",
    32767: "NoData (file sentinel)",
}

CLCD_OFFICIAL = {
    0: "NoData", 1: "Cropland", 2: "Forest", 3: "Shrub", 4: "Grassland",
    5: "Water", 6: "Snow/Ice", 7: "Barren", 8: "Impervious", 9: "Wetland",
}

WANG_OFFICIAL = {
    0: "NoData", 1: "Forest", 2: "Shrub", 3: "Meadow", 4: "Real steppe",
    5: "Dry steppe", 6: "Desert steppe", 7: "Wetland", 8: "Water",
    9: "Cropland", 10: "Built-up", 11: "Barren", 12: "Desert", 13: "Sand",
    14: "Ice",
}

# GLC_FCS30D — 36 codes (abbreviated for readability)
GLC_OFFICIAL = {
    0: "fill", 10: "Cropland rainfed", 11: "Herbaceous cropland",
    12: "Tree/shrub cropland", 20: "Cropland irrigated",
    51: "Open EVG broadleaf", 52: "Closed EVG broadleaf",
    61: "Open DEC broadleaf", 62: "Closed DEC broadleaf",
    71: "Open EVG needleleaf", 72: "Closed EVG needleleaf",
    81: "Open DEC needleleaf", 82: "Closed DEC needleleaf",
    91: "Open mixed leaf", 92: "Closed mixed leaf",
    120: "Shrubland", 121: "EVG shrubland", 122: "DEC shrubland",
    130: "Grassland", 140: "Lichens/mosses",
    150: "Sparse veg", 152: "Sparse shrub", 153: "Sparse herbaceous",
    181: "Swamp", 182: "Marsh", 183: "Flooded flat", 184: "Saline",
    185: "Mangrove", 186: "Salt marsh", 187: "Tidal flat",
    190: "Built-up", 200: "Bare", 201: "Consolidated bare",
    202: "Unconsolidated bare", 210: "Water", 220: "Ice/snow",
    250: "fill", 255: "nodata",
}

MPSDSL_OFFICIAL = {
    0: "non-sandy / nodata",
    2: "sand (sub-type)", 3: "sand (sub-type)",
    4: "desert (sub-type)", 5: "desert (sub-type)",
    6: "desert (sub-type)", 7: "desert (sub-type)",
}

GRASSLAND_IM_OFFICIAL = {0: "non-grassland", 1: "grassland"}


# ============================================================================
# REPRESENTATIVE FILES TO INSPECT
# Each item: (product/group label, file path, source_nodata, official_legend_dict)
# ============================================================================

def get_lulc_targets():
    """Build the LULC targets list dynamically — handles MODIS/WorldCover
    by checking common locations even though they aren't in LULC_PRODUCTS yet."""
    targets = []

    # Wang2024_MP 2020 — first because cross-tab needs it
    cfg = LULC_PRODUCTS["Wang2024_MP"]
    f = cfg["folder"] / cfg["year_template"].format(year=2020)
    targets.append(("Wang2024_MP", 2020, f, None, WANG_OFFICIAL))

    # CCI 2020 — try both filename patterns
    cci_cfg = LULC_PRODUCTS["CCI"]
    cci_f = None
    for tmpl in cci_cfg["year_templates"]:
        cand = cci_cfg["folder"] / tmpl.format(year=2020)
        if cand.exists():
            cci_f = cand
            break
    targets.append(("CCI", 2020, cci_f, 32767, CCI_OFFICIAL))

    # CLCD 2020
    cfg = LULC_PRODUCTS["CLCD"]
    f = cfg["folder"] / cfg["year_template"].format(year=2020)
    targets.append(("CLCD", 2020, f, None, CLCD_OFFICIAL))

    # GLC_FCS30D 2020 (band 21 of the second multi-band file)
    cfg = LULC_PRODUCTS["GLC_FCS30D"]
    targets.append(("GLC_FCS30D", 2020, cfg["files"][1], None, GLC_OFFICIAL))

    # MPSDSL 2020
    cfg = LULC_PRODUCTS["MPSDSL"]
    f = cfg["folder"] / cfg["year_template"].format(year=2020)
    targets.append(("MPSDSL", 2020, f, None, MPSDSL_OFFICIAL))

    # Grassland_IM 2020
    cfg = LULC_PRODUCTS["Grassland_IM"]
    f = cfg["folder"] / cfg["year_template"].format(year=2020)
    targets.append(("Grassland_IM", 2020, f, None, GRASSLAND_IM_OFFICIAL))

    # MODIS (native files — not yet in paths.py)
    modis_path = Path(r"D:\all lulc data\modis\original modis\MODIS_LandCover_IGBP_2020.tif")
    if not modis_path.exists():
        # try 2021 or 2019 as fallback
        for y in [2021, 2019, 2018, 2010, 2005, 2001]:
            cand = modis_path.parent / f"MODIS_LandCover_IGBP_{y}.tif"
            if cand.exists():
                modis_path = cand
                break
    targets.append(("MODIS_native", modis_path.name.replace(".tif", ""),
                    modis_path, 255, MODIS_IGBP_OFFICIAL))

    # WorldCover (native files — not yet in paths.py)
    wc_2020 = Path(r"D:\all lulc data\wordcover 20,21\original\ESA_WorldCover_30m_2020_v100.tif")
    targets.append(("WorldCover_v100", "2020", wc_2020, 0, WORLDCOVER_OFFICIAL))
    wc_2021 = Path(r"D:\all lulc data\wordcover 20,21\original\ESA_WorldCover_30m_2021_v100.tif")
    if not wc_2021.exists():
        # alternative naming
        wc_2021 = wc_2021.parent / "ESA_WorldCover_30m_2021_v200.tif"
    targets.append(("WorldCover_v200", "2021", wc_2021, 0, WORLDCOVER_OFFICIAL))

    return targets


# Driver targets — (group_label, year, file_path, dtype, nodata_value or None)
def get_driver_targets():
    targets = []

    # Climate — pick one index, year 2020
    cdir = DRIVERS["climate_indices_root"] / "CDD"
    if cdir.exists():
        tifs = sorted(cdir.glob("*.tif"))
        if tifs:
            targets.append(("climate_CDD", "sample", tifs[-1]))

    # Grazing total
    f = DRIVERS["grazing_root"] / "LHGI_2020.tif"
    if f.exists():
        targets.append(("grazing_LHGI", 2020, f))
    # Grazing — sheep
    f = DRIVERS["grazing_root"] / "Sheep"
    if f.exists():
        tifs = sorted(f.glob("*.tif"))
        if tifs:
            targets.append(("grazing_Sheep", "sample", tifs[-1]))

    # Phenology — SOS 2020
    f = DRIVERS["phenology_sos_folder"] / DRIVERS["phenology_sos_template"].format(year=2020)
    if f.exists():
        targets.append(("phenology_SOS", 2020, f))
    f = DRIVERS["phenology_eos_folder"] / DRIVERS["phenology_eos_template"].format(year=2020)
    if f.exists():
        targets.append(("phenology_EOS", 2020, f))

    # WorldPop 2020
    f = DRIVERS["worldpop_folder"] / DRIVERS["worldpop_template"].format(year=2020)
    if f.exists():
        targets.append(("popdens", 2020, f))

    # Cropping intensity 2019
    f = DRIVERS["cropping_intensity_folder"] / DRIVERS["cropping_intensity_template"].format(year=2019)
    if f.exists():
        targets.append(("cropping_intensity", 2019, f))

    # GLW
    if DRIVERS["glw_2020"].exists():
        targets.append(("GLW_livestock", 2020, DRIVERS["glw_2020"]))

    return targets


# ============================================================================
# INSPECTION FUNCTIONS
# ============================================================================

def inspect_lulc_file(label, year, path, src_nodata, official, aoi_mask, wang_arr=None):
    """Inspect one LULC raster: native code distribution + cross-tab vs Wang."""
    info = {
        "label": label, "year": year, "path": str(path),
        "exists": False, "error": None,
    }
    if path is None or not Path(path).exists():
        info["error"] = "FILE NOT FOUND"
        log.warning(f"  ⚠ {label} {year}: FILE NOT FOUND: {path}")
        return info

    info["exists"] = True

    try:
        # 1) Open file and capture metadata
        with rasterio.open(path) as src:
            info["driver"] = src.driver
            info["dtype"] = str(src.dtypes[0])
            info["bands"] = src.count
            info["dimensions"] = f"{src.width} × {src.height}"
            info["crs"] = str(src.crs) if src.crs else "None"
            info["bounds"] = tuple(src.bounds)
            info["pixel_size"] = (src.transform.a, abs(src.transform.e))
            info["nodata"] = src.nodata
            info["file_size_mb"] = round(Path(path).stat().st_size / 1024 / 1024, 2)

        # 2) Reproject to target grid for AOI-scoped inspection
        # GLC needs special band handling for 2020
        if label == "GLC_FCS30D" and int(year) == 2020:
            band = 21  # 2020 = band 21 in the 2000-2022 file
        else:
            band = 1
        arr = reproject_to_target(
            Path(path), band=band, resampling=Resampling.nearest,
            src_nodata=src_nodata, target_dtype="int32"
        )

        # 3) Native code distribution INSIDE AOI
        inside = arr[aoi_mask]
        u, c = np.unique(inside, return_counts=True)
        total = c.sum()
        rows = []
        unknown_codes = []
        for code, count in sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1]):
            code = int(code)
            pct = 100 * count / total
            in_legend = code in official
            name = official.get(code, "??? UNKNOWN ???")
            rows.append({
                "code": code, "n_pixels": int(count), "pct_aoi": round(pct, 4),
                "in_legend": in_legend, "name": name,
            })
            if not in_legend:
                unknown_codes.append(code)
        df = pd.DataFrame(rows)
        info["freq_table"] = df
        info["unknown_codes"] = unknown_codes
        info["n_codes_total"] = len(u)
        info["pixels_inside_aoi"] = int(total)

        # 4) Cross-tab vs Wang_MP
        if wang_arr is not None and label != "Wang2024_MP":
            try:
                valid = aoi_mask & (arr > 0) & (wang_arr > 0)
                if src_nodata is not None:
                    valid &= (arr != src_nodata)
                cdf = pd.DataFrame({"product": arr[valid], "wang": wang_arr[valid]})
                ct = (cdf.groupby(["product", "wang"]).size()
                         .unstack(fill_value=0))
                # Rename Wang columns
                ct.columns = [
                    f"W{c}_{WANG_OFFICIAL.get(c, '?')}"
                    for c in ct.columns
                ]
                ct["total"] = ct.sum(axis=1)
                # Dominant Wang per row
                only = ct.drop(columns=["total"])
                ct["dominant_Wang"] = only.idxmax(axis=1)
                ct["dominant_pct"] = (only.max(axis=1) / ct["total"] * 100).round(1)
                info["crosstab"] = ct
            except Exception as e:
                info["crosstab_error"] = str(e)

        return info

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        log.error(f"  ✗ {label} {year}: {e}")
        log.error(traceback.format_exc())
        return info


def inspect_continuous_raster(label, year, path, aoi_mask):
    """Inspect one continuous raster (climate / grazing / pop / phenology)."""
    info = {"label": label, "year": year, "path": str(path), "exists": False}
    if not Path(path).exists():
        info["error"] = "FILE NOT FOUND"
        return info
    info["exists"] = True

    try:
        with rasterio.open(path) as src:
            info["driver"] = src.driver
            info["dtype"] = str(src.dtypes[0])
            info["bands"] = src.count
            info["dimensions"] = f"{src.width} × {src.height}"
            info["crs"] = str(src.crs) if src.crs else "None"
            info["pixel_size"] = (src.transform.a, abs(src.transform.e))
            info["nodata"] = src.nodata
            info["file_size_mb"] = round(Path(path).stat().st_size / 1024 / 1024, 2)

        arr = reproject_to_target(
            Path(path), band=1, resampling=Resampling.bilinear,
            target_dtype="float32"
        )
        inside = arr[aoi_mask]
        finite = inside[np.isfinite(inside)]
        if finite.size > 0:
            info["min"] = float(np.nanmin(finite))
            info["max"] = float(np.nanmax(finite))
            info["mean"] = float(np.nanmean(finite))
            info["std"] = float(np.nanstd(finite))
            info["pct5"] = float(np.percentile(finite, 5))
            info["pct95"] = float(np.percentile(finite, 95))
            info["n_finite"] = int(finite.size)
            info["n_nan"] = int(inside.size - finite.size)
        else:
            info["error"] = "All values NaN inside AOI"
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info


# ============================================================================
# MARKDOWN REPORT BUILDER
# ============================================================================

def format_lulc_section(info):
    lines = [f"\n## {info['label']} ({info['year']})\n"]
    if not info["exists"]:
        lines.append(f"⚠️ **FILE NOT FOUND:** `{info['path']}`\n")
        return "\n".join(lines)
    if info.get("error") and "freq_table" not in info:
        lines.append(f"❌ **ERROR:** {info['error']}\n")
        return "\n".join(lines)

    lines.append(f"- **Path:** `{info['path']}`")
    lines.append(f"- **Size on disk:** {info['file_size_mb']} MB")
    lines.append(f"- **Driver/Dtype/Bands:** {info['driver']} / {info['dtype']} / {info['bands']}")
    lines.append(f"- **Dimensions:** {info['dimensions']}")
    lines.append(f"- **CRS:** `{info['crs']}`")
    lines.append(f"- **Pixel size:** {info['pixel_size']}")
    lines.append(f"- **NoData (file header):** `{info['nodata']}`")
    lines.append(f"- **Pixels inside AOI:** {info['pixels_inside_aoi']:,}")
    lines.append(f"- **Unique codes inside AOI:** {info['n_codes_total']}")

    if info["unknown_codes"]:
        lines.append(f"- ⚠️ **Codes NOT in official legend:** `{info['unknown_codes']}`")
    else:
        lines.append(f"- ✅ All codes match official legend.")

    lines.append("\n**Native code distribution inside AOI (top 25):**\n")
    lines.append(info["freq_table"].head(25).to_markdown(index=False))
    lines.append("")

    if "crosstab" in info:
        lines.append("**Cross-tabulation vs Wang2024_MP 2020 (dominant Wang class per native code):**\n")
        small = info["crosstab"][["total", "dominant_Wang", "dominant_pct"]].copy()
        small["total"] = small["total"].astype(int)
        lines.append(small.to_markdown())
        lines.append("")
    elif "crosstab_error" in info:
        lines.append(f"_Cross-tab error: {info['crosstab_error']}_\n")

    return "\n".join(lines)


def format_continuous_section(info):
    lines = [f"\n## {info['label']} ({info['year']})\n"]
    if not info["exists"]:
        lines.append(f"⚠️ **FILE NOT FOUND:** `{info['path']}`\n")
        return "\n".join(lines)
    if info.get("error"):
        lines.append(f"❌ **ERROR:** {info['error']}\n")
        return "\n".join(lines)

    lines.append(f"- **Path:** `{info['path']}`")
    lines.append(f"- **Size:** {info['file_size_mb']} MB | "
                 f"**Driver/Dtype:** {info['driver']}/{info['dtype']}")
    lines.append(f"- **Dimensions:** {info['dimensions']}")
    lines.append(f"- **CRS:** `{info['crs']}`")
    lines.append(f"- **Pixel size:** {info['pixel_size']}")
    lines.append(f"- **NoData:** `{info['nodata']}`")
    lines.append(f"- **Inside AOI:** {info['n_finite']:,} finite values, {info['n_nan']:,} NaN")
    lines.append(f"- **Statistics (after reprojection to 1km Albers, inside AOI):**")
    lines.append(f"    - min: {info['min']:.4g}, 5th pct: {info['pct5']:.4g}, "
                 f"mean: {info['mean']:.4g}, std: {info['std']:.4g}, "
                 f"95th pct: {info['pct95']:.4g}, max: {info['max']:.4g}")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ensure_dirs()
    log.info("=" * 80)
    log.info("COMPREHENSIVE DIAGNOSTIC — paper 2")
    log.info("=" * 80)

    # AOI
    log.info("\n── AOI ──")
    try:
        aoi = load_aoi_in_target_crs()
        aoi_mask = get_aoi_mask()
        log.info(f"AOI loaded; {aoi_mask.sum():,} pixels inside (target 1km Albers grid)")
    except Exception as e:
        log.error(f"AOI FAILED: {e}")
        return

    # ------------------------------------------------------------------
    # Wang2024_MP first (others need it for cross-tab)
    # ------------------------------------------------------------------
    lulc_targets = get_lulc_targets()
    wang_info = None
    wang_arr = None

    # Compute Wang once and cache the array
    for label, year, path, snd, official in lulc_targets:
        if label == "Wang2024_MP":
            wang_info = inspect_lulc_file(label, year, path, snd, official, aoi_mask, None)
            if wang_info["exists"] and not wang_info.get("error"):
                wang_arr = reproject_to_target(
                    path, band=1, resampling=Resampling.nearest,
                    src_nodata=snd, target_dtype="int32"
                )
            break

    # Now inspect all LULC products
    lulc_results = []
    for label, year, path, snd, official in lulc_targets:
        log.info(f"\n── LULC: {label} {year} ──")
        if label == "Wang2024_MP":
            lulc_results.append(wang_info)
            continue
        info = inspect_lulc_file(label, year, path, snd, official, aoi_mask, wang_arr)
        lulc_results.append(info)
        # Print quick console summary
        if info.get("exists") and not info.get("error"):
            n_unknown = len(info.get("unknown_codes", []))
            log.info(f"  {info['n_codes_total']} unique codes, {n_unknown} unknown, "
                     f"{info['pixels_inside_aoi']:,} px inside AOI")
            if n_unknown:
                log.info(f"  ⚠ Unknown codes: {info['unknown_codes']}")
            # Save CSVs
            info["freq_table"].to_csv(
                OUT_LOGS / f"diag_{label}_{year}_freq.csv", index=False
            )
            if "crosstab" in info:
                info["crosstab"].to_csv(OUT_LOGS / f"diag_{label}_{year}_vs_Wang.csv")

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------
    driver_targets = get_driver_targets()
    driver_results = []
    for label, year, path in driver_targets:
        log.info(f"\n── Driver: {label} {year} ──")
        info = inspect_continuous_raster(label, year, path, aoi_mask)
        driver_results.append(info)
        if info.get("exists") and not info.get("error"):
            log.info(f"  range [{info['min']:.4g} .. {info['max']:.4g}], "
                     f"mean {info['mean']:.4g}, {info['n_finite']:,} finite px")

    # ------------------------------------------------------------------
    # Write Markdown report
    # ------------------------------------------------------------------
    report_lines = [
        "# Comprehensive Dataset Diagnostic — Paper 2",
        "",
        "Generated by `comprehensive_diagnose.py`. All inspections are scoped to the",
        "Inner Mongolia AOI after reprojection to the target 1 km Albers grid.",
        "",
        "## AOI summary",
        f"- AOI CRS: `{aoi.crs}`",
        f"- AOI feature count: {len(aoi)}",
        f"- AOI pixels inside the 1 km target grid: {aoi_mask.sum():,}",
        f"- AOI bounds (target CRS): {tuple(round(b, 0) for b in aoi.total_bounds)}",
        "",
        "---",
        "",
        "# Part 1: LULC products",
        "",
    ]
    for info in lulc_results:
        report_lines.append(format_lulc_section(info))

    report_lines.append("\n---\n\n# Part 2: Driver datasets\n")
    for info in driver_results:
        report_lines.append(format_continuous_section(info))

    # Final summary
    report_lines.append("\n---\n\n# Summary checks\n")
    issues = []
    for info in lulc_results:
        if not info.get("exists"):
            issues.append(f"- ❌ {info['label']} {info['year']}: file not found")
        elif info.get("error"):
            issues.append(f"- ❌ {info['label']} {info['year']}: {info['error']}")
        elif info.get("unknown_codes"):
            issues.append(f"- ⚠️ {info['label']} {info['year']}: codes not in official legend: "
                          f"{info['unknown_codes']}")
    for info in driver_results:
        if not info.get("exists"):
            issues.append(f"- ❌ Driver {info['label']} {info['year']}: file not found")
        elif info.get("error"):
            issues.append(f"- ❌ Driver {info['label']} {info['year']}: {info['error']}")

    if not issues:
        report_lines.append("✅ **No issues detected. All inspected datasets look healthy.**\n")
    else:
        report_lines.append(f"Found {len(issues)} potential issue(s):\n")
        report_lines.extend(issues)
        report_lines.append("")

    report_path = OUT_LOGS / "comprehensive_diagnostic.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log.info("\n" + "=" * 80)
    log.info(f"✓ Comprehensive report: {report_path}")
    log.info(f"✓ Per-file CSVs: {OUT_LOGS}/diag_*.csv")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
