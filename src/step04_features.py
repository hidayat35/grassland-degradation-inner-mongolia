r"""
================================================================================
STEP 04 — FEATURE ENGINEERING
================================================================================
Builds the per-pixel feature matrix that every downstream ML/causal/GNN step
consumes. One Parquet file per anchor year, ~1.1M rows × ~80 columns each.

Each row = one 1 km Albers pixel inside the Inner Mongolia AOI.
Each column = one engineered feature.

Output:
    D:\paper2_outputs\04_features\
        features_2000.parquet
        features_2005.parquet
        features_2010.parquet
        features_2015.parquet
        features_2020.parquet
        features_metadata.json        (column dictionary + provenance)

Column groups
-------------
  geometry          : row, col, x, y, lon, lat (for spatial CV blocking)
  target            : lulc_class (1..14, fused), conf_joint, conf_l1, conf_l2,
                      agreement_count
  prev_target       : lulc_class_prev (from previous anchor year, NaN for 2000),
                      lulc_change_flag (1 if class changed, 0 otherwise)
  climate_state     : 15 ETCCDI indices at year Y (or nearest available year ≤ Y)
  climate_5y_slope  : OLS slope of each index over years [Y-4 .. Y]
  climate_10y_slope : OLS slope of each index over years [Y-9 .. Y]
  climate_anomaly   : index_Y minus 30-yr-baseline (1990-2019 mean)
  drought (SPEI)    : annual_mean / gs_mean / annual_min / annual_max +
                      n_severe_drought / n_severe_wet for each of {03,06,12}
  ndvi              : annual_mean, gs_mean, gs_max at Y + 5y_slope of gs_mean
  phenology         : sos, eos, los at Y + 5y_slope of each
  grazing           : LHGI total + Cattle / Sheep / Goat / BigL densities at Y
  cropping          : GCI (categorical) at Y (or nearest)
  population        : popdens at Y
  livestock         : GLW total density (nearest year: 2010 or 2020)
  terrain (static)  : elevation, slope, aspect_sin, aspect_cos
  anthropogenic     : road_distance_km
  spatial context   : nbr3_<class>_frac for each of the 14 classes
                      (fraction of each unified class in the 3x3 neighborhood)

Memory & time
-------------
- Per-year feature matrix ≈ 1.1M rows × 80 columns × 4 bytes = ~350 MB in RAM,
  ~50-100 MB as Parquet with zstd compression.
- Per-pixel OLS trends use vectorised closed-form (no scipy stats), so trend
  computation is O(n_pixels) per index, not O(n_years × n_pixels).
- Expected runtime: 5-15 min per anchor year. Cached: re-running is fast.

Idempotency
-----------
If features_<year>.parquet exists, the year is skipped unless --force.

Run
---
$ python -m src.step04_features                        # all anchor years
$ python -m src.step04_features --years 2020           # one year
$ python -m src.step04_features --force                # rebuild even if exists
================================================================================
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import (
    OUT_FUSED, OUT_DRIVERS, OUT_FEATURES, OUT_HARMONIZED,
    ANCHOR_YEARS, ensure_dirs, TARGET_CRS, DRIVERS,
)
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import (
    log, get_aoi_mask, compute_target_grid,
)

# Suppress benign nanmean-on-empty warnings (we handle them explicitly)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered.*")


# ============================================================================
# DRIVER REGISTRY — what's where, and what's available for which years
# ============================================================================

# Climate indices: subdir name -> output prefix
CLIMATE_INDICES = ["CDD", "CWD", "GSL", "R10", "R20", "Rx1day", "Rx5day",
                   "Tn10p", "Tn90p", "TNn", "TNx", "Tx10p", "Tx90p", "TXn", "TXx"]
CLIMATE_AVAILABLE_YEARS = list(range(1982, 2021))   # ETCCDI: 1982-2020
CLIMATE_BASELINE_RANGE = (1990, 2019)               # 30-yr baseline window

# SPEI: timescale -> stats kept as features
SPEI_TIMESCALES = ["spei03", "spei06", "spei12"]
SPEI_STATS = ["annual_mean", "gs_mean", "annual_min", "annual_max",
              "n_severe_drought", "n_severe_wet"]
SPEI_AVAILABLE_YEARS = list(range(1990, 2025))

# NDVI: native annual files cover 2001-2024
NDVI_BANDS = ["annual_mean", "gs_mean", "gs_max"]
NDVI_AVAILABLE_YEARS = list(range(2001, 2025))

# Phenology: 2001-2020
PHEN_VARS = ["sos", "eos", "los"]
PHEN_AVAILABLE_YEARS = list(range(2001, 2021))

# Grazing: LHGI total + sub-species
GRAZING_AVAILABLE_YEARS = list(range(1980, 2024))   # tolerant; we check files

# Pop / cropping / GLW: yearly availability set per processor


# ============================================================================
# IO HELPERS
# ============================================================================

def _read_geotiff(path: Path) -> np.ndarray | None:
    """Read a single-band GeoTIFF, return as float32. None if missing."""
    if not Path(path).exists():
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def _nearest_available_year(target: int, available: list[int]) -> int | None:
    """Pick the year in `available` closest to `target`. None if list empty."""
    if not available:
        return None
    return min(available, key=lambda y: abs(y - target))


# ============================================================================
# OLS SLOPE — closed-form vectorised per-pixel trend
# ============================================================================

def per_pixel_slope(years: list[int], stack: np.ndarray) -> np.ndarray:
    """Compute the OLS slope per pixel over the time axis.

    Parameters
    ----------
    years : list of N int, time coordinates
    stack : (N, H, W) float32 array, with NaN allowed for missing values

    Returns
    -------
    slope : (H, W) float32 — OLS slope per pixel.
            NaN where fewer than 3 valid years are available.

    Uses the closed-form OLS solution. To avoid float32 precision loss when
    year values are ~2000 (year² is ~4M, beyond float32's ~7-digit mantissa),
    we CENTER the years before the regression. Centering doesn't change the
    slope estimate but keeps all intermediate sums in float32-safe range.
    NaN-safe: missing values are masked per-pixel; ≥3 valid years required.
    """
    if stack.shape[0] != len(years):
        raise ValueError("stack first dim must match len(years)")
    H, W = stack.shape[1:]
    # Center the time axis so sums are well-conditioned in float32
    y_arr = np.asarray(years, dtype=np.float32)
    y_mean = y_arr.mean()
    x = y_arr - y_mean   # centered; sum(x) ≈ 0 mathematically

    # Mask of valid values per pixel
    valid = np.isfinite(stack)
    n_valid = valid.sum(axis=0).astype(np.float32)

    # Zero-out invalid points (their contribution to the sums is masked out
    # via the corresponding weights)
    safe_stack = np.where(valid, stack, 0.0).astype(np.float32)
    x_b = x[:, None, None].astype(np.float32)
    x_b_valid = np.where(valid, x_b, 0.0).astype(np.float32)

    sum_x  = x_b_valid.sum(axis=0)
    sum_y  = safe_stack.sum(axis=0)
    sum_xy = (x_b_valid * safe_stack).sum(axis=0)
    sum_xx = (x_b_valid * x_b_valid).sum(axis=0)

    denom = n_valid * sum_xx - sum_x * sum_x
    numer = n_valid * sum_xy - sum_x * sum_y
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(denom > 0, numer / denom, np.nan).astype(np.float32)
    # Require at least 3 valid years for a meaningful trend
    slope = np.where(n_valid >= 3, slope, np.nan).astype(np.float32)
    return slope


# ============================================================================
# FEATURE LOADERS — each returns a dict {column_name: (H,W) float32 array}
# ============================================================================

def load_target(year: int, aoi_mask: np.ndarray) -> dict:
    """Fused LULC class + confidence + agreement (this year's anchor)."""
    out = {}
    p = OUT_FUSED / f"fused_{year}.tif"
    if not p.exists():
        log.error(f"  ✗ missing fused class for {year}: {p}")
        return out
    out["lulc_class"]      = _read_geotiff(p)
    out["conf_joint"]      = _read_geotiff(OUT_FUSED / f"confidence_{year}.tif")
    out["conf_l1"]         = _read_geotiff(OUT_FUSED / f"confidence_l1_{year}.tif")
    out["conf_l2"]         = _read_geotiff(OUT_FUSED / f"confidence_l2_{year}.tif")
    out["agreement_count"] = _read_geotiff(OUT_FUSED / f"agreement_{year}.tif")
    return out


def load_prev_target(year: int) -> dict:
    """Previous-anchor-year LULC class + change flag (vs. previous anchor)."""
    out = {}
    try:
        idx = ANCHOR_YEARS.index(year)
    except ValueError:
        return out
    if idx == 0:
        # First anchor (e.g., 2000) — no previous
        return out
    prev_year = ANCHOR_YEARS[idx - 1]
    prev = _read_geotiff(OUT_FUSED / f"fused_{prev_year}.tif")
    cur  = _read_geotiff(OUT_FUSED / f"fused_{year}.tif")
    if prev is None or cur is None:
        return out
    out["lulc_class_prev"] = prev.astype(np.float32)
    change = np.where(
        (prev > 0) & (cur > 0) & (prev != cur), 1.0,
        np.where((prev == 0) | (cur == 0), np.nan, 0.0)
    ).astype(np.float32)
    out["lulc_change_flag"] = change
    return out


def load_climate(year: int) -> dict:
    """Climate state at Y + 5-yr / 10-yr slope + anomaly vs 1990-2019 baseline.

    Up to 15 indices × (1 state + 2 slopes + 1 anomaly) = 60 columns.
    """
    out = {}
    H, W = compute_target_grid()[2], compute_target_grid()[1]
    # NB: compute_target_grid returns (transform, w, h, bounds) — fix
    transform, w, h, bounds = compute_target_grid()
    H, W = h, w

    for idx in CLIMATE_INDICES:
        idx_lc = idx.lower()
        idx_dir = OUT_DRIVERS / idx_lc
        # nearest year for "state" if exact year missing
        state_year = year if year in CLIMATE_AVAILABLE_YEARS else _nearest_available_year(year, CLIMATE_AVAILABLE_YEARS)
        if state_year is None:
            continue
        p_state = idx_dir / f"{idx_lc}_{state_year}.tif"
        state = _read_geotiff(p_state)
        if state is None:
            continue
        out[f"clim_{idx_lc}"] = state

        # 5-year slope on years [Y-4 .. Y]
        years_5 = [y for y in range(year - 4, year + 1) if y in CLIMATE_AVAILABLE_YEARS]
        if len(years_5) >= 3:
            stack = np.stack([
                _read_geotiff(idx_dir / f"{idx_lc}_{y}.tif")
                for y in years_5
            ], axis=0)
            out[f"clim_{idx_lc}_slope5y"] = per_pixel_slope(years_5, stack)
        else:
            out[f"clim_{idx_lc}_slope5y"] = np.full((H, W), np.nan, dtype=np.float32)

        # 10-year slope on years [Y-9 .. Y]
        years_10 = [y for y in range(year - 9, year + 1) if y in CLIMATE_AVAILABLE_YEARS]
        if len(years_10) >= 5:
            stack = np.stack([
                _read_geotiff(idx_dir / f"{idx_lc}_{y}.tif")
                for y in years_10
            ], axis=0)
            out[f"clim_{idx_lc}_slope10y"] = per_pixel_slope(years_10, stack)
        else:
            out[f"clim_{idx_lc}_slope10y"] = np.full((H, W), np.nan, dtype=np.float32)

        # Anomaly: current - baseline mean (1990-2019)
        baseline_years = [y for y in range(CLIMATE_BASELINE_RANGE[0],
                                            CLIMATE_BASELINE_RANGE[1] + 1)
                          if y in CLIMATE_AVAILABLE_YEARS]
        if len(baseline_years) >= 10:
            stack = np.stack([
                _read_geotiff(idx_dir / f"{idx_lc}_{y}.tif")
                for y in baseline_years
            ], axis=0)
            baseline_mean = np.nanmean(stack, axis=0)
            anom = state - baseline_mean
            out[f"clim_{idx_lc}_anomaly"] = anom.astype(np.float32)

    return out


def load_spei(year: int) -> dict:
    """SPEI features at year Y — 6 stats × 3 timescales = 18 columns."""
    out = {}
    state_year = year if year in SPEI_AVAILABLE_YEARS else _nearest_available_year(year, SPEI_AVAILABLE_YEARS)
    if state_year is None:
        return out
    for ts in SPEI_TIMESCALES:
        for stat in SPEI_STATS:
            p = OUT_DRIVERS / ts / f"{ts}_{stat}_{state_year}.tif"
            arr = _read_geotiff(p)
            if arr is not None:
                out[f"{ts}_{stat}"] = arr
    return out


def load_ndvi(year: int) -> dict:
    """NDVI features at year Y + 5-yr slope of gs_mean.

    NDVI only available 2001-2024; for year 2000 we use the nearest year (2001)
    and flag this via the column ``ndvi_year_offset``.
    """
    out = {}
    transform, w, h, bounds = compute_target_grid()
    H, W = h, w
    state_year = year if year in NDVI_AVAILABLE_YEARS else _nearest_available_year(year, NDVI_AVAILABLE_YEARS)
    if state_year is None:
        return out

    for band in NDVI_BANDS:
        p = OUT_DRIVERS / f"ndvi_{band}" / f"ndvi_{band}_{state_year}.tif"
        arr = _read_geotiff(p)
        if arr is not None:
            out[f"ndvi_{band}"] = arr

    # 5-yr slope of GS-mean (key biomass trend feature for paper 2)
    years_5 = [y for y in range(state_year - 4, state_year + 1) if y in NDVI_AVAILABLE_YEARS]
    if len(years_5) >= 3:
        stack = np.stack([
            _read_geotiff(OUT_DRIVERS / "ndvi_gs_mean" / f"ndvi_gs_mean_{y}.tif")
            for y in years_5
        ], axis=0)
        out["ndvi_gs_mean_slope5y"] = per_pixel_slope(years_5, stack)
    return out


def load_phenology(year: int) -> dict:
    """SOS / EOS / LOS at year Y + 5-yr slope of each."""
    out = {}
    state_year = year if year in PHEN_AVAILABLE_YEARS else _nearest_available_year(year, PHEN_AVAILABLE_YEARS)
    if state_year is None:
        return out
    for var in PHEN_VARS:
        # State
        p_state = OUT_DRIVERS / var / f"{var}_{state_year}.tif"
        arr = _read_geotiff(p_state)
        if arr is not None:
            out[f"phen_{var}"] = arr
        # 5-yr slope
        years_5 = [y for y in range(state_year - 4, state_year + 1) if y in PHEN_AVAILABLE_YEARS]
        if len(years_5) >= 3:
            stack_list = []
            for y in years_5:
                a = _read_geotiff(OUT_DRIVERS / var / f"{var}_{y}.tif")
                if a is None:
                    stack_list.append(np.full(stack_list[0].shape if stack_list else (1, 1), np.nan, dtype=np.float32))
                else:
                    stack_list.append(a)
            if all(s.shape == stack_list[0].shape for s in stack_list):
                stack = np.stack(stack_list, axis=0)
                out[f"phen_{var}_slope5y"] = per_pixel_slope(years_5, stack)
    return out


def load_grazing(year: int) -> dict:
    """LHGI total + Cattle / Sheep / Goat / BigL densities at Y."""
    out = {}
    # Find what years actually exist for the total
    total_dir = OUT_DRIVERS / "grazing_total"
    if total_dir.exists():
        avail = sorted(int(f.stem.split("_")[-1])
                       for f in total_dir.glob("grazing_total_*.tif")
                       if f.stem.split("_")[-1].isdigit())
        state_year = year if year in avail else _nearest_available_year(year, avail)
        if state_year is not None:
            p = total_dir / f"grazing_total_{state_year}.tif"
            arr = _read_geotiff(p)
            if arr is not None:
                out["graz_total"] = arr
    # Per-species
    for sp in ["cattle", "sheep", "goat", "bigl"]:
        sp_dir = OUT_DRIVERS / sp
        if not sp_dir.exists():
            continue
        avail = sorted(int(f.stem.split("_")[-1])
                       for f in sp_dir.glob(f"{sp}_*.tif")
                       if f.stem.split("_")[-1].isdigit())
        if not avail:
            continue
        state_year = year if year in avail else _nearest_available_year(year, avail)
        if state_year is None:
            continue
        p = sp_dir / f"{sp}_{state_year}.tif"
        arr = _read_geotiff(p)
        if arr is not None:
            out[f"graz_{sp}"] = arr
    return out


def load_cropping(year: int) -> dict:
    out = {}
    d = OUT_DRIVERS / "cropping_intensity"
    if not d.exists():
        return out
    avail = sorted(int(f.stem.split("_")[-1])
                   for f in d.glob("cropping_intensity_*.tif")
                   if f.stem.split("_")[-1].isdigit())
    if not avail:
        return out
    state_year = year if year in avail else _nearest_available_year(year, avail)
    p = d / f"cropping_intensity_{state_year}.tif"
    arr = _read_geotiff(p)
    if arr is not None:
        # GCI is uint8 categorical; cast to float and treat 255 (nodata) as NaN
        arr = arr.astype(np.float32)
        arr[arr == 255] = np.nan
        out["cropping_intensity"] = arr
    return out


def load_population(year: int) -> dict:
    out = {}
    d = OUT_DRIVERS / "popdens"
    if not d.exists():
        return out
    avail = sorted(int(f.stem.split("_")[-1])
                   for f in d.glob("popdens_*.tif")
                   if f.stem.split("_")[-1].isdigit())
    if not avail:
        return out
    state_year = year if year in avail else _nearest_available_year(year, avail)
    p = d / f"popdens_{state_year}.tif"
    arr = _read_geotiff(p)
    if arr is not None:
        out["popdens"] = arr
    return out


def load_glw(year: int) -> dict:
    """GLW gridded livestock — only 2010 and 2020 are available."""
    out = {}
    glw_dir = OUT_DRIVERS / "glw"
    if not glw_dir.exists():
        return out
    avail = sorted(int(f.stem.split("_")[-1])
                   for f in glw_dir.glob("glw_*.tif")
                   if f.stem.split("_")[-1].isdigit())
    if not avail:
        return out
    state_year = year if year in avail else _nearest_available_year(year, avail)
    p = glw_dir / f"glw_{state_year}.tif"
    arr = _read_geotiff(p)
    if arr is not None:
        out["glw_density"] = arr
    return out


def load_terrain() -> dict:
    """Static terrain features: elevation, slope, aspect_sin, aspect_cos.

    Aspect is decomposed into sin/cos to avoid the 0/360° wrap discontinuity.
    """
    out = {}
    t = OUT_DRIVERS / "terrain"
    elev   = _read_geotiff(t / "elevation.tif")
    slope  = _read_geotiff(t / "slope.tif")
    aspect = _read_geotiff(t / "aspect.tif")
    if elev   is not None: out["elevation"] = elev
    if slope  is not None: out["slope"]     = slope
    if aspect is not None:
        rad = np.deg2rad(aspect)
        out["aspect_sin"] = np.sin(rad).astype(np.float32)
        out["aspect_cos"] = np.cos(rad).astype(np.float32)
    return out


def load_roads() -> dict:
    """Static road-distance feature."""
    out = {}
    d = OUT_DRIVERS / "roads"
    # Filename may vary; we look for any *.tif in the roads folder
    if not d.exists():
        return out
    tifs = sorted(d.glob("*.tif"))
    if tifs:
        arr = _read_geotiff(tifs[0])
        if arr is not None:
            out["road_distance_km"] = arr
    return out


def load_spatial_context(lulc_class_arr: np.ndarray) -> dict:
    """3x3 neighborhood fraction of each LULC class.

    For each pixel and each of the 14 classes, compute the fraction of the
    surrounding 3x3 window that belongs to that class. This is fast for
    the ST-GNN's local-context features in step 07, and useful as a feature
    for XGBoost too (e.g. "fraction of barren in neighborhood" is a strong
    desertification predictor).
    """
    from scipy.ndimage import uniform_filter
    out = {}
    if lulc_class_arr is None:
        return out
    H, W = lulc_class_arr.shape
    # Build boolean masks per class then 3x3 average filter
    for code in range(1, 15):
        mask = (lulc_class_arr == code).astype(np.float32)
        # uniform_filter is a 3x3 mean
        nbr = uniform_filter(mask, size=3, mode="reflect")
        out[f"nbr3_class{code}_frac"] = nbr.astype(np.float32)
    return out


# ============================================================================
# MAIN PIPELINE PER YEAR
# ============================================================================

def build_year(year: int, aoi_mask: np.ndarray, force: bool = False) -> Path | None:
    out_path = OUT_FEATURES / f"features_{year}.parquet"
    if out_path.exists() and not force:
        log.info(f"  ✓ already exists: {out_path.name}")
        return out_path

    log.info(f"\n── Building features for {year} ──")
    columns: dict[str, np.ndarray] = {}

    # 1) Target (fused LULC)
    log.info("   loading target (fused LULC)...")
    columns.update(load_target(year, aoi_mask))
    if "lulc_class" not in columns:
        log.error(f"  ✗ no fused LULC for {year}; skipping year")
        return None

    # 2) Previous-year target
    log.info("   loading previous-year target...")
    columns.update(load_prev_target(year))

    # 3) Climate
    log.info("   loading climate (15 indices × state/slope5y/slope10y/anomaly)...")
    columns.update(load_climate(year))

    # 4) Drought (SPEI)
    log.info("   loading SPEI...")
    columns.update(load_spei(year))

    # 5) NDVI
    log.info("   loading NDVI...")
    columns.update(load_ndvi(year))

    # 6) Phenology
    log.info("   loading phenology...")
    columns.update(load_phenology(year))

    # 7) Grazing
    log.info("   loading grazing...")
    columns.update(load_grazing(year))

    # 8) Cropping
    log.info("   loading cropping intensity...")
    columns.update(load_cropping(year))

    # 9) Population
    log.info("   loading WorldPop...")
    columns.update(load_population(year))

    # 10) GLW
    log.info("   loading GLW livestock...")
    columns.update(load_glw(year))

    # 11) Terrain (static)
    log.info("   loading SRTM terrain...")
    columns.update(load_terrain())

    # 12) Roads (static)
    log.info("   loading road distance...")
    columns.update(load_roads())

    # 13) Spatial context (3x3 LULC-class neighborhood fractions)
    log.info("   computing 3x3 LULC neighborhood fractions...")
    columns.update(load_spatial_context(columns.get("lulc_class")))

    log.info(f"   total raw feature layers: {len(columns)}")

    # ------------------------------------------------------------------
    # Flatten to tabular: rows = pixels inside AOI
    # ------------------------------------------------------------------
    log.info("   flattening to tabular form...")
    H, W = aoi_mask.shape
    transform, _, _, _ = compute_target_grid()
    # Indices of inside-AOI pixels
    rows, cols = np.where(aoi_mask)
    n = rows.size
    log.info(f"   {n:,} pixels inside AOI")

    # Compute x, y in target CRS; and (optionally) lon, lat (reprojected once)
    xs = transform[2] + (cols + 0.5) * transform[0]
    ys = transform[5] + (rows + 0.5) * transform[4]   # transform[4] is negative
    df_data = {
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "x":   xs.astype(np.float32),
        "y":   ys.astype(np.float32),
        "year": np.full(n, year, dtype=np.int16),
    }

    # Reproject (x,y) → (lon,lat) once
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(xs, ys)
        df_data["lon"] = np.asarray(lon, dtype=np.float32)
        df_data["lat"] = np.asarray(lat, dtype=np.float32)
    except Exception as e:
        log.warning(f"   pyproj reproject failed ({e}); skipping lon/lat columns")

    # Extract values at AOI pixels from every column raster
    skipped = []
    for name, arr in columns.items():
        if arr is None:
            skipped.append(name)
            continue
        if arr.shape != (H, W):
            log.warning(f"   ⚠ shape mismatch '{name}': {arr.shape} vs ({H},{W}); skip")
            skipped.append(name)
            continue
        df_data[name] = arr[rows, cols].astype(np.float32)
    if skipped:
        log.warning(f"   skipped columns: {skipped}")

    df = pd.DataFrame(df_data)
    log.info(f"   final shape: {df.shape}")

    # Write parquet (zstd compression for size + speed)
    OUT_FEATURES.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, compression="zstd", index=False)
    except Exception:
        # Fallback to snappy (pyarrow default)
        df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    log.info(f"   ✓ wrote {out_path.name}  ({size_mb:.1f} MB)")

    # Quick QC
    log.info(f"   QC: nan-fractions of first 5 numeric columns:")
    for c in df.columns[6:11]:
        nf = df[c].isna().mean()
        log.info(f"      {c}: {100*nf:.1f}% NaN")
    log.info(f"   QC: target distribution (lulc_class top 5):")
    vc = df["lulc_class"].value_counts().head(5)
    for code, n_ in vc.items():
        name = UNIFIED_LEGEND.get(int(code), ("?", ""))[0]
        log.info(f"      class {int(code):>2} {name:<14}: {n_:>10,} px "
                 f"({100*n_/len(df):5.2f}%)")
    return out_path


# ============================================================================
# METADATA — column dictionary so step 05+ can introspect what's available
# ============================================================================

def write_metadata():
    """Write a column dictionary describing every feature in the parquets."""
    meta_path = OUT_FEATURES / "features_metadata.json"
    parquets = sorted(OUT_FEATURES.glob("features_*.parquet"))
    if not parquets:
        return
    # Read schema of latest parquet to extract column names
    df_sample = pd.read_parquet(parquets[-1])
    cols = list(df_sample.columns)

    # Group columns by prefix
    groups = {
        "geometry":   ["row", "col", "x", "y", "lon", "lat", "year"],
        "target":     ["lulc_class", "conf_joint", "conf_l1", "conf_l2",
                       "agreement_count"],
        "prev_target":[c for c in cols if c.startswith("lulc_") and "prev" in c
                       or c == "lulc_change_flag"],
        "climate":    [c for c in cols if c.startswith("clim_")],
        "spei":       [c for c in cols if c.startswith("spei")],
        "ndvi":       [c for c in cols if c.startswith("ndvi_")],
        "phenology":  [c for c in cols if c.startswith("phen_")],
        "grazing":    [c for c in cols if c.startswith("graz_")],
        "cropping":   [c for c in cols if c.startswith("cropping_")],
        "population": [c for c in cols if c == "popdens"],
        "livestock":  [c for c in cols if c == "glw_density"],
        "terrain":    [c for c in cols if c in ("elevation", "slope",
                                                  "aspect_sin", "aspect_cos")],
        "roads":      [c for c in cols if c == "road_distance_km"],
        "spatial_ctx":[c for c in cols if c.startswith("nbr3_")],
    }
    meta = {
        "anchor_years": ANCHOR_YEARS,
        "n_columns": len(cols),
        "columns_by_group": groups,
        "n_columns_per_group": {g: len(cs) for g, cs in groups.items()},
        "files": {p.name: p.stat().st_size for p in parquets},
        "schema": {c: str(df_sample[c].dtype) for c in cols},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"\n✓ Metadata written: {meta_path}")
    return meta_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="years to build (default: ANCHOR_YEARS)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild parquet even if it exists")
    args = parser.parse_args()

    ensure_dirs()
    log.info("=" * 80)
    log.info("STEP 04 — Feature engineering (per-pixel parquet matrices)")
    log.info("=" * 80)

    aoi_mask = get_aoi_mask()
    years_to_do = args.years if args.years else ANCHOR_YEARS

    written = []
    for year in years_to_do:
        path = build_year(year, aoi_mask, force=args.force)
        if path is not None:
            written.append(path)

    if written:
        write_metadata()
    log.info("=" * 80)
    log.info(f"Step 04 complete. {len(written)} parquet file(s) ready.")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
