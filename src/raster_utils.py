"""
================================================================================
RASTER UTILITIES — common reprojection / resampling / AOI masking
================================================================================
Production-quality helpers used throughout the preprocessing pipeline.

Design priorities
-----------------
1. Memory safe: stream large rasters via rasterio windows when possible.
2. CRS-agnostic: every input may be in a different CRS; we reproject ONCE to
   the target reference grid (Albers, China; 1 km).
3. Robust to non-overlap: if a raster doesn't cover the AOI, return an
   all-NoData array with a logged warning instead of crashing.
4. Reproducible: every output GeoTIFF carries a tag with the source path
   + a SHA1 of the config hash, so figures are traceable to inputs.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window
import geopandas as gpd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import (
    AOI_SHP, TARGET_CRS, TARGET_RES_M, TARGET_BOUNDS, OUT_LOGS
)


# ============================================================================
# LOGGING
# ============================================================================

def get_logger(name: str = "paper2") -> logging.Logger:
    """One logger, file + console handlers, idempotent."""
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(OUT_LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = get_logger()


# ============================================================================
# AOI HANDLING
# ============================================================================

_AOI_CACHE: dict = {}


def load_aoi_in_target_crs() -> gpd.GeoDataFrame:
    """Load the Inner Mongolia AOI, reproject to TARGET_CRS, cache."""
    key = ("aoi", TARGET_CRS)
    if key in _AOI_CACHE:
        return _AOI_CACHE[key]
    if not AOI_SHP.exists():
        raise FileNotFoundError(f"AOI not found: {AOI_SHP}")
    aoi = gpd.read_file(AOI_SHP)
    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS: {AOI_SHP}")
    aoi_t = aoi.to_crs(TARGET_CRS)
    _AOI_CACHE[key] = aoi_t
    log.info(f"Loaded AOI: {len(aoi_t)} feature(s), bounds {tuple(aoi_t.total_bounds)}")
    return aoi_t


def compute_target_grid() -> Tuple[rasterio.Affine, int, int, Tuple[float, float, float, float]]:
    """Compute the canonical (transform, width, height, bounds) for the
    target reference grid that snaps to TARGET_RES_M and contains the AOI."""
    aoi = load_aoi_in_target_crs()
    minx, miny, maxx, maxy = aoi.total_bounds
    # Snap outward to a clean multiple of TARGET_RES_M for reproducible alignment
    res = TARGET_RES_M
    minx = np.floor(minx / res) * res
    miny = np.floor(miny / res) * res
    maxx = np.ceil(maxx / res) * res
    maxy = np.ceil(maxy / res) * res
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    return transform, width, height, (minx, miny, maxx, maxy)


def get_aoi_mask() -> np.ndarray:
    """Rasterize the AOI polygon onto the target grid -> bool mask (True inside)."""
    key = ("aoi_mask", TARGET_CRS, TARGET_RES_M)
    if key in _AOI_CACHE:
        return _AOI_CACHE[key]
    aoi = load_aoi_in_target_crs()
    transform, w, h, _ = compute_target_grid()
    shapes = [(geom, 1) for geom in aoi.geometry if geom is not None]
    mask = features.rasterize(
        shapes, out_shape=(h, w), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)
    _AOI_CACHE[key] = mask
    log.info(f"AOI mask: {mask.sum():,} of {mask.size:,} pixels inside AOI "
             f"({100*mask.sum()/mask.size:.1f}%)")
    return mask


# ============================================================================
# REPROJECT TO TARGET GRID
# ============================================================================

def reproject_to_target(
    src_path: Path,
    band: int = 1,
    resampling: Resampling = Resampling.nearest,
    src_nodata: Optional[float] = None,
    target_dtype: Optional[str] = None,
) -> np.ndarray:
    """Open `src_path`, read band `band`, reproject to the target grid.

    Returns a 2D array aligned to the target grid.
    If the source doesn't overlap the AOI, returns an all-zero array
    and logs a warning — does NOT crash.

    NoData handling
    ---------------
    Categorical data (Resampling.nearest):
        src_nodata pixels become 0 in the output (no interpolation possible).
    Continuous data (Resampling.bilinear / average / cubic / ...):
        src_nodata pixels are first masked to NaN, target is float32, and the
        output also uses NaN for nodata. This prevents the classic bug where
        bilinear interpolation smears huge sentinel values (-2.1e9, -99999,
        -3.4e38) into neighboring pixels and corrupts mean/std statistics.

    Parameters
    ----------
    resampling : nearest for categorical LULC, bilinear/average for continuous.
    src_nodata : override the file's nodata value (e.g. CCI uses 32767 sentinel
                 even when the file header may say something different).
    target_dtype : explicit output dtype. If None, picks float32 for continuous
                   resampling and source dtype for nearest.
    """
    transform, w, h, bounds = compute_target_grid()
    src_path = Path(src_path)
    if not src_path.exists():
        log.warning(f"  ⚠ MISSING: {src_path}; returning zeros")
        return np.zeros((h, w), dtype=target_dtype or "uint8")

    is_categorical = (resampling == Resampling.nearest)

    try:
        with rasterio.open(src_path) as src:
            if src.crs is None:
                log.warning(f"  ⚠ NO CRS: {src_path}; returning zeros")
                return np.zeros((h, w), dtype=target_dtype or "uint8")

            # Overlap test in src CRS
            try:
                src_aoi_b = transform_bounds(TARGET_CRS, src.crs, *bounds, densify_pts=21)
                if not _bounds_intersect(src_aoi_b, src.bounds):
                    log.warning(f"  ⚠ NO AOI OVERLAP: {src_path.name}; returning zeros")
                    return np.zeros((h, w), dtype=target_dtype or "uint8")
            except Exception as e:
                log.warning(f"  ⚠ Overlap check failed for {src_path.name}: {e} — proceeding anyway")

            src_data = src.read(band)
            # Resolve effective source nodata (explicit override > file header > None)
            effective_src_nodata = src_nodata if src_nodata is not None else src.nodata

            # -----------------------------------------------------------------
            # Categorical path (nearest neighbour)
            # -----------------------------------------------------------------
            if is_categorical:
                dtype = target_dtype or src_data.dtype
                dst = np.zeros((h, w), dtype=dtype)
                reproject(
                    source=src_data,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=effective_src_nodata,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    dst_nodata=0,
                    resampling=resampling,
                    num_threads=4,
                )
                return dst

            # -----------------------------------------------------------------
            # Continuous path (bilinear / average / cubic / ...)
            # CRITICAL: pre-mask nodata to NaN BEFORE resampling, otherwise
            # bilinear smears huge sentinel values into real pixels.
            # -----------------------------------------------------------------
            src_float = src_data.astype(np.float32, copy=True)
            if effective_src_nodata is not None and not np.isnan(effective_src_nodata):
                src_float[src_data == effective_src_nodata] = np.nan
            # Also catch float nodata sentinels that may have slight precision drift
            # (e.g. -3.4028235e+38) and any non-finite values
            src_float[~np.isfinite(src_float)] = np.nan

            dst = np.full((h, w), np.nan, dtype=np.float32)
            reproject(
                source=src_float,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                dst_nodata=np.nan,
                resampling=resampling,
                num_threads=4,
            )
            # Final scrub: any non-finite values become NaN
            dst[~np.isfinite(dst)] = np.nan
            if target_dtype and target_dtype != "float32":
                # User explicitly asked for a different dtype — cast (NaN → 0 for int)
                dst = np.where(np.isnan(dst), 0, dst).astype(target_dtype)
            return dst
    except Exception as e:
        log.error(f"  ✗ FAILED reproject {src_path}: {e}; returning zeros")
        return np.zeros((h, w), dtype=target_dtype or "uint8")


def _bounds_intersect(b1, b2) -> bool:
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


# ============================================================================
# WRITE GEOTIFF ON TARGET GRID
# ============================================================================

def write_target_geotiff(
    arr: np.ndarray,
    out_path: Path,
    dtype: str = "uint8",
    nodata: float = 0,
    compress: str = "lzw",
    tags: Optional[dict] = None,
) -> None:
    """Write a 2D array to disk on the canonical target grid (Albers, 1 km)."""
    transform, w, h, _ = compute_target_grid()
    if arr.shape != (h, w):
        raise ValueError(f"shape mismatch {arr.shape} vs target {(h, w)}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": w,
        "height": h,
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def config_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ============================================================================
# SPEI NETCDF READER — extract one year's monthly slice, build per-year stats
# ============================================================================

def spei_yearly_stats(
    nc_path: Path,
    year: int,
    var_name: str = "spei",
) -> dict | None:
    """Open an SPEI NetCDF, extract all 12 monthly values for `year`, and
    compute per-pixel summary statistics on the native 0.5° grid.

    Returns a dict of 2D arrays (each (lat, lon) in the native grid):
      annual_mean         : mean SPEI over the 12 months of the year
      gs_mean             : mean over Apr-Sep (months 4-9)
      annual_min          : worst (most negative) SPEI in the year — worst drought
      annual_max          : highest (wettest) SPEI in the year
      n_severe_drought    : number of months with SPEI < -1.5 (severe drought)
      n_moderate_drought  : number of months with -1.5 ≤ SPEI < -1.0
      n_severe_wet        : number of months with SPEI > +1.5

    Also returns:
      lon, lat            : 1-D coordinate arrays
      crs                 : the geographic CRS string ("EPSG:4326" — SPEI is in WGS84)

    Returns None if the year is outside the file's time range.

    Notes
    -----
    This produces native-grid arrays; the caller is expected to write them
    to a temporary GeoTIFF and feed that into reproject_to_target() so the
    Albers 1 km resampling is consistent with every other driver.
    """
    try:
        import xarray as xr
    except ImportError:
        log.error("xarray not installed — cannot read SPEI NetCDF. "
                  "Install with: pip install xarray netCDF4")
        return None

    nc_path = Path(nc_path)
    if not nc_path.exists():
        log.warning(f"  ⚠ SPEI file missing: {nc_path}")
        return None

    with xr.open_dataset(nc_path) as ds:
        if var_name not in ds.data_vars:
            # Fall back to first data variable that is not "crs"
            candidates = [v for v in ds.data_vars if v != "crs"]
            if not candidates:
                log.error(f"  ✗ no usable variable in {nc_path}")
                return None
            var_name = candidates[0]
        da = ds[var_name]
        # Identify time / lat / lon dim names robustly
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        if "time" not in da.coords:
            log.error(f"  ✗ no 'time' coord in {nc_path}")
            return None

        # Slice the year
        try:
            year_slice = da.sel(time=str(year))
        except KeyError:
            log.warning(f"  ⚠ year {year} not in {nc_path.name}")
            return None
        if year_slice.sizes.get("time", 0) == 0:
            log.warning(f"  ⚠ year {year} not in {nc_path.name}")
            return None

        # Compute stats with NaN-safe reductions
        monthly = year_slice.values.astype(np.float32)   # (12, lat, lon)
        if monthly.shape[0] < 12:
            log.warning(f"  ⚠ {nc_path.name} {year} has only {monthly.shape[0]} months")

        # Growing-season selection: months 4..9 inclusive
        time_months = year_slice["time"].dt.month.values
        gs_mask = (time_months >= 4) & (time_months <= 9)

        annual_mean = np.nanmean(monthly, axis=0)
        gs_mean     = np.nanmean(monthly[gs_mask], axis=0) if gs_mask.any() else annual_mean
        annual_min  = np.nanmin(monthly, axis=0)
        annual_max  = np.nanmax(monthly, axis=0)
        n_severe_drought   = np.sum((monthly < -1.5) & np.isfinite(monthly), axis=0).astype(np.float32)
        n_moderate_drought = np.sum((monthly >= -1.5) & (monthly < -1.0) & np.isfinite(monthly), axis=0).astype(np.float32)
        n_severe_wet       = np.sum((monthly > 1.5) & np.isfinite(monthly), axis=0).astype(np.float32)

        return {
            "annual_mean":        annual_mean,
            "gs_mean":            gs_mean,
            "annual_min":         annual_min,
            "annual_max":         annual_max,
            "n_severe_drought":   n_severe_drought,
            "n_moderate_drought": n_moderate_drought,
            "n_severe_wet":       n_severe_wet,
            "lon": ds[lon_name].values,
            "lat": ds[lat_name].values,
            "crs": "EPSG:4326",
        }


def write_lonlat_geotiff(
    arr: np.ndarray, lon: np.ndarray, lat: np.ndarray,
    out_path: Path, crs: str = "EPSG:4326",
    nodata: float = np.nan, dtype: str = "float32",
) -> Path:
    """Write a 2D (lat, lon) array as a regular WGS84 GeoTIFF, so it can be
    fed to reproject_to_target() for the Albers 1 km warp.

    NOTE: SPEI files have latitudes in ascending order (south-to-north) but
    GeoTIFFs expect north-up. We flip the array on write.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Latitude direction
    if lat[0] < lat[-1]:
        # Ascending — flip both lat and array so the geotiff is north-up
        lat = lat[::-1]
        arr = arr[::-1, :]
    # Build affine transform
    lon_step = float(lon[1] - lon[0])
    lat_step = float(lat[0] - lat[1])  # positive (north-up after flip)
    # Pixel center -> upper-left corner: shift by half a pixel
    west  = float(lon[0]) - lon_step / 2
    north = float(lat[0]) + lat_step / 2
    transform = from_origin(west, north, lon_step, lat_step)
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
        "tiled": False,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)
    return out_path

