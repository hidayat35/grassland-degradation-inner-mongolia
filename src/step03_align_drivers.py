r"""
================================================================================
STEP 03 — DRIVER STACK ALIGNMENT
================================================================================
Aligns every driver layer to the target 1km Albers grid for direct use by
the ML/GNN step. Drivers covered:

  Climate (15 indices, 1982-2020)  : Σ → upsampled from 0.05° (~5 km) to 1 km
  Grazing intensity (LHGI, 1980-2024) : ~10 km → upsampled to 1 km
  Grazing by livestock (Cattle/Sheep/Goat/BigLivestock) : same as LHGI
  Phenology SOS / EOS / LOS (2001-2020) : already ~250 m → resampled to 1 km
  WorldPop population density (2000-2020) : ~1 km → resampled to exact target grid
  Cropping intensity (2001-2019) : ~250 m → resampled to 1 km
  Desertification degree (2001-2021, vector .shp) : rasterized to 1 km
  Sandstorm distribution (2000-2021, vector .shp) : rasterized as annual COUNT
  GRIP roads : distance-to-nearest-road in km (static covariate)
  GLW livestock counts (2010, 2020) : ~10 km → upsampled

Continuous drivers use BILINEAR resampling.
Categorical drivers (desertification degree codes) use NEAREST.

Output:
  D:\paper2_outputs\03_drivers\<variable>\<variable>_<year>.tif
  e.g.  03_drivers/cdd/cdd_2001.tif, 03_drivers/sheep/sheep_2010.tif, ...

Run
---
$ python -m src.step03_align_drivers                        # everything
$ python -m src.step03_align_drivers --group climate        # one group
$ python -m src.step03_align_drivers --group climate --years 2000 2020
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.enums import Resampling
from rasterio import features

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import (
    DRIVERS, OUT_DRIVERS, ANCHOR_YEARS, ensure_dirs, TARGET_CRS
)
from src.raster_utils import (
    log, reproject_to_target, write_target_geotiff, get_aoi_mask,
    compute_target_grid
)


# ============================================================================
# 1) CLIMATE INDICES
# ============================================================================

def process_climate_indices(years_filter=None):
    """Each subfolder = one climate index, each tif = one year.

    Index plausible ranges (used as a final safety filter even after the
    raster_utils nodata fix). Anything outside is treated as NoData.
    """
    # Plausible physical ranges. (min, max) — generous bounds.
    # CDD/CWD/GSL/R10/R20: number of days, 0-366. Rx1day/Rx5day: mm, 0-1000.
    # Tn**/Tx**/TNn/TNx/TXn/TXx: °C (or % for **p indices), -80 to 80 covers both.
    PLAUSIBLE = {
        "CDD":    (0, 366),
        "CWD":    (0, 366),
        "GSL":    (0, 366),
        "R10":    (0, 366),
        "R20":    (0, 366),
        "Rx1day": (0, 1000),
        "Rx5day": (0, 2000),
        "Tn10p":  (-1, 101),   # percentile (0-100), allow tiny slack
        "Tn90p":  (-1, 101),
        "Tx10p":  (-1, 101),
        "Tx90p":  (-1, 101),
        "TNn":    (-80, 80),
        "TNx":    (-80, 80),
        "TXn":    (-80, 80),
        "TXx":    (-80, 80),
    }
    log.info("\n── Climate indices ──")
    root = DRIVERS["climate_indices_root"]
    aoi_mask = get_aoi_mask()
    for subdir in DRIVERS["climate_subdirs"]:
        idx_dir = root / subdir
        if not idx_dir.exists():
            log.warning(f"  ⚠ missing climate subdir: {idx_dir}")
            continue
        tifs = sorted(idx_dir.glob("*.tif"))
        log.info(f"  {subdir}: {len(tifs)} year(s)")
        out_dir = OUT_DRIVERS / subdir.lower()
        lo, hi = PLAUSIBLE.get(subdir, (-1e6, 1e6))
        for f in tifs:
            # Extract year from filename (e.g. 1982CDD_T.tif -> 1982)
            m = re.search(r"(\d{4})", f.name)
            if not m:
                continue
            year = int(m.group(1))
            if years_filter and year not in years_filter:
                continue
            out_path = out_dir / f"{subdir.lower()}_{year}.tif"
            if out_path.exists():
                continue
            arr = reproject_to_target(
                f, band=1, resampling=Resampling.bilinear, target_dtype="float32"
            )
            # Final safety: clamp implausible values to NaN
            arr[(arr < lo) | (arr > hi)] = np.nan
            arr[~aoi_mask] = np.nan
            write_target_geotiff(
                arr, out_path, dtype="float32", nodata=np.nan,
                tags={"variable": subdir, "year": year,
                      "plausible_range": f"[{lo}, {hi}]"},
            )


# ============================================================================
# 2) GRAZING INTENSITY (total LHGI + per-livestock)
# ============================================================================

def process_grazing(years_filter=None):
    log.info("\n── Grazing intensity ──")
    aoi_mask = get_aoi_mask()
    root = DRIVERS["grazing_root"]

    # Total LHGI
    out_dir = OUT_DRIVERS / "grazing_total"
    for f in sorted(root.glob("LHGI_*.tif")):
        m = re.search(r"LHGI_(\d{4})\.tif", f.name)
        if not m:
            continue
        year = int(m.group(1))
        if years_filter and year not in years_filter:
            continue
        out_path = out_dir / f"grazing_total_{year}.tif"
        if out_path.exists():
            continue
        arr = reproject_to_target(f, band=1, resampling=Resampling.bilinear, target_dtype="float32")
        arr[~aoi_mask] = np.nan
        write_target_geotiff(arr, out_path, dtype="float32", nodata=np.nan,
                             tags={"variable": "LHGI", "year": year})
        log.info(f"  ✓ grazing_total {year}")

    # Per-livestock
    for species, rel_tmpl in DRIVERS["grazing_subspecies"].items():
        out_dir = OUT_DRIVERS / species.lower()
        # Glob within sub-dir
        subdir, name_tmpl = rel_tmpl.split("/", 1)
        files = sorted((root / subdir).glob("*.tif"))
        log.info(f"  {species}: {len(files)} year(s)")
        for f in files:
            m = re.search(r"(\d{4})", f.name)
            if not m:
                continue
            year = int(m.group(1))
            if years_filter and year not in years_filter:
                continue
            out_path = out_dir / f"{species.lower()}_{year}.tif"
            if out_path.exists():
                continue
            arr = reproject_to_target(f, band=1, resampling=Resampling.bilinear, target_dtype="float32")
            arr[~aoi_mask] = np.nan
            write_target_geotiff(arr, out_path, dtype="float32", nodata=np.nan,
                                 tags={"variable": species, "year": year})


# ============================================================================
# 3) PHENOLOGY (SOS / EOS / LOS)
# ============================================================================

def process_phenology(years_filter=None):
    """Phenology rasters (SOS/EOS/LOS) are day-of-year values 1-365 with nodata
    typically 65535 (uint16 max). raster_utils masks file nodata to NaN; we
    additionally enforce the physically plausible DOY range."""
    log.info("\n── Phenology (SOS / EOS / LOS) ──")
    aoi_mask = get_aoi_mask()
    spec = [
        ("sos", DRIVERS["phenology_sos_folder"], "{year}SOS.tif"),
        ("eos", DRIVERS["phenology_eos_folder"], "{year}EOS.tif"),
        ("los", DRIVERS["phenology_los_folder"], "{year}LOS.tif"),
    ]
    for name, folder, tmpl in spec:
        out_dir = OUT_DRIVERS / name
        for year in range(2001, 2021):
            if years_filter and year not in years_filter:
                continue
            f = folder / tmpl.format(year=year)
            if not f.exists():
                continue
            out_path = out_dir / f"{name}_{year}.tif"
            if out_path.exists():
                continue
            arr = reproject_to_target(f, band=1, resampling=Resampling.bilinear, target_dtype="float32")
            # Sanity range: DOY must be 1..366 (LOS up to ~365 too).
            # Anything outside the plausible range is treated as NoData.
            arr[(arr < 1) | (arr > 366)] = np.nan
            arr[~aoi_mask] = np.nan
            write_target_geotiff(arr, out_path, dtype="float32", nodata=np.nan,
                                 tags={"variable": name.upper(), "year": year})
            log.info(f"  ✓ {name} {year}")


# ============================================================================
# 4) WORLDPOP POPULATION DENSITY
# ============================================================================

def process_population(years_filter=None):
    """WorldPop nodata is -99999 in the file header. raster_utils handles it
    correctly now — we just need to mask outside-AOI to NaN."""
    log.info("\n── WorldPop ──")
    aoi_mask = get_aoi_mask()
    out_dir = OUT_DRIVERS / "popdens"
    for year in DRIVERS["worldpop_years"]:
        if years_filter and year not in years_filter:
            continue
        f = DRIVERS["worldpop_folder"] / DRIVERS["worldpop_template"].format(year=year)
        if not f.exists():
            continue
        out_path = out_dir / f"popdens_{year}.tif"
        if out_path.exists():
            continue
        arr = reproject_to_target(f, band=1, resampling=Resampling.bilinear, target_dtype="float32")
        arr[~aoi_mask] = np.nan
        write_target_geotiff(arr, out_path, dtype="float32", nodata=np.nan,
                             tags={"variable": "popdens", "year": year})
        log.info(f"  ✓ popdens {year}")


# ============================================================================
# 5) CROPPING INTENSITY
# ============================================================================

def process_cropping_intensity(years_filter=None):
    """GCI is categorical (0=no crop, 1=single, 2=double, 3=triple, 200=fallow, etc).
    Nearest-neighbour resampling is correct — bilinear would invent decimals."""
    log.info("\n── Cropping intensity ──")
    aoi_mask = get_aoi_mask()
    out_dir = OUT_DRIVERS / "cropping_intensity"
    for year in range(2001, 2020):
        if years_filter and year not in years_filter:
            continue
        f = DRIVERS["cropping_intensity_folder"] / DRIVERS["cropping_intensity_template"].format(year=year)
        if not f.exists():
            continue
        out_path = out_dir / f"cropping_intensity_{year}.tif"
        if out_path.exists():
            continue
        # nearest preserves integer class codes; uint8 because native is 0-255
        arr = reproject_to_target(
            f, band=1, resampling=Resampling.nearest, target_dtype="uint8"
        )
        arr[~aoi_mask] = 255  # 255 = NoData outside AOI
        write_target_geotiff(arr, out_path, dtype="uint8", nodata=255,
                             tags={"variable": "GCI", "year": year})
        log.info(f"  ✓ GCI {year}")


# ============================================================================
# 6) DESERTIFICATION DEGREE  (.shp per year, polygon with 'gridcode')
# ============================================================================

def process_desertification(years_filter=None):
    log.info("\n── Desertification degree (rasterize from .shp) ──")
    aoi_mask = get_aoi_mask()
    transform, w, h, _ = compute_target_grid()
    root = DRIVERS["desertification_root"]
    if not root.exists():
        log.warning(f"  ⚠ missing desertification root: {root}")
        return

    out_dir = OUT_DRIVERS / "desertification"
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        m = re.search(r"(\d{4})", year_dir.name)
        if not m:
            continue
        year = int(m.group(1))
        if years_filter and year not in years_filter:
            continue
        out_path = out_dir / f"desertification_{year}.tif"
        if out_path.exists():
            continue

        shps = sorted(year_dir.glob("*.shp"))
        if not shps:
            continue
        # Use the first shp (datasets usually have one main shp per year)
        shp = shps[0]
        try:
            gdf = gpd.read_file(shp).to_crs(TARGET_CRS)
            if "gridcode" not in gdf.columns:
                log.warning(f"  ⚠ no gridcode column in {shp.name}")
                continue
            shapes = ((g, int(c)) for g, c in zip(gdf.geometry, gdf["gridcode"]))
            arr = features.rasterize(
                shapes, out_shape=(h, w), transform=transform,
                fill=0, dtype="uint8", all_touched=False,
            )
            arr[~aoi_mask] = 0
            write_target_geotiff(arr, out_path, dtype="uint8", nodata=0,
                                 tags={"variable": "desertification_degree", "year": year})
            log.info(f"  ✓ desertification {year}")
        except Exception as e:
            log.error(f"  ✗ desertification {year} failed: {e}")


# ============================================================================
# 7) SANDSTORM ANNUAL COUNT (rasterize event polygons, sum events per pixel)
# ============================================================================

def process_sandstorm(years_filter=None):
    log.info("\n── Sandstorm event count ──")
    aoi_mask = get_aoi_mask()
    transform, w, h, _ = compute_target_grid()
    root = DRIVERS["sandstorm_root"]
    if not root.exists():
        log.warning(f"  ⚠ missing sandstorm root")
        return
    out_dir = OUT_DRIVERS / "sandstorm_count"

    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        m = re.search(r"(\d{4})", year_dir.name)
        if not m:
            continue
        year = int(m.group(1))
        if years_filter and year not in years_filter:
            continue
        out_path = out_dir / f"sandstorm_count_{year}.tif"
        if out_path.exists():
            continue

        # Accumulate event presence across all date subfolders
        accum = np.zeros((h, w), dtype=np.uint16)
        for date_dir in sorted(year_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            shps = sorted(date_dir.glob("*.shp"))
            if not shps:
                continue
            try:
                gdf = gpd.read_file(shps[0]).to_crs(TARGET_CRS)
                event = features.rasterize(
                    ((g, 1) for g in gdf.geometry if g is not None),
                    out_shape=(h, w), transform=transform,
                    fill=0, dtype="uint8", all_touched=True,
                )
                accum += event
            except Exception as e:
                log.warning(f"    ⚠ sandstorm {date_dir.name}: {e}")
        accum[~aoi_mask] = 0
        write_target_geotiff(accum, out_path, dtype="uint16", nodata=0,
                             tags={"variable": "sandstorm_count_per_year", "year": year})
        log.info(f"  ✓ sandstorm_count {year} (max events/cell = {accum.max()})")


# ============================================================================
# 8) GRIP ROADS — distance-to-nearest-road (static)
# ============================================================================

def process_roads():
    log.info("\n── GRIP roads (distance-to-nearest, static) ──")
    aoi_mask = get_aoi_mask()
    transform, w, h, _ = compute_target_grid()
    out_path = OUT_DRIVERS / "road_distance_km" / "road_distance_km.tif"
    if out_path.exists():
        log.info("  ✓ road_distance_km already exists")
        return

    roads_shp = DRIVERS["grip_roads"]
    if not roads_shp.exists():
        log.warning(f"  ⚠ GRIP roads file not found: {roads_shp}")
        return

    log.info("  reading GRIP roads (this can take a few minutes — large shapefile)...")
    # Read with a bbox to limit memory: only roads within AOI bounds (with buffer)
    aoi = gpd.read_file(roads_shp.parent / roads_shp.name, rows=None)
    # Above line reads everything. To limit, use the AOI bbox via mask:
    # gpd.read_file supports bbox=, but only with pyogrio engine. Try it:
    from configs.paths import AOI_SHP
    aoi_geom = gpd.read_file(AOI_SHP).to_crs(aoi.crs)
    bbox = tuple(aoi_geom.total_bounds)
    log.info(f"  clipping to AOI bbox in source CRS: {bbox}")
    roads = gpd.read_file(roads_shp, bbox=bbox).to_crs(TARGET_CRS)
    log.info(f"  {len(roads)} road segments after bbox clip")

    # Rasterize roads to binary presence
    road_pres = features.rasterize(
        ((g, 1) for g in roads.geometry if g is not None),
        out_shape=(h, w), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    )
    # Distance transform (cells -> km)
    from scipy.ndimage import distance_transform_edt
    dist_cells = distance_transform_edt(road_pres == 0)
    dist_km = (dist_cells * 1.0).astype(np.float32)  # 1 cell = 1 km
    dist_km[~aoi_mask] = np.nan
    write_target_geotiff(dist_km, out_path, dtype="float32", nodata=np.nan,
                         tags={"variable": "road_distance_km", "source": "GRIP4_region6"})
    log.info(f"  ✓ road_distance_km saved")


# ============================================================================
# 9) GLW livestock totals (2010 & 2020)
# ============================================================================

def process_glw():
    """GLW nodata is -3.4028e38 (float32 min). raster_utils now masks it to NaN
    correctly during reprojection."""
    log.info("\n── GLW gridded livestock (2010, 2020) ──")
    aoi_mask = get_aoi_mask()
    for year, key in [(2010, "glw_2010"), (2020, "glw_2020")]:
        f = DRIVERS[key]
        if not f.exists():
            log.warning(f"  ⚠ missing {f}")
            continue
        out_path = OUT_DRIVERS / "glw" / f"glw_{year}.tif"
        if out_path.exists():
            continue
        arr = reproject_to_target(f, band=1, resampling=Resampling.bilinear, target_dtype="float32")
        arr[~aoi_mask] = np.nan
        write_target_geotiff(arr, out_path, dtype="float32", nodata=np.nan,
                             tags={"variable": "GLW_total_livestock", "year": year})
        log.info(f"  ✓ GLW {year}")


# ============================================================================
# MAIN
# ============================================================================

# ============================================================================
# 10) MODIS NDVI annual statistics (MOD13A2 → annual / GS_mean / GS_max)
# ============================================================================

def process_ndvi(years_filter=None):
    """Align MODIS NDVI annual GeoTIFFs to the target 1km Albers grid.

    Each input has 3 bands: ANNUAL_MEAN, GS_MEAN, GS_MAX (int16, scale 0.0001).
    We write 3 SEPARATE single-band GeoTIFFs per year, in float32 NDVI units
    (multiplied by 0.0001), so the feature engineer can mix-and-match.
    """
    log.info("\n── MODIS NDVI annual ──")
    folder   = DRIVERS["ndvi_folder"]
    template = DRIVERS["ndvi_template"]
    years    = DRIVERS["ndvi_years"]
    band_names = DRIVERS["ndvi_band_names"]
    scale    = DRIVERS["ndvi_scale"]
    aoi_mask = get_aoi_mask()

    for year in years:
        if years_filter and year not in years_filter:
            continue
        f = folder / template.format(year=year)
        if not f.exists():
            log.warning(f"  ⚠ NDVI {year}: missing {f.name}")
            continue
        for band_idx, band_name in enumerate(band_names, start=1):
            out_dir = OUT_DRIVERS / f"ndvi_{band_name}"
            out_path = out_dir / f"ndvi_{band_name}_{year}.tif"
            if out_path.exists():
                continue
            # int16 → use nearest-neighbour-equivalent (continuous data is OK
            # with bilinear, but NDVI has a sharp masking pattern around
            # cloud/water that bilinear would smear). Use bilinear; the fixed
            # reproject_to_target handles nodata properly.
            arr = reproject_to_target(
                f, band=band_idx,
                resampling=Resampling.bilinear,
                src_nodata=None,        # MODIS export has nodata=None; 0 = masked
                target_dtype="float32",
            )
            # Apply NDVI scale (0.0001) AFTER reprojection
            arr = arr * scale
            # Clamp to physically valid NDVI range [-0.2, 1.0]; anything outside
            # is a leftover masked-as-zero pixel or interpolation artifact.
            arr[(arr < -0.2) | (arr > 1.0)] = np.nan
            arr[~aoi_mask] = np.nan
            write_target_geotiff(
                arr, out_path, dtype="float32", nodata=np.nan,
                tags={"variable": f"NDVI_{band_name}", "year": year,
                      "source_band": band_idx, "native_scale": scale},
            )
            log.info(f"  ✓ NDVI {band_name} {year}")


# ============================================================================
# 11) SRTM terrain (elevation / slope / aspect) — STATIC
# ============================================================================

def process_srtm():
    """Align the single SRTM 3-band file to the target 1km Albers grid.

    Writes 3 separate single-band GeoTIFFs: elevation, slope, aspect.
    Static — no year filter.
    """
    log.info("\n── SRTM terrain (elevation / slope / aspect) ──")
    src_path = DRIVERS["srtm_file"]
    band_names = DRIVERS["srtm_band_names"]
    src_nodata = DRIVERS["srtm_src_nodata"]
    aoi_mask = get_aoi_mask()

    if not src_path.exists():
        log.warning(f"  ⚠ SRTM file missing: {src_path}")
        return

    for band_idx, band_name in enumerate(band_names, start=1):
        out_dir = OUT_DRIVERS / "terrain"
        out_path = out_dir / f"{band_name}.tif"
        if out_path.exists():
            log.info(f"  ✓ {band_name} already exists")
            continue
        # Elevation/slope use bilinear. Aspect is circular (degrees 0-360 wrap
        # around) — bilinear can produce nonsense near the 0/360 discontinuity.
        # The cleanest fix is to decompose aspect into sin/cos before resampling,
        # but for our purposes nearest-neighbour is a safe approximation for the
        # 90m→1km step (we lose some smoothness but never have spurious values).
        if band_name == "aspect":
            resamp = Resampling.nearest
        else:
            resamp = Resampling.bilinear
        arr = reproject_to_target(
            src_path, band=band_idx,
            resampling=resamp,
            src_nodata=src_nodata,
            target_dtype="float32",
        )
        # Plausible-range filter
        if band_name == "elevation":
            arr[(arr < -500) | (arr > 9000)] = np.nan
        elif band_name == "slope":
            arr[(arr < 0) | (arr > 90)] = np.nan
        elif band_name == "aspect":
            arr[(arr < 0) | (arr > 360)] = np.nan
        arr[~aoi_mask] = np.nan
        write_target_geotiff(
            arr, out_path, dtype="float32", nodata=np.nan,
            tags={"variable": band_name, "source": "SRTM_1arcsec",
                  "static": "true"},
        )
        log.info(f"  ✓ {band_name}")


# ============================================================================
# 12) SPEI (3 timescales × per-year statistics from monthly NetCDF)
# ============================================================================

def process_spei(years_filter=None):
    """Extract per-year SPEI statistics from the CSIC NetCDF files and align
    them to the target 1km Albers grid.

    For each timescale (spei03, spei06, spei12) and each year, we compute and
    save 7 statistics as separate GeoTIFFs:
       annual_mean, gs_mean, annual_min, annual_max,
       n_severe_drought, n_moderate_drought, n_severe_wet

    Years processed: FULL_PERIOD intersection with the file's time range.
    Implementation: extract native-grid stats → write a temporary WGS84
    GeoTIFF → reproject_to_target() handles the warp to Albers 1 km exactly
    like every other driver. Temporary file is deleted after.
    """
    from src.raster_utils import spei_yearly_stats, write_lonlat_geotiff
    import tempfile

    log.info("\n── SPEI drought index ──")
    folder = DRIVERS["spei_folder"]
    timescales = DRIVERS["spei_timescales"]
    template = DRIVERS["spei_template"]
    var_name = DRIVERS["spei_var"]
    aoi_mask = get_aoi_mask()

    # Default years: 1990-2024 (full analysis period, intersected with file coverage)
    default_years = list(range(1990, 2025))

    stat_names = ["annual_mean", "gs_mean", "annual_min", "annual_max",
                  "n_severe_drought", "n_moderate_drought", "n_severe_wet"]

    for ts in timescales:
        nc_path = folder / template.format(ts=ts)
        if not nc_path.exists():
            log.warning(f"  ⚠ missing {nc_path}")
            continue
        log.info(f"  {ts}: opening {nc_path.name}")

        for year in default_years:
            if years_filter and year not in years_filter:
                continue
            # Check if all outputs already exist for this (ts, year)
            out_dir = OUT_DRIVERS / ts
            all_exist = all(
                (out_dir / f"{ts}_{stat}_{year}.tif").exists()
                for stat in stat_names
            )
            if all_exist:
                continue

            stats = spei_yearly_stats(nc_path, year, var_name=var_name)
            if stats is None:
                continue

            for stat in stat_names:
                out_path = out_dir / f"{ts}_{stat}_{year}.tif"
                if out_path.exists():
                    continue
                arr_native = stats[stat]
                # Write temp WGS84 GeoTIFF, then reproject to Albers 1 km
                with tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    write_lonlat_geotiff(
                        arr_native, stats["lon"], stats["lat"],
                        tmp_path, crs=stats["crs"],
                        nodata=np.nan, dtype="float32",
                    )
                    arr = reproject_to_target(
                        tmp_path, band=1,
                        resampling=Resampling.bilinear,
                        target_dtype="float32",
                    )
                finally:
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                arr[~aoi_mask] = np.nan
                write_target_geotiff(
                    arr, out_path, dtype="float32", nodata=np.nan,
                    tags={"variable": f"{ts}_{stat}", "year": year,
                          "source": "CSIC SPEIbase v2.10"},
                )
            log.info(f"  ✓ {ts} {year}")



GROUPS = {
    "climate": process_climate_indices,
    "grazing": process_grazing,
    "phenology": process_phenology,
    "population": process_population,
    "cropping": process_cropping_intensity,
    "desertification": process_desertification,
    "sandstorm": process_sandstorm,
    "roads": lambda yf=None: process_roads(),  # static, ignores year filter
    "glw": lambda yf=None: process_glw(),      # 2010 & 2020 only
    "ndvi": process_ndvi,
    "srtm": lambda yf=None: process_srtm(),    # static, ignores year filter
    "spei": process_spei,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, default=None,
                        choices=list(GROUPS.keys()),
                        help="single driver group (default: all)")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    args = parser.parse_args()

    ensure_dirs()
    log.info("=" * 80)
    log.info("STEP 03 — Align driver datasets to target 1km Albers grid")
    log.info("=" * 80)

    groups_to_run = [args.group] if args.group else list(GROUPS.keys())
    for g in groups_to_run:
        GROUPS[g](args.years)

    log.info("=" * 80)
    log.info("Driver alignment complete.")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
