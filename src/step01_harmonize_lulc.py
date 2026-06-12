r"""
================================================================================
STEP 01 — HARMONIZE LULC PRODUCTS
================================================================================
For every (product, year) combination available:
  1. Open native raster (may be in any CRS / resolution).
  2. Reproject to target grid (Albers, 1 km) using NEAREST resampling.
     Nearest is correct for categorical class codes (no interpolation across
     class boundaries — that would create non-existent intermediate codes).
  3. Remap native codes -> unified 13-class legend via class_harmonization.py.
  4. Apply AOI mask -> set non-IM pixels to 0 (NoData).
  5. Save as compressed GeoTIFF.

Output structure:
  D:\paper2_outputs\01_harmonized_lulc\<product>\<product>_<year>.tif

Why 1 km, not 30 m?
- At 30 m the IM AOI is ~2 billion pixels: too big for fusion + ML stacking.
- 1 km Albers gives ~1.5M pixels — fits in 64 GB RAM with all driver layers.
- For paper 1, area statistics at 1 km lose <2% accuracy vs 30 m (we'll
  validate this in supplementary materials).
- The 1 km decision also matches WorldPop + grazing intensity native res,
  avoiding artificial oversampling artifacts.

Robustness
- Missing files logged and skipped.
- Non-overlapping rasters logged and skipped.
- Corrupt rasters logged with traceback and skipped.
- GLC_FCS30D multi-band files: bands are mapped to years automatically.

Run
---
$ python -m src.step01_harmonize_lulc                 # all products
$ python -m src.step01_harmonize_lulc --product CLCD  # one product
$ python -m src.step01_harmonize_lulc --years 2000 2020  # one year
================================================================================
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import (
    LULC_PRODUCTS, OUT_HARMONIZED, ensure_dirs, ANCHOR_YEARS
)
from configs.class_harmonization import PRODUCT_REGISTRY, remap_array
from src.raster_utils import (
    log, reproject_to_target, write_target_geotiff, get_aoi_mask
)


# ============================================================================
# PER-PRODUCT FILE RESOLUTION
# ============================================================================

def list_year_file(product: str, year: int) -> tuple[Path | None, int]:
    """Return (source file path, band index) for (product, year), or (None, 1) if unavailable.

    Returns a tuple so the caller knows which BAND to read from a multi-band file
    (relevant for GLC_FCS30D and potentially MODIS).
    """
    cfg = LULC_PRODUCTS[product]

    # CCI — multiple filename patterns to try
    if product == "CCI":
        if year not in cfg["years"]:
            return None, 1
        for tmpl in cfg["year_templates"]:
            path = cfg["folder"] / tmpl.format(year=year)
            if path.exists():
                return path, 1
        return None, 1

    # WorldCover — multiple filename patterns (v100 / v200)
    if product == "WorldCover":
        if year not in cfg["years"]:
            return None, 1
        for tmpl in cfg["year_templates"]:
            path = cfg["folder"] / tmpl.format(year=year)
            if path.exists():
                return path, 1
        return None, 1

    # MODIS — one file per year (verified by inspect_modis_bands.py).
    # The file has 13 bands but we only want band 1 (LC_Type1 = IGBP).
    if product == "MODIS":
        if year not in cfg["years"]:
            return None, 1
        path = cfg["folder"] / cfg["year_template"].format(year=year)
        band = cfg.get("band", 1)
        return (path, band) if path.exists() else (None, band)

    # Simple year-per-file layout (Wang, CLCD, MPSDSL, Grassland_IM)
    if "folder" in cfg and "year_template" in cfg:
        if year not in cfg["years"]:
            return None, 1
        fn = cfg["year_template"].format(year=year)
        path = cfg["folder"] / fn
        return (path, 1) if path.exists() else (None, 1)

    # GLC_FCS30D — TWO multi-band files
    if product == "GLC_FCS30D":
        if year <= 1995:
            src_path = cfg["files"][0]
        else:
            src_path = cfg["files"][1]
        # Band number is computed by get_glc_band() — return it
        try:
            band = get_glc_band(year, src_path)
        except Exception as e:
            log.warning(f"  ⚠ GLC band lookup failed for {year}: {e}; using band 1")
            band = 1
        return src_path, band

    return None, 1


def get_glc_band(year: int, src_path: Path) -> int:
    """Pick the band index (1-based) inside a GLC_FCS30D multi-band file.

    Per the official GLC_FCS30D user guide (Zhang et al. 2024):
    - "GLC_FCS30D_19851995_5years" file has 3 bands = 1985, 1990, 1995.
    - "GLC_FCS30D_20002022" file has 23 bands = 2000, 2001, ..., 2022.
    So 2000 is band 1 of the SECOND file, NOT the last band of the first.
    """
    fname = src_path.name.lower()
    with rasterio.open(src_path) as src:
        n_bands = src.count

    if "1985" in fname or "19851995" in fname.replace("-", "").replace("_", ""):
        # 1985-1995 file: 3 bands = 1985, 1990, 1995
        if n_bands == 3:
            mapping = {1985: 1, 1990: 2, 1995: 3}
            if year in mapping:
                return mapping[year]
            # If asked for 2000 here, fall back to last band but WARN
            log.warning(f"  ⚠ requested year {year} not in 1985-1995 file (only 1985/1990/1995); using band {n_bands}")
            return n_bands
        else:
            # Unexpected band count — proportional fallback
            frac = (year - 1985) / (1995 - 1985)
            return max(1, min(n_bands, round(1 + frac * (n_bands - 1))))
    else:
        # 2000-2022 file: 23 bands = 2000..2022 inclusive
        if n_bands == 23:
            band = year - 2000 + 1
            if 1 <= band <= 23:
                return band
            log.warning(f"  ⚠ year {year} out of 2000-2022 range; clamping")
            return max(1, min(23, band))
        # Unexpected band count — proportional fallback
        frac = (year - 2000) / (2022 - 2000)
        return max(1, min(n_bands, round(1 + frac * (n_bands - 1))))


# ============================================================================
# PROCESS ONE (product, year)
# ============================================================================

def process_one(product: str, year: int, aoi_mask: np.ndarray, force: bool = False) -> bool:
    out_dir = OUT_HARMONIZED / product
    out_path = out_dir / f"{product}_{year}.tif"
    if out_path.exists() and not force:
        log.info(f"  ✓ already done: {out_path.name}")
        return True

    src_path, band = list_year_file(product, year)
    if src_path is None:
        log.info(f"  - {product} {year}: no source file (not available)")
        return False

    try:
        log.info(f"  → {product} {year}: reproject from {src_path.name} (band {band})")
        # Per-product source nodata (CCI 32767, MODIS 255, WorldCover 0, etc.)
        cfg = LULC_PRODUCTS[product]
        src_nodata = cfg.get("src_nodata")  # None = let rasterio infer from file
        native = reproject_to_target(
            src_path,
            band=band,
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            target_dtype="int32",
        )
        # Remap native codes -> unified legend (uint8)
        unified = remap_array(native, product)
        # AOI mask
        unified[~aoi_mask] = 0
        # Save
        write_target_geotiff(
            unified, out_path, dtype="uint8", nodata=0,
            tags={
                "source": str(src_path),
                "band_used": band,
                "product": product,
                "year": year,
                "legend": "unified14_paper2",
            },
        )
        # Quick QC
        unique, counts = np.unique(unified, return_counts=True)
        log.info(f"  ✓ saved {out_path.name}; class hist (top 5): "
                 f"{dict(sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1])[:5])}")
        return True
    except Exception as e:
        log.error(f"  ✗ {product} {year} FAILED: {e}")
        log.error(traceback.format_exc())
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default=None,
                        help="single product to process (default: all)")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="subset of years (default: all available)")
    parser.add_argument("--anchor-only", action="store_true",
                        help="process only ANCHOR_YEARS (2000, 2005, 2010, 2015, 2020)")
    parser.add_argument("--force", action="store_true",
                        help="re-process even if output exists")
    args = parser.parse_args()

    ensure_dirs()
    log.info("=" * 80)
    log.info("STEP 01 — Harmonize LULC products to unified 13-class legend @ 1km Albers")
    log.info("=" * 80)

    aoi_mask = get_aoi_mask()
    log.info(f"AOI mask shape: {aoi_mask.shape}, inside-AOI: {aoi_mask.sum():,}")

    products = [args.product] if args.product else list(LULC_PRODUCTS.keys())
    success, fail = 0, 0
    for product in products:
        cfg = LULC_PRODUCTS[product]
        years_avail = cfg.get("years",
                              list(range(1985, 2024)) if product == "GLC_FCS30D" else [])
        if args.anchor_only:
            years_to_do = [y for y in ANCHOR_YEARS if y in years_avail]
        elif args.years:
            years_to_do = [y for y in args.years if y in years_avail]
        else:
            years_to_do = years_avail

        log.info(f"\n── {product}: {len(years_to_do)} year(s) to process ──")
        for year in years_to_do:
            if process_one(product, year, aoi_mask, force=args.force):
                success += 1
            else:
                fail += 1

    log.info("=" * 80)
    log.info(f"DONE — {success} succeeded, {fail} failed/skipped")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
