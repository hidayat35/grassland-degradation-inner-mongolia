r"""
================================================================================
mapfig.py — publication map figures with boundary overlay, GeoTIFF + CSV
================================================================================
Standard helper for EVERY map figure in Paper 2, enforcing the Q1 conventions:

  1. The Inner Mongolia boundary (from AOI_SHP) is reprojected to the map CRS
     (TARGET_CRS, Albers) and drawn as an outline on every panel, so the region
     is always oriented and legible.
  2. Each map is written as a georeferenced GeoTIFF (data-availability artifact)
     AND as PNG/PDF/SVG (manuscript).
  3. Each figure emits a CSV companion of the underlying per-cell values
     (node/pixel coordinates + value) so Results text can be written from real
     numbers and the data is reproducible.

Use from any render script:

    from src.mapfig import render_map, render_categorical_map, boundary_in_albers

    render_map(
        array=arr2d, transform=tf, title="...", cbar_label="...",
        out_stem=OUT/"fig_xxx", cmap="RdBu_r", vmin=-0.5, vmax=0.5,
        csv_values=dict(node=..., x=..., y=..., lon=..., lat=..., value=...),  # optional
    )

The boundary is cached after first load. If geopandas/pyproj are unavailable
or the shapefile is missing, the map still renders (without the outline) and a
warning is logged — figures never silently fail.
================================================================================
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import AOI_SHP, TARGET_CRS
from src.raster_utils import log

_BOUNDARY_CACHE = {"geom": None, "loaded": False, "extent": None}


def boundary_in_albers():
    """Load the AOI shapefile and reproject to TARGET_CRS (Albers).

    Returns a list of (xs, ys) exterior-ring arrays ready to plot, plus the
    bounding extent (minx, miny, maxx, maxy). Cached after first call.
    Returns (None, None) if the shapefile or geopandas is unavailable.
    """
    if _BOUNDARY_CACHE["loaded"]:
        return _BOUNDARY_CACHE["geom"], _BOUNDARY_CACHE["extent"]
    _BOUNDARY_CACHE["loaded"] = True
    try:
        import geopandas as gpd
        if not Path(AOI_SHP).exists():
            log.warning(f"  [mapfig] boundary shapefile not found: {AOI_SHP} "
                        f"— maps will render without outline")
            return None, None
        gdf = gpd.read_file(AOI_SHP).to_crs(TARGET_CRS)
        rings = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            for poly in polys:
                xs, ys = poly.exterior.xy
                rings.append((np.asarray(xs), np.asarray(ys)))
                for interior in poly.interiors:
                    ix, iy = interior.xy
                    rings.append((np.asarray(ix), np.asarray(iy)))
        minx, miny, maxx, maxy = gdf.total_bounds
        _BOUNDARY_CACHE["geom"] = rings
        _BOUNDARY_CACHE["extent"] = (minx, miny, maxx, maxy)
        log.info(f"  [mapfig] boundary loaded: {len(rings)} ring(s), reprojected to Albers")
        return rings, (minx, miny, maxx, maxy)
    except Exception as e:
        log.warning(f"  [mapfig] could not load boundary ({e}) — maps render without outline")
        return None, None


def _overlay_boundary(ax, transform, arr_shape, lw=1.0, color="black"):
    """Draw boundary rings on ax, converting Albers metres -> pixel coordinates
    matching imshow's default (origin upper-left, no extent)."""
    rings, _ = boundary_in_albers()
    if rings is None:
        return
    a = transform.a; e = transform.e            # pixel sizes (e<0)
    c = transform.c; f = transform.f            # origin (top-left x, y)
    for xs, ys in rings:
        col = (xs - c) / a
        row = (ys - f) / e
        ax.plot(col, row, color=color, lw=lw, zorder=10)


def _write_geotiff(array, transform, out_tif):
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": array.shape[0], "width": array.shape[1],
               "crs": TARGET_CRS, "transform": transform,
               "nodata": np.nan, "compress": "lzw"}
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def _write_csv(csv_values, out_csv):
    if csv_values is None:
        return
    pd.DataFrame(csv_values).to_csv(out_csv, index=False)


def render_map(array, transform, title, cbar_label, out_stem,
               cmap="viridis", vmin=None, vmax=None, interpolation="nearest",
               subtitle=None, csv_values=None, write_tif=True,
               figsize=(12, 9), boundary_lw=1.0,
               land_base=None, cbar_extend="neither"):
    """Render a continuous-valued map with boundary overlay + multi-format save
    + GeoTIFF + optional CSV companion.

    out_stem: Path without extension (e.g. OUT/"fig_spillover"); produces
              fig_spillover.{png,pdf,svg,tif} and (if csv_values) .csv

    land_base: optional 2-D boolean array (same shape as `array`) marking valid
               land cells. Where True, a light-grey base is drawn UNDER the data
               so that valid-but-near-zero cells are visibly distinguished from
               the no-data page background (avoids a 'washed-out / blank' look).
    cbar_extend: 'neither' | 'max' | 'min' | 'both' — draw an arrow on the
               colour bar when the scale is capped (e.g. vmax below the data max).
    write_tif writes the RAW `array` (not the display-clipped version) for data
               availability.
    """
    out_stem = Path(out_stem)
    masked = np.ma.masked_invalid(array)
    fig, ax = plt.subplots(figsize=figsize)
    # grey land base (drawn first, beneath the data layer)
    if land_base is not None:
        from matplotlib.colors import ListedColormap
        base = np.where(land_base, 1.0, np.nan)
        ax.imshow(np.ma.masked_invalid(base),
                  cmap=ListedColormap(["#e9e9e9"]),
                  interpolation="nearest", zorder=0)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation=interpolation, zorder=1)
    _overlay_boundary(ax, transform, array.shape, lw=boundary_lw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, extend=cbar_extend)
    cb.set_label(cbar_label, fontsize=10)
    if subtitle:
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, style="italic")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_stem}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    if write_tif:
        _write_geotiff(array, transform, f"{out_stem}.tif")
    _write_csv(csv_values, f"{out_stem}.csv")
    log.info(f"  [mapfig] wrote {out_stem.name}.[png/pdf/svg"
             + ("/tif" if write_tif else "")
             + ("/csv" if csv_values is not None else "") + "]")


def render_categorical_map(array, transform, title, out_stem, legend,
                           subtitle=None, csv_values=None, write_tif=True,
                           figsize=(13, 9), boundary_lw=1.0):
    """Render an integer class map with a discrete legend + boundary overlay.

    legend: dict {code(int): (label(str), color(hex))}, codes 1..K; 0 = masked.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    out_stem = Path(out_stem)
    codes = sorted(legend)
    cmap = ListedColormap([legend[c][1] for c in codes])
    masked = np.ma.masked_where(~np.isfinite(array) | (array == 0), array)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, cmap=cmap, vmin=min(codes) - 0.5, vmax=max(codes) + 0.5,
                   interpolation="nearest")
    _overlay_boundary(ax, transform, array.shape, lw=boundary_lw)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    patches = [Patch(color=legend[c][1], label=legend[c][0]) for c in codes]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=11, frameon=False)
    if subtitle:
        ax.text(0.5, -0.05, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, style="italic")
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{out_stem}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    if write_tif:
        _write_geotiff(array.astype(np.float32), transform, f"{out_stem}.tif")
    _write_csv(csv_values, f"{out_stem}.csv")
    log.info(f"  [mapfig] wrote {out_stem.name} (categorical)")


if __name__ == "__main__":
    # self-test: load boundary, report
    rings, extent = boundary_in_albers()
    if rings is not None:
        print(f"Boundary OK: {len(rings)} ring(s)")
        print(f"Extent (Albers m): {extent}")
    else:
        print("Boundary not available — check AOI_SHP path and geopandas install")
