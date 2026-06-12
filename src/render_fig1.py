r"""
Figure 1 — study-area map of Inner Mongolia.

A publication study-area figure with full cartographic furniture:
  * SRTM elevation base with hillshade relief
  * the seven ecological/aridity zones (used in the causal analysis) as
    coloured boundaries overlaid on the relief
  * the Inner Mongolia outline (AOI_SHP)
  * a scale bar in kilometres
  * a north arrow
  * an OUTER graticule with degree labels (e.g. 105°E, 45°N), no minutes/seconds
  * an inset locator showing Inner Mongolia's position within China

Everything is plotted in the project Albers CRS (TARGET_CRS); the graticule is
produced by transforming whole-degree lon/lat lines into Albers, so the grid is
correctly curved for the projection.

Run locally:
    python -m src.render_fig1

Reads:
    SRTM_DEM           D:\other datasets\SRTM\SRTM_DEM_slope_aspect_IM.tif  (band 1 = elevation)
    AOI_SHP            Inner Mongolia outline (configs.paths)
    fused_2020.tif     (OUT_FUSED) — used only to derive the ecological zones
Optional (for the inset):
    CHINA_SHP          a national/provincial boundary shapefile, if available;
                       set the path below. If absent, the inset is drawn from the
                       IM outline alone (still shows position via the graticule).
Writes (to OUT_FIGURES/"fig1"):
    fig1_study_area.{png,pdf,svg}
    fig1_elevation_zones.tif         elevation clipped to AOI (data availability)
    fig1_zone_areas.csv              per-zone area (km2 and %) companion table
"""
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyArrow
from matplotlib import patheffects
from pyproj import Transformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import AOI_SHP, OUT_FUSED, OUT_FIGURES, TARGET_CRS
from src.raster_utils import log

OUT1 = OUT_FIGURES / "fig1"
SRTM = Path(r"D:\other datasets\SRTM\SRTM_DEM_slope_aspect_IM.tif")
# Optional national boundary for the inset (set to a real path if you have one):
CHINA_SHP = None   # e.g. Path(r"D:\shape files\china\china_provinces.shp")

# Ecological zones (same mapping as the causal analysis build_zones)
ZONE_MAP = {1: [1, 3], 2: [4], 3: [5], 4: [6], 5: [9], 6: [13], 7: [11, 12]}
ZONE_NAMES = {1: "Forest / meadow steppe", 2: "Typical steppe", 3: "Dry steppe",
              4: "Desert steppe", 5: "Cropland", 6: "Sandy land", 7: "Desert / bare"}
ZONE_COLORS = {1: "#1a6b3c", 2: "#74c476", 3: "#c2a23a", 4: "#9c7a30",
               5: "#e8722a", 6: "#efe0a8", 7: "#cda15a"}


def _albers_grid_from(src_path):
    """Reproject a source raster (any CRS) to TARGET_CRS at ~1 km and return
    (array, transform, (minx,miny,maxx,maxy))."""
    with rasterio.open(src_path) as src:
        dst_crs = TARGET_CRS
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=1000)
        dst = np.full((h, w), np.nan, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear, dst_nodata=np.nan)
    minx = transform.c; maxy = transform.f
    maxx = minx + transform.a * w; miny = maxy + transform.e * h
    return dst, transform, (minx, miny, maxx, maxy)


def _read_albers(path):
    with rasterio.open(path) as s:
        return s.read(1).astype("float32"), s.transform


def _boundary_rings():
    import geopandas as gpd
    gdf = gpd.read_file(AOI_SHP).to_crs(TARGET_CRS)
    rings = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            rings.append((np.asarray(xs), np.asarray(ys)))
    return rings, gdf


def _extent_xy(transform, shape):
    H, W = shape
    minx = transform.c; maxy = transform.f
    maxx = minx + transform.a * W; miny = maxy + transform.e * H
    return (minx, maxx, miny, maxy)


def _draw_graticule(ax, extent, fwd, lons, lats):
    """Draw curved whole-degree lon/lat lines and label them on the outer frame."""
    minx, maxx, miny, maxy = extent
    # meridians
    yy = np.linspace(miny, maxy, 200)
    inv = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    # we need lat range to draw a meridian; sample using inverse at left/right mid
    for lon in lons:
        lat_line = np.linspace(35, 54, 200)
        xs, ys = fwd.transform(np.full_like(lat_line, lon), lat_line)
        ax.plot(xs, ys, color="grey", lw=0.4, alpha=0.6, zorder=3)
        # label at bottom where the line crosses miny
        idx = np.argmin(np.abs(ys - miny))
        if minx <= xs[idx] <= maxx:
            ax.annotate(f"{lon}°E", (xs[idx], miny), xytext=(0, -16),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=11, fontweight="bold")
    # parallels
    for lat in lats:
        lon_line = np.linspace(95, 128, 200)
        xs, ys = fwd.transform(lon_line, np.full_like(lon_line, lat))
        ax.plot(xs, ys, color="grey", lw=0.4, alpha=0.6, zorder=3)
        idx = np.argmin(np.abs(xs - minx))
        if miny <= ys[idx] <= maxy:
            ax.annotate(f"{lat}°N", (minx, ys[idx]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=11, fontweight="bold")


def _scale_bar(ax, extent, length_km=400, n_seg=4):
    minx, maxx, miny, maxy = extent
    seg = length_km * 1000 / n_seg
    x0 = minx + 0.04 * (maxx - minx)
    y0 = miny + 0.045 * (maxy - miny)
    h = 0.014 * (maxy - miny)
    for i in range(n_seg):
        c = "black" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((x0 + i * seg, y0), seg, h, facecolor=c,
                                   edgecolor="black", lw=1.0, zorder=12))
    for i in range(n_seg + 1):
        ax.annotate(f"{int(i*length_km/n_seg)}", (x0 + i * seg, y0 + h),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=11, fontweight="bold", zorder=12)
    ax.annotate("km", (x0 + n_seg * seg, y0 + h), xytext=(16, 3),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=11, fontweight="bold", zorder=12)


def _north_arrow(ax, extent):
    minx, maxx, miny, maxy = extent
    x = minx + 0.965 * (maxx - minx)
    y = miny + 0.90 * (maxy - miny)
    dy = 0.055 * (maxy - miny)
    ax.add_patch(FancyArrow(x, y, 0, dy, width=0.0, head_width=0.020 * (maxx - minx),
                            head_length=0.024 * (maxy - miny), fc="black", ec="black",
                            length_includes_head=True, zorder=12))
    ax.annotate("N", (x, y + dy), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=15, fontweight="bold", zorder=12)


def main():
    OUT1.mkdir(parents=True, exist_ok=True)
    log.info("Figure 1 — study-area map")

    # ---- elevation -> Albers ----
    if not SRTM.exists():
        raise SystemExit(f"SRTM not found at {SRTM} — edit the SRTM path in render_fig1.py")
    elev, tf, ext_m = _albers_grid_from(SRTM)
    extent = (ext_m[0], ext_m[2], ext_m[1], ext_m[3])  # minx,maxx,miny,maxy
    log.info(f"  elevation reprojected: {elev.shape}, extent {extent}")

    # ---- ecological zones from fused 2020 (already Albers, ~1km) ----
    fused, tf_f = _read_albers(OUT_FUSED / "fused_2020.tif")
    # align zones to the elevation grid by simple resample if shapes differ
    zones_src = np.zeros_like(fused, dtype="float32")
    for z, codes in ZONE_MAP.items():
        zones_src[np.isin(fused, codes)] = z
    if zones_src.shape != elev.shape:
        zr = np.zeros(elev.shape, dtype="float32")
        reproject(zones_src, zr, src_transform=tf_f, src_crs=TARGET_CRS,
                  dst_transform=tf, dst_crs=TARGET_CRS, resampling=Resampling.nearest)
        zones = zr
    else:
        zones = zones_src
    aoi = zones > 0
    elev_aoi = np.where(aoi, elev, np.nan)

    # ---- boundary ----
    rings, gdf = _boundary_rings()

    # ============ FIGURE: two panels — (a) topography, (b) ecological zones ====
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(22, 10))

    # shared hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    elev_filled = np.where(np.isfinite(elev_aoi), elev_aoi, np.nanmin(elev_aoi))
    hill = ls.hillshade(elev_filled, vert_exag=0.0008, dx=tf.a, dy=abs(tf.e))
    hill_m = np.where(aoi, hill, np.nan)

    # padded extent for furniture/labels
    dx = extent[1] - extent[0]; dy = extent[3] - extent[2]
    pad_x = 0.06 * dx; pad_y = 0.07 * dy
    padded = (extent[0] - pad_x, extent[1] + pad_x,
              extent[2] - pad_y, extent[3] + pad_y)
    fwd = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    GLONS = [100, 105, 110, 115, 120, 125]
    GLATS = [40, 42, 44, 46, 48, 50]

    # ---- panel (a): topography ----
    axA.imshow(hill_m, cmap="gray", extent=extent, origin="upper", alpha=0.55, zorder=1)
    imA = axA.imshow(elev_aoi, cmap="terrain", extent=extent, origin="upper",
                     alpha=0.7, zorder=2)
    for xs, ys in rings:
        axA.plot(xs, ys, color="black", lw=1.2, zorder=6)
    _draw_graticule(axA, padded, fwd, GLONS, GLATS)
    _scale_bar(axA, padded, length_km=400, n_seg=4)
    _north_arrow(axA, padded)
    cbA = plt.colorbar(imA, ax=axA, shrink=0.5, pad=0.02)
    cbA.set_label("Elevation (m a.s.l.)", fontsize=12, fontweight="bold")
    cbA.ax.tick_params(labelsize=10)
    axA.set_xlim(padded[0], padded[1]); axA.set_ylim(padded[2], padded[3])
    axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_linewidth(1.3)
    axA.set_title("(a) Topography", fontsize=15, fontweight="bold")

    # ---- panel (b): ecological zones ----
    zcmap = ListedColormap([ZONE_COLORS[z] for z in sorted(ZONE_NAMES)])
    znorm = BoundaryNorm([z - 0.5 for z in sorted(ZONE_NAMES)] + [7.5], zcmap.N)
    axB.imshow(np.ma.masked_where(~aoi, zones), cmap=zcmap, norm=znorm,
               extent=extent, origin="upper", alpha=1.0, zorder=1)
    axB.imshow(hill_m, cmap="gray", extent=extent, origin="upper", alpha=0.25, zorder=2)
    for xs, ys in rings:
        axB.plot(xs, ys, color="black", lw=1.2, zorder=6)
    _draw_graticule(axB, padded, fwd, GLONS, GLATS)
    _scale_bar(axB, padded, length_km=400, n_seg=4)
    _north_arrow(axB, padded)
    zpatches = [Patch(facecolor=ZONE_COLORS[z], edgecolor="black", label=ZONE_NAMES[z])
                for z in sorted(ZONE_NAMES)]
    legB = axB.legend(handles=zpatches, title="Ecological zone", loc="upper left",
                      fontsize=11, title_fontsize=12.5, frameon=True, framealpha=0.92,
                      borderpad=0.9, labelspacing=0.6, handlelength=1.8, handleheight=1.5)
    legB.set_zorder(13); legB.get_title().set_fontweight("bold")
    axB.set_xlim(padded[0], padded[1]); axB.set_ylim(padded[2], padded[3])
    axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_linewidth(1.3)
    axB.set_title("(b) Ecological zones", fontsize=15, fontweight="bold")

    # ---- inset locator (IM within China) — upper-left of panel (a) ----
    axins = fig.add_axes([0.075, 0.66, 0.13, 0.20])
    if CHINA_SHP is not None and Path(CHINA_SHP).exists():
        import geopandas as gpd
        china = gpd.read_file(CHINA_SHP).to_crs(TARGET_CRS)
        china.plot(ax=axins, facecolor="#f0f0f0", edgecolor="grey", lw=0.4)
    gdf.plot(ax=axins, facecolor="#d6604d", edgecolor="black", lw=0.6)
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title("Location in China", fontsize=8)
    for s in axins.spines.values():
        s.set_linewidth(0.8)

    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT1 / f"fig1_study_area.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  ✓ fig1_study_area.[png/pdf/svg]")

    # ---- GeoTIFF of clipped elevation ----
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1,
            "height": elev_aoi.shape[0], "width": elev_aoi.shape[1],
            "crs": TARGET_CRS, "transform": tf, "nodata": np.nan, "compress": "lzw"}
    with rasterio.open(OUT1 / "fig1_elevation_zones.tif", "w", **prof) as dst:
        dst.write(elev_aoi, 1)

    # ---- zone-area CSV ----
    px_km2 = (tf.a * abs(tf.e)) / 1e6
    rows = []
    for z in sorted(ZONE_NAMES):
        n = int((zones == z).sum())
        rows.append({"zone": ZONE_NAMES[z], "n_pixels": n,
                     "area_km2": round(n * px_km2, 1)})
    zdf = pd.DataFrame(rows)
    tot = zdf["area_km2"].sum()
    zdf["area_pct"] = (100 * zdf["area_km2"] / tot).round(2)
    zdf.to_csv(OUT1 / "fig1_zone_areas.csv", index=False)
    log.info(f"  ✓ fig1_zone_areas.csv  (total {tot:,.0f} km²)")
    log.info("\nUpload fig1_study_area.png + fig1_zone_areas.csv to review.")


if __name__ == "__main__":
    main()
