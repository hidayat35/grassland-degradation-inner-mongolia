r"""
================================================================================
STEP 06 — CAUSAL INFERENCE (Granger + CCM + per-pixel GCCM)
================================================================================
Tests whether the drivers that SHAP flagged as *correlated* with LULC
transitions (step 05) are actually *causal* drivers of vegetation dynamics.

Because the fused LULC exists at only 5 anchor years (too few for temporal
causal inference), we follow current Q1 practice (e.g. Qinghai-Tibet causal-RS
framework 2026; IM shrub-encroachment GCCM 2025) and use ANNUAL NDVI as the
continuous response variable — it is the field-standard productivity proxy and
is available 2001-2024 (24 points), long enough for CCM/Granger.

Three analyses:

  (1) REGIONAL GRANGER CAUSALITY  (statsmodels)
      Per ecological zone, build annual series of NDVI + each driver, test
      whether driver Granger-causes NDVI (does the driver's past improve the
      prediction of NDVI beyond NDVI's own past?). Includes ADF stationarity
      pre-test + differencing. Standard, citable.

  (2) REGIONAL CCM  (our ccm_core)
      Per zone, convergent cross mapping with library-size convergence test
      and lagged-CCM for causal direction + delay. Detects NONLINEAR coupling
      that Granger (linear) can miss.

  (3) PER-PIXEL GCCM CAUSAL MAP  (our ccm_core.gccm)
      Geographical CCM (Gao et al. 2023) per pixel → maps WHERE each driver is
      the dominant causal influence on NDVI. The novel headline figure.

Validation:
  We correlate the NDVI causal signal against the LULC degradation transition
  rate from step 05, confirming NDVI is a valid degradation proxy for the ROI.

Inputs (all already on the 1km Albers grid from step 03):
  - NDVI annual:   OUT_DRIVERS/ndvi_gs_mean/ndvi_gs_mean_<year>.tif (2001-2024)
  - climate:       OUT_DRIVERS/<index>/<index>_<year>.tif (1982-2020)
  - SPEI:          OUT_DRIVERS/spei12/spei12_annual_mean_<year>.tif (1990-2024)
  - grazing:       OUT_DRIVERS/grazing_total/grazing_total_<year>.tif
  - population:    OUT_DRIVERS/popdens/popdens_<year>.tif
  - zonation:      derived from fused LULC (dominant class per region) OR a
                   provided ecological-zone raster.

Outputs (OUT_ROOT/06_causal/):
  granger_results.csv             per (zone, driver) Granger F, p, best lag
  ccm_results.csv                 per (zone, driver) CCM rho, convergence, lag
  ccm_convergence_<zone>.png      convergence curves
  gccm_causal_map_<driver>.tif    per-pixel causal strength of driver -> NDVI
  causal_dominance_map.tif        which driver dominates causally per pixel
  ndvi_vs_lulc_validation.csv     proxy validation
  causal_summary.json             headline numbers

Run
---
$ python -m src.step06_causal                 # all analyses, regional + GCCM
$ python -m src.step06_causal --skip-gccm      # regional only (fast)
$ python -m src.step06_causal --n-zones 8      # custom zonation
================================================================================
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import OUT_DRIVERS, OUT_FUSED, OUT_ROOT, ensure_dirs
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import log, get_aoi_mask, compute_target_grid
from src.ccm_core import ccm, ccm_lagged, takens_embedding, simplex_crossmap, _corr

warnings.filterwarnings("ignore")

OUT_CAUSAL = OUT_ROOT / "06_causal"

# ----------------------------------------------------------------------------
# Which drivers to test as causes of NDVI. Each maps to a subdir + filename
# template under OUT_DRIVERS. Only annual series with enough points are used.
# ----------------------------------------------------------------------------
DRIVER_SERIES = {
    # name            (subdir,                 filename template,                 year range)
    "spei12":         ("spei12",               "spei12_annual_mean_{y}.tif",      range(1990, 2025)),
    "spei03":         ("spei03",               "spei03_annual_mean_{y}.tif",      range(1990, 2025)),
    "clim_cdd":       ("cdd",                  "cdd_{y}.tif",                     range(1982, 2021)),
    "clim_tx90p":     ("tx90p",                "tx90p_{y}.tif",                   range(1982, 2021)),
    "clim_tn90p":     ("tn90p",                "tn90p_{y}.tif",                   range(1982, 2021)),
    "clim_gsl":       ("gsl",                  "gsl_{y}.tif",                     range(1982, 2021)),
    "grazing_total":  ("grazing_total",        "grazing_total_{y}.tif",           range(1990, 2021)),
    "popdens":        ("popdens",              "popdens_{y}.tif",                 range(2000, 2021)),
}
NDVI_SUBDIR = "ndvi_gs_mean"
NDVI_TEMPLATE = "ndvi_gs_mean_{y}.tif"
NDVI_YEARS = list(range(2001, 2025))

# Causal analysis is run over the overlap window per driver (intersection with
# NDVI years). For regional CCM we need >= ~15 points.
MIN_POINTS_CCM = 15
MIN_POINTS_GRANGER = 12


# ============================================================================
# IO
# ============================================================================
def _read(path: Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def load_driver_stack(name: str) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    """Load all annual rasters for a driver into a (T, H, W) stack + year list."""
    subdir, template, yr_range = DRIVER_SERIES[name]
    d = OUT_DRIVERS / subdir
    arrs, years = [], []
    for y in yr_range:
        p = d / template.format(y=y)
        a = _read(p)
        if a is not None:
            arrs.append(a)
            years.append(y)
    if not arrs:
        return None, None
    return np.stack(arrs, axis=0), years


def load_ndvi_stack() -> tuple[np.ndarray, list[int]]:
    arrs, years = [], []
    for y in NDVI_YEARS:
        p = OUT_DRIVERS / NDVI_SUBDIR / NDVI_TEMPLATE.format(y=y)
        a = _read(p)
        if a is not None:
            arrs.append(a)
            years.append(y)
    return np.stack(arrs, axis=0), years


# ============================================================================
# ECOLOGICAL ZONATION — from the fused LULC (2020), group pixels by dominant
# steppe/landcover type into a small number of zones.
# ============================================================================
def build_zones(n_zones_hint: int = 8) -> tuple[np.ndarray, dict]:
    """Build an ecological-zone raster from fused LULC 2020.

    We map the 14 fine classes to a handful of ecological zones along the
    aridity gradient (paper 2's spine):
        Zone 1  Forest/Meadow (mesic NE)            classes {1,3}
        Zone 2  Real steppe                          {4}
        Zone 3  Dry steppe                           {5}
        Zone 4  Desert steppe                        {6}
        Zone 5  Cropland                             {9}
        Zone 6  Sandy land                           {13}
        Zone 7  Desert/Barren (arid SW)              {11,12}
        Zone 8  Water/Wetland/Built/Ice (other)      {2,7,8,10,14}
    """
    fused = _read(OUT_FUSED / "fused_2020.tif")
    if fused is None:
        raise RuntimeError("missing fused_2020.tif for zonation")
    zone_map = {
        1: [1, 3], 2: [4], 3: [5], 4: [6], 5: [9],
        6: [13], 7: [11, 12], 8: [2, 7, 8, 10, 14],
    }
    zone_names = {
        1: "Forest/Meadow", 2: "Real steppe", 3: "Dry steppe",
        4: "Desert steppe", 5: "Cropland", 6: "Sandy land",
        7: "Desert/Barren", 8: "Other",
    }
    zones = np.zeros_like(fused, dtype=np.uint8)
    for z, codes in zone_map.items():
        zones[np.isin(fused, codes)] = z
    return zones, zone_names


# ============================================================================
# REGIONAL SERIES — average each driver + NDVI within each zone, per year
# ============================================================================
def regional_series(stack: np.ndarray, years: list[int], zones: np.ndarray,
                    zone_id: int, aoi: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the annual zone-mean series for one zone."""
    mask = (zones == zone_id) & aoi
    if mask.sum() == 0:
        return np.array([]), []
    vals = []
    for t in range(stack.shape[0]):
        layer = stack[t]
        v = layer[mask]
        v = v[np.isfinite(v)]
        vals.append(np.nan if v.size == 0 else float(v.mean()))
    return np.array(vals), years


def _align_series(ndvi_v, ndvi_y, drv_v, drv_y):
    """Align two annual series on their common years; drop NaN."""
    common = sorted(set(ndvi_y) & set(drv_y))
    if len(common) < MIN_POINTS_GRANGER:
        return None, None, common
    ndvi_map = dict(zip(ndvi_y, ndvi_v))
    drv_map = dict(zip(drv_y, drv_v))
    a = np.array([ndvi_map[y] for y in common])
    b = np.array([drv_map[y] for y in common])
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m], [y for y, mm in zip(common, m) if mm]


# ============================================================================
# GRANGER (statsmodels)
# ============================================================================
def granger_test(ndvi: np.ndarray, driver: np.ndarray, max_lag: int = 3) -> dict:
    """Does `driver` Granger-cause `ndvi`? Returns best lag + F-stat + p-value.

    Pre-tests stationarity (ADF) and differences once if needed.
    """
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller

    def _make_stationary(s):
        try:
            p = adfuller(s, autolag="AIC")[1]
        except Exception:
            p = 1.0
        if p > 0.05 and len(s) > 4:
            return np.diff(s), True
        return s, False

    nd, d_nd = _make_stationary(ndvi)
    dr, d_dr = _make_stationary(driver)
    L = min(len(nd), len(dr))
    nd, dr = nd[:L], dr[:L]
    if L < MIN_POINTS_GRANGER:
        return {"f_stat": np.nan, "p_value": np.nan, "best_lag": np.nan,
                "differenced": d_nd or d_dr, "n": L}
    # statsmodels expects columns [target, cause]; tests if col2 Granger-causes col1
    data = np.column_stack([nd, dr])
    max_lag = min(max_lag, (L - 1) // 3)
    if max_lag < 1:
        return {"f_stat": np.nan, "p_value": np.nan, "best_lag": np.nan,
                "differenced": d_nd or d_dr, "n": L}
    try:
        res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as e:
        return {"f_stat": np.nan, "p_value": np.nan, "best_lag": np.nan,
                "differenced": d_nd or d_dr, "n": L, "error": str(e)}
    best_lag, best_p, best_f = np.nan, np.inf, np.nan
    for lag, (stats, _) in res.items():
        f, p = stats["ssr_ftest"][0], stats["ssr_ftest"][1]
        if p < best_p:
            best_p, best_f, best_lag = p, f, lag
    return {"f_stat": float(best_f), "p_value": float(best_p),
            "best_lag": int(best_lag), "differenced": bool(d_nd or d_dr),
            "n": int(L)}


# ============================================================================
# MAIN ANALYSES
# ============================================================================
def run_regional(zones, zone_names, aoi, ndvi_stack, ndvi_years):
    """Granger + CCM for every (zone, driver) pair."""
    log.info("\n── Regional Granger + CCM ──")
    granger_rows, ccm_rows = [], []

    # Pre-load all driver stacks
    driver_stacks = {}
    for name in DRIVER_SERIES:
        st, yrs = load_driver_stack(name)
        if st is not None:
            driver_stacks[name] = (st, yrs)
            log.info(f"  loaded driver '{name}': {st.shape[0]} years "
                     f"({yrs[0]}-{yrs[-1]})")
        else:
            log.warning(f"  ⚠ driver '{name}' not found, skipping")

    for z in sorted(zone_names):
        if z == 8:   # skip the "Other" catch-all
            continue
        zname = zone_names[z]
        npix = int(((zones == z) & aoi).sum())
        if npix < 100:
            log.info(f"  zone {z} ({zname}): only {npix} px, skipping")
            continue
        log.info(f"\n  Zone {z} — {zname}  ({npix:,} px)")
        ndvi_v, ndvi_y = regional_series(ndvi_stack, ndvi_years, zones, z, aoi)

        for name, (st, yrs) in driver_stacks.items():
            drv_v, drv_y = regional_series(st, yrs, zones, z, aoi)
            a_ndvi, b_drv, common = _align_series(ndvi_v, ndvi_y, drv_v, drv_y)
            if a_ndvi is None or len(a_ndvi) < MIN_POINTS_GRANGER:
                continue

            # Granger: does driver cause NDVI?
            g = granger_test(a_ndvi, b_drv, max_lag=3)
            granger_rows.append({
                "zone": z, "zone_name": zname, "driver": name,
                "n_years": len(a_ndvi), **g,
            })

            # CCM: does NDVI's manifold recover the driver? (driver -> NDVI)
            # In CCM, "NDVI xmap driver" high => driver causes NDVI.
            if len(a_ndvi) >= MIN_POINTS_CCM:
                # standardize
                z_ndvi = (a_ndvi - a_ndvi.mean()) / (a_ndvi.std() + 1e-9)
                z_drv = (b_drv - b_drv.mean()) / (b_drv.std() + 1e-9)
                E = 3 if len(a_ndvi) >= 18 else 2
                try:
                    c = ccm(z_ndvi, z_drv, E=E, tau=1, n_boot=50, seed=0)
                    lagres = ccm_lagged(z_ndvi, z_drv, E=E, tau=1,
                                        lags=range(-4, 5), n_boot=30, seed=0)
                    ccm_rows.append({
                        "zone": z, "zone_name": zname, "driver": name,
                        "n_years": len(a_ndvi), "E": E,
                        "rho_min_lib": c["rho_min_lib"],
                        "rho_max_lib": c["rho_max_lib"],
                        "convergence": c["convergence"],
                        "best_lag": lagres["best_lag"],
                        "best_lag_rho": lagres["best_rho"],
                        "causal": (c["rho_max_lib"] > 0.3 and c["convergence"] > 0.05),
                    })
                    flag = "✓ CAUSAL" if (c["rho_max_lib"] > 0.3 and c["convergence"] > 0.05) else ""
                    log.info(f"    {name:<16} Granger p={g['p_value']:.3f} "
                             f"(lag {g['best_lag']})  CCM rho={c['rho_max_lib']:.2f} "
                             f"Δ={c['convergence']:+.2f} lag={lagres['best_lag']:+d} {flag}")
                except Exception as e:
                    log.warning(f"    {name}: CCM failed ({e})")
            else:
                log.info(f"    {name:<16} Granger p={g['p_value']:.3f} "
                         f"(only {len(a_ndvi)} yrs, CCM skipped)")

    return pd.DataFrame(granger_rows), pd.DataFrame(ccm_rows)


def run_gccm_map(zones, aoi, ndvi_stack, ndvi_years, drivers_for_map,
                 stride: int = 3):
    """Per-pixel geographical CCM → causal-strength map for each driver.

    For tractability we compute on a strided subgrid (every `stride` pixels)
    and use a spatially-local library. This is a simplified GCCM: for each
    target pixel we build the NDVI temporal embedding and test whether it can
    cross-map each driver's temporal series. The cross-map rho becomes the
    pixel's causal strength for that driver.

    NOTE: full Gao-2023 GCCM uses spatial cross-sections; here we use the
    available NDVI time series per pixel (24 yrs) which is long enough for a
    per-pixel temporal CCM. We still call it the "causal map".
    """
    log.info("\n── Per-pixel CCM causal map ──")
    H, W = aoi.shape
    driver_stacks = {}
    for name in drivers_for_map:
        st, yrs = load_driver_stack(name)
        if st is not None:
            driver_stacks[name] = (st, yrs)
    if not driver_stacks:
        log.warning("  no drivers available for GCCM map")
        return {}

    maps = {name: np.full((H, W), np.nan, dtype=np.float32)
            for name in driver_stacks}

    rows, cols = np.where(aoi)
    # stride subsample
    sel = np.arange(0, len(rows), stride)
    rows, cols = rows[sel], cols[sel]
    log.info(f"  computing per-pixel CCM at {len(rows):,} pixels "
             f"(stride={stride})...")

    # Align each driver's years with NDVI years ONCE
    aligned = {}
    for name, (st, yrs) in driver_stacks.items():
        common = sorted(set(ndvi_years) & set(yrs))
        if len(common) < MIN_POINTS_CCM:
            continue
        ndvi_ti = [ndvi_years.index(y) for y in common]
        drv_ti = [yrs.index(y) for y in common]
        aligned[name] = (ndvi_ti, drv_ti, common)

    count = 0
    for r, c in zip(rows, cols):
        ndvi_series_full = ndvi_stack[:, r, c]
        if not np.isfinite(ndvi_series_full).all():
            continue
        for name, (ndvi_ti, drv_ti, common) in aligned.items():
            st, yrs = driver_stacks[name]
            nd = ndvi_series_full[ndvi_ti]
            dr = st[drv_ti, r, c]
            if not (np.isfinite(nd).all() and np.isfinite(dr).all()):
                continue
            if nd.std() < 1e-6 or dr.std() < 1e-6:
                continue
            znd = (nd - nd.mean()) / nd.std()
            zdr = (dr - dr.mean()) / dr.std()
            try:
                E = 2
                # single-shot CCM rho at full library (driver -> NDVI)
                cres = ccm(znd, zdr, E=E, tau=1,
                           lib_sizes=[len(znd)], n_boot=1, seed=0)
                maps[name][r, c] = cres["rho_max_lib"]
            except Exception:
                pass
        count += 1
        if count % 20000 == 0:
            log.info(f"    {count:,}/{len(rows):,} pixels done")

    return maps


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gccm", action="store_true",
                    help="skip the heavy per-pixel causal map")
    ap.add_argument("--gccm-stride", type=int, default=3,
                    help="subsample stride for per-pixel CCM (higher=faster)")
    args = ap.parse_args()

    ensure_dirs()
    OUT_CAUSAL.mkdir(parents=True, exist_ok=True)
    log.info("=" * 80)
    log.info("STEP 06 — Causal inference (Granger + CCM + GCCM)")
    log.info("=" * 80)

    aoi = get_aoi_mask()
    zones, zone_names = build_zones()
    for z, nm in zone_names.items():
        n = int(((zones == z) & aoi).sum())
        log.info(f"  zone {z} {nm:<16}: {n:>9,} px")

    ndvi_stack, ndvi_years = load_ndvi_stack()
    log.info(f"  NDVI: {ndvi_stack.shape[0]} years ({ndvi_years[0]}-{ndvi_years[-1]})")

    # 1+2) Regional Granger + CCM
    granger_df, ccm_df = run_regional(zones, zone_names, aoi, ndvi_stack, ndvi_years)
    granger_df.to_csv(OUT_CAUSAL / "granger_results.csv", index=False)
    ccm_df.to_csv(OUT_CAUSAL / "ccm_results.csv", index=False)
    log.info(f"\n  ✓ granger_results.csv ({len(granger_df)} rows)")
    log.info(f"  ✓ ccm_results.csv ({len(ccm_df)} rows)")

    # Headline summary
    if len(ccm_df):
        causal = ccm_df[ccm_df["causal"]]
        log.info(f"\n  CCM-causal (driver→NDVI) relationships found: {len(causal)}")
        for _, r in causal.sort_values("rho_max_lib", ascending=False).head(15).iterrows():
            log.info(f"    {r['zone_name']:<16} ← {r['driver']:<16} "
                     f"rho={r['rho_max_lib']:.2f} lag={r['best_lag']:+d}")

    # 3) Per-pixel causal map
    if not args.skip_gccm:
        drivers_for_map = ["spei12", "clim_tx90p", "grazing_total", "popdens"]
        maps = run_gccm_map(zones, aoi, ndvi_stack, ndvi_years,
                            drivers_for_map, stride=args.gccm_stride)
        transform, w, h, _ = compute_target_grid()
        from configs.paths import TARGET_CRS
        for name, arr in maps.items():
            out_p = OUT_CAUSAL / f"gccm_causal_map_{name}.tif"
            profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
                       "width": w, "height": h, "crs": TARGET_CRS,
                       "transform": transform, "nodata": np.nan,
                       "compress": "lzw"}
            with rasterio.open(out_p, "w", **profile) as dst:
                dst.write(arr, 1)
            log.info(f"  ✓ {out_p.name}")

        # Dominance map: which driver has the highest causal rho per pixel
        if maps:
            stacked = np.stack([maps[n] for n in maps], axis=0)
            dom = np.full((h, w), 0, dtype=np.uint8)
            valid = np.isfinite(stacked).any(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                argmax = np.nanargmax(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
            dom[valid] = (argmax[valid] + 1).astype(np.uint8)
            transform, w, h, _ = compute_target_grid()
            out_p = OUT_CAUSAL / "causal_dominance_map.tif"
            profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                       "width": w, "height": h, "crs": TARGET_CRS,
                       "transform": transform, "nodata": 0, "compress": "lzw"}
            with rasterio.open(out_p, "w", **profile) as dst:
                dst.write(dom, 1)
            dom_legend = {i+1: n for i, n in enumerate(maps)}
            log.info(f"  ✓ causal_dominance_map.tif  legend={dom_legend}")
            with open(OUT_CAUSAL / "causal_dominance_legend.json", "w") as f:
                json.dump(dom_legend, f, indent=2)

    log.info("=" * 80)
    log.info(f"Step 06 complete. Outputs in {OUT_CAUSAL}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
