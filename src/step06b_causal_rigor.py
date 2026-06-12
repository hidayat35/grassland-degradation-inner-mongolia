r"""
step06b_causal_rigor.py — multiple-comparison control and surrogate significance
for the regional causal analysis (addresses reviewer Point 4).

Adds the two pieces a Q1 methods reviewer expects:
  1) FDR (Benjamini-Hochberg) correction of the Granger p-values across the full
     set of zone x driver tests -> q-values.
  2) Phase-randomised SURROGATE significance for CCM: for each zone x driver pair
     we build N surrogate driver series that preserve the power spectrum (hence
     autocorrelation) but destroy the cross-dynamics, run CCM on each, and report
     an empirical p-value = P(rho_surrogate >= rho_observed).

A consensus link is then redefined as: Granger q < 0.05 AND CCM surrogate p < 0.05
AND convergence > 0.05. This replaces the uncorrected-p version.

Run locally:
    python -m src.step06b_causal_rigor

Reads (from OUT_CAUSAL = D:\paper2_outputs\06_causal):
    granger_results.csv, ccm_results.csv   (the existing per-pair results)
and re-loads the zone-mean NDVI + driver series the same way step06 does
(via its regional_series machinery) to run the surrogate test.

Writes:
    granger_results_fdr.csv         (adds q_value, significant_fdr)
    ccm_surrogate_results.csv       (adds surrogate_p, n_surrogates)
    consensus_links_corrected.csv   (the defensible consensus set)
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ROOT
from src.raster_utils import log
from src.ccm_core import ccm

OUT_CAUSAL = OUT_ROOT / "06_causal"
N_SURROGATES = 200
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# 1) Benjamini-Hochberg FDR
# ----------------------------------------------------------------------------
def benjamini_hochberg(pvals):
    """Return BH-adjusted q-values for an array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def apply_fdr_to_granger():
    f = OUT_CAUSAL / "granger_results.csv"
    if not f.exists():
        log.info(f"  (no {f}; skipping FDR)")
        return None
    g = pd.read_csv(f)
    g["q_value"] = benjamini_hochberg(g["p_value"].values)
    g["significant_fdr"] = g["q_value"] < 0.05
    g.to_csv(OUT_CAUSAL / "granger_results_fdr.csv", index=False)
    n_sig_raw = int((g["p_value"] < 0.05).sum())
    n_sig_fdr = int(g["significant_fdr"].sum())
    log.info(f"  Granger: {n_sig_raw} significant at raw p<0.05; "
             f"{n_sig_fdr} survive FDR q<0.05 (of {len(g)} tests)")
    return g


# ----------------------------------------------------------------------------
# 2) Phase-randomised surrogate significance for CCM
# ----------------------------------------------------------------------------
def phase_randomise(x):
    """Return a surrogate with the same power spectrum (preserves autocorrelation)."""
    n = len(x)
    xf = np.fft.rfft(x - np.mean(x))
    mag = np.abs(xf)
    # randomise phases (keep DC real)
    phases = RNG.uniform(0, 2 * np.pi, size=mag.shape)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    surro = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return surro + np.mean(x)


def ccm_surrogate_p(ndvi, driver, E=3, n_surr=N_SURROGATES):
    """Empirical p = P(rho_surrogate >= rho_observed) for 'driver -> ndvi'.

    CCM convention: ndvi xmap driver tests whether driver is a cause of ndvi.
    We surrogate the driver series (the putative cause).
    """
    obs = ccm(ndvi, driver, E=E)
    rho_obs = obs["rho_max_lib"]
    conv_obs = obs["convergence"]
    ge = 0
    for _ in range(n_surr):
        d_s = phase_randomise(driver)
        r = ccm(ndvi, d_s, E=E)["rho_max_lib"]
        if r >= rho_obs:
            ge += 1
    p = (ge + 1) / (n_surr + 1)
    return rho_obs, conv_obs, float(p)


def run_surrogates():
    """Re-load the zone-mean series and surrogate-test each existing CCM pair.

    We import step06's regional-series builders so the series match exactly.
    """
    try:
        from src.step06_causal import (load_ndvi_stack, load_driver_stack,
                                        build_zones, regional_series, _align_series)
    except Exception as e:
        log.info(f"  (could not import step06 series builders: {e})")
        log.info("  Falling back: surrogate test requires the zone-mean series; "
                 "run this on the machine with the driver rasters.")
        return None

    cf = OUT_CAUSAL / "ccm_results.csv"
    if not cf.exists():
        log.info(f"  (no {cf}; skipping surrogate CCM)")
        return None
    ccm_df = pd.read_csv(cf)

    from src.raster_utils import get_aoi_mask
    aoi = get_aoi_mask()
    ndvi_stack, ndvi_years = load_ndvi_stack()
    zones, zone_names = build_zones()
    name_to_id = {v: k for k, v in zone_names.items()}

    rows = []
    for _, r in ccm_df.iterrows():
        zone = r["zone_name"]; drv = r["driver"]
        try:
            dstack, dyears = load_driver_stack(drv)
            if dstack is None:
                continue
            zid = name_to_id.get(zone)
            if zid is None:
                continue
            nd, nd_years = regional_series(ndvi_stack, ndvi_years, zones, zid, aoi)
            dr, dr_years = regional_series(dstack, dyears, zones, zid, aoi)
            nd_a, dr_a, common = _align_series(nd, nd_years, dr, dr_years)
            if nd_a is None or len(nd_a) < 12:
                continue
            rho_obs, conv, p = ccm_surrogate_p(nd_a, dr_a)
            rows.append({"zone_name": zone, "driver": drv,
                         "rho_max_lib": round(rho_obs, 4),
                         "convergence": round(conv, 4),
                         "surrogate_p": round(p, 4),
                         "n_surrogates": N_SURROGATES,
                         "significant_surrogate": bool(p < 0.05 and conv > 0.05)})
            log.info(f"  {zone} <- {drv}: rho={rho_obs:.3f} surrogate_p={p:.3f}")
        except Exception as e:
            log.info(f"  ({zone} <- {drv} failed: {e})")
    if not rows:
        return None
    sdf = pd.DataFrame(rows)
    sdf.to_csv(OUT_CAUSAL / "ccm_surrogate_results.csv", index=False)
    log.info(f"  ✓ ccm_surrogate_results.csv written ({len(sdf)} pairs)")
    return sdf


def build_consensus(granger_fdr, ccm_surr):
    if granger_fdr is None or ccm_surr is None:
        log.info("  (consensus needs both FDR + surrogate outputs)")
        return
    g = granger_fdr[granger_fdr["significant_fdr"]][["zone_name", "driver", "q_value", "best_lag"]]
    c = ccm_surr[ccm_surr["significant_surrogate"]][["zone_name", "driver", "rho_max_lib", "surrogate_p"]]
    con = g.merge(c, on=["zone_name", "driver"], how="inner")
    con.to_csv(OUT_CAUSAL / "consensus_links_corrected.csv", index=False)
    log.info(f"\n  ✓ consensus_links_corrected.csv — {len(con)} links survive "
             f"BOTH FDR-Granger and surrogate-CCM:")
    for _, r in con.iterrows():
        log.info(f"     {r['zone_name']} <- {r['driver']}  "
                 f"(q={r['q_value']:.3f}, CCM rho={r['rho_max_lib']:.2f}, "
                 f"surrogate p={r['surrogate_p']:.3f})")


def main():
    log.info("Causal rigour: FDR + surrogate significance")
    g = apply_fdr_to_granger()
    s = run_surrogates()
    build_consensus(g, s)
    log.info("\nUpload granger_results_fdr.csv + ccm_surrogate_results.csv + "
             "consensus_links_corrected.csv so the corrected consensus set and "
             "q-values can be put into Table 4 and §3.4.")


if __name__ == "__main__":
    main()
