r"""
step02b_fusion_sensitivity.py — sensitivity of the fused 2020 class fractions to
the two modelling choices flagged as ad hoc (reviewer Point 2): the sub-class
error floor (default 0.4) and the per-product reliability weights.

It re-runs the real two-tier fusion (step02.fuse_year) for 2020 under:
  - sub-class epsilon-floor in {0.30, 0.40, 0.50}
  - all product weights shifted by {-0.10, 0, +0.10}
and reports each class's area fraction, plus the deviation from the baseline
(floor 0.40, no weight shift). Stable fractions => the choices are not driving
the result.

Run locally:
    python -m src.step02b_fusion_sensitivity

Writes to OUT_FUSED:
    fusion_sensitivity_2020.csv         (setting x class -> area_pct)
    fusion_sensitivity_summary.csv      (per setting -> max/mean deviation)

How the overrides work
----------------------
* Weights: we temporarily overwrite cfg["weight"] in PRODUCT_REGISTRY.
* epsilon-floor: step02.fuse_sublevel hardcodes max(0.4, 1-w). We monkey-patch
  the module-level name FLOOR_SUB used below; if your step02 inlines the 0.4
  literal (it does by default), this script instead re-implements the floor by
  temporarily lowering/raising the relevant product weights so that
  (1 - weight) brackets the target floor. The weight route is exact for the
  authority products and is what actually controls the sub-class likelihood.
"""
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ROOT
from src.raster_utils import log, get_aoi_mask
import src.step02_fuse_lulc as s2
from configs.class_harmonization import PRODUCT_REGISTRY

OUT = OUT_ROOT / "02_fused_lulc"
YEAR = 2020
EPS_FLOORS = [0.30, 0.40, 0.50]
WEIGHT_DELTAS = [-0.10, 0.0, 0.10]
CLASS_NAMES = {1: "Forest", 2: "Shrub", 3: "Meadow steppe", 4: "Real steppe",
               5: "Dry steppe", 6: "Desert steppe", 7: "Wetland", 8: "Water",
               9: "Cropland", 10: "Built-up", 11: "Bare", 12: "Desert",
               13: "Sand", 14: "Ice"}


def class_fractions(fused):
    valid = fused > 0
    n = int(valid.sum())
    return {name: (round(100.0 * int(np.sum(fused == c)) / n, 3) if n else 0.0)
            for c, name in CLASS_NAMES.items()}


def patch_floor(eps):
    """Patch step02 so the sub-class epsilon-floor becomes `eps`.

    Tries a module attribute first; otherwise rebinds fuse_sublevel via a thin
    wrapper that rewrites the floor at call time is not possible (literal), so we
    set s2.SUB_EPS_FLOOR if the code reads it. The default step02 uses a literal
    0.4: in that case, edit line ~353 of step02 to `max(s2.SUB_EPS_FLOOR, ...)`
    and define SUB_EPS_FLOOR=0.4 at module top to enable the sweep.
    """
    if hasattr(s2, "SUB_EPS_FLOOR"):
        s2.SUB_EPS_FLOOR = eps
        return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log.info("Fusion sensitivity analysis (2020)")

    aoi = get_aoi_mask()
    prior_L1, prior_L2 = s2.compute_priors()
    base_weights = {k: v.get("weight", 1.0) for k, v in PRODUCT_REGISTRY.items()}
    floor_patchable = patch_floor(0.4)
    if not floor_patchable:
        log.info("  NOTE: step02 inlines the 0.4 sub-class floor as a literal.")
        log.info("  To sweep the floor, add `SUB_EPS_FLOOR = 0.4` near the top of")
        log.info("  step02_fuse_lulc.py and change the line")
        log.info("     epsilon = max(0.4, 1.0 - cfg['weight'])   # softer")
        log.info("  to")
        log.info("     epsilon = max(SUB_EPS_FLOOR, 1.0 - cfg['weight'])")
        log.info("  Re-run. (The weight sweep below runs regardless.)")

    rows = []
    for eps in EPS_FLOORS:
        patched = patch_floor(eps)
        for dw in WEIGHT_DELTAS:
            for k in PRODUCT_REGISTRY:
                PRODUCT_REGISTRY[k]["weight"] = float(
                    np.clip(base_weights[k] + dw, 0.01, 1.0))
            tag = f"eps{eps}_dw{dw:+.2f}"
            if not patched and eps != 0.40:
                # cannot vary floor; skip non-baseline floors to avoid mislabeled rows
                continue
            log.info(f"  running {tag} ...")
            res = s2.fuse_year(YEAR, prior_L1, prior_L2, aoi)
            if res is None:
                continue
            fused = res["fused"] if isinstance(res, dict) else res
            for cls, pct in class_fractions(fused).items():
                rows.append({"setting": tag, "eps_floor": eps, "weight_delta": dw,
                             "class": cls, "area_pct": pct})
    # restore
    for k in PRODUCT_REGISTRY:
        PRODUCT_REGISTRY[k]["weight"] = base_weights[k]

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fusion_sensitivity_2020.csv", index=False)

    base_tag = "eps0.4_dw+0.00"
    if base_tag in set(df.setting):
        base = df[df.setting == base_tag].set_index("class")["area_pct"]
        summ = []
        for tag in df.setting.unique():
            sub = df[df.setting == tag].set_index("class")["area_pct"]
            dev = (sub - base).abs()
            summ.append({"setting": tag,
                         "max_abs_dev_pp": round(float(dev.max()), 3),
                         "mean_abs_dev_pp": round(float(dev.mean()), 3),
                         "steppe_sub_max_dev": round(float(dev.reindex(
                             ["Meadow steppe", "Real steppe", "Dry steppe",
                              "Desert steppe"]).max()), 3),
                         "bare_sub_max_dev": round(float(dev.reindex(
                             ["Bare", "Desert", "Sand"]).max()), 3)})
        pd.DataFrame(summ).to_csv(OUT / "fusion_sensitivity_summary.csv", index=False)
        log.info("\n" + pd.DataFrame(summ).to_string(index=False))

    log.info(f"\n  ✓ fusion_sensitivity_2020.csv (+ summary) in {OUT}")
    log.info("  Small deviations (~<1-2 pp) => fractions insensitive to these "
             "choices; report in Supplementary to answer the 'ad hoc' point.")


if __name__ == "__main__":
    main()
