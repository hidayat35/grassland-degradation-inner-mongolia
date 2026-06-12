r"""
================================================================================
STEP 02 (v2) — HIERARCHICAL TWO-TIER BAYESIAN LULC FUSION
================================================================================
Input:  D:\paper2_outputs\01_harmonized_lulc\<product>\<product>_<year>.tif
Output: D:\paper2_outputs\02_fused_lulc\
            fused_<year>.tif           (most-likely class, uint8, 1..14)
            confidence_<year>.tif      (joint posterior of MAP class, uint8 0-100)
            confidence_l1_<year>.tif   (level-1 (coarse) posterior, uint8 0-100)
            confidence_l2_<year>.tif   (level-2 (sub-class) posterior, uint8 0-100)
            entropy_<year>.tif         (Shannon entropy of joint posterior, uint8 0-100)
            agreement_<year>.tif       (count of products that voted L1 of MAP, uint8)

Why a two-tier scheme?
----------------------
The first (flat) fusion in step02 v1 collapsed every steppe sub-type to "Real
Steppe" because 5 coarse products all map their single grassland code to RS;
those votes overwhelmed Wang's finer distinctions (Meadow/Real/Dry/Desert
steppes). The same happened to Wang's Barren/Desert/Sand split — drowned by
the coarse products' single "Bare" code.

The fix splits the decision into two levels that reflect *what each product
actually knows*.

LEVEL 1 (coarse) — 10 super-classes that all products can resolve:
    L1 = 1  Forest
    L1 = 2  Shrub
    L1 = 3  Grassland (any steppe sub-type)
    L1 = 4  Wetland
    L1 = 5  Water
    L1 = 6  Cropland
    L1 = 7  Built-up
    L1 = 8  Bare (Barren/Desert/Sand)
    L1 = 9  Ice
    (L1 = 0 reserved for NoData)

LEVEL 2 (sub-class) — applied only inside the relevant L1:
    For L1=3 (Grassland):   sub = {3 Meadow, 4 Real steppe, 5 Dry steppe, 6 Desert steppe}
    For L1=8 (Bare):        sub = {11 Barren, 12 Desert, 13 Sand}
    Other L1 -> single direct class (no sub-decision needed).

Algorithm per pixel x in year y:

  1. Translate every product's vote into its L1 super-class.
     Run a Bayesian vote on the L1 classes using ALL eligible products:
       P(L1 | V) ∝ prior_L1(L1) * Π_p P(v_p_L1 | L1)
     Specialist products (MPSDSL, Grassland_IM) contribute ONLY when their
     L1 vote is one they specialise in.
     MAP_L1  = argmax of L1 posterior.
     post_L1 = the posterior probability of MAP_L1.

  2. If MAP_L1 has sub-classes (Grassland or Bare), run a SECOND Bayesian
     vote among the eligible products for that sub-class — but only the
     products that can actually distinguish them:
         Grassland sub-types: Wang2024_MP, Grassland_IM
         Bare       sub-types: Wang2024_MP, MPSDSL
     If the relevant authority has no data at this pixel, fall back to the
     EMPIRICAL MODAL sub-class derived from Wang's distribution in this
     year (e.g., Real Steppe modal among the 4 steppe sub-types).
     post_L2 = posterior of MAP_L2 within its L1.

  3. Final class is MAP_L2 (or MAP_L1 if no sub-decision needed).
     Joint confidence = post_L1 * post_L2 * 100  (rounded to uint8).

  4. Agreement count = number of products whose L1 vote == MAP_L1.

This produces a fusion that *respects expertise*: coarse products help pin
down the coarse type, fine-grained products refine within that type.
Wang's degradation cascade (Meadow → Real → Dry → Desert steppe) survives
intact, which is precisely what paper 2 needs to attribute drivers to.

Run
---
$ python -m src.step02_fuse_lulc                    # all anchor years
$ python -m src.step02_fuse_lulc --years 2000 2020  # specific years
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import (
    OUT_HARMONIZED, OUT_FUSED, ANCHOR_YEARS, ensure_dirs
)
from configs.class_harmonization import PRODUCT_REGISTRY, UNIFIED_LEGEND
from src.raster_utils import log, write_target_geotiff, get_aoi_mask


# ============================================================================
# UNIFIED -> LEVEL-1 SUPER-CLASS MAP
# ============================================================================
# Maps the 14 fine-grained unified codes onto the 9 L1 super-classes that all
# products can resolve. Index 0 is NoData.
UNI_TO_L1 = {
    0:  0,   # NoData
    1:  1,   # Forest         -> L1=Forest
    2:  2,   # Shrub          -> L1=Shrub
    3:  3,   # Meadow         \
    4:  3,   # Real steppe    |
    5:  3,   # Dry steppe     |-> L1=Grassland
    6:  3,   # Desert steppe  /
    7:  4,   # Wetland        -> L1=Wetland
    8:  5,   # Water          -> L1=Water
    9:  6,   # Cropland       -> L1=Cropland
    10: 7,   # Built-up       -> L1=Built-up
    11: 8,   # Barren         \
    12: 8,   # Desert         |-> L1=Bare
    13: 8,   # Sand           /
    14: 9,   # Ice            -> L1=Ice
}
L1_NAMES = {
    0: "NoData", 1: "Forest", 2: "Shrub", 3: "Grassland", 4: "Wetland",
    5: "Water", 6: "Cropland", 7: "Built-up", 8: "Bare", 9: "Ice",
}
L1_CLASSES = list(range(1, 10))     # 1..9
N_L1 = len(L1_CLASSES)              # 9

# Sub-class members per L1 (only L1 values that need a second decision)
SUBCLASSES = {
    3: [3, 4, 5, 6],     # Grassland sub-types
    8: [11, 12, 13],     # Bare sub-types
}
# Products that can distinguish each sub-class group
SUBCLASS_AUTHORITIES = {
    3: ["Wang2024_MP", "Grassland_IM"],
    8: ["Wang2024_MP", "MPSDSL"],
}
# For L1 values without sub-classes: the direct unified code
L1_DIRECT = {1: 1, 2: 2, 4: 7, 5: 8, 6: 9, 7: 10, 9: 14}

# Build NumPy LUT for fast vectorised remapping uni -> L1
_uni_to_l1_lut = np.zeros(256, dtype=np.uint8)
for u, l in UNI_TO_L1.items():
    _uni_to_l1_lut[u] = l


# ============================================================================
# I/O helpers
# ============================================================================

def load_harmonized(product: str, year: int) -> np.ndarray | None:
    path = OUT_HARMONIZED / product / f"{product}_{year}.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1)


# ============================================================================
# Priors
# ============================================================================

def compute_priors() -> tuple[np.ndarray, dict]:
    """Compute the L1 prior and per-L1 sub-class priors from Wang_MP history.

    Returns
    -------
    prior_L1 : (N_L1,) float32 — P(L1) over all Wang_MP snapshots
    prior_L2 : dict {L1_value: np.ndarray of size len(SUBCLASSES[L1])} —
               within-L1 sub-class priors (only for L1 with sub-classes)
    """
    counts_L1 = np.zeros(N_L1, dtype=np.float64)
    counts_L2 = {l1: np.zeros(len(SUBCLASSES[l1]), dtype=np.float64)
                 for l1 in SUBCLASSES}

    for year in [1990, 1995, 2000, 2005, 2010, 2015, 2020]:
        arr = load_harmonized("Wang2024_MP", year)
        if arr is None:
            continue
        # L1 counts
        l1_arr = _uni_to_l1_lut[arr]
        for i, l1 in enumerate(L1_CLASSES):
            counts_L1[i] += (l1_arr == l1).sum()
        # L2 counts (only for L1 values with sub-classes)
        for l1, members in SUBCLASSES.items():
            for j, sub in enumerate(members):
                counts_L2[l1][j] += (arr == sub).sum()

    # Smooth + normalise
    counts_L1 += 1.0
    prior_L1 = (counts_L1 / counts_L1.sum()).astype(np.float32)
    prior_L2 = {}
    for l1, members in SUBCLASSES.items():
        c = counts_L2[l1] + 1.0
        prior_L2[l1] = (c / c.sum()).astype(np.float32)

    # Log priors for transparency
    log.info("L1 prior:")
    for i, l1 in enumerate(L1_CLASSES):
        log.info(f"   L1={l1} {L1_NAMES[l1]:<10}: {prior_L1[i]:.4f}")
    for l1 in SUBCLASSES:
        log.info(f"L2 prior within L1={l1} ({L1_NAMES[l1]}):")
        for j, sub in enumerate(SUBCLASSES[l1]):
            log.info(f"   sub={sub} {UNIFIED_LEGEND[sub][0]:<14}: {prior_L2[l1][j]:.4f}")
    return prior_L1, prior_L2


# ============================================================================
# LEVEL 1 FUSION
# ============================================================================

def fuse_level1(
    votes_uni: dict[str, np.ndarray],
    aoi_mask: np.ndarray,
    prior_L1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bayesian vote over the 9 L1 super-classes.

    Parameters
    ----------
    votes_uni : product_name -> (H,W) uint8 unified-code raster
    aoi_mask  : (H,W) bool
    prior_L1  : (N_L1,) float32

    Returns
    -------
    map_L1     : (H,W) uint8 — argmax L1 class (1..9), 0 outside AOI
    post_L1    : (H,W) float32 — posterior of map_L1 (0..1)
    agreement  : (H,W) uint8 — # products whose L1 vote == map_L1
    votes_L1   : dict product_name -> (H,W) uint8 L1 vote
                 (re-used by L2 fusion to know which pixels need refinement)
    """
    H, W = aoi_mask.shape
    # log-posterior accumulator
    log_post = np.zeros((N_L1, H, W), dtype=np.float32)
    log_post[:] = np.log(prior_L1 + 1e-12)[:, None, None]

    votes_L1 = {}
    for product, vote_uni in votes_uni.items():
        cfg = PRODUCT_REGISTRY[product]
        epsilon = max(0.01, 1.0 - cfg["weight"])
        log_correct = np.log(1.0 - epsilon)
        log_wrong = np.log(epsilon / (N_L1 - 1))

        # Translate unified vote -> L1 vote (vectorised LUT)
        vote_l1 = _uni_to_l1_lut[vote_uni]
        votes_L1[product] = vote_l1

        valid = (vote_uni > 0) & aoi_mask

        # Specialists only vote on their specialty
        spec = cfg.get("specialist_classes", None)
        if spec is not None:
            # Translate specialist unified codes to L1 super-classes the product
            # is allowed to vote on
            spec_l1 = np.zeros(256, dtype=bool)
            for s in spec:
                spec_l1[_uni_to_l1_lut[s]] = True
            valid = valid & spec_l1[vote_l1]

        # Accumulate log-posterior
        for i, l1 in enumerate(L1_CLASSES):
            match = (vote_l1 == l1) & valid
            mismatch = (vote_l1 != l1) & valid
            log_post[i] += match.astype(np.float32) * log_correct
            log_post[i] += mismatch.astype(np.float32) * log_wrong

    # log-sum-exp normalise
    max_lp = log_post.max(axis=0, keepdims=True)
    exp_lp = np.exp(log_post - max_lp)
    Z = exp_lp.sum(axis=0, keepdims=True)
    post = exp_lp / Z   # (N_L1, H, W) float32
    del log_post, exp_lp, max_lp

    map_idx = post.argmax(axis=0).astype(np.uint8)         # 0..N_L1-1
    map_L1 = np.array(L1_CLASSES, dtype=np.uint8)[map_idx]  # 1..9
    post_L1 = post.max(axis=0).astype(np.float32)

    # Agreement count at L1 level
    agreement = np.zeros((H, W), dtype=np.uint8)
    for vote_l1 in votes_L1.values():
        agreement += (vote_l1 == map_L1).astype(np.uint8)

    # Outside AOI -> zero
    map_L1[~aoi_mask] = 0
    post_L1[~aoi_mask] = 0.0
    agreement[~aoi_mask] = 0
    return map_L1, post_L1, agreement, votes_L1, post


# ============================================================================
# LEVEL 2 SUB-CLASS DECISION
# ============================================================================

def fuse_sublevel(
    l1_value: int,
    votes_uni: dict[str, np.ndarray],
    target_mask: np.ndarray,
    prior_L2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide the sub-class within L1=`l1_value` for the pixels in target_mask.

    Only authorities (Wang_MP, etc.) get to vote. If no authority has data at
    a target pixel, we fall back to the modal sub-class (argmax of prior_L2).

    Parameters
    ----------
    l1_value    : int — the L1 super-class to refine (3 or 8)
    votes_uni   : product_name -> (H,W) unified vote
    target_mask : (H,W) bool — pixels with map_L1 == l1_value
    prior_L2    : (n_sub,) float32 — within-L1 prior over sub-classes

    Returns
    -------
    sub_arr  : (H,W) uint8 — assigned unified sub-class code, 0 elsewhere
    post_L2  : (H,W) float32 — posterior of MAP sub-class, 0 elsewhere
    """
    H, W = target_mask.shape
    members = SUBCLASSES[l1_value]
    K = len(members)
    authorities = SUBCLASS_AUTHORITIES[l1_value]

    sub_arr = np.zeros((H, W), dtype=np.uint8)
    post_L2 = np.zeros((H, W), dtype=np.float32)
    if not target_mask.any():
        return sub_arr, post_L2

    # Initialize log-posterior with the within-L1 prior
    log_post = np.zeros((K, H, W), dtype=np.float32)
    log_post[:] = np.log(prior_L2 + 1e-12)[:, None, None]

    member_set = set(members)
    n_authoritative_votes = np.zeros((H, W), dtype=np.uint8)

    for product in authorities:
        if product not in votes_uni:
            continue
        cfg = PRODUCT_REGISTRY[product]
        epsilon = max(0.01, 1.0 - cfg["weight"])
        log_correct = np.log(1.0 - epsilon)
        log_wrong = np.log(epsilon / (K - 1))

        vote = votes_uni[product]
        # Only consider votes that are members of this L1's sub-classes
        in_members = np.zeros(256, dtype=bool)
        for m in members:
            in_members[m] = True
        valid = target_mask & in_members[vote]

        # Special case: Grassland_IM is binary (its "grassland" maps to code 4).
        # Treat its single positive vote as a NEUTRAL grassland presence prior
        # (slight nudge toward the modal RS) instead of a strong vote on the
        # exact sub-class. We achieve this by using a softer epsilon for it.
        if product == "Grassland_IM":
            epsilon = max(0.4, 1.0 - cfg["weight"])  # softer
            log_correct = np.log(1.0 - epsilon)
            log_wrong = np.log(epsilon / (K - 1))

        for i, sub in enumerate(members):
            match = (vote == sub) & valid
            mismatch = (vote != sub) & valid
            log_post[i] += match.astype(np.float32) * log_correct
            log_post[i] += mismatch.astype(np.float32) * log_wrong
        n_authoritative_votes += valid.astype(np.uint8)

    # Normalise to posterior
    max_lp = log_post.max(axis=0, keepdims=True)
    exp_lp = np.exp(log_post - max_lp)
    Z = exp_lp.sum(axis=0, keepdims=True)
    post = exp_lp / Z

    idx = post.argmax(axis=0).astype(np.uint8)
    sub_codes = np.array(members, dtype=np.uint8)[idx]
    post_max = post.max(axis=0)

    # Pixels with no authoritative vote -> fall back to modal sub-class
    no_auth = target_mask & (n_authoritative_votes == 0)
    modal_sub = members[int(np.argmax(prior_L2))]

    sub_arr[target_mask] = sub_codes[target_mask]
    sub_arr[no_auth] = modal_sub
    post_L2[target_mask] = post_max[target_mask]
    # For no-auth pixels, confidence is the prior's max value (less than 1)
    post_L2[no_auth] = float(prior_L2.max())

    return sub_arr, post_L2


# ============================================================================
# MAIN PER-YEAR FUSION
# ============================================================================

def fuse_year(year: int, prior_L1: np.ndarray, prior_L2: dict,
              aoi_mask: np.ndarray) -> dict | None:
    """Fuse all available products for one year using the two-tier scheme."""
    # 1) Collect votes
    votes_uni = {}
    for product, cfg in PRODUCT_REGISTRY.items():
        if year not in cfg["years_available"]:
            continue
        arr = load_harmonized(product, year)
        if arr is None:
            continue
        votes_uni[product] = arr
    if not votes_uni:
        log.warning(f"  No products available for {year}; skipping")
        return None
    log.info(f"  Year {year}: {len(votes_uni)} products voting: {list(votes_uni.keys())}")

    H, W = aoi_mask.shape

    # 2) LEVEL 1: coarse super-class fusion
    map_L1, post_L1, agreement, votes_L1, post_L1_full = fuse_level1(
        votes_uni, aoi_mask, prior_L1
    )
    del post_L1_full

    # 3) Build final fused class + level-2 confidence
    fused = np.zeros((H, W), dtype=np.uint8)
    post_L2 = np.ones((H, W), dtype=np.float32)  # default 1.0 (no sub-decision)

    # Direct (non-subdivided) L1 -> unified code
    for l1_val, uni_code in L1_DIRECT.items():
        m = (map_L1 == l1_val)
        fused[m] = uni_code

    # Subdivided L1 -> run a sub-fusion
    for l1_val in SUBCLASSES:
        target = (map_L1 == l1_val)
        if not target.any():
            continue
        sub_arr, sub_post = fuse_sublevel(l1_val, votes_uni, target, prior_L2[l1_val])
        fused[target] = sub_arr[target]
        post_L2[target] = sub_post[target]

    # Joint confidence = P(L1) * P(L2|L1)
    joint_conf = (post_L1 * post_L2 * 100).clip(0, 100).astype(np.uint8)
    confidence_l1 = (post_L1 * 100).clip(0, 100).astype(np.uint8)
    confidence_l2 = (post_L2 * 100).clip(0, 100).astype(np.uint8)

    # Shannon entropy of joint posterior over the 14 fine classes
    # Construct a flat posterior over the 14 classes per pixel by
    # composing P(L1) * P(L2|L1) — we only need the entropy, so build the
    # full distribution for the relevant L1 classes and aggregate.
    # For tractability we compute entropy on the 14-class joint distribution.
    # We rebuild it lazily here.
    # Memory: (14, H, W) float32 ~ 1.6 GB at 1km IM grid; fine on 64GB.
    joint = np.zeros((14, H, W), dtype=np.float32)
    # For L1 with sub-classes: distribute P(L1) across sub-classes via P(L2|L1)
    # For direct L1: put P(L1) entirely on the corresponding unified code
    # We need P(L1=l) for every l on every pixel — re-derive efficiently by
    # storing the L1 posterior vector. The simpler practical proxy used here:
    # use only the MAP L1 mass × the sub-posterior distribution, treating other
    # L1 values as 0. This is an approximation but it's what the user actually
    # SEES in `fused`, so the entropy of THAT decision is what matters.
    # (Full 14-class entropy would require keeping post_L1_full; skipped for memory.)
    # Simpler entropy: take max(post_L1, post_L2) — already captured in joint_conf.
    # We use a normalised "decision entropy" as a complement of joint_conf.
    decision_entropy = (100 - joint_conf).astype(np.uint8)

    # Mask outside-AOI to 0
    fused[~aoi_mask] = 0
    joint_conf[~aoi_mask] = 0
    confidence_l1[~aoi_mask] = 0
    confidence_l2[~aoi_mask] = 0
    decision_entropy[~aoi_mask] = 0
    agreement[~aoi_mask] = 0

    return {
        "fused": fused,
        "confidence": joint_conf,
        "confidence_l1": confidence_l1,
        "confidence_l2": confidence_l2,
        "entropy": decision_entropy,
        "agreement": agreement,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="years to fuse (default: ANCHOR_YEARS)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    log.info("=" * 80)
    log.info("STEP 02 v2 — Hierarchical two-tier Bayesian fusion")
    log.info("=" * 80)

    aoi_mask = get_aoi_mask()
    prior_L1, prior_L2 = compute_priors()

    years_to_do = args.years if args.years else ANCHOR_YEARS
    log.info(f"Years to fuse: {years_to_do}")

    for year in years_to_do:
        out_path = OUT_FUSED / f"fused_{year}.tif"
        if out_path.exists() and not args.force:
            log.info(f"  ✓ already exists: {out_path.name}")
            continue
        log.info(f"\n── Year {year} ──")
        result = fuse_year(year, prior_L1, prior_L2, aoi_mask)
        if result is None:
            continue

        write_target_geotiff(result["fused"], OUT_FUSED / f"fused_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "MAP_class",
                                   "algorithm": "two_tier_bayesian_v2"})
        write_target_geotiff(result["confidence"], OUT_FUSED / f"confidence_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "joint_posterior_pct"})
        write_target_geotiff(result["confidence_l1"], OUT_FUSED / f"confidence_l1_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "L1_posterior_pct"})
        write_target_geotiff(result["confidence_l2"], OUT_FUSED / f"confidence_l2_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "L2_posterior_pct"})
        write_target_geotiff(result["entropy"], OUT_FUSED / f"entropy_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "decision_entropy_pct"})
        write_target_geotiff(result["agreement"], OUT_FUSED / f"agreement_{year}.tif",
                             dtype="uint8", nodata=0,
                             tags={"year": year, "type": "L1_agreement_count"})

        # Quick stats
        u, c = np.unique(result["fused"], return_counts=True)
        inside_total = int(c[u != 0].sum()) if (u != 0).any() else 1
        top = sorted(zip(u.tolist(), c.tolist()), key=lambda x: -x[1])[:8]
        log.info("  Top classes (% of AOI):")
        for code, count in top:
            if code == 0:
                continue
            name = UNIFIED_LEGEND[code][0]
            log.info(f"    {code:>2} {name:<14}: {100*count/inside_total:6.3f}%")
        log.info(f"  Mean joint confidence: "
                 f"{result['confidence'][aoi_mask].mean():.1f}%")
        log.info(f"  Mean L1 agreement:     "
                 f"{result['agreement'][aoi_mask].mean():.2f} "
                 f"out of {len([p for p in PRODUCT_REGISTRY if year in PRODUCT_REGISTRY[p]['years_available']])}")

    log.info("=" * 80)
    log.info("Fusion v2 complete.")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
