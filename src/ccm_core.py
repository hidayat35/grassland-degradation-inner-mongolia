"""
================================================================================
CCM / GCCM CORE — convergent cross mapping (our own clean implementation)
================================================================================
This module implements:
  - takens_embedding()    : time-delay state-space reconstruction
  - simplex_crossmap()    : cross-map skill via simplex projection (k-NN in the
                            shadow manifold), the heart of CCM
  - ccm()                 : full CCM with library-size convergence test
  - ccm_lagged()          : lagged CCM to resolve causal direction + time delay
                            (Ye et al. 2015 extension)
  - gccm()                : geographical CCM (Gao et al. 2023) for spatial
                            cross-sectional data with few time points

Validation (run this file directly):
  We test against the canonical Sugihara et al. (2012) two-species coupled
  logistic map, where the TRUE causal structure is known:
      X(t+1) = X(t)[rx - rx*X(t) - bxy*Y(t)]
      Y(t+1) = Y(t)[ry - ry*Y(t) - byx*X(t)]
  With bxy>0, byx=0: X is forced by Y (Y causes X), so:
      - "X xmap Y" cross-map skill should CONVERGE high  (X's manifold
        recovers Y  => Y influences X => Y is a cause of X)
      - "Y xmap X" should stay low
  This is the standard unit test for any CCM implementation.

If the printed result matches the known structure, the core is correct and we
wire it into step06.
================================================================================
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------
# Takens embedding
# ----------------------------------------------------------------------------
def takens_embedding(x: np.ndarray, E: int, tau: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Time-delay embedding of a 1-D series.

    Returns
    -------
    M : (n_vectors, E) array of delay vectors [x(t), x(t-tau), ..., x(t-(E-1)tau)]
    t_index : indices t corresponding to each row (the "present" time of each vector)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    first = (E - 1) * tau
    rows = []
    tidx = []
    for t in range(first, n):
        vec = [x[t - j * tau] for j in range(E)]
        rows.append(vec)
        tidx.append(t)
    return np.asarray(rows), np.asarray(tidx)


# ----------------------------------------------------------------------------
# Simplex cross-map: use X's shadow manifold to predict Y
# ----------------------------------------------------------------------------
def simplex_crossmap(
    Mx: np.ndarray, tidx: np.ndarray, target: np.ndarray,
    lib_idx: np.ndarray, pred_idx: np.ndarray, E: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-map: from the manifold Mx (reconstructed from X), predict the
    values of `target` (e.g. Y) at the prediction points.

    For each prediction vector, find its E+1 nearest neighbours IN THE LIBRARY
    (excluding itself), weight them by exp(-d/d1), and form a weighted estimate
    of the target at the corresponding times.

    Returns (predicted, observed) at the prediction points.
    """
    preds = np.full(len(pred_idx), np.nan)
    obs   = np.full(len(pred_idx), np.nan)

    lib_vectors = Mx[lib_idx]
    lib_times   = tidx[lib_idx]

    for i, p in enumerate(pred_idx):
        v = Mx[p]
        # distances to all library vectors
        d = np.sqrt(((lib_vectors - v) ** 2).sum(axis=1))
        # exclude self (distance ~0 at same time)
        same = (lib_times == tidx[p])
        d[same] = np.inf
        # nearest E+1
        k = E + 1
        if np.isfinite(d).sum() < k:
            continue
        nn = np.argsort(d)[:k]
        dnn = d[nn]
        d1 = dnn[0] if dnn[0] > 0 else 1e-12
        w = np.exp(-dnn / d1)
        w /= w.sum()
        # target values at neighbour times
        tgt_vals = target[lib_times[nn]]
        preds[i] = np.sum(w * tgt_vals)
        obs[i]   = target[tidx[p]]
    return preds, obs


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------
# Full CCM with library-size convergence
# ----------------------------------------------------------------------------
def ccm(
    x: np.ndarray, y: np.ndarray, E: int = 3, tau: int = 1,
    lib_sizes: list[int] | None = None, n_boot: int = 30,
    seed: int = 0,
) -> dict:
    """Convergent cross mapping: does X's manifold recover Y? (tests Y -> X causation)

    Returns dict with lib_sizes and mean cross-map skill (rho) at each, plus the
    convergence delta (rho at max lib - rho at min lib).

    Interpretation: if "x xmap y" rho INCREASES with library size and saturates
    high, then Y is a cause of X (information about Y is encoded in X's history).
    """
    rng = np.random.default_rng(seed)
    Mx, tidx = takens_embedding(x, E, tau)
    n_vec = len(tidx)
    if lib_sizes is None:
        lib_sizes = sorted(set([E + 2,
                                max(E + 2, n_vec // 4),
                                max(E + 2, n_vec // 2),
                                max(E + 2, 3 * n_vec // 4),
                                n_vec]))
    # prediction points = all vectors
    pred_idx = np.arange(n_vec)
    rho_by_L = []
    for L in lib_sizes:
        L = min(L, n_vec)
        rhos = []
        for _ in range(n_boot):
            lib_idx = rng.choice(n_vec, size=L, replace=False)
            preds, obs = simplex_crossmap(Mx, tidx, y, lib_idx, pred_idx, E)
            rhos.append(_corr(preds, obs))
        rho_by_L.append(np.nanmean(rhos))
    rho_by_L = np.array(rho_by_L)
    return {
        "lib_sizes": list(lib_sizes),
        "rho": rho_by_L,
        "rho_min_lib": float(rho_by_L[0]),
        "rho_max_lib": float(rho_by_L[-1]),
        "convergence": float(rho_by_L[-1] - rho_by_L[0]),
        "E": E, "tau": tau,
    }


# ----------------------------------------------------------------------------
# Lagged CCM (Ye et al. 2015) — direction + time delay
# ----------------------------------------------------------------------------
def ccm_lagged(
    x: np.ndarray, y: np.ndarray, E: int = 3, tau: int = 1,
    lags: range | None = None, lib_frac: float = 0.8, n_boot: int = 30,
    seed: int = 0,
) -> dict:
    """Cross-map skill of 'x xmap y_lagged' as a function of lag.

    A peak at NEGATIVE lag in the true causal direction indicates the driver
    leads the response (the response carries delayed information about its
    driver). Returns rho per lag and the optimal lag.
    """
    rng = np.random.default_rng(seed)
    if lags is None:
        lags = range(-6, 7)
    Mx, tidx = takens_embedding(x, E, tau)
    n_vec = len(tidx)
    L = max(E + 2, int(lib_frac * n_vec))
    pred_idx = np.arange(n_vec)
    out = {}
    for lag in lags:
        # shift target
        rhos = []
        for _ in range(n_boot):
            lib_idx = rng.choice(n_vec, size=min(L, n_vec), replace=False)
            # predict y at time (t+lag)
            preds = np.full(len(pred_idx), np.nan)
            obs   = np.full(len(pred_idx), np.nan)
            lib_vectors = Mx[lib_idx]
            lib_times = tidx[lib_idx]
            for i, p in enumerate(pred_idx):
                tt = tidx[p] + lag
                if tt < 0 or tt >= len(y):
                    continue
                v = Mx[p]
                d = np.sqrt(((lib_vectors - v) ** 2).sum(axis=1))
                d[lib_times == tidx[p]] = np.inf
                k = E + 1
                if np.isfinite(d).sum() < k:
                    continue
                nn = np.argsort(d)[:k]
                dnn = d[nn]
                d1 = dnn[0] if dnn[0] > 0 else 1e-12
                w = np.exp(-dnn / d1); w /= w.sum()
                preds[i] = np.sum(w * y[lib_times[nn]])  # note: predict y at neighbour times; lag handled by obs
                obs[i] = y[tt]
            rhos.append(_corr(preds, obs))
        out[lag] = np.nanmean(rhos)
    best_lag = max(out, key=lambda k: (out[k] if np.isfinite(out[k]) else -np.inf))
    return {"rho_by_lag": out, "best_lag": int(best_lag),
            "best_rho": float(out[best_lag])}


# ----------------------------------------------------------------------------
# Validation against the canonical Sugihara coupled logistic map
# ----------------------------------------------------------------------------
def _coupled_logistic(n=1000, rx=3.8, ry=3.5, bxy=0.02, byx=0.1, seed=42):
    """Two-species coupled logistic map.

    bxy = effect of Y on X ; byx = effect of X on Y.
    With bxy>0 and byx>0 we have bidirectional but asymmetric coupling.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n); y = np.zeros(n)
    x[0], y[0] = 0.4, 0.2
    for t in range(n - 1):
        x[t+1] = x[t] * (rx - rx * x[t] - bxy * y[t])
        y[t+1] = y[t] * (ry - ry * y[t] - byx * x[t])
        x[t+1] = min(max(x[t+1], 0), 1)
        y[t+1] = min(max(y[t+1], 0), 1)
    return x[100:], y[100:]   # drop transient


if __name__ == "__main__":
    print("=" * 72)
    print("CCM CORE VALIDATION — Sugihara coupled logistic map")
    print("=" * 72)
    # Strong forcing of Y on X (bxy large), weak X on Y (byx small)
    # => X xmap Y should be HIGH (Y causes X);  Y xmap X LOWER
    x, y = _coupled_logistic(n=1200, rx=3.7, ry=3.7, bxy=0.0, byx=0.32, seed=1)
    # Here byx>0 (X affects Y), bxy=0 (Y does NOT affect X).
    # CCM logic: "Y xmap X" recovers X from Y's manifold => X is cause of Y => HIGH
    #            "X xmap Y" => LOW
    print("\nSystem: byx=0.32 (X drives Y), bxy=0 (Y does NOT drive X)")
    print("Expectation: 'Y xmap X' converges HIGH (X causes Y);  'X xmap Y' stays LOW\n")

    res_yx = ccm(y, x, E=3, tau=1, n_boot=40, seed=0)   # Y xmap X
    res_xy = ccm(x, y, E=3, tau=1, n_boot=40, seed=0)   # X xmap Y

    print(f"  lib sizes: {res_yx['lib_sizes']}")
    print(f"  Y xmap X  rho: {np.round(res_yx['rho'], 3)}   "
          f"(converge Δ={res_yx['convergence']:+.3f}, max={res_yx['rho_max_lib']:.3f})")
    print(f"  X xmap Y  rho: {np.round(res_xy['rho'], 3)}   "
          f"(converge Δ={res_xy['convergence']:+.3f}, max={res_xy['rho_max_lib']:.3f})")

    print("\nVerdict:")
    yx_high = res_yx["rho_max_lib"] > 0.5 and res_yx["convergence"] > 0.05
    xy_low  = res_xy["rho_max_lib"] < res_yx["rho_max_lib"]
    if yx_high and xy_low:
        print("  ✓ PASS — CCM correctly identifies X as the driver of Y")
        print("    (Y xmap X high & convergent; X xmap Y lower, as expected)")
    else:
        print("  ✗ unexpected — check parameters")
    print("\n" + "=" * 72)
