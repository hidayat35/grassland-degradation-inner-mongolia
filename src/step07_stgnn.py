r"""
================================================================================
STEP 07 — SPATIO-TEMPORAL GRAPH NEURAL NETWORK (spatial spillover of degradation)
================================================================================
Tests whether grassland degradation is a SPATIALLY PROPAGATING process — i.e.
whether a location's vegetation trajectory depends on its NEIGHBOURS' states
beyond what local drivers explain. Steps 05-06 are point-wise and cannot see
this; the graph model can.

Approach (confirmed by design discussion):
  - Coarsen the 1-km Albers grid to a coarser analysis grid (default 3 km) by
    block-averaging, giving a tractable graph (~10^5 nodes max on 12 GB).
  - Each node = one coarse cell inside the AOI.
  - Node temporal signal = annual growing-season NDVI (2001-2024, 24 steps),
    the same response used in step 06; node static features = terrain, mean
    driver climatology, and dominant fused class (context).
  - Edges = queen contiguity between coarse cells; adjacency symmetric-
    normalized (Kipf-Welling).
  - Task = one-step-ahead NDVI forecasting (predict year t from years <t).

Spillover EVIDENCE = an ablation across three models on identical splits:
    NodeMLP  (features only)        -> local baseline
    NodeGRU  (temporal, no graph)   -> + temporal dynamics
    A3TGCN   (graph + temporal)     -> + spatial spillover
The skill gap NodeGRU -> A3TGCN quantifies the spillover contribution.

CRITICAL training detail (from sandbox validation): full-batch training badly
underfits these recurrent/graph models. We MINIBATCH OVER NODES — for the GCN
we compute all-node embeddings on the full graph (message passing needs the
whole adjacency) but backpropagate the loss on a random node subset each step.

Outputs (OUT_ROOT/07_stgnn/):
  ablation_metrics.json            test R2/RMSE for the 3 models + spillover gap
  training_curves.csv              per-epoch val loss for each model
  temporal_attention.csv           A3T-GCN attention weights per year (mean)
  node_table.parquet               node id, x, y, lon, lat, dominant class
  pred_vs_obs_test.parquet         observed vs predicted NDVI on test years
  spillover_influence_map.tif      per-node neighbour-ablation influence
  forecast_ndvi_<year>.tif         projected NDVI 2025-2027
  degradation_risk_map.tif         relative near-future degradation-risk score
  fig_ablation.{png,pdf,svg}       skill bar chart
  fig_attention.{png,pdf,svg}      temporal attention curve

Run
---
$ python -m src.step07_stgnn --gpu                      # 3 km, GPU
$ python -m src.step07_stgnn --gpu --grid-km 5          # coarser/faster
$ python -m src.step07_stgnn --gpu --grid-km 10 --epochs 100   # quick prototype
$ python -m src.step07_stgnn --gpu --skip-forecast      # ablation only
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

from configs.paths import (
    OUT_DRIVERS, OUT_FUSED, OUT_ROOT, TARGET_CRS, ensure_dirs,
)
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import log, get_aoi_mask, compute_target_grid

warnings.filterwarnings("ignore")

OUT_GNN = OUT_ROOT / "07_stgnn"

# NDVI temporal signal
NDVI_SUBDIR = "ndvi_gs_mean"
NDVI_TEMPLATE = "ndvi_gs_mean_{y}.tif"
NDVI_YEARS = list(range(2001, 2025))   # 24 steps

# Static node-context drivers (climatological mean over available years)
STATIC_DRIVERS = {
    "elevation":   ("terrain", "elevation.tif"),
    "slope":       ("terrain", "slope.tif"),
}
# Time-varying context drivers (we use their per-year values as extra node feats)
# Kept small to control memory; NDVI is the primary signal.
CONTEXT_DRIVERS = {
    "spei12":   ("spei12", "spei12_annual_mean_{y}.tif", range(1990, 2025)),
    "grazing":  ("grazing_total", "grazing_total_{y}.tif", range(1990, 2021)),
}

# Temporal holdout
TRAIN_END = 2018      # train years <= 2018
VAL_YEARS = (2019, 2021)
TEST_YEARS = (2022, 2024)

RNG = 17


# ============================================================================
# IO + coarsening
# ============================================================================
def _read(path: Path) -> np.ndarray | None:
    if not Path(path).exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def block_coarsen(arr: np.ndarray, factor: int, how: str = "mean") -> np.ndarray:
    """Coarsen a 2-D array by an integer block factor, ignoring NaNs."""
    H, W = arr.shape
    Hc, Wc = H // factor, W // factor
    a = arr[:Hc * factor, :Wc * factor].reshape(Hc, factor, Wc, factor)
    if how == "mean":
        return np.nanmean(a, axis=(1, 3))
    elif how == "mode":
        # integer mode per block (for class maps)
        a2 = a.transpose(0, 2, 1, 3).reshape(Hc, Wc, factor * factor)
        out = np.zeros((Hc, Wc), dtype=np.float32)
        for i in range(Hc):
            for j in range(Wc):
                vals = a2[i, j]
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size:
                    u, c = np.unique(vals.astype(int), return_counts=True)
                    out[i, j] = u[np.argmax(c)]
        return out
    raise ValueError(how)


# ============================================================================
# Graph construction
# ============================================================================
def build_graph(grid_km: int):
    """Coarsen all layers to `grid_km`, define nodes (valid coarse cells),
    build queen-contiguity edges, and assemble the node feature tensors.

    Returns a dict with:
      ndvi_seq : (T, N) standardized NDVI per node per year
      static   : (N, S) standardized static features
      ctx_seq  : (T, N, C) standardized context drivers aligned to NDVI years
      edges    : (E, 2) node-index pairs
      node_rc  : (N, 2) coarse (row, col) of each node
      node_xy  : (N, 2) Albers x, y of each node centre
      dom_class: (N,) dominant fused class per node
      coarse_shape, transform_coarse
    """
    transform, W1, H1, bounds = compute_target_grid()
    res1 = transform.a                      # 1000 m
    factor = int(round(grid_km * 1000 / res1))
    log.info(f"  coarsening factor: {factor} (1km -> {grid_km}km)")

    aoi = get_aoi_mask().astype(np.float32)
    aoi_c = block_coarsen(aoi, factor, "mean")     # fraction of cell inside AOI
    node_mask = aoi_c >= 0.5                        # majority-in-AOI cells
    Hc, Wc = node_mask.shape
    log.info(f"  coarse grid: {Hc}×{Wc}, candidate nodes: {int(node_mask.sum()):,}")

    # NDVI stack -> coarsen each year
    ndvi_layers = []
    for y in NDVI_YEARS:
        a = _read(OUT_DRIVERS / NDVI_SUBDIR / NDVI_TEMPLATE.format(y=y))
        if a is None:
            log.warning(f"  missing NDVI {y}")
            ndvi_layers.append(np.full((Hc, Wc), np.nan, np.float32))
        else:
            ndvi_layers.append(block_coarsen(a, factor, "mean"))
    ndvi_grid = np.stack(ndvi_layers, axis=0)       # (T, Hc, Wc)

    # Require nodes to have NDVI in (nearly) all years
    valid_ndvi = np.isfinite(ndvi_grid).mean(axis=0) >= 0.9
    node_mask = node_mask & valid_ndvi
    log.info(f"  nodes with NDVI coverage: {int(node_mask.sum()):,}")

    # Node indexing
    rc = np.argwhere(node_mask)                     # (N,2) coarse row,col
    N = len(rc)
    idx_of = -np.ones((Hc, Wc), dtype=np.int64)
    idx_of[rc[:, 0], rc[:, 1]] = np.arange(N)

    # NDVI per node, fill residual NaN by temporal mean
    ndvi_seq = ndvi_grid[:, rc[:, 0], rc[:, 1]]     # (T, N)
    for n in range(N):
        s = ndvi_seq[:, n]
        if not np.isfinite(s).all():
            m = np.nanmean(s)
            s[~np.isfinite(s)] = m
            ndvi_seq[:, n] = s

    # Static features
    static_list = []
    for name, (sub, fn) in STATIC_DRIVERS.items():
        a = _read(OUT_DRIVERS / sub / fn)
        if a is None:
            static_list.append(np.zeros(N, np.float32))
        else:
            ac = block_coarsen(a, factor, "mean")
            v = ac[rc[:, 0], rc[:, 1]]
            v[~np.isfinite(v)] = np.nanmean(v[np.isfinite(v)]) if np.isfinite(v).any() else 0.0
            static_list.append(v)
    static = np.stack(static_list, axis=1) if static_list else np.zeros((N, 0), np.float32)

    # Context drivers aligned to NDVI years (nearest available year)
    ctx_layers = []
    for name, (sub, tmpl, yr_range) in CONTEXT_DRIVERS.items():
        avail = list(yr_range)
        per_year = []
        for y in NDVI_YEARS:
            yy = y if y in avail else min(avail, key=lambda a: abs(a - y))
            a = _read(OUT_DRIVERS / sub / tmpl.format(y=yy))
            if a is None:
                per_year.append(np.full(N, np.nan, np.float32))
            else:
                ac = block_coarsen(a, factor, "mean")
                per_year.append(ac[rc[:, 0], rc[:, 1]])
        arr = np.stack(per_year, axis=0)            # (T, N)
        # fill NaN with column mean
        cm = np.nanmean(arr, axis=0)
        inds = np.where(~np.isfinite(arr))
        arr[inds] = np.take(cm, inds[1])
        arr[~np.isfinite(arr)] = 0.0
        ctx_layers.append(arr)
    ctx_seq = np.stack(ctx_layers, axis=2) if ctx_layers else np.zeros((len(NDVI_YEARS), N, 0), np.float32)

    # Dominant fused class per node (from 2020) for context + interpretation
    fused = _read(OUT_FUSED / "fused_2020.tif")
    dom = block_coarsen(fused, factor, "mode") if fused is not None else np.zeros((Hc, Wc), np.float32)
    dom_class = dom[rc[:, 0], rc[:, 1]].astype(int)

    # Edges: queen contiguity among node cells
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    elist = []
    for n in range(N):
        r, c = rc[n]
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if 0 <= rr < Hc and 0 <= cc < Wc:
                m = idx_of[rr, cc]
                if m >= 0 and m > n:           # undirected, dedup
                    elist.append((n, m))
    edges = np.array(elist, dtype=np.int64)
    log.info(f"  edges: {len(edges):,}  (avg degree {2*len(edges)/max(N,1):.1f})")

    # Node centre coordinates in Albers
    res_c = res1 * factor
    minx, miny, maxx, maxy = bounds
    node_x = minx + (rc[:, 1] + 0.5) * res_c
    node_y = maxy - (rc[:, 0] + 0.5) * res_c
    node_xy = np.stack([node_x, node_y], axis=1)

    transform_coarse = rasterio.transform.from_origin(minx, maxy, res_c, res_c)

    return {
        "ndvi_seq": ndvi_seq.astype(np.float32),
        "static": static.astype(np.float32),
        "ctx_seq": ctx_seq.astype(np.float32),
        "edges": edges,
        "node_rc": rc,
        "node_xy": node_xy,
        "dom_class": dom_class,
        "coarse_shape": (Hc, Wc),
        "transform_coarse": transform_coarse,
        "factor": factor,
    }


# ============================================================================
# Assemble model tensors (features per timestep) + standardization
# ============================================================================
def assemble_tensors(graph: dict):
    """Build X_seq (T, N, F) and the per-year target y_t (NDVI at year t).

    Feature at year t for a node = [NDVI_t, context_drivers_t..., static...].
    Target for a forecasting step ending at year t = NDVI_t, predicted from
    the window of years < t.
    """
    ndvi = graph["ndvi_seq"]            # (T,N)
    ctx = graph["ctx_seq"]             # (T,N,C)
    static = graph["static"]           # (N,S)
    T, N = ndvi.shape

    # standardize NDVI globally (keep mean/std to invert for forecast maps)
    ndvi_mu, ndvi_sd = float(np.mean(ndvi)), float(np.std(ndvi) + 1e-6)
    ndvi_z = (ndvi - ndvi_mu) / ndvi_sd

    # standardize context per channel
    if ctx.shape[2] > 0:
        cmu = ctx.mean((0, 1), keepdims=True)
        csd = ctx.std((0, 1), keepdims=True) + 1e-6
        ctx_z = (ctx - cmu) / csd
    else:
        ctx_z = ctx

    # standardize static per channel
    if static.shape[1] > 0:
        smu = static.mean(0, keepdims=True)
        ssd = static.std(0, keepdims=True) + 1e-6
        static_z = (static - smu) / ssd
    else:
        static_z = static

    # Per-timestep feature: [ndvi_t, ctx_t..., static...]
    static_b = np.repeat(static_z[None, :, :], T, axis=0)   # (T,N,S)
    feats = [ndvi_z[:, :, None]]
    if ctx_z.shape[2] > 0:
        feats.append(ctx_z)
    if static_b.shape[2] > 0:
        feats.append(static_b)
    X_seq = np.concatenate(feats, axis=2).astype(np.float32)   # (T,N,F)

    return X_seq, ndvi_z, (ndvi_mu, ndvi_sd)


# ============================================================================
# Training (MINIBATCHED OVER NODES — the validated fix)
# ============================================================================
def train_model(model, X_seq, ndvi_z, edges, N, device,
                use_graph, train_t, val_t, test_t,
                epochs=300, lr=0.005, batch_nodes=4096, patience=40,
                target_mode="anomaly"):
    """One-step-ahead NDVI forecasting with node-minibatched training.

    target_mode:
      'level'   : predict NDVI_z[t] directly (dominated by persistence; high R2
                  but a trivial 'copy last year' baseline already nears the
                  ceiling — weak test of spillover).
      'anomaly' : predict the CHANGE  NDVI_z[t] - NDVI_z[t-1].  This removes
                  the trivial persistence signal and forces all models to
                  compete on the year-to-year deviation, which is where spatial
                  spillover would manifest. This is the rigorous target for the
                  spillover ablation and answers the reviewer's persistence
                  objection. The model still predicts a per-node scalar; we add
                  back NDVI_z[t-1] only when reconstructing levels for reporting.

    Returns test metrics on the LEVEL scale (so R2 is comparable across modes)
    plus the anomaly-scale R2 and the skill-over-persistence score.

    use_graph=True passes the sparse adjacency to the model (A3T-GCN); else MLP/GRU.
    """
    import torch
    import torch.nn as nn
    from src.gnn_models import build_normalized_adjacency

    A_hat = build_normalized_adjacency(edges, N, device=device) if use_graph else None
    Xt = torch.tensor(X_seq, device=device)          # (T,N,F)
    yt = torch.tensor(ndvi_z, device=device)         # (T,N) standardized NDVI level

    def target_at(t):
        if target_mode == "anomaly":
            return yt[t] - yt[t - 1]
        return yt[t]

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(RNG)

    def windows(year_list):
        return [t for t in year_list if t >= 3]

    train_years = windows(train_t)
    val_years = windows(val_t)
    test_years = windows(test_t)

    best_val = np.inf; best_state = None; bad = 0
    curve = []
    for ep in range(epochs):
        model.train()
        rng.shuffle(train_years)
        node_perm = rng.permutation(N)
        ep_loss = 0.0; nb = 0
        for t in train_years:
            Xwin = Xt[:t]
            target = target_at(t)
            for s in range(0, N, batch_nodes):
                idx = torch.tensor(node_perm[s:s + batch_nodes], device=device)
                opt.zero_grad()
                if use_graph:
                    pred = model(Xwin, A_hat)
                    loss = lossf(pred[idx], target[idx])
                else:
                    pred = model(Xwin[:, idx, :], None)
                    loss = lossf(pred, target[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                ep_loss += loss.item(); nb += 1

        # validation (on the training target scale)
        model.eval()
        with torch.no_grad():
            vlosses = []
            for t in val_years:
                Xwin = Xt[:t]
                pred = model(Xwin, A_hat) if use_graph else model(Xwin, None)
                vlosses.append(lossf(pred, target_at(t)).item())
            vloss = float(np.mean(vlosses)) if vlosses else np.nan
        curve.append({"epoch": ep, "train_loss": ep_loss / max(nb, 1), "val_loss": vloss})
        if vloss < best_val - 1e-5:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test metrics ----
    model.eval()
    pred_level, obs_level, pred_anom, obs_anom = [], [], [], []
    persist_level = []   # copy-last-year baseline on the level scale
    with torch.no_grad():
        for t in test_years:
            Xwin = Xt[:t]
            out = (model(Xwin, A_hat) if use_graph else model(Xwin, None)).cpu().numpy()
            prev = yt[t - 1].cpu().numpy()
            lvl = yt[t].cpu().numpy()
            if target_mode == "anomaly":
                p_level = prev + out          # reconstruct level = prev + predicted change
                p_anom = out
            else:
                p_level = out
                p_anom = out - prev
            pred_level.append(p_level); obs_level.append(lvl)
            pred_anom.append(p_anom); obs_anom.append(lvl - prev)
            persist_level.append(prev)        # persistence predicts level = prev
    pred_level = np.concatenate(pred_level); obs_level = np.concatenate(obs_level)
    pred_anom = np.concatenate(pred_anom); obs_anom = np.concatenate(obs_anom)
    persist_level = np.concatenate(persist_level)

    def _r2(obs, pred):
        ss = ((obs - pred) ** 2).sum(); tot = ((obs - obs.mean()) ** 2).sum()
        return float(1 - ss / (tot + 1e-9))

    r2_level = _r2(obs_level, pred_level)
    r2_anom = _r2(obs_anom, pred_anom)
    rmse_level = float(np.sqrt(((obs_level - pred_level) ** 2).mean()))
    # persistence baseline skill on the level scale
    r2_persist = _r2(obs_level, persist_level)
    # skill over persistence: fraction of persistence's residual variance removed
    mse_model = ((obs_level - pred_level) ** 2).mean()
    mse_persist = ((obs_level - persist_level) ** 2).mean()
    skill_over_persist = float(1 - mse_model / (mse_persist + 1e-12))

    return {"r2": r2_level, "rmse": rmse_level, "r2_anomaly": r2_anom,
            "r2_persistence": r2_persist, "skill_over_persistence": skill_over_persist,
            "curve": curve, "test_pred": pred_level, "test_obs": obs_level,
            "A_hat": A_hat}


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--grid-km", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-nodes", type=int, default=4096)
    ap.add_argument("--skip-forecast", action="store_true")
    ap.add_argument("--target-mode", choices=["anomaly", "level"], default="anomaly",
                    help="anomaly = predict NDVI change (rigorous spillover test, "
                         "removes trivial persistence); level = predict NDVI level")
    args = ap.parse_args()

    import torch
    from src.gnn_models import NodeMLP, NodeGRU, A3TGCN

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    ensure_dirs(); OUT_GNN.mkdir(parents=True, exist_ok=True)
    log.info("=" * 80)
    log.info("STEP 07 — Spatio-temporal GNN (spatial spillover)")
    log.info("=" * 80)
    log.info(f"  device: {device}  grid: {args.grid_km}km  epochs: {args.epochs}")
    log.info(f"  target mode: {args.target_mode}"
             + ("  (predicting NDVI year-to-year change — persistence removed)"
                if args.target_mode == "anomaly" else "  (predicting NDVI level)"))

    # ---- graph ----
    log.info("\n── Building graph ──")
    graph = build_graph(args.grid_km)
    N = len(graph["node_rc"])
    X_seq, ndvi_z, (ndvi_mu, ndvi_sd) = assemble_tensors(graph)
    T, _, F = X_seq.shape
    log.info(f"  tensors: X_seq {X_seq.shape}, nodes={N}, features={F}")

    # ---- temporal split (indices into NDVI_YEARS) ----
    yr_idx = {y: i for i, y in enumerate(NDVI_YEARS)}
    train_t = [i for y, i in yr_idx.items() if y <= TRAIN_END]
    val_t = [i for y, i in yr_idx.items() if VAL_YEARS[0] <= y <= VAL_YEARS[1]]
    test_t = [i for y, i in yr_idx.items() if TEST_YEARS[0] <= y <= TEST_YEARS[1]]
    log.info(f"  train years <= {TRAIN_END} ({len(train_t)}), "
             f"val {VAL_YEARS} ({len(val_t)}), test {TEST_YEARS} ({len(test_t)})")

    # ---- ablation: 3 models ----
    results = {}
    specs = [
        ("NodeMLP", lambda: NodeMLP(F, hidden=64), False),
        ("NodeGRU", lambda: NodeGRU(F, hidden=64), False),
        ("A3TGCN",  lambda: A3TGCN(F, gc_dim=32, hidden=64), True),
    ]
    for name, ctor, use_graph in specs:
        log.info(f"\n── Training {name} ({'graph' if use_graph else 'no graph'}) ──")
        torch.manual_seed(RNG)
        model = ctor().to(device)
        res = train_model(model, X_seq, ndvi_z, graph["edges"], N, device,
                          use_graph, train_t, val_t, test_t,
                          epochs=args.epochs, batch_nodes=args.batch_nodes,
                          target_mode=args.target_mode)
        log.info(f"  {name}: level R2={res['r2']:.3f}  RMSE={res['rmse']:.3f}  "
                 f"| anomaly R2={res['r2_anomaly']:+.3f}  "
                 f"skill-over-persistence={res['skill_over_persistence']:+.3f}")
        results[name] = {"model": model, **res}

    # persistence baseline (copy-last-year) is identical for all models; report once
    persist_r2 = results["NodeGRU"]["r2_persistence"]
    log.info(f"\n  Persistence baseline (copy-last-year) level R2: {persist_r2:.3f}")

    # ---- spillover gap ----
    # The rigorous comparison is on the ANOMALY scale (year-to-year change),
    # where trivial persistence is removed. We report the gap on both the
    # anomaly R2 and the skill-over-persistence metric; the level-R2 gap is
    # retained for continuity but is the weak (persistence-dominated) one.
    gap_anom = results["A3TGCN"]["r2_anomaly"] - results["NodeGRU"]["r2_anomaly"]
    gap_skill = (results["A3TGCN"]["skill_over_persistence"]
                 - results["NodeGRU"]["skill_over_persistence"])
    gap_level = results["A3TGCN"]["r2"] - results["NodeGRU"]["r2"]
    log.info(f"\n  SPATIAL SPILLOVER GAP (A3TGCN - NodeGRU):")
    log.info(f"    anomaly R2 gap:              {gap_anom:+.3f}   ← primary (rigorous)")
    log.info(f"    skill-over-persistence gap:  {gap_skill:+.3f}")
    log.info(f"    level R2 gap:                {gap_level:+.3f}   (persistence-dominated)")
    if gap_anom > 0.03:
        log.info("    → graph adds meaningful spatial-spillover skill")
    elif gap_anom > 0.0:
        log.info("    → graph adds marginal spatial-spillover skill")
    else:
        log.info("    → no spillover benefit: dynamics are locally driven at this scale")

    ablation = {
        "grid_km": args.grid_km, "n_nodes": int(N), "n_features": int(F),
        "target_mode": args.target_mode,
        "persistence_baseline_r2_level": persist_r2,
        "models": {k: {"r2_level": v["r2"], "rmse_level": v["rmse"],
                       "r2_anomaly": v["r2_anomaly"],
                       "skill_over_persistence": v["skill_over_persistence"]}
                   for k, v in results.items()},
        "spillover_gap_anomaly_r2": gap_anom,
        "spillover_gap_skill_over_persistence": gap_skill,
        "spillover_gap_level_r2": gap_level,
        "temporal_split": {"train_end": TRAIN_END, "val": VAL_YEARS, "test": TEST_YEARS},
    }
    with open(OUT_GNN / "ablation_metrics.json", "w") as f:
        json.dump(ablation, f, indent=2)
    log.info(f"  ✓ ablation_metrics.json")

    # training curves
    curve_rows = []
    for name, v in results.items():
        for c in v["curve"]:
            curve_rows.append({"model": name, **c})
    pd.DataFrame(curve_rows).to_csv(OUT_GNN / "training_curves.csv", index=False)

    # node table + pred-vs-obs
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(graph["node_xy"][:, 0], graph["node_xy"][:, 1])
    except Exception:
        lon = lat = np.zeros(N)
    node_df = pd.DataFrame({
        "node": np.arange(N),
        "row_c": graph["node_rc"][:, 0], "col_c": graph["node_rc"][:, 1],
        "x": graph["node_xy"][:, 0], "y": graph["node_xy"][:, 1],
        "lon": lon, "lat": lat, "dom_class": graph["dom_class"],
    })
    node_df.to_parquet(OUT_GNN / "node_table.parquet", index=False)

    # ---- temporal attention (A3T-GCN) ----
    a3t = results["A3TGCN"]["model"]
    if getattr(a3t, "last_attn", None) is not None:
        attn = a3t.last_attn.squeeze(-1).mean(dim=1).cpu().numpy()  # mean over nodes -> (T_window,)
        # last_attn corresponds to the LAST forward (a test year window)
        pd.DataFrame({"window_step": np.arange(len(attn)), "attention": attn}).to_csv(
            OUT_GNN / "temporal_attention.csv", index=False)

    # ---- spillover influence map (neighbour ablation on A3T-GCN) ----
    log.info("\n── Spillover influence map (neighbour ablation) ──")
    _spillover_map(results["A3TGCN"]["model"], X_seq, graph, N, device, test_t)

    # ---- forecast ----
    if not args.skip_forecast:
        log.info("\n── Forecast 2025-2027 + degradation risk ──")
        _forecast(results["A3TGCN"]["model"], X_seq, ndvi_z, graph, N, device,
                  ndvi_mu, ndvi_sd, target_mode=args.target_mode)

    # ---- figures ----
    _make_figures(ablation, results)

    log.info("=" * 80)
    log.info(f"Step 07 complete. Outputs in {OUT_GNN}")
    log.info("=" * 80)


# ============================================================================
# Spillover influence: how much each node's prediction shifts when its
# neighbours' inputs are replaced by the global mean (graph signal removed).
# ============================================================================
def _spillover_map(model, X_seq, graph, N, device, test_t):
    import torch
    from src.gnn_models import build_normalized_adjacency
    A_hat = build_normalized_adjacency(graph["edges"], N, device=device)
    Xt = torch.tensor(X_seq, device=device)
    t = max(test_t)
    Xwin = Xt[:t]
    model.eval()
    with torch.no_grad():
        base = model(Xwin, A_hat).cpu().numpy()
        # ablate spatial signal: isolated-node adjacency (identity only)
        iso_edges = np.zeros((0, 2), dtype=np.int64)
        A_iso = build_normalized_adjacency(iso_edges, N, device=device)
        iso = model(Xwin, A_iso).cpu().numpy()
    influence = np.abs(base - iso)
    _write_node_raster(influence, graph, OUT_GNN / "spillover_influence_map.tif")
    log.info(f"  mean |Δ| from neighbour ablation: {float(np.nanmean(influence)):.4f}")


# ============================================================================
# Forecast: roll the A3T-GCN forward autoregressively for 2025-2027.
# ============================================================================
def _forecast(model, X_seq, ndvi_z, graph, N, device, ndvi_mu, ndvi_sd,
              target_mode="anomaly"):
    import torch
    from src.gnn_models import build_normalized_adjacency
    A_hat = build_normalized_adjacency(graph["edges"], N, device=device)
    T, _, F = X_seq.shape
    seq = torch.tensor(X_seq, device=device).clone()   # (T,N,F)
    model.eval()
    forecasts = {}
    fc_years = [2025, 2026, 2027]
    with torch.no_grad():
        cur = seq
        for fy in fc_years:
            out = model(cur, A_hat)                      # model output (level or change)
            prev_ndvi_z = cur[-1, :, 0]                  # last step's NDVI (feature 0)
            if target_mode == "anomaly":
                next_ndvi_z = prev_ndvi_z + out          # reconstruct level = prev + change
            else:
                next_ndvi_z = out
            # build next-step feature row: NDVI=reconstructed level, context/static carried
            last = cur[-1].clone()                       # (N,F)
            last[:, 0] = next_ndvi_z                      # feature 0 is NDVI level
            cur = torch.cat([cur, last[None]], dim=0)     # append the predicted step
            ndvi_pred = next_ndvi_z.cpu().numpy() * ndvi_sd + ndvi_mu
            forecasts[fy] = ndvi_pred
            _write_node_raster(ndvi_pred, graph, OUT_GNN / f"forecast_ndvi_{fy}.tif")
            log.info(f"  forecast {fy}: mean NDVI {float(np.nanmean(ndvi_pred)):.3f}")

    # Degradation risk: relative NDVI decline (2027 vs recent observed mean)
    recent_obs = (ndvi_z[-3:].mean(axis=0)) * ndvi_sd + ndvi_mu   # 2022-2024 mean
    decline = recent_obs - forecasts[2027]
    # normalize to 0..1 risk (higher decline -> higher risk), only positive declines
    risk = np.clip(decline, 0, None)
    if risk.max() > 0:
        risk = risk / np.percentile(risk[risk > 0], 95)
        risk = np.clip(risk, 0, 1)
    _write_node_raster(risk, graph, OUT_GNN / "degradation_risk_map.tif")
    log.info(f"  degradation-risk map written (mean risk {float(np.nanmean(risk)):.3f})")


# ============================================================================
# Helper: paint a per-node vector back onto the coarse raster grid
# ============================================================================
def _write_node_raster(values: np.ndarray, graph: dict, out_path: Path):
    Hc, Wc = graph["coarse_shape"]
    grid = np.full((Hc, Wc), np.nan, dtype=np.float32)
    rc = graph["node_rc"]
    grid[rc[:, 0], rc[:, 1]] = values
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "width": Wc, "height": Hc, "crs": TARGET_CRS,
               "transform": graph["transform_coarse"], "nodata": np.nan,
               "compress": "lzw"}
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(grid, 1)


# ============================================================================
# Figures
# ============================================================================
def _make_figures(ablation, results):
    import matplotlib.pyplot as plt
    # Ablation bar — use the ANOMALY-scale R2 (the rigorous, persistence-removed
    # metric); annotate the persistence baseline as a reference.
    names = ["NodeMLP", "NodeGRU", "A3TGCN"]
    tm = ablation.get("target_mode", "anomaly")
    if tm == "anomaly":
        r2s = [ablation["models"][n]["r2_anomaly"] for n in names]
        ylab = "Test $R^2$ on NDVI change (persistence removed)"
        gap = ablation["spillover_gap_anomaly_r2"]
    else:
        r2s = [ablation["models"][n]["r2_level"] for n in names]
        ylab = "Test $R^2$ (one-step-ahead NDVI level)"
        gap = ablation["spillover_gap_level_r2"]
    labels = ["MLP\n(local)", "GRU\n(+temporal)", "A3T-GCN\n(+spatial)"]
    colors = ["#bdbdbd", "#74a9cf", "#045a8d"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, r2s, color=colors)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(ylab)
    ax.set_title(f"Spatio-temporal ablation — spillover gap {gap:+.3f} $R^2$"
                 + ("" if tm == "anomaly" else "  (persistence-dominated)"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT_GNN / f"fig_ablation.{ext}",
                    dpi=200 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close()

    # Attention curve
    attn_path = OUT_GNN / "temporal_attention.csv"
    if attn_path.exists():
        a = pd.read_csv(attn_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(a["window_step"], a["attention"], marker="o", color="#045a8d")
        ax.set_xlabel("Window step (earlier → most recent year)")
        ax.set_ylabel("Mean temporal attention")
        ax.set_title("A3T-GCN temporal attention over the input window")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        for ext in ("png", "pdf", "svg"):
            plt.savefig(OUT_GNN / f"fig_attention.{ext}",
                        dpi=200 if ext == "png" else None, bbox_inches="tight",
                        facecolor="white")
        plt.close()


if __name__ == "__main__":
    main()
