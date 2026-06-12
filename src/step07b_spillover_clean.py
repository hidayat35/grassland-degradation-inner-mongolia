r"""
step07b_spillover_clean.py — clean, architecture-fixed spillover test
(addresses reviewer Point 5). Holds the model FIXED (NodeGRU) and changes ONLY
the spatial inputs: own-features vs own-features + neighbour-mean features.
If adding neighbours does not raise held-out skill on the change target, there
is no spillover, with no graph-training confound.

Run locally:
    python -m src.step07b_spillover_clean

Writes OUT_GNN/spillover_clean_metrics.json.
"""
from pathlib import Path
import json
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_ROOT
from src.raster_utils import log

OUT_GNN = OUT_ROOT / "07_stgnn"
GRID_KM = 3
TRAIN_END = 2018
VAL_YEARS = (2019, 2021)
TEST_YEARS = (2022, 2024)


def add_neighbour_mean_features(X_seq, edges, N):
    """X_seq (T,N,F) -> (T,N,2F): own features | neighbour-mean features."""
    T, Nn, F = X_seq.shape
    nbrs = [[] for _ in range(N)]
    for a, b in edges:
        nbrs[a].append(b); nbrs[b].append(a)
    nbr_feats = np.zeros_like(X_seq)
    for t in range(T):
        Xt = X_seq[t]
        for i in range(N):
            if nbrs[i]:
                nbr_feats[t, i] = Xt[nbrs[i]].mean(axis=0)
    return np.concatenate([X_seq, nbr_feats], axis=2)


def main():
    try:
        import torch
        from src.gnn_models import NodeGRU
        from src.step07_stgnn import (build_graph, assemble_tensors, train_model,
                                      NDVI_YEARS)
    except Exception as e:
        raise SystemExit(f"need torch + step07 modules available: {e}")

    OUT_GNN.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Clean spillover test (device={device}, grid={GRID_KM} km)")

    graph = build_graph(GRID_KM)
    edges = graph["edges"]
    N = len(graph["node_rc"])
    X_seq, ndvi_z, _ = assemble_tensors(graph)
    T, _, F = X_seq.shape
    log.info(f"  nodes={N:,}  years={T}  own-features={F}")

    # temporal split (indices into NDVI_YEARS), same as step07
    yr_idx = {y: i for i, y in enumerate(NDVI_YEARS)}
    train_t = [i for y, i in yr_idx.items() if y <= TRAIN_END]
    val_t = [i for y, i in yr_idx.items() if VAL_YEARS[0] <= y <= VAL_YEARS[1]]
    test_t = [i for y, i in yr_idx.items() if TEST_YEARS[0] <= y <= TEST_YEARS[1]]

    # ---- Model 1: temporal only ----
    log.info("  training NodeGRU (temporal only)...")
    torch.manual_seed(0)
    m1 = NodeGRU(F, hidden=64).to(device)
    r1 = train_model(m1, X_seq, ndvi_z, edges, N, device,
                     use_graph=False, train_t=train_t, val_t=val_t, test_t=test_t,
                     target_mode="anomaly")
    r2_temporal = r1["r2_anomaly"]

    # ---- Model 2: temporal + neighbour-mean features (same architecture) ----
    log.info("  building neighbour-mean features...")
    X_aug = add_neighbour_mean_features(X_seq, edges, N)
    log.info(f"  augmented features={X_aug.shape[2]} (own {F} + neighbour {F})")
    log.info("  training NodeGRU (temporal + neighbours)...")
    torch.manual_seed(0)
    m2 = NodeGRU(X_aug.shape[2], hidden=64).to(device)
    r2 = train_model(m2, X_aug, ndvi_z, edges, N, device,
                     use_graph=False, train_t=train_t, val_t=val_t, test_t=test_t,
                     target_mode="anomaly")
    r2_plus_nbr = r2["r2_anomaly"]

    gap = float(r2_plus_nbr - r2_temporal)
    out = {
        "grid_km": GRID_KM, "n_nodes": int(N),
        "target": "anomaly (NDVI year-to-year change)",
        "r2_temporal_only": round(float(r2_temporal), 4),
        "r2_temporal_plus_neighbours": round(float(r2_plus_nbr), 4),
        "spillover_gap": round(gap, 4),
        "skill_over_persistence_temporal": round(float(r1["skill_over_persistence"]), 4),
        "skill_over_persistence_plus_nbr": round(float(r2["skill_over_persistence"]), 4),
        "conclusion": ("no spillover" if gap <= 0.01 else "possible spillover — investigate"),
        "note": ("Architecture held fixed (NodeGRU); only spatial inputs change, "
                 "so the gap is attributable to neighbour information alone, with "
                 "no graph-training confound."),
    }
    (OUT_GNN / "spillover_clean_metrics.json").write_text(json.dumps(out, indent=2))
    log.info(f"\n  R2 temporal only          = {out['r2_temporal_only']}")
    log.info(f"  R2 temporal + neighbours  = {out['r2_temporal_plus_neighbours']}")
    log.info(f"  spillover gap             = {out['spillover_gap']}  -> {out['conclusion']}")
    log.info(f"  ✓ spillover_clean_metrics.json written to {OUT_GNN}")
    log.info("\nUpload spillover_clean_metrics.json so the clean gap can replace "
             "the A3T-GCN numbers in §3.5.")


if __name__ == "__main__":
    main()
