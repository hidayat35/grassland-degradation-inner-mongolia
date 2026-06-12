r"""
================================================================================
STEP 05 — MULTINOMIAL TRANSITION ATTRIBUTION (XGBoost + SHAP)
================================================================================
First sanity-check model for paper 2's attribution story:

  TASK:  given the feature vector of a pixel observed at year t (and its
         class at t), predict its class at year t+5.
  MODEL: one multi-class XGBoost classifier (14 output classes)
  EVAL : spatial-blocked cross-validation (100 km blocks → no train/test
         leakage from spatial autocorrelation)
  ATTR : TreeSHAP — Shapley value per feature, per pixel, per class

Why multinomial first?
  - One model captures EVERY transition type at once, not just degradation.
  - Gives an honest "global" attribution view that the binary models will
    refine class-pair by class-pair in step 05b.
  - Trains in ~30 min on RTX 3060 Ti, ~2 h on CPU.

Two parallel versions:
  - VERSION A (primary): filtered to high-confidence pixels (conf_joint ≥ 80
                         at BOTH start and end of the transition pair)
  - VERSION B (sensitivity): all pixels, no confidence filter
  Both are compared in the supplementary materials.

Outputs (in D:\paper2_outputs\05_attribution\multinomial\)
---------------------------------------------------------
  model_A_high_conf.pkl                trained XGBoost (filtered)
  model_B_all_pixels.pkl               trained XGBoost (unfiltered)
  cv_metrics_A.json                    spatial-CV accuracy/F1/AUC per fold
  cv_metrics_B.json
  shap_values_A.npy                    SHAP values (n_samples × n_features × n_classes)
  shap_sample_index_A.csv              which pixels the SHAP sample comes from
  shap_summary_A.csv                   mean(|SHAP|) per feature per class
  feature_importance_A.csv             gain + cover + permutation importance
  confusion_matrix_A.csv               predicted vs actual class
  prediction_map_A_<year>.tif          predicted class raster (anchor years)
  And the same set with suffix _B

  shap_summary_combined.png/pdf/svg    side-by-side A vs B comparison
  feature_importance_top20.png/pdf     bar chart of top features

Run
---
$ python -m src.step05_attribute_multinomial
$ python -m src.step05_attribute_multinomial --gpu     # use CUDA
$ python -m src.step05_attribute_multinomial --version A   # only one version
================================================================================
"""

from __future__ import annotations
import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.paths import (
    OUT_FEATURES, OUT_ATTRIBUTION, ANCHOR_YEARS, ensure_dirs,
)
from configs.class_harmonization import UNIFIED_LEGEND
from src.raster_utils import log

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUT_MN = OUT_ATTRIBUTION / "multinomial"

# Min confidence for the high-confidence filter (Version A)
MIN_CONF_HI = 80.0

# Spatial CV: block size in km
BLOCK_SIZE_KM = 100

# SHAP sample size — TreeSHAP is exact but slow; we compute it on a stratified
# sample so we get representative SHAP without blowing up runtime/RAM.
# 5.7M total rows × 128 features × 14 classes = too big. Sample ~50k pixels.
SHAP_SAMPLE_SIZE = 50_000

# Random seeds for reproducibility
RNG_SEED = 17

# Features that CANNOT be inputs (they are/contain the target, or are geometry)
EXCLUDED_COLUMN_PREFIXES = ("lulc_class", "conf_joint", "conf_l1", "conf_l2",
                            "agreement_count", "lulc_change_flag")
NON_FEATURE_GEOM_COLS = ("row", "col", "x", "y", "lon", "lat", "year")


# ============================================================================
# DATA PREP
# ============================================================================

def load_all_features() -> pd.DataFrame:
    """Stack all 5 anchor-year parquet files."""
    frames = []
    for year in ANCHOR_YEARS:
        p = OUT_FEATURES / f"features_{year}.parquet"
        if not p.exists():
            log.warning(f"  ⚠ missing parquet for {year}: {p}")
            continue
        df = pd.read_parquet(p)
        frames.append(df)
    if not frames:
        raise RuntimeError("No feature parquets found")
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(df):,} total pixel-years across {len(frames)} parquets")
    return df


def build_transitions(df: pd.DataFrame, high_conf_only: bool,
                       min_class_pixels: int = 50) -> pd.DataFrame:
    """Build per-pixel-pair transition rows.

    For each pixel observed at years (t, t+5), produce one row:
        features at year t  (X)
        target = class at year t+5  (y)

    The "from class" at year t is encoded via the categorical column
    `from_class` (XGBoost can handle integers as categories).

    Drops 2020 (no t+5 successor) and any pixel whose class is 0 (NoData)
    at either endpoint.

    Also drops classes with fewer than `min_class_pixels` rows in the target
    or source — these are statistically meaningless for ML attribution AND
    cause non-contiguous-label crashes in XGBoost's spatial CV folds.
    """
    log.info("Building transition rows...")
    pix_keys = ["row", "col"]
    by_year = {int(y): g.set_index(pix_keys) for y, g in df.groupby("year")}

    rows = []
    for i in range(len(ANCHOR_YEARS) - 1):
        y1, y2 = ANCHOR_YEARS[i], ANCHOR_YEARS[i + 1]
        if y1 not in by_year or y2 not in by_year:
            continue
        g1 = by_year[y1]
        g2 = by_year[y2]
        common_idx = g1.index.intersection(g2.index)
        if len(common_idx) == 0:
            continue
        a = g1.loc[common_idx].copy()
        b = g2.loc[common_idx]
        a["target_class"]   = b["lulc_class"].values.astype(np.int8)
        a["target_confidence"] = b["conf_joint"].values
        a["from_class"]     = a["lulc_class"].values.astype(np.int8)
        a["from_conf"]      = a["conf_joint"].values
        a["from_year"]      = y1
        a["to_year"]        = y2
        mask = (a["from_class"] > 0) & (a["target_class"] > 0)
        a = a[mask]
        if high_conf_only:
            mask_hc = (a["from_conf"] >= MIN_CONF_HI) & (a["target_confidence"] >= MIN_CONF_HI)
            a = a[mask_hc]
        rows.append(a.reset_index())
    if not rows:
        raise RuntimeError("No transitions built")
    out = pd.concat(rows, ignore_index=True)
    log.info(f"  Built {len(out):,} pixel-pair rows "
             f"({'high-conf filter ON' if high_conf_only else 'no conf filter'})")

    # Drop rare classes BOTH from target and from source — they cause
    # non-contiguous-label crashes in XGBoost spatial-CV folds, and a class
    # with <50 pixels can't support stable inference anyway.
    target_counts = out["target_class"].value_counts()
    source_counts = out["from_class"].value_counts()
    rare_target = target_counts[target_counts < min_class_pixels].index.tolist()
    rare_source = source_counts[source_counts < min_class_pixels].index.tolist()
    rare = set(rare_target) | set(rare_source)
    if rare:
        rare_names = [f"{int(c)} {UNIFIED_LEGEND.get(int(c), ('?',''))[0]}"
                      for c in sorted(rare)]
        log.warning(f"  ⚠ Dropping rare classes (< {min_class_pixels} pixels): "
                    f"{rare_names}")
        before = len(out)
        out = out[~out["target_class"].isin(rare) & ~out["from_class"].isin(rare)]
        log.warning(f"    {before:,} → {len(out):,} rows after filter")

    log.info(f"  Target class distribution:")
    vc = out["target_class"].value_counts().sort_index()
    for code, n in vc.items():
        name = UNIFIED_LEGEND.get(int(code), ("?", ""))[0]
        log.info(f"    class {int(code):>2} {name:<15}: {n:>10,} "
                 f"({100*n/len(out):5.2f}%)")
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (X, y, feature_names).

    X = feature matrix (all columns except target/geometry/from_year/to_year/conf)
    y = target class (1..14)
    """
    # Build the feature column list
    drop_cols = set(NON_FEATURE_GEOM_COLS) | {"target_class", "target_confidence",
                                               "from_conf", "from_year", "to_year"}
    drop_cols |= {c for c in df.columns
                  if any(c.startswith(p) for p in EXCLUDED_COLUMN_PREFIXES)}
    # But keep `from_class` (the current-year class is a valid input)
    drop_cols.discard("from_class")
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    # Convert from_class to integer dtype that XGBoost categorical-aware mode
    # can use (we just pass as numeric and let the tree decide; cleaner)
    X["from_class"] = X["from_class"].astype(np.int8)
    y = df["target_class"].values.astype(np.int8)
    log.info(f"  Feature matrix: {X.shape}  ({len(feature_cols)} features)")
    return X, y, feature_cols


# ============================================================================
# SPATIAL BLOCK CV
# ============================================================================

def spatial_block_folds(df: pd.DataFrame, n_folds: int = 5,
                       block_size_km: int = BLOCK_SIZE_KM) -> list[np.ndarray]:
    """Generate fold indices that respect 100 km spatial blocks.

    Each pixel gets a `block_id` based on its (x, y) in Albers (1 km units),
    floor-divided by block_size_km. Folds are made by randomly partitioning
    blocks (not pixels) into n_folds groups. This prevents the "train pixel
    is the next-door neighbour of a test pixel" leakage.

    Returns a list of test-fold pixel-index arrays (length n_folds).
    """
    if "x" not in df.columns or "y" not in df.columns:
        log.warning("  no x/y columns — falling back to random KFold")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
        return [test for _, test in kf.split(np.arange(len(df)))]

    # Target Albers grid units are meters; convert to km for block size
    bx = (df["x"].values / 1000.0 / block_size_km).astype(np.int32)
    by = (df["y"].values / 1000.0 / block_size_km).astype(np.int32)
    block_id = bx * 1_000_000 + by

    unique_blocks = np.unique(block_id)
    n_blocks = len(unique_blocks)
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(unique_blocks)
    # Assign blocks to folds
    block_to_fold = {b: i % n_folds for i, b in enumerate(unique_blocks)}
    pixel_fold = np.array([block_to_fold[b] for b in block_id])

    folds = []
    for k in range(n_folds):
        test_idx = np.where(pixel_fold == k)[0]
        folds.append(test_idx)
    sizes = [len(f) for f in folds]
    log.info(f"  Spatial block CV: {n_blocks} blocks of {block_size_km}km "
             f"split into {n_folds} folds; sizes: {sizes}")
    return folds


# ============================================================================
# TRAIN + EVAL ONE VERSION
# ============================================================================

def train_one_version(
    df: pd.DataFrame,
    label: str,
    use_gpu: bool,
    n_folds: int = 5,
    n_estimators: int = 800,
    early_stopping: int = 50,
) -> dict:
    """Train and evaluate one multinomial XGBoost model with spatial CV.

    Returns a dict containing the trained model + CV metrics + SHAP outputs.
    """
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, f1_score,
                                 confusion_matrix, classification_report)

    log.info(f"\n{'='*72}")
    log.info(f"TRAINING VERSION {label}  ({'GPU' if use_gpu else 'CPU'})")
    log.info(f"{'='*72}")

    X, y, feat_cols = split_features_target(df)

    # XGBoost expects target labels 0..K-1. Build a GLOBAL contiguous remap of
    # all classes present in the full dataset. Using the native xgb.train() API
    # (rather than XGBClassifier.fit) avoids the "non-contiguous labels inferred
    # from this fold" crash that happens when a class is globally present but
    # absent from one spatial fold's training subset.
    classes_present = sorted(np.unique(y).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes_present)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int32)
    n_classes = len(classes_present)
    log.info(f"  Classes present in target: {classes_present}  "
             f"(n={n_classes})")

    # Spatial fold splits
    folds = spatial_block_folds(df, n_folds=n_folds)

    # ----- CV TRAINING -----
    fold_metrics = []
    oof_pred = np.full(len(df), -1, dtype=np.int32)   # out-of-fold predictions
    feat_gain_total = np.zeros(len(feat_cols), dtype=np.float64)
    feat_cover_total = np.zeros(len(feat_cols), dtype=np.float64)

    # Native-API parameters (note: num_class is GLOBAL and fixed, so even a
    # fold missing some class still trains with the full class space)
    native_params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RNG_SEED,
        "tree_method": "hist",
        "device": ("cuda" if use_gpu else "cpu"),
        "verbosity": 1,
    }

    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx, assume_unique=True)
        log.info(f"\n  Fold {k+1}/{n_folds}: train={len(train_idx):,}  "
                 f"test={len(test_idx):,}")

        # Sample 10% of training fold as internal validation for early stopping
        rng = np.random.default_rng(RNG_SEED + k)
        n_val = max(5000, len(train_idx) // 10)
        val_pos = rng.choice(len(train_idx), size=n_val, replace=False)
        val_mask = np.zeros(len(train_idx), dtype=bool)
        val_mask[val_pos] = True
        sub_train = train_idx[~val_mask]
        sub_val = train_idx[val_mask]

        X_train = X.iloc[sub_train]
        y_train = y_idx[sub_train]
        X_val = X.iloc[sub_val]
        y_val = y_idx[sub_val]
        X_test = X.iloc[test_idx]
        y_test = y_idx[test_idx]

        # Warn (don't crash) if a class is missing from this fold's training set
        train_classes = set(np.unique(y_train).tolist())
        missing = set(range(n_classes)) - train_classes
        if missing:
            missing_codes = [idx_to_class[i] for i in sorted(missing)]
            log.warning(f"    ⚠ fold {k+1} training set missing class(es) "
                        f"{missing_codes}; native API handles this gracefully.")

        # Native API: DMatrix + xgb.train. num_class is fixed globally, so a
        # fold missing some class trains fine (predicts 0 prob for that class).
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        dtest  = xgb.DMatrix(X_test)
        booster = xgb.train(
            native_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "validation_0")],
            early_stopping_rounds=early_stopping,
            verbose_eval=100,
        )

        # Predict on held-out test fold (softprob → argmax)
        proba = booster.predict(dtest)            # (n_test, n_classes)
        y_pred = proba.argmax(axis=1)
        oof_pred[test_idx] = np.array([idx_to_class[i] for i in y_pred],
                                       dtype=np.int32)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        log.info(f"    fold {k+1}: acc={acc:.4f}  macroF1={f1m:.4f}  "
                 f"weightedF1={f1w:.4f}")
        fold_metrics.append({"fold": k+1, "n_train": int(len(sub_train)),
                              "n_test": int(len(test_idx)),
                              "accuracy": float(acc),
                              "macro_f1": float(f1m),
                              "weighted_f1": float(f1w)})

        # Accumulate feature importance (gain) and cover
        gain = booster.get_score(importance_type="gain")
        cover = booster.get_score(importance_type="cover")
        for fi, fname in enumerate(feat_cols):
            # native DMatrix uses the actual column names, not f0/f1/...
            feat_gain_total[fi]  += gain.get(fname,  0.0)
            feat_cover_total[fi] += cover.get(fname, 0.0)

    # ----- OVERALL OUT-OF-FOLD METRICS -----
    valid = oof_pred > 0
    overall_acc = accuracy_score(y[valid], oof_pred[valid])
    overall_f1m = f1_score(y[valid], oof_pred[valid], average="macro", zero_division=0)
    overall_f1w = f1_score(y[valid], oof_pred[valid], average="weighted", zero_division=0)
    log.info(f"\n  OVERALL OUT-OF-FOLD (Version {label}):")
    log.info(f"    accuracy:    {overall_acc:.4f}")
    log.info(f"    macro F1:    {overall_f1m:.4f}")
    log.info(f"    weighted F1: {overall_f1w:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y[valid], oof_pred[valid],
                          labels=classes_present)
    cm_df = pd.DataFrame(cm,
                         index=[f"actual_{c}" for c in classes_present],
                         columns=[f"pred_{c}" for c in classes_present])

    # ----- FINAL MODEL ON ALL DATA (for SHAP) -----
    log.info(f"\n  Training final model on ALL data for SHAP analysis...")
    rng = np.random.default_rng(RNG_SEED)
    n_val_final = max(5000, len(X) // 10)
    val_pos = rng.choice(len(X), size=n_val_final, replace=False)
    val_mask = np.zeros(len(X), dtype=bool)
    val_mask[val_pos] = True
    dtrain_final = xgb.DMatrix(X.iloc[~val_mask], label=y_idx[~val_mask])
    dval_final   = xgb.DMatrix(X.iloc[val_mask],  label=y_idx[val_mask])
    final_booster = xgb.train(
        native_params,
        dtrain_final,
        num_boost_round=n_estimators,
        evals=[(dval_final, "validation_0")],
        early_stopping_rounds=early_stopping,
        verbose_eval=200,
    )

    # ----- FEATURE IMPORTANCE TABLE -----
    feat_gain_norm = feat_gain_total / (feat_gain_total.sum() + 1e-9)
    feat_imp_df = pd.DataFrame({
        "feature": feat_cols,
        "cv_gain_sum": feat_gain_total,
        "cv_gain_norm": feat_gain_norm,
        "cv_cover_sum": feat_cover_total,
    }).sort_values("cv_gain_norm", ascending=False)
    log.info(f"\n  Top 20 features by gain (mean across folds):")
    for _, row in feat_imp_df.head(20).iterrows():
        log.info(f"    {row['feature']:<32}  gain%={100*row['cv_gain_norm']:5.2f}")

    return {
        "label": label,
        "model": final_booster,
        "feat_cols": feat_cols,
        "classes_present": classes_present,
        "idx_to_class": idx_to_class,
        "fold_metrics": fold_metrics,
        "overall": {"accuracy": float(overall_acc),
                    "macro_f1": float(overall_f1m),
                    "weighted_f1": float(overall_f1w)},
        "confusion_matrix": cm_df,
        "feat_imp_df": feat_imp_df,
        "X_for_shap": X,                 # full feature matrix
        "df_for_shap": df.reset_index(drop=True),
    }


# ============================================================================
# SHAP
# ============================================================================

def compute_shap(result: dict, sample_size: int = SHAP_SAMPLE_SIZE) -> dict:
    """Compute SHAP values for a stratified sample of pixels.

    Returns:
      shap_values   : (n_sample, n_features, n_classes) np.ndarray
      sample_idx    : pixel indices that the sample was drawn from
      summary_df    : feature × class mean(|SHAP|) table
    """
    import shap

    log.info(f"\n  Computing TreeSHAP on a {sample_size:,}-pixel stratified sample...")
    clf = result["model"]
    X = result["X_for_shap"]
    classes = result["classes_present"]
    feat_cols = result["feat_cols"]

    # Stratified sample by (from_class, target_class) so every transition type
    # is represented in SHAP
    df_for_strat = result["df_for_shap"]
    strat_key = (df_for_strat["from_class"].astype(str) + "_" +
                 df_for_strat["target_class"].astype(str))
    rng = np.random.default_rng(RNG_SEED)
    # Allocate sample slots proportional to log(n_pixels) so rare transitions
    # are over-represented relative to natural counts (better SHAP coverage)
    counts = strat_key.value_counts()
    weights = np.log1p(counts).rename_axis("k").reset_index(name="w")
    weights["n_sample"] = (weights["w"] / weights["w"].sum() * sample_size).round().astype(int)
    sample_idx_list = []
    for k, n in zip(weights["k"], weights["n_sample"]):
        if n == 0:
            continue
        pool = np.where(strat_key == k)[0]
        n_take = min(n, len(pool))
        sample_idx_list.append(rng.choice(pool, size=n_take, replace=False))
    sample_idx = np.concatenate(sample_idx_list)
    log.info(f"  Sample contains {len(sample_idx):,} pixels covering "
             f"{(weights['n_sample'] > 0).sum()} transition strata")

    X_sample = X.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    # shap_values is either a list-per-class (n_classes lists of (n,F) arrays)
    # or a 3D array (n, F, n_classes) depending on shap version. Standardise:
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)  # (n, F, n_classes)
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 2:
            # 2D means binary or single output — pad to 3D
            shap_array = shap_array[..., None]

    log.info(f"  SHAP shape: {shap_array.shape}")

    # Summary: mean(|SHAP|) per feature, per class
    mean_abs = np.abs(shap_array).mean(axis=0)   # (F, n_classes)
    summary_df = pd.DataFrame(mean_abs,
                              index=feat_cols,
                              columns=[f"class_{c}" for c in classes])
    summary_df["overall"] = mean_abs.mean(axis=1)
    summary_df = summary_df.sort_values("overall", ascending=False)
    log.info(f"  Top 20 features by mean(|SHAP|) overall:")
    for fname, row in summary_df.head(20).iterrows():
        log.info(f"    {fname:<32}  meanShap={row['overall']:.4f}")

    return {
        "shap_values": shap_array,
        "sample_idx": sample_idx,
        "summary_df": summary_df,
        "X_sample": X_sample,
    }


# ============================================================================
# WRITE OUTPUTS
# ============================================================================

def write_outputs(result: dict, shap_out: dict, label: str):
    """Write all artifacts for one version to OUT_MN/<label>/."""
    out_dir = OUT_MN
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model pickle
    pkl_path = out_dir / f"model_{label}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model": result["model"],
            "feat_cols": result["feat_cols"],
            "classes_present": result["classes_present"],
            "idx_to_class": result["idx_to_class"],
        }, f)
    log.info(f"  ✓ {pkl_path.name}")

    # CV metrics
    cv_path = out_dir / f"cv_metrics_{label}.json"
    with open(cv_path, "w", encoding="utf-8") as f:
        json.dump({"label": label,
                   "fold_metrics": result["fold_metrics"],
                   "overall": result["overall"]}, f, indent=2)
    log.info(f"  ✓ {cv_path.name}")

    # Feature importance
    fi_path = out_dir / f"feature_importance_{label}.csv"
    result["feat_imp_df"].to_csv(fi_path, index=False)
    log.info(f"  ✓ {fi_path.name}")

    # Confusion matrix
    cm_path = out_dir / f"confusion_matrix_{label}.csv"
    result["confusion_matrix"].to_csv(cm_path)
    log.info(f"  ✓ {cm_path.name}")

    # SHAP raw values + sample index
    if shap_out is not None:
        np.save(out_dir / f"shap_values_{label}.npy", shap_out["shap_values"])
        log.info(f"  ✓ shap_values_{label}.npy  shape={shap_out['shap_values'].shape}")
        sidx_path = out_dir / f"shap_sample_index_{label}.csv"
        pd.DataFrame({"sample_idx": shap_out["sample_idx"]}).to_csv(sidx_path, index=False)
        log.info(f"  ✓ {sidx_path.name}")
        summary_path = out_dir / f"shap_summary_{label}.csv"
        shap_out["summary_df"].to_csv(summary_path)
        log.info(f"  ✓ {summary_path.name}")


def plot_comparison(results_a: dict, shap_a: dict,
                     results_b: dict, shap_b: dict):
    """Generate the A vs B comparison plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Top 20 features by mean|SHAP| for each version
    for ax, label, shap_out in [(axes[0], "A (high-conf)", shap_a),
                                  (axes[1], "B (all pixels)", shap_b)]:
        if shap_out is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        top20 = shap_out["summary_df"]["overall"].head(20).iloc[::-1]
        ax.barh(range(len(top20)), top20.values, color="#3a7bd5")
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20.index, fontsize=9)
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"Top 20 features — Version {label}")
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        plt.savefig(OUT_MN / f"shap_summary_combined.{ext}",
                    dpi=200 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"  ✓ shap_summary_combined.[png/pdf/svg]")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true",
                        help="use CUDA (RTX 3060 Ti) for XGBoost")
    parser.add_argument("--version", choices=["A", "B", "both"], default="both",
                        help="A=high-conf only, B=all pixels, both=both (default)")
    parser.add_argument("--shap-sample", type=int, default=SHAP_SAMPLE_SIZE,
                        help=f"SHAP sample size (default {SHAP_SAMPLE_SIZE})")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=800)
    args = parser.parse_args()

    ensure_dirs()
    OUT_MN.mkdir(parents=True, exist_ok=True)
    log.info("=" * 80)
    log.info("STEP 05 — Multinomial transition attribution (XGBoost + SHAP)")
    log.info("=" * 80)
    log.info(f"  device:       {'CUDA (GPU)' if args.gpu else 'CPU'}")
    log.info(f"  versions:     {args.version}")
    log.info(f"  SHAP sample:  {args.shap_sample:,}")
    log.info(f"  CV folds:     {args.n_folds} ({BLOCK_SIZE_KM}km spatial blocks)")

    # Load features ONCE
    df_all = load_all_features()

    results = {}
    shap_outs = {}

    versions_to_run = ["A", "B"] if args.version == "both" else [args.version]
    for v in versions_to_run:
        high_conf = (v == "A")
        df_trans = build_transitions(df_all, high_conf_only=high_conf)
        result = train_one_version(
            df_trans, label=v, use_gpu=args.gpu,
            n_folds=args.n_folds, n_estimators=args.n_estimators,
        )
        shap_out = compute_shap(result, sample_size=args.shap_sample)
        write_outputs(result, shap_out, label=v)
        results[v] = result
        shap_outs[v] = shap_out
        # Free memory before next version
        del result["X_for_shap"]
        del result["df_for_shap"]

    # Comparison plot (if both versions ran)
    if "A" in results and "B" in results:
        plot_comparison(results["A"], shap_outs["A"],
                         results["B"], shap_outs["B"])

    log.info("=" * 80)
    log.info("Step 05 complete.")
    log.info(f"  Outputs: {OUT_MN}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
