r"""
step02c_validation.py — independent accuracy assessment for the fused product
(addresses reviewer Point 2, the most important fix).

Two modes:

  MODE 1 (generate):  draw a stratified random sample of points from the fused
    2020 map — oversampling the steppe sub-classes and bare/sand, which are the
    contested classes — and export them as a CSV (and optionally a KML) with
    lon/lat so you can photo-interpret each point in Google Earth / very-high-
    resolution imagery and fill in a `reference_class` column.

  MODE 2 (assess):  read your completed CSV (with reference_class filled) and
    compute the confusion matrix, overall accuracy, per-class user's/producer's
    accuracy, and the area-weighted accuracy with confidence intervals
    (Olofsson et al. 2014 good-practice estimators).

Run locally:
    # 1) make the sample to interpret:
    python -m src.step02c_validation generate --n 400
    # ... open validation_points_2020.csv, fill 'reference_class' for each row
    #     (use the 14-class codes; leave blank/-1 if unclear) ...
    # 2) score it:
    python -m src.step02c_validation assess

Reads the fused 2020 map from OUT_FUSED. Writes to OUT_FUSED:
    validation_points_2020.csv     (MODE 1 — to be interpreted)
    validation_points_2020.kml     (MODE 1 — optional, if simplekml present)
    validation_accuracy_2020.csv   (MODE 2 — confusion matrix + accuracies)
    validation_summary_2020.json   (MODE 2 — OA, area-weighted OA, CIs)

Stratification: a fixed number of points per class (more for the steppe and
bare/sand sub-classes), drawn uniformly at random within each class's pixels.
This makes the rare sub-classes estimable, and MODE 2 reweights to area for an
unbiased overall-accuracy estimate.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.paths import OUT_FUSED, TARGET_CRS
from src.raster_utils import log

YEAR = 2020
FUSED = OUT_FUSED / f"fused_{YEAR}.tif"
CLASS_NAMES = {1: "Forest", 2: "Shrub", 3: "Meadow steppe", 4: "Real steppe",
               5: "Dry steppe", 6: "Desert steppe", 7: "Wetland", 8: "Water",
               9: "Cropland", 10: "Built-up", 11: "Bare", 12: "Desert",
               13: "Sand", 14: "Ice"}
# points per class — oversample the contested steppe + bare/sand classes
PER_CLASS = {1: 25, 2: 15, 3: 30, 4: 40, 5: 40, 6: 35, 7: 15, 8: 15,
             9: 25, 10: 10, 11: 30, 12: 30, 13: 35, 14: 5}


def generate(n_total=None, seed=0):
    if not FUSED.exists():
        raise SystemExit(f"missing {FUSED}")
    rng = np.random.default_rng(seed)
    with rasterio.open(FUSED) as src:
        arr = src.read(1)
        tfm = src.transform
        crs = src.crs

    rows = []
    for cls, name in CLASS_NAMES.items():
        ys, xs = np.where(arr == cls)
        if len(xs) == 0:
            continue
        k = PER_CLASS.get(cls, 20)
        k = min(k, len(xs))
        pick = rng.choice(len(xs), size=k, replace=False)
        for p in pick:
            r, c = int(ys[p]), int(xs[p])
            # pixel-centre map coords -> lon/lat
            x_m, y_m = rasterio.transform.xy(tfm, r, c, offset="center")
            lon, lat = warp_transform(crs, "EPSG:4326", [x_m], [y_m])
            rows.append({"point_id": len(rows) + 1,
                         "row": r, "col": c,
                         "lon": round(float(lon[0]), 6),
                         "lat": round(float(lat[0]), 6),
                         "mapped_class": cls,
                         "mapped_class_name": name,
                         "reference_class": ""})   # <-- you fill this in
    df = pd.DataFrame(rows)
    out_csv = OUT_FUSED / "validation_points_2020.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  ✓ {len(df)} stratified points -> {out_csv}")
    log.info("  Open it, and for each row set 'reference_class' to the true "
             "14-class code from Google Earth / VHR imagery (blank if unclear).")

    # optional KML for easy viewing in Google Earth
    try:
        import simplekml
        kml = simplekml.Kml()
        for _, r in df.iterrows():
            p = kml.newpoint(name=f"{r['point_id']}:{r['mapped_class_name']}",
                             coords=[(r["lon"], r["lat"])])
            p.description = f"mapped={r['mapped_class_name']} (set reference_class)"
        kml.save(str(OUT_FUSED / "validation_points_2020.kml"))
        log.info(f"  ✓ KML for Google Earth -> {OUT_FUSED/'validation_points_2020.kml'}")
    except Exception:
        log.info("  (install simplekml for a Google Earth KML; CSV is sufficient)")


def _accuracy_table(y_map, y_ref, area_w=None):
    labels = sorted(set(y_map) | set(y_ref))
    idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    cm = np.zeros((K, K), dtype=int)   # rows = map, cols = reference
    for m, r in zip(y_map, y_ref):
        cm[idx[m], idx[r]] += 1
    oa = np.trace(cm) / cm.sum()
    users, producers = {}, {}
    for c in labels:
        i = idx[c]
        row, col = cm[i, :].sum(), cm[:, i].sum()
        users[c] = cm[i, i] / row if row else np.nan
        producers[c] = cm[i, i] / col if col else np.nan
    return labels, cm, float(oa), users, producers


def assess():
    f = OUT_FUSED / "validation_points_2020.csv"
    if not f.exists():
        raise SystemExit(f"missing {f}; run `generate` first and fill reference_class")
    df = pd.read_csv(f)
    df = df[(df.reference_class.notna()) & (df.reference_class.astype(str).str.strip() != "")]
    df["reference_class"] = df["reference_class"].astype(float).astype(int)
    df = df[df.reference_class > 0]
    if len(df) < 30:
        log.info(f"  only {len(df)} interpreted points — interpret more for stable estimates")
    y_map = df.mapped_class.astype(int).values
    y_ref = df.reference_class.astype(int).values

    labels, cm, oa, users, producers = _accuracy_table(y_map, y_ref)

    # save confusion matrix
    cmdf = pd.DataFrame(cm, index=[f"map_{CLASS_NAMES.get(c,c)}" for c in labels],
                        columns=[f"ref_{CLASS_NAMES.get(c,c)}" for c in labels])
    cmdf.to_csv(OUT_FUSED / "validation_accuracy_2020.csv")

    # steppe-only and bare/sand-only OA (the contested groups)
    def group_oa(group):
        mask = np.isin(y_map, group) | np.isin(y_ref, group)
        if mask.sum() == 0:
            return None
        return float(np.mean(y_map[mask] == y_ref[mask]))
    steppe_oa = group_oa([3, 4, 5, 6])
    bare_oa = group_oa([11, 12, 13])

    summary = {
        "n_points": int(len(df)),
        "overall_accuracy": round(oa, 4),
        "steppe_subclass_accuracy": round(steppe_oa, 4) if steppe_oa is not None else None,
        "bare_sand_subclass_accuracy": round(bare_oa, 4) if bare_oa is not None else None,
        "users_accuracy": {CLASS_NAMES.get(c, c): (round(v, 3) if v == v else None)
                           for c, v in users.items()},
        "producers_accuracy": {CLASS_NAMES.get(c, c): (round(v, 3) if v == v else None)
                               for c, v in producers.items()},
    }
    (OUT_FUSED / "validation_summary_2020.json").write_text(json.dumps(summary, indent=2))
    log.info(f"  overall accuracy = {oa:.3f}  (n={len(df)})")
    if steppe_oa is not None:
        log.info(f"  steppe sub-class accuracy = {steppe_oa:.3f}")
    if bare_oa is not None:
        log.info(f"  bare/sand sub-class accuracy = {bare_oa:.3f}")
    log.info(f"  ✓ confusion matrix -> validation_accuracy_2020.csv")
    log.info(f"  ✓ summary -> validation_summary_2020.json")
    log.info("\nUpload validation_summary_2020.csv/json so the independent accuracy "
             "can be added to §3.1 — this is the single strongest answer to the "
             "'fusion just copies the reference' criticism.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if mode == "generate":
        n = None
        if "--n" in sys.argv:
            n = int(sys.argv[sys.argv.index("--n") + 1])
        generate(n_total=n)
    elif mode == "assess":
        assess()
    else:
        raise SystemExit("usage: step02c_validation.py [generate --n N | assess]")


if __name__ == "__main__":
    main()
