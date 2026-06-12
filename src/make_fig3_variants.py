"""
Generate ALL Figure-3 variants into ./Figure 3/ for visual selection.

Combinations produced (2 label styles x 2 classification schemes x 2 width
scalings = 8 flow diagrams), each as a 2-panel figure (flow + change-rate),
plus a CSV companion of the classified transitions.

Run (sandbox or local — only needs transitions_to_model.csv + matplotlib):
    python proto_fig3_variants.py /path/to/transitions_to_model.csv

If no path is given it looks in /mnt/user-data/uploads/ then the CWD.
Output: ./Figure 3/fig3_<labels>_<scheme>_<width>.png  + classified CSVs.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

# ---------------------------------------------------------------- data ------
def find_csv():
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        return Path(sys.argv[1])
    for p in [Path("/mnt/user-data/uploads/transitions_to_model.csv"),
              Path("transitions_to_model.csv")]:
        if p.exists():
            return p
    raise SystemExit("transitions_to_model.csv not found — pass its path as arg 1")

CSV = find_csv()
OUT = Path("Figure 3"); OUT.mkdir(exist_ok=True)
df0 = pd.read_csv(CSV)

NAMES = {1:"Forest",3:"Meadow\nsteppe",4:"Typical\nsteppe",5:"Dry\nsteppe",
         6:"Desert\nsteppe",9:"Cropland",11:"Bare\nland",12:"Desert",13:"Sand"}
ORDER = [1, 3, 4, 5, 6, 13, 11, 12, 9]
YPOS = {c: len(ORDER) - 1 - i for i, c in enumerate(ORDER)}
RANK = {1:0, 3:1, 4:2, 5:3, 6:4, 13:5, 11:6, 12:7}
COL = {1:"#0b4d2c",3:"#5cab5a",4:"#94c465",5:"#c2a23a",6:"#9c7a30",
       9:"#e8722a",11:"#b0a08f",12:"#cda15a",13:"#efe0a8"}
STEPPE = {1, 3, 4, 5, 6}; ARID = {13, 11, 12}

# ---------------------------------------------------- classification schemes -
def scheme_gradient(f, t):
    if f in RANK and t in RANK:
        if RANK[t] < RANK[f]: return "recovery"
        if RANK[t] > RANK[f]: return "degradation"
    return "agricultural"

def scheme_strict(f, t):
    if f in RANK and t in RANK:
        if RANK[t] < RANK[f]:
            return "recovery" if t in STEPPE else "arid reshuffle"
        if RANK[t] > RANK[f]:
            return "arid reshuffle" if (f in ARID and t in ARID) else "degradation"
    return "agricultural"

SCHEMES = {"gradient": scheme_gradient, "strict": scheme_strict}
DIRCOL = {"recovery": "#1a9850", "degradation": "#d6604d",
          "agricultural": "#9970ab", "arid reshuffle": "#969696"}

LABELS = {
    "years":  ("2000", "2020", "Land-cover transitions 2000–2020"),
    "fromto": ("From", "To  (all intervals pooled)",
               "Land-cover transitions (flows pooled over 2000–2020)"),
}
WIDTHS = {"sqrt": (lambda n, m: 0.5 + 6.5 * np.sqrt(n / m)),
          "linear": (lambda n, m: 0.4 + 9.0 * (n / m))}

RATE_INT = ["2000–\n2005", "2005–\n2010", "2010–\n2015", "2015–\n2020"]
# Change rates per interval. Verified against the fused maps via
# compute_change_rates.py (34.46 / 27.95 / 30.48 / 21.92 → rounded below).
# If change_rate_per_interval.csv is present in the output folder, read it so
# the panel is fully reproducible; otherwise use the verified constants.
RATES = [34.5, 27.9, 30.5, 21.9]
try:
    import pandas as _pd
    _rc = OUT / "change_rate_per_interval.csv"
    if _rc.exists():
        _df = _pd.read_csv(_rc)
        RATES = [round(float(v), 1) for v in _df.sort_values("interval")["change_rate_pct"]]
except Exception:
    pass


def draw(scheme_name, label_name, width_name):
    fn = SCHEMES[scheme_name]
    df = df0.copy()
    df["dir"] = [fn(int(f), int(t)) for f, t in zip(df.from_cls, df.to_cls)]
    lsrc, ltgt, title = LABELS[label_name]
    wfun = WIDTHS[width_name]
    maxpix = df.total_pixels.max()

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(20, 11),
                                  gridspec_kw={"width_ratios": [3, 1]})
    x_src, x_tgt, node_w = 0.0, 1.0, 0.14

    for _, r in df.sort_values("total_pixels").iterrows():
        f, t, n = int(r.from_cls), int(r.to_cls), r.total_pixels
        if f not in YPOS or t not in YPOS:
            continue
        y0, y1 = YPOS[f], YPOS[t]
        lw = wfun(n, maxpix)
        verts = [(x_src + node_w/2, y0), (0.5, y0), (0.5, y1), (x_tgt - node_w/2, y1)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]
        ax.add_patch(mpatches.PathPatch(MPath(verts, codes), fill=False,
                     edgecolor=DIRCOL[r["dir"]], lw=lw, alpha=0.5, capstyle="round"))

    for c in ORDER:
        y = YPOS[c]
        ax.add_patch(mpatches.Rectangle((x_src - node_w/2, y - 0.32), node_w, 0.64,
                     facecolor=COL[c], edgecolor="black", lw=0.6, zorder=5))
        ax.add_patch(mpatches.Rectangle((x_tgt - node_w/2, y - 0.32), node_w, 0.64,
                     facecolor=COL[c], edgecolor="black", lw=0.6, zorder=5))
        ax.text(x_src - node_w/2 - 0.04, y, NAMES[c], ha="right", va="center", fontsize=11)
        ax.text(x_tgt + node_w/2 + 0.04, y, NAMES[c], ha="left", va="center", fontsize=11)

    ax.text(x_src, len(ORDER) - 0.05, lsrc, ha="center", fontsize=13, fontweight="bold")
    ax.text(x_tgt, len(ORDER) - 0.05, ltgt, ha="center", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.55, 1.7); ax.set_ylim(-0.9, len(ORDER) + 0.2); ax.axis("off")
    ax.set_title(f"(a) {title} — width ∝ {width_name} area", fontsize=14, loc="left")

    cats = ["recovery", "degradation"] + \
           (["arid reshuffle"] if scheme_name == "strict" else []) + ["agricultural"]
    handles = [mpatches.Patch(color=DIRCOL[k], label=k.capitalize()) for k in cats]
    ax.legend(handles=handles, loc="lower center", ncol=len(cats), fontsize=12,
              frameon=False, bbox_to_anchor=(0.5, -0.02))

    rec = df[df["dir"] == "recovery"].total_pixels.sum()
    deg = df[df["dir"] == "degradation"].total_pixels.sum()
    ax.text(0.5, -0.62, f"Recovery {rec/1e3:.0f}k px  vs  Degradation {deg/1e3:.0f}k px"
            f"   (ratio {rec/max(deg,1):.1f} : 1)", ha="center", fontsize=13,
            fontweight="bold", transform=ax.transData)

    axr.bar(RATE_INT, RATES, color="#4575b4", alpha=0.85)
    for i, v in enumerate(RATES):
        axr.text(i, v + 0.4, f"{v}%", ha="center", fontsize=11)
    axr.set_ylabel("area changing class (%)", fontsize=12)
    axr.set_title("(b) Change rate per interval", fontsize=14, loc="left")
    axr.set_ylim(0, 40); axr.grid(axis="y", alpha=0.3)
    axr.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Bidirectional land-cover dynamics over the Inner Mongolia plateau",
                 fontsize=17, y=0.98)
    plt.tight_layout()
    stem = f"fig3_{label_name}_{scheme_name}_{width_name}"
    plt.savefig(OUT / f"{stem}.png", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close()
    return stem, rec, deg


def main():
    made = []
    for s in SCHEMES:
        # one classified CSV per scheme
        fn = SCHEMES[s]
        d = df0.copy()
        d["from_name"] = d.from_cls.map(lambda c: NAMES.get(c, str(c)).replace("\n", " "))
        d["to_name"] = d.to_cls.map(lambda c: NAMES.get(c, str(c)).replace("\n", " "))
        d["direction"] = [fn(int(f), int(t)) for f, t in zip(d.from_cls, d.to_cls)]
        d.to_csv(OUT / f"transitions_classified_{s}.csv", index=False)
        for lab in LABELS:
            for w in WIDTHS:
                stem, rec, deg = draw(s, lab, w)
                made.append((stem, rec, deg))
    print(f"Wrote {len(made)} variants + 2 classified CSVs to '{OUT}/'")
    for stem, rec, deg in made:
        print(f"  {stem:42s}  rec={rec:>8,}  deg={deg:>8,}  ratio={rec/max(deg,1):.2f}")


if __name__ == "__main__":
    main()
