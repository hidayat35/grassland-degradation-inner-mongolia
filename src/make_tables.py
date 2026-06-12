"""
Build the three main-text tables for Paper 2 as CSV data-availability files,
and emit a markdown rendering for the manuscript.

Tables:
  T1  Input land-cover products used in the fusion (8 products)
  T2  Spatial block cross-validation metrics for the transition model (A vs B)
  T3  Consensus causal links (Granger AND CCM significant)

Run:
    python make_tables.py            # writes ./tables/*.csv and tables.md
"""
from pathlib import Path
import pandas as pd

OUT = Path("tables"); OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------- T1 --------
# Input products to the hierarchical Bayesian fusion, taken directly from
# configs/class_harmonization.py PRODUCT_REGISTRY (names, weights, native
# resolution, year coverage, and fusion role).
T1 = pd.DataFrame([
    ["Wang et al. 2024 (Mongolian Plateau)", "1990–2020 (5-yr)", "30 m", "1.00",
     "Super-class + grassland sub-class (only fine-grained product)"],
    ["GLC-FCS30D", "1985–2022 (annual)", "30 m", "0.90", "Super-class voter"],
    ["CLCD (China Land Cover Dataset)", "1985–2023 (annual)", "30 m", "0.85", "Super-class voter"],
    ["ESA CCI Land Cover", "1992–2022 (annual)", "300 m", "0.85", "Super-class voter"],
    ["MODIS MCD12Q1 (IGBP)", "2001–2024 (annual)", "500 m", "0.75", "Super-class voter"],
    ["ESA WorldCover", "2020, 2021", "10 m", "0.80", "Super-class voter (recent, high-resolution)"],
    ["MPSDSL sandy-land / desert", "1990–2020 (5-yr)", "30 m", "0.95",
     "Specialist: desert + sandy-land sub-classes only"],
    ["Grassland map (Inner Mongolia)", "1991–2020 (annual)", "30 m", "0.50",
     "Specialist prior: grassland sub-classes (meadow/typical/dry/desert steppe)"],
], columns=["Product", "Temporal coverage", "Native resolution",
            "Fusion weight", "Role in fusion"])

# ----------------------------------------------------------------- T2 --------
# Spatial 100-km block cross-validation, multinomial transition model.
T2 = pd.DataFrame([
    ["A — high-confidence pixels (conf ≥ 80 at both endpoints)", "3.24", "12", "82.2", "0.597", "0.813"],
    ["B — all pixels", "4.59", "13", "77.8", "0.600", "0.772"],
], columns=["Model version", "Training rows (millions)", "Classes",
            "Overall accuracy (%)", "Macro-F1", "Weighted-F1"])

# ----------------------------------------------------------------- T3 --------
# Consensus causal links: significant under BOTH Granger causality and CCM.
# Values from step06 (granger_results.csv / ccm_results.csv); strongest first.
T3 = pd.DataFrame([
    ["Sandy land", "Grazing intensity", "0.0004", "0.80", "1–2", "Strongest consensus link; arid managed zone"],
    ["Sandy land", "Population density", "0.006", "0.58", "1–2", "Human pressure in the sandy south"],
    ["Desert / barren", "Grazing intensity", "0.027", "0.62", "1–2", "Livestock pressure on bare/desert margin"],
    ["Desert / barren", "SPEI-12 drought", "0.045", "0.47", "1–2", "Drought control on desert-margin productivity"],
    ["Dry steppe", "Grazing intensity", "0.048", "0.48", "1–2", "Grazing pressure on the steppe gradient"],
], columns=["Response (NDVI in zone)", "Driver", "Granger p", "CCM ρ",
            "Lag (yr)", "Interpretation"])


def main():
    T1.to_csv(OUT / "table1_input_products.csv", index=False)
    T2.to_csv(OUT / "table2_cv_metrics.csv", index=False)
    T3.to_csv(OUT / "table3_causal_consensus.csv", index=False)

    def md(df, title, note):
        s = [f"### {title}\n", df.to_markdown(index=False), f"\n*{note}*\n"]
        return "\n".join(s)

    doc = "\n\n".join([
        "# Paper 2 — Main-text tables\n",
        md(T1, "Table 1. Input land-cover products used in the hierarchical Bayesian fusion.",
           "Eight products contribute to the fusion, all harmonised to the 14-class unified "
           "legend and resampled to the 1-km Albers analysis grid. The fusion weight reflects "
           "each product's reliability and is used in the Level-1 (super-class) Bayesian vote. "
           "Three products play specialist roles: Wang et al. (2024) is the only product that "
           "resolves grassland sub-classes; the MPSDSL map is trusted only for the desert and "
           "sandy-land sub-classes; and the Inner Mongolia grassland map serves as a sub-class "
           "prior. WorldCover (2020–2021) is exported at 30 m from its native 10 m via Google "
           "Earth Engine."),
        md(T2, "Table 2. Spatial block cross-validation of the multinomial transition model.",
           "Skill is evaluated under 100-km spatial-block cross-validation (no spatial "
           "leakage between train and test). Version A restricts to high-confidence pixels "
           "(fusion confidence ≥ 80 at both the origin and destination year); Version B uses "
           "all pixels. The close agreement between versions indicates the conclusions are "
           "not artefacts of the confidence threshold."),
        md(T3, "Table 3. Consensus causal links between drivers and vegetation productivity.",
           "Links significant under BOTH Granger causality (p < 0.05) AND convergent "
           "cross-mapping (CCM). Response is the zonal mean growing-season NDVI (2001–2024); "
           "CCM ρ is the convergent cross-map skill at the optimal library size; lag is the "
           "dominant causal lag. Grazing intensity is the most consistent causal driver "
           "across the arid degradation zones."),
    ])
    (OUT / "tables.md").write_text(doc)
    print("Wrote 3 CSVs + tables.md to ./tables/")
    print(doc)


if __name__ == "__main__":
    main()
