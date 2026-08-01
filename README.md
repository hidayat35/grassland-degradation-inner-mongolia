# Causal attribution of grassland degradation in Inner Mongolia

Code for the study *"Locally governed grassland recovery supports zone-specific restoration management in Inner Mongolia, 2000–2024"*

The pipeline (1) fuses eight independent land-cover products into a 14-class record with per-pixel uncertainty using a hierarchical two-tier Bayesian scheme, (2) attributes the resulting land-cover transitions to environmental and anthropogenic drivers using gradient boosting with SHAP for association and FDR-corrected Granger causality with surrogate-tested convergent cross mapping (CCM) for causation, and (3) tests whether grassland change propagates spatially using a spatio-temporal graph-neural-network comparison.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_all.py                  # orchestrates the full pipeline (steps 00–07)
├── check_env.py                # verifies the Python environment / dependencies
├── comprehensive_diagnose.py   # input-data diagnostics
├── transitions_diagnostic.py   # transition sanity checks
├── configs/
│   ├── paths.py                # ALL input/output paths + target grid (EDIT THIS)
│   └── class_harmonization.py  # per-product class cross-walk + reliability weights
└── src/
    ├── raster_utils.py         # reprojection, AOI masking, NetCDF reading, grid helpers
    ├── io_helpers.py           # shared I/O utilities
    ├── ccm_core.py             # convergent cross mapping (Takens embedding, simplex)
    ├── gnn_models.py           # NodeMLP, NodeGRU, A3T-GCN architectures
    │
    ├── step00_diagnose.py            # inspect raw products, verify class mappings
    ├── step01_harmonize_lulc.py      # raw products -> unified 14-class, 1 km Albers
    ├── step02_fuse_lulc.py           # hierarchical two-tier Bayesian fusion (+ uncertainty)
    ├── step02b_fusion_sensitivity.py # robustness to epsilon-floor and weights
    ├── step02c_validation.py         # stratified validation point sampling + accuracy
    ├── step03_align_drivers.py       # reproject/resample all driver datasets to the grid
    ├── step04_features.py            # build ML-ready per-pixel feature tables (parquet)
    ├── step05_attribute_multinomial.py # XGBoost transition model + SHAP attribution
    ├── step05b_baselines.py          # persistence/Markov baselines + neighbour ablation
    ├── step06_causal.py              # regional Granger + CCM causal screening
    ├── step06b_causal_rigor.py       # FDR correction + CCM surrogate significance
    ├── step07_stgnn.py               # spatio-temporal GNN training (spillover test)
    ├── step07b_spillover_clean.py    # clean GRU + neighbour-feature spillover test
    │
    ├── compute_change_rates.py       # per-interval areal change rate
    ├── make_transition_matrix.py     # full 14×14 transition matrix (Supp. Table S8)
    ├── make_tables.py                # assemble main-text tables
    ├── make_causal_fig.py            # causal heatmap figure
    ├── smooth_causal_maps.py         # smoothing of per-pixel CCM maps (illustrative)
    ├── mapfig.py                     # shared map-figure styling (boundary, scalebar, etc.)
    ├── render_fig1.py                # Fig 1 study area
    ├── render_fig2.py                # Fig 2 fused land cover + uncertainty
    ├── make_fig3_variants.py         # Fig 3 transition-flow variants
    ├── render_fig4_shap.py           # Fig 4 SHAP attribution
    ├── render_causal_figs.py         # Fig 5 causal panels
    ├── render_gnn_maps.py            # GNN spatial outputs
    ├── render_step07_figs.py         # Fig 6 spillover + risk maps
    └── render_workflow.py            # workflow / graphical abstract
```

---

## Setup

1. **Python.** Python 3.9 is recommended (developed under a conda environment).

   ```bash
   conda create -n grassland python=3.9
   conda activate grassland
   pip install -r requirements.txt
   ```

   For GPU training of the graph models, install the PyTorch build matching your CUDA
   version from <https://pytorch.org> before `pip install -r requirements.txt`.
   The graph models use **pure PyTorch** (no `torch-geometric` dependency).

2. **Verify the environment:**

   ```bash
   python check_env.py
   ```

3. **Configure paths.** All input and output locations are centralised in
   [`configs/paths.py`](configs/paths.py). Edit the four root paths at the top to
   match your machine:

   ```python
   DATA_ROOT  = Path(r"...")   # folder holding the land-cover products
   OTHER_ROOT = Path(r"...")   # folder holding the driver datasets
   AOI_SHP    = Path(r"...")   # Inner Mongolia boundary shapefile
   OUT_ROOT   = Path(r"...")   # where all processed outputs are written
   ```

   The per-product file templates, the 14-class legend, the reliability weights,
   the target grid (1 km Albers), and the study period are all defined in
   `configs/paths.py` and `configs/class_harmonization.py`.

---

## Input data

The study uses **eight land-cover products** and a stack of **driver datasets**. These
are third-party datasets and are **not** redistributed here; download them from their
original providers and point `configs/paths.py` at them. The exact products, native
resolutions, bands, and the unified-legend cross-walk are listed in
`configs/class_harmonization.py` (and in the paper's supplementary Table S6).

Land-cover products: Mongolian-Plateau LULC (the fine-grained reference), GLC_FCS30D,
CLCD, ESA CCI / C3S, MODIS MCD12Q1, ESA WorldCover, a Mongolian-Plateau sandy-land/desert
specialist, and an Inner Mongolia annual grassland mask.

Driver datasets: MODIS NDVI (MOD13A2), SPEI drought (CSIC SPEIbase), ETCCDI climate
extremes, vegetation phenology, grassland grazing intensity and Gridded Livestock of the
World, WorldPop population density, cropping intensity, SRTM terrain, and GRIP roads.

All layers are reprojected and resampled to a common **1 km Albers Equal-Area (China)**
grid clipped to the Inner Mongolia boundary (1,146,509 valid cells).

---

## Reproducing the pipeline

Each step reads the outputs of the previous one. Run them in order (either via
`run_all.py`, or individually as modules from the repository root):

```bash
# 0. diagnostics (optional but recommended on first run)
python -m src.step00_diagnose

# 1. harmonise the eight products to the unified 14-class legend on the 1 km grid
python -m src.step01_harmonize_lulc

# 2. hierarchical two-tier Bayesian fusion (+ per-pixel uncertainty)
python -m src.step02_fuse_lulc
python -m src.step02b_fusion_sensitivity     # robustness check
python -m src.step02c_validation generate --n 400   # then interpret points, then:
python -m src.step02c_validation assess

# 3. align all driver datasets to the grid
python -m src.step03_align_drivers

# 4. build ML-ready feature tables
python -m src.step04_features

# 5. attribution: XGBoost transition model + SHAP, with baselines
python -m src.step05_attribute_multinomial
python -m src.step05b_baselines

# 6. causal screening: Granger + CCM, with FDR + surrogate significance
python -m src.step06_causal
python -m src.step06b_causal_rigor

# 7. spatial-spillover test: spatio-temporal GNN + clean neighbour-feature comparison
python -m src.step07_stgnn
python -m src.step07b_spillover_clean

# figures, tables, transition matrix
python -m src.compute_change_rates
python -m src.make_transition_matrix
python -m src.make_tables
python -m src.render_fig1
python -m src.render_fig2
python -m src.make_fig3_variants
python -m src.render_fig4_shap
python -m src.render_causal_figs
python -m src.render_step07_figs
python -m src.render_workflow
```

Outputs are written to the sub-directories of `OUT_ROOT` defined in `configs/paths.py`
(`01_harmonized_lulc/`, `02_fused_lulc/`, `03_drivers/`, `04_features/`, attribution,
causal, GNN, figures, and tables).

---

## Method summary

- **Two-tier Bayesian fusion** (`step02_fuse_lulc.py`). Each product votes at the thematic
  level it can support: a super-class stage over nine classes all products resolve, then a
  sub-class stage (steppe densities; bare/desert/sand) using only the products competent to
  make each distinction. A symmetric per-product error model derived from the reliability
  weights gives a posterior per pixel; the most-likely class, joint confidence, decision
  entropy, and product agreement are all retained.

- **Dual attribution** (`step05_*`, `step06_*`). Gradient boosting with SHAP identifies the
  associative drivers of five-year transitions (judged against persistence and Markov
  baselines). Granger causality (FDR-corrected) and convergent cross mapping
  (phase-randomised surrogate significance) are then required to *agree* before a
  relationship is treated as causal.

- **Spatial-spillover test** (`step07_*`). A node GRU using only a location's own temporal
  trajectory is compared, under a fixed architecture, to the same model augmented with its
  neighbours' signal. No skill gain from the neighbour information indicates no spillover at
  the tested scale.

---

## Citation

If you use this code, please cite the paper (citation to be added on publication) and this
repository.

## License

Released under the MIT License (see [LICENSE](LICENSE)).

## Contact

Questions and issues: please open a GitHub issue, or contact the corresponding author.
