"""
================================================================================
PATHS CONFIGURATION — Inner Mongolia Paper 2
================================================================================
Single source of truth for every input/output path. Edit ONLY this file when
your drive layout changes.
================================================================================
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# ROOT PATHS  — edit only if you move data
# ----------------------------------------------------------------------------
DATA_ROOT = Path(r"D:\all lulc data")
OTHER_ROOT = Path(r"D:\other datasets")
AOI_SHP = Path(r"D:\shape files\Inner mongolai shape\neimeng shape\neimeng_reprojected_EPSG32649.shp")

# All preprocessing outputs go here. Will be created on demand.
OUT_ROOT = Path(r"D:\paper2_outputs")

# ----------------------------------------------------------------------------
# TARGET REFERENCE GRID
# ----------------------------------------------------------------------------
# Every harmonized layer is reprojected and resampled to this exact grid.
# 1km Albers (China) chosen to:
#   - keep equal-area properties (correct for area statistics)
#   - match the native CRS of CLCD/Wang_MP/GLC_FCS30D (no extra reproject for them)
#   - 1km matches WorldPop, extreme climate (5km is upsampled), grazing intensity
TARGET_CRS = "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
TARGET_RES_M = 1000  # 1 km
# Reasonable bounds for Inner Mongolia in target CRS (slightly padded)
# Will be refined by clipping with the AOI shapefile.
TARGET_BOUNDS = (-650_000, 4_000_000, 1_510_000, 5_900_000)

# ----------------------------------------------------------------------------
# OUTPUT SUB-DIRECTORIES
# ----------------------------------------------------------------------------
OUT_HARMONIZED = OUT_ROOT / "01_harmonized_lulc"  # each product, each year, unified legend, 1km AOI-masked
OUT_FUSED      = OUT_ROOT / "02_fused_lulc"        # Bayesian fusion outputs (most-likely class + uncertainty)
OUT_DRIVERS    = OUT_ROOT / "03_drivers"           # aligned driver stack (climate, grazing, etc.)
OUT_FEATURES   = OUT_ROOT / "04_features"          # ML-ready tabular pixel/cell datasets (parquet)
OUT_MODELS     = OUT_ROOT / "05_models"            # trained XGBoost + GNN checkpoints
OUT_FIGURES    = OUT_ROOT / "06_figures"           # final figures
OUT_TABLES     = OUT_ROOT / "07_tables"            # final tables
OUT_LOGS       = OUT_ROOT / "logs"

ALL_OUT_DIRS = [
    OUT_HARMONIZED, OUT_FUSED, OUT_DRIVERS,
    OUT_FEATURES, OUT_MODELS, OUT_FIGURES, OUT_TABLES, OUT_LOGS
]

# ----------------------------------------------------------------------------
# DATASET PATHS  — explicit listing of EVERY input the pipeline knows about
# ----------------------------------------------------------------------------
LULC_PRODUCTS = {
    "Wang2024_MP": {
        "folder": DATA_ROOT / "updated long-term monoglia" / "MON_LULC",
        "year_template": "{year}.tif",
        "years": [1990, 1995, 2000, 2005, 2010, 2015, 2020],
    },
    "GLC_FCS30D": {
        # Two multi-band files:
        #   File 1 (19851995): 3 bands = 1985, 1990, 1995
        #   File 2 (20002022): 23 bands = 2000, 2001, ..., 2022
        "files": [
            DATA_ROOT / "GLC" / "maps" / "preprcessed" / "GLC_alber_1985_2000.tif",
            DATA_ROOT / "GLC" / "maps" / "preprcessed" / "GLC_alber_2000-2022.tif",
        ],
        "year_to_band": "auto",  # determined at runtime
    },
    "CLCD": {
        "folder": DATA_ROOT / "CLCD" / "Inner Mongolia",
        "year_template": "CLCD_v01_{year}_albert_neimeng.tif",
        "years": list(range(1985, 2024)),
    },
    "CCI": {
        # NATIVE CCI files (22-class LCCS legend, Albers China, 30 m, int16 nodata=32767)
        # Two filename patterns coexist in the folder for different years:
        #   v2.0.7cds (older years): ESACCI-LC-L4-LCCS-Map-300m-P1Y-<year>-v2.0.7cds.area-subset.54.127.37.97.tif
        #   v2.1.1    (newer years): C3S-LC-L4-LCCS-Map-300m-P1Y-<year>-v2.1.1.area-subset.54.127.37.97.tif
        "folder": DATA_ROOT / "CCI" / "new" / "TIFF",
        "year_templates": [
            "ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7cds.area-subset.54.127.37.97.tif",
            "C3S-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.1.1.area-subset.54.127.37.97.tif",
        ],
        "years": list(range(1992, 2023)),
        "src_nodata": 32767,
    },
    "MODIS": {
        # NATIVE MODIS MCD12Q1 v6.1 files, exported from GEE.
        # Verified by inspect_modis_bands.py:
        #   - 24 separate annual files: MODIS_LandCover_IGBP_2001.tif ... 2024.tif
        #   - Each file has 13 BANDS (GEE exports the whole image, not just
        #     the selected sub-band):
        #       Band 1: LC_Type1 (IGBP 17-class)        <-- WE USE THIS
        #       Band 2: LC_Type2 (UMD)
        #       Band 3: LC_Type3 (LAI)
        #       Band 4: LC_Type4 (BGC)
        #       Band 5: LC_Type5 (PFT)
        #       Bands 6-8: LC_Prop1/2/3_Assessment (confidence 0-99)
        #       Bands 9-11: LC_Prop1/2/3 (FAO-LCCS variants)
        #       Band 12: QC, Band 13: LW (land/water mask)
        # FUTURE IDEA: bands 6-8 give per-pixel confidence — could be used as
        # spatially-varying weights in the Bayesian fusion. Not used now to
        # keep methodology simple, but worth a supplementary experiment.
        "folder": DATA_ROOT / "modis" / "original modis",
        "year_template": "MODIS_LandCover_IGBP_{year}.tif",
        "years": list(range(2001, 2025)),
        "band": 1,           # always LC_Type1 (IGBP)
        "src_nodata": 255,
    },
    "WorldCover": {
        # NATIVE ESA WorldCover files exported from GEE.
        # Note: v100 file is for 2020, v200 (or sometimes v100) for 2021.
        "folder": DATA_ROOT / "wordcover 20,21" / "original",
        "year_templates": [
            "ESA_WorldCover_30m_{year}_v100.tif",
            "ESA_WorldCover_30m_{year}_v200.tif",
        ],
        "years": [2020, 2021],
        "src_nodata": 0,
    },
    "MPSDSL": {
        "folder": DATA_ROOT / "MP_sandy_land and sandy_desesrt" / "MPSDSL" / "MPSDSL",
        "year_template": "{year}.tif",
        "years": [1990, 1995, 2000, 2005, 2010, 2015, 2020],
    },
    "Grassland_IM": {
        "folder": DATA_ROOT / "A 30-m annual grassland dataset from 1991 to 2020 for Inner Mongolia, China" / "data",
        "year_template": "{year}.tif",
        "years": list(range(1991, 2021)),
    },
}

# ----------------------------------------------------------------------------
# DRIVER PATHS
# ----------------------------------------------------------------------------
DRIVERS = {
    "climate_indices_root": OTHER_ROOT / "0.05° Grid Dataset of Extreme Climate Index in Inner Mongolia of China (1982-2020)" / "ExtClimIndicesInnerMongolia1982-2020",
    "climate_subdirs": ["CDD", "CWD", "GSL", "R10", "R20", "Rx1day", "Rx5day",
                        "Tn10p", "Tn90p", "TNn", "TNx", "Tx10p", "Tx90p", "TXn", "TXx"],
    "grazing_root": OTHER_ROOT / "A_long-term_high-resolution_dataset_of_grasslands_grazing_intensity_in_China" / "26195684",
    "grazing_total_template": "LHGI_{year}.tif",
    "grazing_subspecies": {
        "BigL":   "BigL/bigL_{year}.tif",
        "Cattle": "Cattle/cattle_{year}.tif",
        "Goat":   "goat/goat_{year}.tif",
        "Sheep":  "Sheep/sheep_{year}.tif",
    },
    "phenology_sos_folder": OTHER_ROOT / "A dataset of vegetation phenology of Inner Mongolia (2001-2020)" / "IMSOS",
    "phenology_eos_folder": OTHER_ROOT / "A dataset of vegetation phenology of Inner Mongolia (2001-2020)" / "IMEOS",
    "phenology_los_folder": OTHER_ROOT / "A dataset of vegetation phenology of Inner Mongolia (2001-2020)" / "IMLOS",
    "phenology_sos_template": "{year}SOS.tif",
    "phenology_eos_template": "{year}EOS.tif",
    "phenology_los_template": "{year}LOS.tif",
    "worldpop_folder": OTHER_ROOT / "WorldPop Data(Population Density)",
    "worldpop_template": "chn_pd_{year}_1km_UNadj.tif",
    "worldpop_years": list(range(2000, 2021)),
    "cropping_intensity_folder": OTHER_ROOT / "Annual_global_cropping_intensity_from_2001_to_2019" / "14099402" / "GCI&QC",
    "cropping_intensity_template": "GCI_{year}.tif",
    "desertification_root": OTHER_ROOT / "A dataset of desertification degree monitoring in Inner Mongolia from 2001 to 2021" / "内蒙荒漠化程度年度数据",
    "sandstorm_root": OTHER_ROOT / "Dataset of spring sandstorm distribution in the Mongolian Plateau from 2000 to 2021",
    "ndvi_factors_root": OTHER_ROOT / "Dataset of NDVI Change Trends and Impact Factors in Inner Mongolia (2000-2015)" / "NDVITrends&FactorsInnerMongolia",
    "grip_roads": OTHER_ROOT / "GRIP global roads database" / "GRIP4_Region6_vector_shp" / "GRIP4_region6.shp",
    "glw_2010": OTHER_ROOT / "GLW (Gridded Livestock of the World)" / "GLW 3(2010)" / "all livestock" / "DATA_GLW3_MAPSET_D-DA_GLW3.D-DA.GLEAM2-ALL-LU.tif",
    "glw_2020": OTHER_ROOT / "GLW (Gridded Livestock of the World)" / "GLW 4(2020)" / "all livestock" / "GLW4-2020.D-DA.GLEAM3-ALL-LU.tif",

    # -----------------------------------------------------------------------
    # NEWLY ADDED DRIVERS (verified by inspect_new_drivers.py)
    # -----------------------------------------------------------------------
    # MODIS NDVI annual statistics (MOD13A2 v6.1) — 24 annual GeoTIFFs, each
    # with 3 bands: ANNUAL_MEAN, GS_MEAN, GS_MAX. Native int16, scale 0.0001.
    "ndvi_folder": DATA_ROOT / "modis" / "NDVI_annual",
    "ndvi_template": "MODIS_NDVI_IM_{year}.tif",
    "ndvi_years": list(range(2001, 2025)),
    "ndvi_band_names": ["annual_mean", "gs_mean", "gs_max"],
    "ndvi_scale": 0.0001,         # native int16 stores NDVI * 10000

    # SRTM 90 m terrain — one static 3-band file (elevation, slope, aspect).
    # GEE export uses +3.4e38 as nodata (float32 max), confirmed by inspector.
    "srtm_file": OTHER_ROOT / "SRTM" / "SRTM_DEM_slope_aspect_IM.tif",
    "srtm_band_names": ["elevation", "slope", "aspect"],
    "srtm_src_nodata": 3.4e38,    # GEE float32 export sentinel (POSITIVE)

    # SPEI from CSIC v2.10 — global 0.5° monthly NetCDF, 1901-2024.
    # Three timescales capture different drought windows:
    #   spei03: 3-month — grassland's typical response window
    #   spei06: 6-month — growing-season cumulative water deficit
    #   spei12: 12-month — annual / long-term drought, best NDVI correlation
    "spei_folder": OTHER_ROOT / "SPEI",
    "spei_timescales": ["spei03", "spei06", "spei12"],
    "spei_template": "{ts}.nc",   # e.g. "spei03.nc"
    "spei_var": "spei",           # variable name inside the NetCDF
}

# ----------------------------------------------------------------------------
# STUDY PERIOD
# ----------------------------------------------------------------------------
# Years where MOST drivers + LC products are available (intersection).
# Climate indices: 1982-2020. Phenology: 2001-2020. WorldPop: 2000-2020.
# Wang_MP: 5-year snapshots. We use the 5-year snapshots as anchor years
# and interpolate annual drivers between them.
ANCHOR_YEARS = [2000, 2005, 2010, 2015, 2020]
FULL_PERIOD = list(range(1990, 2024))


def ensure_dirs():
    """Create all output directories."""
    for d in ALL_OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("AOI:", AOI_SHP)
    print("Target CRS:", TARGET_CRS)
    print("Target res:", TARGET_RES_M, "m")
    print("LULC products:", list(LULC_PRODUCTS.keys()))
    print("Anchor years:", ANCHOR_YEARS)
    print("\nOutput dirs that will be created:")
    for d in ALL_OUT_DIRS:
        print(" ", d)
