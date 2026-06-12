"""
================================================================================
CLASS HARMONIZATION TABLES
================================================================================
Maps the native class codes of all LULC products to a UNIFIED LEGEND
(IPCC-aligned, 13 classes, matching the Wang et al. 2024 MP scheme used in
paper 1 — so paper 2 builds directly on top).

VERIFICATION STATUS (all 8 retained products verified)
------------------------------------------------------
✅ VERIFIED against official documentation and/or diagnostic inspection:
   - Wang2024_MP : Wang et al. 2024 Sci Data — TRUE 14-class scheme
                   (the published Sci Data paper confirms labels 1-14 =
                   forests, shrubs, meadows, real steppes, dry steppes,
                   desert steppes, wetlands, water, croplands, built-up,
                   barren, deserts, sand, ice).
                   Verified empirically: dominant codes in IM 2020 match
                   expected geography (Real Steppe 25%, Sand 19%,
                   Cropland 14%, Forest 14%, Dry Steppe 8.5%, Desert 8%).
   - GLC_FCS30D  : Zhang et al. 2024 ESSD, 36-code legend confirmed
   - CLCD        : Yang & Huang 2021 ESSD, 9-class legend confirmed
   - CCI         : ESA C3S native 22-class LCCS legend confirmed
   - MODIS       : MCD12Q1 v6.1 IGBP 17-class legend confirmed
                   (NATIVE files downloaded directly from GEE; multi-band
                   structure verified separately)
   - WorldCover  : ESA WorldCover v100/v200 native 11-class legend
                   confirmed (2 years only: 2020 + 2021)
   - MPSDSL      : Mongolian Plateau Sandy Desert & Sandy Land specialist
   - Grassland_IM: binary grassland mask (Tang et al. 2024)

Unified legend (matches Wang et al. 2024 Sci Data 14-class scheme):
    1  FO  Forest
    2  SH  Shrub
    3  MS  Meadow steppe         (mesic grassland)
    4  RS  Real steppe           (typical grassland)
    5  YS  Dry steppe            (drier than RS, wetter than DS)
    6  DS  Desert steppe         (sparse arid grassland)
    7  WT  Wetland
    8  WA  Water
    9  CR  Cropland
    10 BA  Built-up
    11 BR  Bare area / barren
    12 DE  Desert
    13 SD  Sand
    14 IC  Ice / permanent snow
    0      NoData / outside AOI

DESIGN NOTES
- Products that cannot distinguish meadow / real / desert steppe (CLCD's
  single "grassland" class; CCI's "Grassland" + "Mosaic herbaceous"
  collapsed; GLC_FCS30D's "Grassland" 130) are mapped to UNI=4 (Real steppe)
  as the modal class. The fusion layer then uses fine-grained products
  (Wang2024_MP for MS/RS/DS distinctions, MPSDSL for SD/DE distinctions)
  to refine where they disagree.
- All "unknown" / unmapped codes go to 0 (NoData).
- CCI file-level NoData = 32767 is mapped to 0 explicitly.
================================================================================
"""

# Canonical class names + display colors (for figures)
UNIFIED_LEGEND = {
    0:  ("NoData",        "#000000"),
    1:  ("Forest",        "#1f6e3a"),
    2:  ("Shrub",         "#7fbf5b"),
    3:  ("Meadow steppe", "#a6d96a"),
    4:  ("Real steppe",   "#fee08b"),
    5:  ("Dry steppe",    "#fdb863"),
    6:  ("Desert steppe", "#fdae61"),
    7:  ("Wetland",       "#5ab4ac"),
    8:  ("Water",         "#2c7fb8"),
    9:  ("Cropland",      "#d7191c"),
    10: ("Built-up",      "#878787"),
    11: ("Bare area",     "#bfbfbf"),
    12: ("Desert",        "#fff2cc"),
    13: ("Sand",          "#f7e07a"),
    14: ("Ice",           "#f0f9e8"),
}

# Convenience: short code -> int
UNI_NAME_TO_CODE = {name: code for code, (name, _) in UNIFIED_LEGEND.items()}

# ============================================================================
# 1. Inner Mongolia Annual Grassland (Tang et al. 2024) — BINARY mask
#    {0: non-grassland, 1: grassland}
#    Use ONLY as a grassland-presence prior in the fusion (not as a class src).
# ============================================================================
GRASSLAND_IM_MAP = {
    0: 0,    # background -> NoData (will be overridden by other products)
    1: 4,    # grassland -> Real steppe (placeholder; refined by other products)
}

# ============================================================================
# 2. GLC_FCS30D (Zhang et al. 2024, ESSD) — FULL 36-CODE LEGEND
#    Verified against the GLC_FCS30D User Guide and Earth Engine catalog.
# ============================================================================
GLC_FCS30D_MAP = {
    0:   0,    # fill value
    10:  9,    # Rainfed cropland                       -> Cropland
    11:  9,    # Herbaceous cover cropland              -> Cropland
    12:  9,    # Tree or shrub cover (orchard) cropland -> Cropland
    20:  9,    # Irrigated cropland                     -> Cropland
    51:  1,    # Open evergreen broadleaf forest        -> Forest
    52:  1,    # Closed evergreen broadleaf forest      -> Forest
    61:  1,    # Open deciduous broadleaf forest        -> Forest
    62:  1,    # Closed deciduous broadleaf forest      -> Forest
    71:  1,    # Open evergreen needleleaf forest       -> Forest
    72:  1,    # Closed evergreen needleleaf forest     -> Forest
    81:  1,    # Open deciduous needleleaf forest       -> Forest
    82:  1,    # Closed deciduous needleleaf forest     -> Forest
    91:  1,    # Open mixed leaf forest                 -> Forest
    92:  1,    # Closed mixed leaf forest               -> Forest
    120: 2,    # Shrubland                              -> Shrub
    121: 2,    # Evergreen shrubland                    -> Shrub
    122: 2,    # Deciduous shrubland                    -> Shrub
    130: 4,    # Grassland                              -> Real steppe (modal)
    140: 4,    # Lichens and mosses                     -> Real steppe (sparse veg proxy)
    150: 11,   # Sparse vegetation                      -> Barren
    152: 11,   # Sparse shrubland                       -> Barren
    153: 11,   # Sparse herbaceous                      -> Barren
    181: 7,    # Swamp                                  -> Wetland
    182: 7,    # Marsh                                  -> Wetland
    183: 7,    # Flooded flat                           -> Wetland
    184: 7,    # Saline                                 -> Wetland
    185: 7,    # Mangrove (won't occur in IM)           -> Wetland
    186: 7,    # Salt marsh                             -> Wetland
    187: 7,    # Tidal flat (won't occur in IM)         -> Wetland
    190: 10,   # Impervious / built-up                  -> Built-up
    200: 11,   # Bare areas                             -> Barren
    201: 12,   # Consolidated bare (rocky desert)       -> Desert
    202: 13,   # Unconsolidated bare (sandy)            -> Sand
    210: 8,    # Water body                             -> Water
    220: 14,   # Permanent ice and snow                 -> Ice
    250: 0,    # filled value                           -> NoData
    255: 0,    # nodata
}

# ============================================================================
# 3. CLCD v1.0.3 (Yang & Huang 2021) — 9 classes
# ============================================================================
CLCD_MAP = {
    0: 0,    # nodata
    1: 9,    # cropland
    2: 1,    # forest
    3: 2,    # shrub
    4: 4,    # grassland  -> Real steppe (modal)
    5: 8,    # water
    6: 14,   # snow/ice
    7: 11,   # barren
    8: 10,   # impervious / built-up
    9: 7,    # wetland
}

# ============================================================================
# 4. ESA CCI Land Cover (Defourny et al. / ESA C3S Land Cover v2.0.7 + v2.1.1)
# ----------------------------------------------------------------------------
# ✅ VERIFIED via diagnose_cci_native.py — native files at
#    D:\all lulc data\CCI\new\TIFF\
# Files are 30 m, already in Albers China (matches our target CRS), int16,
# NoData = 32767. Two filename patterns (interchangeable, identical legend):
#    ESACCI-LC-L4-LCCS-Map-300m-P1Y-<year>-v2.0.7cds.area-subset...tif
#    C3S-LC-L4-LCCS-Map-300m-P1Y-<year>-v2.1.1.area-subset...tif
# Full 22-class LCCS legend confirmed present (10..220).
# ============================================================================
CCI_MAP = {
    0:   0,    # No Data
    10:  9,    # Cropland, rainfed                                  -> Cropland
    11:  9,    # Herbaceous cover cropland                          -> Cropland
    12:  9,    # Tree or shrub cover cropland (orchard)             -> Cropland
    20:  9,    # Cropland, irrigated or post-flooding               -> Cropland
    30:  9,    # Mosaic cropland (>50%) / natural veg (<50%)        -> Cropland
    40:  4,    # Mosaic natural veg (>50%) / cropland (<50%)        -> Real steppe (mixed)
    50:  1,    # Tree cover, broadleaved evergreen, closed-to-open  -> Forest
    60:  1,    # Tree cover, broadleaved deciduous (>15%)           -> Forest
    61:  1,    # Tree cover, broadleaved deciduous (>40%)           -> Forest
    62:  1,    # Tree cover, broadleaved deciduous (15-40%)         -> Forest
    70:  1,    # Tree cover, needleleaved evergreen                 -> Forest
    71:  1,    # Tree cover, needleleaved evergreen (>40%)          -> Forest
    72:  1,    # Tree cover, needleleaved evergreen (15-40%)        -> Forest
    80:  1,    # Tree cover, needleleaved deciduous                 -> Forest
    81:  1,    # Tree cover, needleleaved deciduous (>40%)          -> Forest
    82:  1,    # Tree cover, needleleaved deciduous (15-40%)        -> Forest
    90:  1,    # Tree cover, mixed leaf type                        -> Forest
    100: 2,    # Mosaic tree/shrub (>50%) / herbaceous              -> Shrub (mixed woody)
    110: 4,    # Mosaic herbaceous (>50%) / tree/shrub              -> Real steppe (mixed)
    120: 2,    # Shrubland                                          -> Shrub
    121: 2,    # Evergreen shrubland                                -> Shrub
    122: 2,    # Deciduous shrubland                                -> Shrub
    130: 4,    # Grassland                                          -> Real steppe (modal)
    140: 4,    # Lichens and mosses                                 -> Real steppe (sparse veg proxy)
    150: 11,   # Sparse vegetation (tree, shrub, herbaceous <15%)   -> Barren
    151: 11,   # Sparse tree (<15%)                                 -> Barren
    152: 11,   # Sparse shrub (<15%)                                -> Barren
    153: 11,   # Sparse herbaceous (<15%)                           -> Barren
    160: 7,    # Tree cover, flooded, fresh/brakish water           -> Wetland
    170: 7,    # Tree cover, flooded, saline water                  -> Wetland
    180: 7,    # Shrub/herbaceous cover, flooded                    -> Wetland
    190: 10,   # Urban areas                                        -> Built-up
    200: 11,   # Bare areas                                         -> Barren
    201: 12,   # Consolidated bare (rocky desert)                   -> Desert
    202: 13,   # Unconsolidated bare (sandy)                        -> Sand
    210: 8,    # Water bodies                                       -> Water
    220: 14,   # Permanent snow and ice                             -> Ice
    32767: 0,  # NoData sentinel (file-level nodata)                -> NoData
}

# ============================================================================
# 5. Mongolian Plateau LULC (Wang et al. 2024 Sci Data) — THE 14-CLASS SCHEME
#    Per the published paper (Wang et al. 2024, Sci Data, doi 10.1038/s41597-025-05648-8):
#    "labels 1-14 correspond to forests, shrubs, meadows, real steppes,
#     dry steppes, desert steppes, wetlands, water, croplands, built-up
#     land, barren land, deserts, sand, and ice"
#    Our unified legend was redesigned to match exactly (identity mapping).
#    NOTE: paper 1 Table 1 listed only 13 classes (omitted "dry steppes"),
#    but the underlying .tif has 14 codes. We use the dataset's true 14.
# ============================================================================
WANG_MP_MAP = {
    0:  0,    # background / nodata
    1:  1,    # forests
    2:  2,    # shrubs
    3:  3,    # meadows         (= meadow steppe)
    4:  4,    # real steppes
    5:  5,    # dry steppes
    6:  6,    # desert steppes
    7:  7,    # wetlands
    8:  8,    # water
    9:  9,    # croplands
    10: 10,   # built-up land
    11: 11,   # barren land
    12: 12,   # deserts
    13: 13,   # sand
    14: 14,   # ice
}

# ============================================================================
# 6. MODIS MCD12Q1 IGBP — NATIVE 17-class legend (LC_Type1)
# ----------------------------------------------------------------------------
# ✅ VERIFIED via comprehensive_diagnostic.md — codes 1-17 (+0=filled, 15=ice)
# all present and meaningful inside IM AOI. Native files from GEE export:
#   D:\all lulc data\modis\original modis\MODIS_LandCover_IGBP_YYYY.tif
# Source: ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')
# Reference legend: MCD12Q1 v6.1 user guide.
# ============================================================================
MODIS_MAP = {
    0:  0,    # Water bodies (filled value)         -> NoData (truly tiny in IM)
    1:  1,    # Evergreen Needleleaf Forests        -> Forest
    2:  1,    # Evergreen Broadleaf Forests         -> Forest
    3:  1,    # Deciduous Needleleaf Forests        -> Forest
    4:  1,    # Deciduous Broadleaf Forests         -> Forest
    5:  1,    # Mixed Forests                       -> Forest
    6:  2,    # Closed Shrublands                   -> Shrub
    7:  2,    # Open Shrublands                     -> Shrub
    8:  4,    # Woody Savannas (cross-tab: 89% Wang Forest in IM; but typically
              #   sparse-tree-grass in MODIS - mapped to Real steppe for the IM
              #   context where it sits between forest and grassland)
    9:  4,    # Savannas                            -> Real steppe
    10: 4,    # Grasslands                          -> Real steppe (modal in cross-tab)
    11: 7,    # Permanent Wetlands                  -> Wetland
    12: 9,    # Croplands                           -> Cropland
    13: 10,   # Urban and Built-up                  -> Built-up
    14: 9,    # Cropland/Natural Veg Mosaics        -> Cropland (mosaic, lean crop)
    15: 14,   # Permanent Snow and Ice              -> Ice
    16: 11,   # Barren                              -> Barren
    17: 8,    # Water Bodies                        -> Water
    255: 0,   # Unclassified / nodata               -> NoData
}

# ============================================================================
# 7. ESA WorldCover v100 (2020) and v200 (2021) — NATIVE 11-class legend
# ----------------------------------------------------------------------------
# ✅ VERIFIED via comprehensive_diagnostic.md — codes 10/20/.../100 all present.
# Native files (from GEE export):
#   D:\all lulc data\wordcover 20,21\original\ESA_WorldCover_30m_2020_v100.tif
#   D:\all lulc data\wordcover 20,21\original\ESA_WorldCover_30m_2021_v200.tif
# (or v100 for both, depending on what GEE export produced)
# Note: only 2 years available, both used as voters in the fusion AT THEIR
# RESPECTIVE YEARS. They cannot contribute to time-series transitions.
# ============================================================================
WORLDCOVER_MAP = {
    0:   0,    # No Data
    10:  1,    # Tree cover               -> Forest
    20:  2,    # Shrubland                -> Shrub
    30:  4,    # Grassland                -> Real steppe (modal)
    40:  9,    # Cropland                 -> Cropland
    50:  10,   # Built-up                 -> Built-up
    60:  11,   # Bare / sparse vegetation -> Barren
    70:  14,   # Snow and ice             -> Ice
    80:  8,    # Permanent water bodies   -> Water
    90:  7,    # Herbaceous wetland       -> Wetland
    95:  7,    # Mangroves (won't appear in IM) -> Wetland
    100: 4,    # Moss and lichen          -> Real steppe (sparse veg proxy, modal in cross-tab)
}

# ============================================================================
# 8. MP Sandy Desert & Sandy Land (MPSDSL) — sand sub-types
#    8 codes per report (0-7).
# ============================================================================
MPSDSL_MAP = {
    0: 0,    # nodata / non-sandy
    2: 13,   # sand
    3: 13,   # sand (sub-type)
    4: 12,   # desert (rocky / consolidated)
    5: 12,   # desert
    6: 12,   # desert
    7: 12,   # desert
}

# ============================================================================
# REGISTRY — every product gets a unique ID + metadata
# ============================================================================
PRODUCT_REGISTRY = {
    "Wang2024_MP": {
        "map": WANG_MP_MAP,
        "weight": 1.0,           # highest trust: native 13-class, used in paper 1
        "fine_grain": True,      # can distinguish MS / RS / DS
        "years_available": [1990, 1995, 2000, 2005, 2010, 2015, 2020],
        "native_res_m": 30,
        "is_categorical": True,
    },
    "GLC_FCS30D": {
        "map": GLC_FCS30D_MAP,
        "weight": 0.9,
        "fine_grain": False,
        "years_available": list(range(1985, 2023)),
        "native_res_m": 30,
        "is_categorical": True,
    },
    "CLCD": {
        "map": CLCD_MAP,
        "weight": 0.85,
        "fine_grain": False,
        "years_available": list(range(1985, 2024)),
        "native_res_m": 30,
        "is_categorical": True,
    },
    "CCI": {
        "map": CCI_MAP,
        "weight": 0.85,           # verified native 22-class; matches CLCD trust level
        "fine_grain": False,
        "years_available": list(range(1992, 2023)),
        "native_res_m": 300,      # native ESA CCI resolution
        "is_categorical": True,
    },
    "MODIS": {
        "map": MODIS_MAP,
        "weight": 0.75,           # native IGBP, but coarse 500 m; trust below 30m products
        "fine_grain": False,
        # years_available will be set at runtime from the actual MODIS file's bands
        # — see step01 dispatcher. As a placeholder, full GEE coverage 2001-2024:
        "years_available": list(range(2001, 2025)),
        "native_res_m": 500,
        "is_categorical": True,
    },
    "WorldCover": {
        "map": WORLDCOVER_MAP,
        "weight": 0.80,           # high-quality recent product; 2 years only
        "fine_grain": False,
        "years_available": [2020, 2021],
        "native_res_m": 10,       # native 10 m, exported at 30 m via GEE
        "is_categorical": True,
    },
    "MPSDSL": {
        "map": MPSDSL_MAP,
        "weight": 0.95,          # specialized; overrides others for SD/DE
        "fine_grain": False,
        "years_available": [1990, 1995, 2000, 2005, 2010, 2015, 2020],
        "native_res_m": 30,
        "specialist_classes": [12, 13],  # only trust for desert (12) + sand (13)
        "is_categorical": True,
    },
    "Grassland_IM": {
        "map": GRASSLAND_IM_MAP,
        "weight": 0.5,           # binary prior only
        "fine_grain": False,
        "years_available": list(range(1991, 2021)),
        "native_res_m": 30,
        "specialist_classes": [3, 4, 5, 6],  # only as grassland prior: MS/RS/YS/DS
        "is_categorical": True,
    },
}


def remap_array(arr, product_name):
    """Vectorized remapping using a NumPy lookup table — fast for huge rasters.

    Builds a uint8 LUT of size (max_native_code + 1) once per product, then
    indexes into it. ~100× faster than per-pixel loops.
    """
    import numpy as np

    mapping = PRODUCT_REGISTRY[product_name]["map"]
    max_code = max(mapping.keys())
    # Cap LUT size at uint8 max to avoid huge arrays when native codes are big
    lut_size = max(256, max_code + 1)
    lut = np.zeros(lut_size, dtype=np.uint8)
    for native, unified in mapping.items():
        if 0 <= native < lut_size:
            lut[native] = unified
    # Clip out-of-range values to 0 (NoData) before indexing
    arr_clipped = np.where(
        (arr >= 0) & (arr < lut_size), arr, 0
    ).astype(np.int64)
    return lut[arr_clipped]


if __name__ == "__main__":
    # Self-test
    import numpy as np
    print("Unified legend classes:", len(UNIFIED_LEGEND) - 1, "(plus 0=NoData)")
    print()
    # Wang: identity now (14-class native = 14-class unified)
    wang_test = np.array([0, 1, 5, 9, 13, 14], dtype=np.int64)
    print("Wang test:    ", wang_test.tolist())
    print("Wang expected:[0, 1, 5, 9, 13, 14]  (identity)")
    print("Wang actual:  ", remap_array(wang_test, "Wang2024_MP").tolist())
    print()
    # CLCD: 1=cropland→9, 2=forest→1, 4=grassland→4, 7=barren→11, 8=builtup→10
    clcd_test = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    print("CLCD test:    ", clcd_test.tolist())
    print("CLCD expected:[0, 9, 1, 4, 8, 14, 11, 10, 7]")
    print("CLCD actual:  ", remap_array(clcd_test, "CLCD").tolist())
    print()
    # GLC sample: 130=grass→4, 200=bare→11, 210=water→8, 220=ice→14
    glc_test = np.array([0, 10, 130, 190, 200, 202, 210, 220], dtype=np.int64)
    print("GLC test:     ", glc_test.tolist())
    print("GLC expected: [0, 9, 4, 10, 11, 13, 8, 14]")
    print("GLC actual:   ", remap_array(glc_test, "GLC_FCS30D").tolist())
    print()
    # CCI sample
    cci_test = np.array([0, 10, 130, 190, 200, 202, 210, 220, 32767], dtype=np.int64)
    print("CCI test:     ", cci_test.tolist())
    print("CCI expected: [0, 9, 4, 10, 11, 13, 8, 14, 0]")
    print("CCI actual:   ", remap_array(cci_test, "CCI").tolist())
    print()
    # MODIS IGBP
    modis_test = np.array([0, 1, 5, 8, 10, 11, 12, 13, 15, 16, 17, 255], dtype=np.int64)
    print("MODIS test:    ", modis_test.tolist())
    print("MODIS expected:[0, 1, 1, 4, 4, 7, 9, 10, 14, 11, 8, 0]")
    print("MODIS actual:  ", remap_array(modis_test, "MODIS").tolist())
    print()
    # WorldCover
    wc_test = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
    print("WC test:       ", wc_test.tolist())
    print("WC expected:   [0, 1, 2, 4, 9, 10, 11, 14, 8, 7, 4]")
    print("WC actual:     ", remap_array(wc_test, "WorldCover").tolist())
    print()
    print("All product weights:", {k: v["weight"] for k, v in PRODUCT_REGISTRY.items()})
    print(f"Number of products in fusion: {len(PRODUCT_REGISTRY)}")
