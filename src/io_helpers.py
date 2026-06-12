"""
================================================================================
MULTI-FORMAT OUTPUT HELPERS
================================================================================
Every figure / table / raster the pipeline produces is saved in MULTIPLE
formats so reviewers and collaborators can re-use them:

  Figure  -> .png (300 dpi, for inline viewing)
          -> .pdf (vector, for journal submission)
          -> .svg (vector, for editing in Inkscape / Illustrator)
          -> .csv (the underlying data, for re-plotting)

  Table   -> .csv (universal)
          -> .xlsx (for supervisor / non-technical readers)
          -> .md (markdown table for pasting into manuscripts)
          -> .tex (LaTeX booktabs format, for the paper)

  Raster  -> .tif (GeoTIFF, the source of truth)
          -> .png (preview rendering with class colors)
          -> .csv (per-class area summary)

Usage
-----
    from src.io_helpers import save_figure, save_table, save_raster_with_preview

    fig, ax = plt.subplots()
    ax.plot(years, areas)
    save_figure(fig, OUT_FIGURES / "grassland_area_trend",
                data={"year": years, "area_km2": areas})

    df = pd.DataFrame({...})
    save_table(df, OUT_TABLES / "class_transitions_2020", caption="Class transitions in 2020")
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ============================================================================
# FIGURES
# ============================================================================

def save_figure(
    fig,
    stem: Union[str, Path],
    data: Optional[Union[pd.DataFrame, dict]] = None,
    dpi: int = 300,
    formats: tuple = ("png", "pdf", "svg"),
) -> dict:
    """Save a matplotlib figure to multiple formats + companion CSV.

    Parameters
    ----------
    fig    : matplotlib Figure
    stem   : output path WITHOUT extension (e.g. OUT_FIGURES/"fig3a_trend")
    data   : DataFrame or dict-of-arrays — the data plotted in the figure.
             Saved as <stem>.csv so the chart can be re-plotted by anyone.
    dpi    : raster resolution for png (300 = journal standard)
    formats: subset of ("png", "pdf", "svg")

    Returns
    -------
    dict of {format: path} for all files written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    for ext in formats:
        out = stem.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(out, bbox_inches="tight", facecolor="white")
        written[ext] = out

    # Companion CSV with the underlying data
    if data is not None:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        csv_path = stem.with_suffix(".csv")
        data.to_csv(csv_path, index=False)
        written["csv"] = csv_path

    return written


# ============================================================================
# TABLES
# ============================================================================

def save_table(
    df: pd.DataFrame,
    stem: Union[str, Path],
    caption: Optional[str] = None,
    label: Optional[str] = None,
    formats: tuple = ("csv", "xlsx", "md", "tex"),
    float_fmt: str = "%.3f",
) -> dict:
    """Save a DataFrame to multiple table formats."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = {}

    if "csv" in formats:
        p = stem.with_suffix(".csv")
        df.to_csv(p, index=False, float_format=float_fmt)
        written["csv"] = p

    if "xlsx" in formats:
        p = stem.with_suffix(".xlsx")
        try:
            df.to_excel(p, index=False)
            written["xlsx"] = p
        except Exception:
            pass  # openpyxl missing — skip silently

    if "md" in formats:
        p = stem.with_suffix(".md")
        with open(p, "w", encoding="utf-8") as f:
            if caption:
                f.write(f"**Table: {caption}**\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n")
        written["md"] = p

    if "tex" in formats:
        p = stem.with_suffix(".tex")
        try:
            tex = df.to_latex(
                index=False,
                float_format=float_fmt,
                caption=caption or "",
                label=label or stem.stem,
                escape=True,
            )
            with open(p, "w", encoding="utf-8") as f:
                f.write(tex)
            written["tex"] = p
        except Exception:
            pass

    return written


# ============================================================================
# RASTERS — save GeoTIFF + PNG preview + per-class area CSV
# ============================================================================

def save_raster_with_preview(
    arr: np.ndarray,
    stem: Union[str, Path],
    legend: Optional[dict] = None,   # {code: (name, hex_color)}
    write_geotiff: bool = True,
    dtype: str = "uint8",
    nodata: int = 0,
    title: Optional[str] = None,
) -> dict:
    """Write a categorical raster as GeoTIFF + PNG preview + class-area CSV.

    `legend` is expected to look like `UNIFIED_LEGEND` from class_harmonization.
    """
    from src.raster_utils import write_target_geotiff  # local import to avoid cycle
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = {}

    if write_geotiff:
        tif = stem.with_suffix(".tif")
        write_target_geotiff(arr, tif, dtype=dtype, nodata=nodata,
                             tags={"figure": stem.stem})
        written["tif"] = tif

    # PNG preview
    if legend is not None:
        codes = sorted(legend.keys())
        colors = [legend[c][1] for c in codes]
        cmap = ListedColormap(colors)
        bounds = [c - 0.5 for c in codes] + [codes[-1] + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=12)
        # Compact legend on the right
        patches = [Patch(color=legend[c][1], label=f"{c} {legend[c][0]}")
                   for c in codes if c != 0]
        ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=8, frameon=False)
        png = stem.with_suffix(".png")
        fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written["png"] = png

    # Per-class area CSV (each pixel = 1 km² @ 1 km grid)
    u, c = np.unique(arr, return_counts=True)
    rows = []
    for code, count in zip(u, c):
        code = int(code)
        if code == nodata:
            continue
        name = legend[code][0] if (legend and code in legend) else f"class_{code}"
        rows.append({"code": code, "name": name, "n_pixels": int(count),
                     "area_km2": float(count)})  # 1 px = 1 km² at 1 km grid
    if rows:
        df = pd.DataFrame(rows).sort_values("area_km2", ascending=False)
        csv = stem.with_suffix(".csv")
        df.to_csv(csv, index=False, float_format="%.2f")
        written["csv"] = csv

    return written


# ============================================================================
# LIGHTWEIGHT JSON RUN-METADATA
# ============================================================================

def save_run_metadata(stem: Union[str, Path], **kv) -> Path:
    """Save a small JSON sidecar of the parameters used to produce an output.

    Useful for reproducibility — every figure can carry the exact configuration
    that generated it.
    """
    import json
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    p = stem.with_suffix(".json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(kv, f, indent=2, default=str)
    return p
