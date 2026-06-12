"""
================================================================================
ENVIRONMENT CHECK — run this BEFORE the pipeline
================================================================================
Verifies:
  1. Python version (3.9+ recommended)
  2. All required packages installed at usable versions
  3. GPU + CUDA visible (for the future ML/GNN steps)
  4. The data folders on D: actually exist and are readable
  5. The AOI shapefile loads with a valid CRS
  6. Writability of the output folder
  7. Free disk space on D:

No side effects (creates a temp file briefly to test write permission,
then deletes it). Run from the Paper2 project root:

    python check_env.py
================================================================================
"""

from __future__ import annotations

import sys
import platform
import shutil
from pathlib import Path


# ANSI color codes (work in PyCharm + modern Windows terminals)
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
END   = "\033[0m"


def ok(msg):     print(f"  {GREEN}✓{END} {msg}")
def fail(msg):   print(f"  {RED}✗{END} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{END} {msg}")
def info(msg):   print(f"  {CYAN}ℹ{END} {msg}")
def header(msg): print(f"\n{BOLD}── {msg} ──{END}")


# ============================================================================
# CHECKS
# ============================================================================

errors = []
warnings_list = []


def check_python():
    header("1. Python version")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} {platform.release()}")
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor} is supported")
    else:
        fail(f"Python 3.9+ required, got {v.major}.{v.minor}")
        errors.append("Upgrade Python to 3.9+")


def check_package(name, import_name=None, min_version=None, optional=False):
    """Try to import a package, report version, optionally check min version."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "(unknown)")
        if min_version and ver != "(unknown)":
            # Lightweight version comparison
            try:
                cur = tuple(int(x) for x in ver.split(".")[:3] if x.isdigit())
                req = tuple(int(x) for x in min_version.split(".")[:3])
                if cur < req:
                    warn(f"{name:20s} {ver}  (recommend ≥ {min_version})")
                    warnings_list.append(f"{name} {ver} < recommended {min_version}")
                    return ver
            except Exception:
                pass
        ok(f"{name:20s} {ver}")
        return ver
    except ImportError as e:
        if optional:
            warn(f"{name:20s} not installed (optional)")
            warnings_list.append(f"optional package missing: {name}")
        else:
            fail(f"{name:20s} NOT INSTALLED  ({e})")
            errors.append(f"pip install {name}")
        return None


def check_required_packages():
    header("2. Required packages (preprocessing pipeline)")
    check_package("numpy",        min_version="1.20")
    check_package("pandas",       min_version="1.3")
    check_package("scipy",        min_version="1.7")
    check_package("rasterio",     min_version="1.3")
    check_package("geopandas",    min_version="0.10")
    check_package("pyogrio",      min_version="0.6")   # fast vector IO used by GRIP roads
    check_package("matplotlib",   min_version="3.5")
    check_package("openpyxl",     min_version="3.0")   # for .xlsx (yearbook, factors)
    check_package("xlrd",         min_version="2.0")   # for legacy .xls (yearbook)
    check_package("xarray",       min_version="2022.0", optional=True)   # NetCDF
    check_package("netCDF4",      min_version="1.5",    optional=True)
    check_package("tabulate",     optional=True)        # nicer markdown tables

    header("3. ML / GNN packages (needed AFTER preprocessing — install later if not now)")
    check_package("sklearn",      import_name="sklearn", optional=True)
    check_package("xgboost",      optional=True)
    check_package("shap",         optional=True)
    check_package("torch",        optional=True)
    check_package("torch_geometric", optional=True)


def check_gpu():
    header("4. GPU (CUDA) for ML/GNN steps")
    try:
        import torch
    except ImportError:
        warn("torch not installed — skipping GPU check (install later for ML steps)")
        warnings_list.append("torch missing (ok for now)")
        return
    info(f"PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        ok(f"CUDA available: {n} device(s)")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info(f"   [{i}] {name}  ({mem:.1f} GB VRAM)")
        # Cuda compile version
        if hasattr(torch.version, "cuda"):
            info(f"   built against CUDA {torch.version.cuda}")
    else:
        warn("CUDA not available — ML steps will run on CPU (much slower)")
        warnings_list.append("CUDA not available; consider 'pip install torch --index-url https://download.pytorch.org/whl/cu121'")


def check_paths():
    header("5. Data folders + AOI shapefile")
    # Lazy import so we don't fail before announcing what's wrong
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from configs.paths import (
            DATA_ROOT, OTHER_ROOT, AOI_SHP, OUT_ROOT,
            LULC_PRODUCTS, DRIVERS,
        )
    except Exception as e:
        fail(f"Could not import configs/paths.py: {e}")
        errors.append("Fix configs/paths.py")
        return

    # Roots
    for label, p in [("DATA_ROOT", DATA_ROOT), ("OTHER_ROOT", OTHER_ROOT)]:
        if Path(p).exists():
            ok(f"{label}: {p}")
        else:
            fail(f"{label} not found: {p}")
            errors.append(f"Verify {label} path in configs/paths.py")

    # AOI
    if Path(AOI_SHP).exists():
        ok(f"AOI shapefile: {AOI_SHP}")
        try:
            import geopandas as gpd
            aoi = gpd.read_file(AOI_SHP)
            if aoi.crs is None:
                fail("AOI has no CRS!")
                errors.append("AOI shapefile must have a valid CRS")
            else:
                ok(f"AOI CRS: {aoi.crs}")
                ok(f"AOI features: {len(aoi)}, bounds: {tuple(round(b, 1) for b in aoi.total_bounds)}")
        except Exception as e:
            fail(f"AOI load failed: {e}")
            errors.append("AOI shapefile cannot be opened")
    else:
        fail(f"AOI not found: {AOI_SHP}")
        errors.append(f"Place AOI shapefile at {AOI_SHP}")

    header("6. Per-product folder / file existence")
    for product, cfg in LULC_PRODUCTS.items():
        if "folder" in cfg:
            p = Path(cfg["folder"])
            if p.exists():
                ok(f"{product:18s} folder: {p}")
            else:
                fail(f"{product:18s} folder MISSING: {p}")
                errors.append(f"Missing folder for {product}")
        if "files" in cfg:
            for f in cfg["files"]:
                if Path(f).exists():
                    ok(f"{product:18s} file:   {Path(f).name}")
                else:
                    fail(f"{product:18s} file MISSING: {f}")
                    errors.append(f"Missing file for {product}: {Path(f).name}")

    header("7. Driver folders / files")
    for key, p in DRIVERS.items():
        if isinstance(p, dict):
            continue  # skip the species dict
        if isinstance(p, list):
            continue
        if not p:
            continue
        # paths.py uses pathlib; some entries are templates with {year} placeholders
        p_str = str(p)
        if "{year}" in p_str or key.endswith("_template"):
            continue
        if Path(p).exists():
            ok(f"{key:30s}: {p}")
        else:
            warn(f"{key:30s} missing: {p}")
            warnings_list.append(f"driver missing: {key}")


def check_output_writability():
    header("8. Output folder writability + disk space")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from configs.paths import OUT_ROOT
    except Exception as e:
        fail(f"Cannot import OUT_ROOT: {e}")
        return

    OUT_ROOT = Path(OUT_ROOT)
    try:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        test_file = OUT_ROOT / ".env_check_test"
        test_file.write_text("ok")
        test_file.unlink()
        ok(f"Can write to {OUT_ROOT}")
    except Exception as e:
        fail(f"Cannot write to {OUT_ROOT}: {e}")
        errors.append(f"Output folder {OUT_ROOT} is not writable")
        return

    # Disk space
    try:
        total, used, free = shutil.disk_usage(OUT_ROOT.anchor)
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        info(f"Disk on {OUT_ROOT.anchor}: {free_gb:.1f} GB free of {total_gb:.1f} GB")
        if free_gb < 50:
            warn(f"Less than 50 GB free — pipeline may run out (need ~80 GB total)")
            warnings_list.append("low disk space")
        elif free_gb < 100:
            warn(f"~80 GB recommended for full pipeline; you have {free_gb:.0f}")
        else:
            ok(f"Plenty of free space ({free_gb:.0f} GB)")
    except Exception as e:
        warn(f"Could not check disk space: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"{BOLD}{CYAN}")
    print("=" * 78)
    print("  Paper 2 — Environment Check")
    print("=" * 78)
    print(f"{END}")
    info(f"Python executable: {sys.executable}")

    check_python()
    check_required_packages()
    check_gpu()
    check_paths()
    check_output_writability()

    # Final summary
    print(f"\n{BOLD}{'=' * 78}{END}")
    if errors:
        print(f"{RED}{BOLD}✗  {len(errors)} BLOCKING ISSUE(S) FOUND:{END}")
        for e in errors:
            print(f"   • {e}")
    if warnings_list:
        print(f"{YELLOW}{BOLD}⚠  {len(warnings_list)} warning(s):{END}")
        for w in warnings_list:
            print(f"   • {w}")
    if not errors and not warnings_list:
        print(f"{GREEN}{BOLD}✓  All checks passed. You are ready to run the pipeline.{END}")
    elif not errors:
        print(f"{GREEN}{BOLD}✓  No blocking issues. Warnings above are OK to proceed.{END}")
    else:
        print(f"\n{RED}Fix the blocking issues above before running the pipeline.{END}")
    print(f"{BOLD}{'=' * 78}{END}\n")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
