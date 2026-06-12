r"""
================================================================================
MASTER PIPELINE — Inner Mongolia Paper 2
================================================================================
Runs the full preprocessing chain in order:
   01) harmonize LULC products  →  unified 13-class, 1km, AOI-masked
   02) Bayesian fusion          →  fused class + uncertainty maps
   03) align driver stack       →  climate, grazing, phenology, etc.

Each step is idempotent (won't re-do existing outputs unless --force).
Each step writes a log to D:\paper2_outputs\logs\paper2.log

Run
---
$ python run_all.py                 # everything, anchor years only
$ python run_all.py --full          # every year for every product (slow)
$ python run_all.py --step 01       # one step only
$ python run_all.py --force         # re-do existing outputs
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.paths import ensure_dirs
from src.raster_utils import log

from src.step00_diagnose import main as step00_main
from src.step01_harmonize_lulc import main as step01_main
from src.step02_fuse_lulc import main as step02_main
from src.step03_align_drivers import main as step03_main
from src.step04_features import main as step04_main
from src.step05_attribute_multinomial import main as step05_main
from src.step06_causal import main as step06_main
from src.step07_stgnn import main as step07_main


def main():
    parser = argparse.ArgumentParser(description="Inner Mongolia Paper 2 pipeline")
    parser.add_argument("--step", type=str, default=None,
                        choices=["00", "01", "02", "03", "04", "05", "06", "07"],
                        help="run only one step (default: 01-05; 00 is diagnostic, run manually)")
    parser.add_argument("--full", action="store_true",
                        help="harmonize ALL years (default: anchor years only)")
    parser.add_argument("--force", action="store_true",
                        help="force re-process even if outputs exist")
    parser.add_argument("--gpu", action="store_true",
                        help="use CUDA (RTX 3060 Ti) for step 05 XGBoost training")
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.time()
    log.info("\n" + "█" * 80)
    log.info("█  INNER MONGOLIA PAPER 2 PIPELINE")
    log.info("█  step: " + (args.step if args.step else "ALL"))
    log.info("█  full-year mode: " + str(args.full))
    log.info("█" * 80 + "\n")

    # Build argv simulation for each sub-step
    def call_step(step_main, extra_argv):
        old_argv = sys.argv[:]
        sys.argv = [step_main.__module__] + extra_argv
        try:
            step_main()
        finally:
            sys.argv = old_argv

    if args.step == "00":
        call_step(step00_main, [])
        return

    if args.step in (None, "01"):
        extra = []
        if not args.full:
            extra.append("--anchor-only")
        if args.force:
            extra.append("--force")
        call_step(step01_main, extra)

    if args.step in (None, "02"):
        extra = []
        if args.force:
            extra.append("--force")
        call_step(step02_main, extra)

    if args.step in (None, "03"):
        call_step(step03_main, [])

    if args.step in (None, "04"):
        extra = ["--force"] if args.force else []
        call_step(step04_main, extra)

    if args.step in (None, "05"):
        extra = ["--gpu"] if args.gpu else []
        call_step(step05_main, extra)

    if args.step == "06":
        # Heavy (per-pixel CCM); only run when explicitly requested, not in
        # the default all-steps chain.
        call_step(step06_main, [])

    if args.step == "07":
        # Heavy (GNN training); only run when explicitly requested.
        extra = ["--gpu"] if args.gpu else []
        call_step(step07_main, extra)

    elapsed = time.time() - t0
    log.info(f"\n█ Pipeline finished in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
