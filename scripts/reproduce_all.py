"""Reproduce every number, map and figure in the report.

    python scripts/reproduce_all.py            # full run (~90 min on an 8-thread CPU laptop)
    python scripts/reproduce_all.py --quick    # seed 0 only, no ablations (~12 min); report then mixes in committed ablation results

Steps: download -> crop experiments (2 scenes x 2 splits x 2 models x 3 seeds) -> band / S2 /
block-size experiments -> Cuprite mineral mapping -> outputs + results.csv -> report PDF -> tests.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args):
    t = time.time()
    print(f"\n>>> {' '.join(args)}", flush=True)
    subprocess.check_call([PY] + args, cwd=ROOT)
    print(f"<<< {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    run(["scripts/download_data.py"])
    if a.quick:
        run(["scripts/train_crop_models.py", "--seeds", "0"])
        run(["scripts/run_mineral_mapping.py"])
    else:
        run(["scripts/train_crop_models.py"])
        run(["scripts/band_experiments.py"])
        run(["scripts/train_crop_models.py", "--splits", "spatial", "--block", "30", "--tag", "b30", "--no_save_models"])
        run(["scripts/run_mineral_mapping.py"])
    run(["scripts/generate_outputs.py"])
    run(["scripts/gap_analysis.py"])
    run(["scripts/make_report.py"])
    run(["-m", "pytest", "-q", "tests"])
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
