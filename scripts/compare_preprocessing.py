"""Measure preprocessing variants on the benchmark, each with its own refitted scorer.

Runs check_changes.py once per variant (unit tests, held-out refit, gap
benchmark, PSNR/SSIM), one after another, and prints a Markdown table of the
results next to the default. A change to preprocessing changes the candidates,
so a variant is only judged fairly with a scorer refitted on its own output;
check_changes.py does that.

Usage:
    python scripts/compare_preprocessing.py                       # the variants below
    python scripts/compare_preprocessing.py --only sauvola,h4     # a subset
    python scripts/compare_preprocessing.py --table               # re-print the table from earlier runs

Each run takes about 2.5 minutes with 8 workers. Everything goes to --output-dir
(a folder in the system's temp directory by default); nothing is written to
restoration/outputs/, the shipped model or the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "restoration_preprocessing")

# name -> (hypothesis, PreprocessConfig overrides)
VARIANTS = {
    "default": ("current behaviour (Otsu, no blur, zhang skeleton)", {}),
    "median3": ("H1: gray-level median 3 before Otsu", {"denoise": "median", "denoise_size": 3}),
    "sauvola": ("H2: Sauvola, window 25, k 0.2", {"threshold": "sauvola"}),
    "sauvola51": ("H2: Sauvola, window 51, k 0.2", {"threshold": "sauvola", "window": 51}),
    "adaptive": ("H2: adaptive Gaussian, window 25, C 10", {"threshold": "adaptive_gaussian"}),
    "nlm_adaptive_close": ("H3: nlmeans h10 + adaptive Gaussian + closing 3",
                           {"denoise": "nlmeans", "denoise_size": 10, "threshold": "adaptive_gaussian", "close_px": 3}),
    "h4": ("H4: junction-aware spur pruning", {"junction_aware_pruning": True}),
    "lee": ("skeleton: Lee instead of Zhang", {"skeleton": "lee"}),
    "h4_median3": ("H4 + H1 combined", {"junction_aware_pruning": True, "denoise": "median", "denoise_size": 3}),
    "h4_sauvola51": ("H4 + Sauvola 51 combined", {"junction_aware_pruning": True, "threshold": "sauvola", "window": 51}),
}

COLUMNS = (
    ("gap_recall", "recall %"),
    ("off_drawing_widths", "off-drawing (lw)"),
    ("wrong_strokes", "wrong strokes"),
    ("strokes_on_undamaged", "strokes on undamaged"),
    ("psnr", "PSNR"),
    ("ssim", "SSIM"),
)
STAGES = ("endpoints_missing", "no_candidate_generated", "candidate_not_selected", "bridged_by_extraction")


def run(name: str, output_dir: str, workers: int) -> None:
    hypothesis, overrides = VARIANTS[name]
    out = os.path.join(output_dir, name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "check_changes.py"), "--output-dir", out, "--workers", str(workers)]
    if overrides:
        cmd += ["--preprocess", json.dumps(overrides)]
    print(f"=== {name}: {hypothesis}", flush=True)
    with open(os.path.join(out, "check_changes.log"), "w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)


def table(output_dir: str, names) -> str:
    head = ["variant"] + [label for _, label in COLUMNS] + [s.replace("_", " ") + " %" for s in STAGES]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in names:
        path = os.path.join(output_dir, name, "check_results.json")
        if not os.path.exists(path):
            lines.append(f"| {name} | (no result) |")
            continue
        with open(path) as f:
            result = json.load(f)
        summary, stages = result["summary"], result.get("stages", {})
        cells = [name] + [str(summary.get(key, "")) for key, _ in COLUMNS] + [str(stages.get(s, "")) for s in STAGES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="comma-separated variant names")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--table", action="store_true", help="only print the table of earlier runs")
    args = parser.parse_args()

    names = args.only.split(",") if args.only else list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}; known: {list(VARIANTS)}")
    if not args.table:
        for name in names:
            run(name, args.output_dir, args.workers)
    print(table(args.output_dir, names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
