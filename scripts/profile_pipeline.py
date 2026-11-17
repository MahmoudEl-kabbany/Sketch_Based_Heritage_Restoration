"""Where does the pipeline spend its time, and how does that grow with the drawing?

Scaling probe: restores every damaged benchmark drawing one after another
(no parallel workers, so timings are not distorted) and tabulates each stage's
time against the number of free stroke ends. For each stage, it fits the
exponent k of time ~ endpoints^k.

Profile: runs cProfile on one drawing (by default the one with the most free
ends) with and without the figures, prints the 30 functions with the most
cumulative time, and saves the .prof files. Each profiled run is preceded by an
unprofiled warm-up, so compile and import costs stay out of the profile.

Usage:
    python scripts/profile_pipeline.py                     # scaling probe + profile of the largest drawing
    python scripts/profile_pipeline.py --image test_images/difficult_test_cases/ankh_damaged.png
    python scripts/profile_pipeline.py --no-scaling --markdown

Everything is written to --output-dir (a folder in the system's temp directory by default).
"""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import os
import pstats
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate_restoration import discover_pairs  # noqa: E402
from restoration.pipeline import restore  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "restoration_profile")
STAGES = (
    ("phase1_extraction_s", "P1 extract"),
    ("phase2_candidate_generation_s", "P2 candidates"),
    ("phase3_scoring_s", "P3 scoring"),
    ("phase4_asp_total_s", "P4 ASP"),
    ("phase5_synthesis_s", "P5 synthesis"),
    ("phase7_output_s", "P7 output"),
    ("total_pipeline_s", "total"),
)


def _quiet_restore(image: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return restore(image, output_dir=output_dir, **kwargs).report


def scaling_probe(images: List[str], output_dir: str) -> List[Dict[str, Any]]:
    _quiet_restore(images[0], output_dir, save_images=False)  # warm-up
    rows = []
    for image in images:
        report = _quiet_restore(image, output_dir, save_images=False)
        timings = report["summary"]["stage_timings_s"]
        rows.append({
            "image": os.path.relpath(image, ROOT).replace("\\", "/"),
            "endpoints": report["analysis"]["extraction"]["endpoints"],
            "candidates": report["analysis"]["candidate_generation"]["total"],
            **{key: timings.get(key, 0.0) for key, _ in STAGES},
        })
    rows.sort(key=lambda r: r["endpoints"])
    return rows


def growth_exponents(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Least-squares slope of log(time) on log(endpoints), per stage."""
    use = [r for r in rows if r["endpoints"] > 0]
    x = np.log([r["endpoints"] for r in use])
    out = {}
    for key, label in STAGES:
        t = np.array([max(r[key], 1e-6) for r in use])
        out[label] = float(np.polyfit(x, np.log(t), 1)[0]) if len(use) >= 2 else float("nan")
    return out


def print_scaling(rows: List[Dict[str, Any]], markdown: bool) -> None:
    head = ["image", "ends", "cands"] + [label for _, label in STAGES]
    lines = [head]
    for r in rows:
        name = os.path.basename(r["image"])
        lines.append([name, str(r["endpoints"]), str(r["candidates"])]
                     + [f"{r[key] * 1000:.0f}" for key, _ in STAGES])
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for line in lines[1:]:
            print("| " + " | ".join(line) + " |")
    else:
        widths = [max(len(l[i]) for l in lines) for i in range(len(head))]
        for line in lines:
            print("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(line, widths))))
    print("\n(times in ms)\nGrowth exponent k in time ~ endpoints^k:")
    for label, k in growth_exponents(rows).items():
        print(f"  {label:<14s} {k:5.2f}")
    totals = {label: sum(r[key] for r in rows) for key, label in STAGES}
    print("\nShare of total time over all drawings:")
    for label, value in totals.items():
        if label != "total":
            print(f"  {label:<14s} {100 * value / max(totals['total'], 1e-9):5.1f}%")


def profile(image: str, output_dir: str, save_images: bool, top: int) -> str:
    _quiet_restore(image, output_dir, save_images=save_images)  # warm-up
    profiler = cProfile.Profile()
    with contextlib.redirect_stdout(io.StringIO()):
        profiler.enable()
        restore(image, output_dir=output_dir, save_images=save_images)
        profiler.disable()
    path = os.path.join(output_dir, f"profile_{'images' if save_images else 'noimages'}.prof")
    profiler.dump_stats(path)
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative").print_stats(top)
    print(f"\n=== cProfile, save_images={save_images}: {os.path.relpath(image, ROOT)} -> {path}")
    print(buffer.getvalue().split("\n", 4)[-1] if "ncalls" in buffer.getvalue() else buffer.getvalue())
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="drawing to profile (default: the one with most free ends)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-scaling", action="store_true", help="skip the scaling probe")
    parser.add_argument("--no-profile", action="store_true", help="skip cProfile")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--markdown", action="store_true", help="print the scaling table as Markdown")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if os.path.commonpath([output_dir, os.path.join(ROOT, "restoration", "outputs")]) == \
            os.path.join(ROOT, "restoration", "outputs"):
        parser.error("refusing to write into restoration/outputs/")
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(ROOT)

    images = [os.path.join(ROOT, damaged) for _, _, damaged in discover_pairs()]
    if not images:
        print("No benchmark images found (test_images/ is not tracked by git).")
        return 1

    target = args.image
    if not args.no_scaling:
        rows = scaling_probe(images, output_dir)
        print_scaling(rows, args.markdown)
        target = target or os.path.join(ROOT, rows[-1]["image"])
    if not args.no_profile:
        if target is None:
            counts = {img: _quiet_restore(img, output_dir, save_images=False)["analysis"]["extraction"]["endpoints"]
                      for img in images}
            target = max(counts, key=counts.get)
        target = os.path.abspath(target)
        profile(target, output_dir, save_images=False, top=args.top)
        profile(target, output_dir, save_images=True, top=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
