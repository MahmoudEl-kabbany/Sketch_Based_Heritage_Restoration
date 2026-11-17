"""Exact snapshot of the pipeline's output, to prove that a refactor changed nothing.

check_changes.py judges whether a change makes restoration better or worse.
This module answers a stricter question: is the output *identical*? For every
benchmark drawing (damaged and undamaged), it records which stroke ends were
bridged and every final path's segments and control points, once with the
default settings and once with Phase 6 on. It uses the shipped scoring model.

Usage:
    python regression_snapshot.py            # compare the code as it is now with the snapshot
    python regression_snapshot.py --rebless  # record the current output as the new snapshot

The same comparison runs under pytest as test_regression.py. Nothing is
written to restoration/outputs/: the pipeline's reports go to a temporary folder.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(ROOT, "regression", "snapshot")
TOLERANCE = 1e-6

# (variant name, restore() keyword arguments)
VARIANTS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("default", {}),
    ("phase6", {"efd_closure": True}),
)


def benchmark_images() -> List[str]:
    """The damaged and the undamaged image of every benchmark pair, relative to the repo root."""
    from evaluate_restoration import discover_pairs

    with contextlib.chdir(ROOT):
        pairs = discover_pairs()
    images = []
    for _name, original, damaged in pairs:
        images += [damaged, original]
    return [os.path.relpath(os.path.join(ROOT, p), ROOT).replace("\\", "/") for p in images]


def snapshot_path(image: str) -> str:
    # Drop the leading test_images/: .gitignore ignores a folder of that name at any depth.
    relative = image.split("/", 1)[1] if image.startswith("test_images/") else image
    return os.path.join(SNAPSHOT_DIR, os.path.splitext(relative)[0] + ".json")


def _paths_summary(paths) -> List[Dict[str, Any]]:
    return [
        {
            "closed": bool(p.is_closed),
            "segments": len(p.segments),
            "control_points": [s.control_points.round(9).tolist() for s in p.segments],
        }
        for p in paths
    ]


def snapshot_of(image: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Restore *image* (relative to the repo root) under every variant and keep what must not change."""
    from restoration.pipeline import restore

    record: Dict[str, Any] = {"image": image}
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(ROOT):
        for variant, kwargs in VARIANTS:
            with contextlib.redirect_stdout(io.StringIO()):
                result = restore(image, output_dir=output_dir or scratch, save_images=False, **kwargs)
            analysis = result.report["analysis"]
            record[variant] = {
                "accepted_endpoint_pairs": analysis["selection"]["accepted_endpoint_pairs"],
                "paths": _paths_summary(result.restored_paths),
                "phase6": {k: v for k, v in analysis["efd_closure_phase"].items()},
            }
    return record


def differences(expected: Any, actual: Any, where: str = "", tol: float = TOLERANCE) -> List[str]:
    """Human-readable differences between two snapshot records; numbers compare within *tol*."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected == actual else [f"{where}: {expected!r} != {actual!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return [] if abs(expected - actual) <= tol else [f"{where}: {expected!r} != {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = [f"{where}.{k}: missing" for k in expected if k not in actual]
        out += [f"{where}.{k}: unexpected" for k in actual if k not in expected]
        for k in expected:
            if k in actual:
                out += differences(expected[k], actual[k], f"{where}.{k}", tol)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(expected)} != {len(actual)}"]
        out: List[str] = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += differences(e, a, f"{where}[{i}]", tol)
        return out
    return [] if expected == actual else [f"{where}: {expected!r} != {actual!r}"]


def load_snapshot(image: str) -> Optional[Dict[str, Any]]:
    path = snapshot_path(image)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_snapshot(record: Dict[str, Any]) -> None:
    path = snapshot_path(record["image"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, separators=(",", ":"))
        f.write("\n")


def _check_one(image: str) -> Tuple[str, Dict[str, Any], List[str]]:
    """Module-level so that it can run in a worker process."""
    record = snapshot_of(image)
    expected = load_snapshot(image)
    diffs = ["no snapshot recorded"] if expected is None else differences(expected, record, image)
    return image, record, diffs


def run(images: Sequence[str], workers: int, rebless: bool) -> int:
    from restoration.pipeline import default_workers, map_in_processes

    workers = workers or default_workers()
    changed = 0
    for image, record, diffs in map_in_processes(_check_one, [(i,) for i in images], workers):
        if rebless:
            write_snapshot(record)
            print(f"  recorded  {image}")
        elif diffs:
            changed += 1
            print(f"  CHANGED   {image}  ({len(diffs)} difference(s))")
            for d in diffs[:5]:
                print(f"              {d}")
        else:
            print(f"  same      {image}")
    if not rebless:
        print(f"\n{len(images) - changed}/{len(images)} images identical to the snapshot")
    return 1 if changed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebless", action="store_true", help="record the current output as the snapshot")
    parser.add_argument("--filter", type=str, default=None, help="only images whose path contains this text")
    parser.add_argument("--workers", type=int, default=0, help="worker processes (default: cores - 2, at most 8)")
    args = parser.parse_args()

    images = [i for i in benchmark_images() if not args.filter or args.filter.lower() in i.lower()]
    if not images:
        print("No benchmark images found (test_images/ is not tracked by git).")
        return 1
    return run(images, args.workers, args.rebless)


if __name__ == "__main__":
    sys.exit(main())
