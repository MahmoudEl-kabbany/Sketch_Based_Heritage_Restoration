"""Build the synthetic damage dataset (see synthetic_damage.py for what each variant contains).

Sources (decision D3): the clean originals of the benchmark and, if present, the
rasterized Quick, Draw! drawings written by scripts/fetch_quickdraw.py.

Usage:
    python scripts/make_synthetic.py                          # both sources -> data/synthetic
    python scripts/make_synthetic.py --variants 3 --seed 1
    python scripts/make_synthetic.py --no-originals --quickdraw data/quickdraw --out data/qd_only

Splits are by source drawing (train 70% / val 15% / test 15%, from a hash of
the drawing name), never by variant. The output is deterministic for a seed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate_restoration import ORIGINAL_DIR  # noqa: E402
from synthetic_damage import DIFFICULTY, write_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join("data", "synthetic"))
    parser.add_argument("--originals", default=ORIGINAL_DIR, help="folder of clean benchmark originals")
    parser.add_argument("--no-originals", action="store_true")
    parser.add_argument("--quickdraw", default=os.path.join("data", "quickdraw"))
    parser.add_argument("--variants", type=int, default=2, help="variants per drawing and difficulty")
    parser.add_argument("--difficulties", default=",".join(DIFFICULTY))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)

    sources = []
    if not args.no_originals:
        for path in sorted(glob.glob(os.path.join(args.originals, "*.png"))):
            sources.append((os.path.splitext(os.path.basename(path))[0], path))
    if args.quickdraw and os.path.isdir(args.quickdraw):
        for path in sorted(glob.glob(os.path.join(args.quickdraw, "qd_*.png"))):
            sources.append((os.path.splitext(os.path.basename(path))[0], path))
    if not sources:
        print("No source drawings found.")
        return 1

    entries = write_dataset(sources, args.out, args.variants, args.difficulties.split(","), args.seed)
    by_split = {}
    for e in entries:
        by_split.setdefault(e["split"], set()).add(e["drawing"])
    print(f"{len(sources)} drawings, {len(entries)} variants, "
          f"{sum(e['gap_count'] for e in entries)} gaps -> {args.out}")
    for split, drawings in sorted(by_split.items()):
        print(f"  {split}: {len(drawings)} drawings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
