"""Fetch a small, seeded subset of Google's Quick, Draw! dataset and rasterize it as clean line drawings.

Source: https://github.com/googlecreativelab/quickdraw-dataset (simplified
drawings, ndjson). The data is licensed CC BY 4.0 by Google; keep the
attribution in data/quickdraw/ATTRIBUTION.txt with any copy.

Only the first --scan lines of each category file are streamed (not the whole
file, which is tens of megabytes); --per-category recognized drawings are
chosen from them with a fixed seed, so the same command gives the same set.

Usage:
    python scripts/fetch_quickdraw.py                                  # default categories, 25 each
    python scripts/fetch_quickdraw.py --categories bird,fish --per-category 10
    python scripts/fetch_quickdraw.py --out data/quickdraw --size 512 --line-width 4

Output: <out>/<category>_<key_id>.png (black on white) and <out>/index.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List

import cv2
import numpy as np

URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
# Subjects close to heritage line art: animals and plants, symbols, vessels and buildings.
DEFAULT_CATEGORIES = (
    "bird", "fish", "cat", "snake", "butterfly", "flower", "tree", "leaf",
    "sun", "moon", "star", "eye", "crown", "key", "vase", "cup",
    "house", "castle", "sailboat", "hand",
)
ATTRIBUTION = (
    "Drawings from the Quick, Draw! dataset by Google, licensed CC BY 4.0\n"
    "(https://github.com/googlecreativelab/quickdraw-dataset, https://creativecommons.org/licenses/by/4.0/).\n"
    "Rasterized by scripts/fetch_quickdraw.py.\n"
)


def stream_lines(category: str, limit: int) -> List[Dict[str, Any]]:
    url = URL.format(urllib.parse.quote(category))
    rows = []
    with urllib.request.urlopen(url, timeout=60) as response:
        for raw in response:
            rows.append(json.loads(raw))
            if len(rows) >= limit:
                break
    return rows


def rasterize(strokes: List[List[List[int]]], size: int, line_width: int, margin: int) -> np.ndarray:
    """Black strokes on white; the 0-255 drawing box is scaled to fit inside the margin."""
    image = np.full((size, size), 255, np.uint8)
    xs = np.concatenate([np.asarray(s[0], float) for s in strokes])
    ys = np.concatenate([np.asarray(s[1], float) for s in strokes])
    span = max(np.ptp(xs), np.ptp(ys), 1.0)
    scale = (size - 2 * margin) / span
    ox = margin + ((size - 2 * margin) - np.ptp(xs) * scale) / 2.0 - xs.min() * scale
    oy = margin + ((size - 2 * margin) - np.ptp(ys) * scale) / 2.0 - ys.min() * scale
    for s in strokes:
        pts = np.column_stack([np.asarray(s[0], float) * scale + ox, np.asarray(s[1], float) * scale + oy])
        cv2.polylines(image, [np.round(pts * 16).astype(np.int32).reshape(-1, 1, 2)], False, 0,
                      line_width, cv2.LINE_AA, shift=4)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--scan", type=int, default=2000, help="lines streamed per category to choose from")
    parser.add_argument("--min-strokes", type=int, default=3, help="skip drawings with fewer strokes")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--margin", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join("data", "quickdraw"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ATTRIBUTION.txt"), "w") as f:
        f.write(ATTRIBUTION)
    index = []
    for category in [c.strip() for c in args.categories.split(",") if c.strip()]:
        try:
            rows = stream_lines(category, args.scan)
        except Exception as error:   # a misspelt category gives HTTP 404
            print(f"  {category}: failed ({error})")
            continue
        usable = [r for r in rows if r.get("recognized") and len(r["drawing"]) >= args.min_strokes]
        chosen = random.Random(f"{args.seed}-{category}").sample(usable, min(args.per_category, len(usable)))
        for row in sorted(chosen, key=lambda r: r["key_id"]):
            name = f"qd_{category.replace(' ', '_')}_{row['key_id']}"
            cv2.imwrite(os.path.join(args.out, name + ".png"),
                        rasterize(row["drawing"], args.size, args.line_width, args.margin))
            index.append({"name": name, "category": category, "key_id": row["key_id"],
                          "countrycode": row.get("countrycode"), "strokes": len(row["drawing"])})
        print(f"  {category}: {len(chosen)} drawings (from {len(usable)} usable of {len(rows)} streamed)")
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"source": URL, "seed": args.seed, "size": args.size, "line_width": args.line_width,
                   "drawings": index}, f, indent=1)
    print(f"{len(index)} drawings -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
