"""Why are junction gaps not repaired? A diagnostic on the synthetic dataset.

For every junction-type gap (synthetic_damage.py) of the chosen split and levels,
the shipped pipeline restores the damaged image, and each gap that is not
repaired is put down to one reason:

    no single bridge covers it        no candidate covers >= 60% of the gap (it would need
                                      two bridges, or a stroke end joined to the middle of a stroke)
    covering bridge below threshold   the scorer gives the covering bridge too low a probability
    covering bridge lost a conflict   it is above the threshold, but the solver chose another
                                      bridge sharing one of its ends (what a joint pairing model could fix)

It then compares the measurements of the under-scored covering bridges with
those of drawn bridges that repair a gap.

Usage:
    python scripts/diagnose_junctions.py                    # val split, easy + medium
    python scripts/diagnose_junctions.py --split test --levels easy,medium,hard
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from evaluate_gaps import COVERAGE_THRESHOLD, build_ground_truth, gap_coverage, load_pair  # noqa: E402
from restoration.pipeline import map_in_processes, restore  # noqa: E402
from synthetic_damage import load_dataset  # noqa: E402


def one(entry):
    from evaluate_synthetic import gap_types

    original, gapped, agree = load_pair(entry["clean"], entry["gapped"])
    if agree < 0.98:
        return []
    gt = build_ground_truth(original, gapped)
    with open(entry["gaps"]) as f:
        types = gap_types(entry["clean"], entry["gapped"], json.load(f)["gaps"])
    with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
        r = restore(entry["damaged"], output_dir=out, save_images=False)
    threshold = r.report["analysis"]["scoring_model"]["threshold"]
    accepted = {c.id for c in r.accepted}
    drawn = gap_coverage(gt, [c.bridge_points for c in r.accepted]) if r.accepted else np.zeros(len(gt.gaps))
    any_path = gap_coverage(gt, [p.sample(30) for p in r.restored_paths])
    cover = np.array([gap_coverage(gt, [c.bridge_points]) for c in r.candidates]).reshape(len(r.candidates), len(gt.gaps))
    ends = np.array([e.position for e in r.extraction.endpoints]).reshape(-1, 2)
    rows = []
    rows_extra = {}
    for k, gap in enumerate(gt.gaps):
        if types.get(gap["id"]) != "junction":
            continue
        ys, xs = np.nonzero(gt.gap_labels == gap["id"])
        centre = np.array([xs.mean(), ys.mean()])
        near_ends = int((np.linalg.norm(ends - centre, axis=1) <= gap["length_px"] / 2 + 3 * gt.line_width).sum()) if len(ends) else 0
        if drawn[k] >= COVERAGE_THRESHOLD or any_path[k] >= COVERAGE_THRESHOLD:
            reason = "repaired"
        else:
            covering = [i for i in range(len(r.candidates)) if cover[i, k] >= COVERAGE_THRESHOLD]
            if not covering:
                reason = "no single bridge covers it"
            else:
                best = max(covering, key=lambda i: r.candidates[i].probability)
                if r.candidates[best].probability <= threshold:
                    reason = "covering bridge below threshold"
                    c = r.candidates[best]
                    rows_extra = {"prob": c.probability, "contrib": dict(c.contributions), "values": dict(c.features), "shape": c.shape}
                else:
                    reason = "covering bridge lost a conflict"
        extra = rows_extra if reason == "covering bridge below threshold" else {}
        rows_extra = {}
        rows.append({**extra, "kind": "junction", "stubs": gap["stubs"], "near_ends": near_ends, "reason": reason,
                     "best_single_cover": float(cover[:, k].max()) if len(cover) else 0.0,
                     "drawn_cover": float(drawn[k])})
    for i, c in enumerate(r.candidates):
        if c.id in accepted and len(gt.gaps) and cover[i].max() >= COVERAGE_THRESHOLD:
            rows.append({"kind": "good", "shape": c.shape, "values": dict(c.features)})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--levels", default="easy,medium")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.chdir(ROOT)
    levels = args.levels.split(",")
    entries = [e for e in load_dataset(args.data, None if args.split == "all" else args.split) if e["difficulty"] in levels]
    allrows = [row for result in map_in_processes(one, [(e,) for e in entries], args.workers) for row in result]
    good = [r for r in allrows if r["kind"] == "good"]
    rows = [r for r in allrows if r["kind"] == "junction"]
    print(f"{len(entries)} variants, {len(rows)} junction gaps")
    by_stubs = Counter(min(r["stubs"], 5) for r in rows)
    print("stubs (5 = 5+):", dict(sorted(by_stubs.items())))
    for stubs in sorted(by_stubs):
        sel = [r for r in rows if min(r["stubs"], 5) == stubs]
        reasons = Counter(r["reason"] for r in sel)
        print(f"  {stubs} stubs ({len(sel)}): " + ", ".join(f"{k} {100 * v / len(sel):.0f}%" for k, v in reasons.most_common()))
    failed = [r for r in rows if r["reason"] != "repaired"]
    print("all failures:", {k: f"{100 * v / max(len(failed), 1):.0f}%" for k, v in Counter(r["reason"] for r in failed).items()})
    nb = [r for r in failed if r["reason"] == "no single bridge covers it"]
    if nb:
        print("  'no single bridge': median best single cover %.2f, median drawn cover %.2f"
              % (np.median([r["best_single_cover"] for r in nb]), np.median([r["drawn_cover"] for r in nb])))

    under = [r for r in rows if "contrib" in r]
    print(f"under-scored covering bridges: {len(under)}; median probability {np.median([r['prob'] for r in under]):.2f}; shapes {Counter(r['shape'] for r in under)}")
    total = Counter()
    for r in under:
        for k, v in r["contrib"].items():
            total[k] += v
    print("summed log-odds per measurement (most negative first):")
    for k, v in sorted(total.items(), key=lambda kv: kv[1])[:8]:
        vals = [r["values"][k] for r in under if k in r["values"]]
        print(f"  {k:16s} {v / len(under):+.2f} per bridge   median value {np.median(vals):.3g}")

    for shape in ("smooth", "corner"):
        u = [r for r in under if r["shape"] == shape]
        g = [r for r in good if r["shape"] == shape]
        if not u or not g:
            continue
        print(f"{shape}: under-scored junction bridges {len(u)} vs good drawn bridges {len(g)} (median values)")
        for k in u[0]["values"]:
            print(f"  {k:16s} junction {np.median([r['values'][k] for r in u]):9.3g}   good {np.median([r['values'][k] for r in g]):9.3g}")
