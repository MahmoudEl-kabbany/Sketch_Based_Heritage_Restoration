"""Gap-level evaluation on the synthetic damage dataset, by difficulty, gap type and source.

For every variant, the pipeline restores damaged.png. Ground truth comes from
(clean.png, gapped.png) (see synthetic_damage.py), and each true gap is matched
to the typed gap recorded in gaps.json. The numbers are those of
evaluate_gaps.py: gap recall, the stage each gap reached
(endpoints_missing = extraction miss, no_candidate_generated = beyond the
candidate-recall ceiling, ...), wrong strokes and line drawn off the drawing.

Usage:
    python scripts/evaluate_synthetic.py --split test
    python scripts/evaluate_synthetic.py --split test --models cv_models/   # held-out models by drawing
    python scripts/evaluate_synthetic.py --data data/synthetic --difficulty hard --limit 20

The undamaged clean drawing of each source is restored once (with its first
variant) to measure what the pipeline draws where nothing is missing.
Results: <output-dir>/synthetic_evaluation.json and a Markdown summary on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate_gaps import STAGES, build_ground_truth, evaluate_pair, load_pair, summarize  # noqa: E402
from restoration.pipeline import default_workers, map_in_processes  # noqa: E402
from restoration.scoring import MODEL_PATH, held_out_models  # noqa: E402
from synthetic_damage import GAP_TYPES, load_dataset  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "restoration_synthetic_eval")


def gap_types(clean_path: str, gapped_path: str, records: List[Dict[str, Any]]) -> Dict[int, str]:
    """Ground-truth gap id -> recorded type ("other" for gaps no record explains)."""
    original, gapped, _ = load_pair(clean_path, gapped_path)
    gt = build_ground_truth(original, gapped)
    h, w = gt.shape
    radius = max(2, int(round(gt.line_width)))
    votes: Dict[int, Counter] = defaultdict(Counter)
    for record in records:
        points = np.asarray(record["centre_line"] + [record["centre"]] + record["cut_points"], float).reshape(-1, 2)
        labels = []
        for x, y in np.round(points).astype(int):
            window = gt.gap_labels[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
            labels += window[window > 0].tolist()
        if labels:
            gap_id = Counter(labels).most_common(1)[0][0]
            votes[gap_id][record["type"]] += 1
    return {gap["id"]: (votes[gap["id"]].most_common(1)[0][0] if votes[gap["id"]] else "other") for gap in gt.gaps}


def evaluate_entry(entry: Dict[str, Any], output_dir: str, model_path: str, include_undamaged: bool) -> Dict[str, Any]:
    name = f"{entry['drawing']}/{entry['difficulty']}_{entry['variant']}"
    with contextlib.redirect_stdout(io.StringIO()):
        row = evaluate_pair(
            name, entry["clean"], entry["damaged"], os.path.join(output_dir, entry["drawing"], name.split("/")[1]),
            model_path=model_path, reference_path=entry["gapped"], include_undamaged=include_undamaged,
        )
    row.update(drawing=entry["drawing"], difficulty=entry["difficulty"], split=entry["split"],
               source="quickdraw" if entry["drawing"].startswith("qd_") else "own")
    if "skipped" not in row:
        with open(entry["gaps"]) as f:
            types = gap_types(entry["clean"], entry["gapped"], json.load(f)["gaps"])
        for gap in row["gaps"]:
            gap["type"] = types.get(gap["id"], "other")
    return row


def _recall(gaps: List[Dict[str, Any]]) -> Optional[float]:
    return round(100.0 * sum(g["stage"] == "repaired" for g in gaps) / len(gaps), 1) if gaps else None


def breakdown(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in rows if "skipped" not in r]
    gaps = [g for r in rows for g in r["gaps"]]

    def stage_shares(selection):
        counts = Counter(g["stage"] for g in selection)
        return {s: round(100.0 * counts[s] / max(len(selection), 1), 1) for s in STAGES}

    out: Dict[str, Any] = {"overall": summarize(rows)}
    for key, values in (("difficulty", ("easy", "medium", "hard")), ("source", ("own", "quickdraw"))):
        out[f"by_{key}"] = {}
        for value in values:
            chosen = [r for r in rows if r[key] == value]
            if not chosen:
                continue
            s = summarize(chosen)
            out[f"by_{key}"][value] = {
                "images": len(chosen), "gaps": s["gaps"], "recall": _recall([g for r in chosen for g in r["gaps"]]),
                "wrong_strokes": s["damaged"]["wrong_strokes"], "off_drawing_widths": s["damaged"]["off_drawing_widths"],
                "stages": stage_shares([g for r in chosen for g in r["gaps"]]),
            }
    out["by_type"] = {}
    for kind in GAP_TYPES + ("other",):
        chosen = [g for g in gaps if g.get("type") == kind]
        if chosen:
            out["by_type"][kind] = {"gaps": len(chosen), "recall": _recall(chosen), "stages": stage_shares(chosen)}
    return out


def markdown(result: Dict[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        f"Images {overall['images']}, gaps {overall['gaps']}, recall {100 * (overall['gap_recall'] or 0):.1f}%, "
        f"wrong strokes {overall['damaged']['wrong_strokes']}, off-drawing {overall['damaged']['off_drawing_widths']} lw "
        f"(+ undamaged: {overall['undamaged']['strokes']} strokes, {overall['undamaged']['off_drawing_widths']} lw off)",
        "",
    ]
    stage_head = " | ".join(s.replace("_", " ") + " %" for s in STAGES)
    for key in ("by_difficulty", "by_source"):
        lines += [f"| {key[3:]} | images | gaps | recall % | wrong strokes | off-drawing (lw) | {stage_head} |",
                  "|" + "---|" * (6 + len(STAGES))]
        for value, r in result[key].items():
            lines.append(f"| {value} | {r['images']} | {r['gaps']} | {r['recall']} | {r['wrong_strokes']} | "
                         f"{r['off_drawing_widths']} | " + " | ".join(str(r["stages"][s]) for s in STAGES) + " |")
        lines.append("")
    lines += [f"| gap type | gaps | recall % | {stage_head} |", "|" + "---|" * (3 + len(STAGES))]
    for kind, r in result["by_type"].items():
        lines.append(f"| {kind} | {r['gaps']} | {r['recall']} | " + " | ".join(str(r["stages"][s]) for s in STAGES) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("data", "synthetic"))
    parser.add_argument("--split", default="test", help="train, val, test or all")
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--source", default=None, choices=("own", "quickdraw"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", default=None, help="train_scoring.py --cv-dir folder: held-out model per drawing")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    os.chdir(ROOT)

    entries = load_dataset(args.data, None if args.split == "all" else args.split, args.difficulty)
    if args.source:
        entries = [e for e in entries if (e["drawing"].startswith("qd_")) == (args.source == "quickdraw")]
    entries = entries[:args.limit] if args.limit else entries
    if not entries:
        print("No variants selected.")
        return 1
    drawings = sorted({e["drawing"] for e in entries})
    model_of: Dict[str, str] = {}
    if args.models:
        # Held-out model for every drawing the models were fitted with; the others
        # (e.g. Quick, Draw! drawings when the models saw only the benchmark) were never
        # seen by any model, so the shipped one is a fair choice for them.
        with open(os.path.join(args.models, "folds.json")) as f:
            known = set(json.load(f))
        model_of = held_out_models(args.models, [d for d in drawings if d in known])
        print(f"  held-out models for {sum(d in known for d in drawings)} of {len(drawings)} drawings; "
              f"the shipped model for the rest")

    first_variant = {}
    for e in entries:
        first_variant.setdefault(e["drawing"], (e["difficulty"], e["variant"]))
    jobs = [(e, args.output_dir, model_of.get(e["drawing"], MODEL_PATH),
             first_variant[e["drawing"]] == (e["difficulty"], e["variant"])) for e in entries]
    rows = []
    for i, row in enumerate(map_in_processes(evaluate_entry, jobs, args.workers), 1):
        rows.append(row)
        if i % 25 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} variants evaluated", flush=True)

    result = breakdown(rows)
    result["config"] = {"data": args.data, "split": args.split, "difficulty": args.difficulty,
                        "source": args.source, "models": args.models, "variants": len(entries),
                        "held_out_drawings": sorted(d for d in drawings if d in model_of),
                        "drawings": len(drawings), "skipped": [r["name"] for r in rows if "skipped" in r]}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "synthetic_evaluation.json"), "w") as f:
        json.dump({"result": result, "per_image": rows}, f, indent=1, default=float)
    print(markdown(result))
    print(f"\nResults -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
