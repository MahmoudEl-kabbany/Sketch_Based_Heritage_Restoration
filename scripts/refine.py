"""Correct a restoration by hand: forbid, force or hint bridges, and re-solve the whole drawing.

Opens an OpenCV window with the damaged sketch, the bridges the pipeline drew
(green, labelled R1, R2, ...) and every free stroke end (blue dots).

    left-click two stroke ends   select a pair (red); then press
        f   force a bridge between them
        h   hint a bridge between them
        x   forbid a bridge between them
    right-click a drawn bridge   forbid it
    u   undo the last constraint
    s   save the constraints (JSON)
    q / Esc   save and quit

After every change the solver re-optimizes the whole drawing: a forbidden
bridge frees its two ends for other partners; a forced one takes its ends from
their old partners. Every action is logged with the affected candidates'
measurements to the feedback file (train_scoring.py --include-feedback).

Usage:
    python scripts/refine.py path/to/damaged.png
    python scripts/refine.py damaged.png --constraints my_fixes.json --feedback feedback/feedback.jsonl

The constraints file can then be passed to restore(..., user_constraints="my_fixes.json").
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from restoration.constraints import DEFAULT_TOLERANCE_PX, KINDS, ConstraintError  # noqa: E402
from restoration.feedback import DEFAULT_FEEDBACK_PATH, append_feedback, feedback_rows  # noqa: E402
from restoration.pipeline import restore  # noqa: E402
from restoration.scoring import MODEL_PATH  # noqa: E402

PICK_RADIUS_PX = 12.0   # how close (in image pixels) a click must be to a stroke end or bridge


class RefineSession:
    """The state of one refinement: constraints, the current restoration, and the feedback log."""

    def __init__(self, image_path: str, constraints: Optional[Dict[str, Any]] = None,
                 model_path: str = MODEL_PATH, feedback_path: Optional[str] = DEFAULT_FEEDBACK_PATH):
        self.image_path = image_path
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.constraints: Dict[str, Any] = {"image": os.path.basename(image_path), "forbid": [], "force": [],
                                            "hint": [], "tolerance_px": DEFAULT_TOLERANCE_PX}
        for kind in KINDS:
            self.constraints[kind] = list((constraints or {}).get(kind, []))
        self.history: List[Tuple[str, Dict[str, Any]]] = []
        self.output_dir = tempfile.mkdtemp(prefix="refine_")
        self.result = None
        self.message = ""
        self.solve()

    # ---- solving
    def solve(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.result = restore(self.image_path, model_path=self.model_path, output_dir=self.output_dir,
                                  save_images=False, user_constraints=self.constraints)

    @property
    def endpoints(self):
        return self.result.extraction.endpoints

    def endpoint_at(self, xy, radius: float = PICK_RADIUS_PX):
        """The free stroke end nearest to *xy* within *radius*, or None."""
        if not self.endpoints:
            return None
        positions = np.array([ep.position for ep in self.endpoints])
        d = np.linalg.norm(positions - np.asarray(xy, float), axis=1)
        i = int(np.argmin(d))
        return self.endpoints[i] if d[i] <= radius else None

    def bridge_at(self, xy, radius: float = PICK_RADIUS_PX):
        """The drawn bridge (accepted candidate) passing nearest to *xy* within *radius*, or None."""
        best, best_d = None, radius
        for c in self.result.accepted or []:
            d = float(np.min(np.linalg.norm(c.bridge_points - np.asarray(xy, float), axis=1)))
            if d <= best_d:
                best, best_d = c, d
        return best

    # ---- editing
    def add(self, kind: str, a_xy, b_xy) -> bool:
        """Add a constraint between the stroke ends at *a_xy* and *b_xy*, re-solve and log it.
        Returns False (and keeps the old state) if the constraint cannot be satisfied."""
        item = {"a": [round(float(v), 1) for v in a_xy], "b": [round(float(v), 1) for v in b_xy]}
        self.constraints[kind].append(item)
        try:
            self.solve()
        except ConstraintError as error:
            self.constraints[kind].pop()
            self.solve()
            self.message = f"not applied: {error}"
            return False
        self.history.append((kind, item))
        self._log(kind, a_xy, b_xy)
        self.message = f"{kind}: {item['a']} - {item['b']}"
        return True

    def undo(self) -> bool:
        if not self.history:
            return False
        kind, item = self.history.pop()
        self.constraints[kind].remove(item)
        self.solve()
        self.message = f"undid {kind}"
        return True

    def _log(self, kind: str, a_xy, b_xy) -> None:
        if not self.feedback_path:
            return
        ep_a, ep_b = self.endpoint_at(a_xy, 1e9), self.endpoint_at(b_xy, 1e9)
        rows = feedback_rows(self.image_path, kind, (ep_a.endpoint_id, ep_b.endpoint_id),
                             self.result.candidates or [], self.result.accepted or [], self.result.extraction)
        append_feedback(rows, self.feedback_path)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.constraints, f, indent=2)
        self.message = f"saved {path}"

    # ---- drawing
    def render(self, selected=()) -> np.ndarray:
        gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        width = max(1, int(round(self.result.extraction.line_width)))
        colors = {"forbid": (0, 0, 255), "force": (255, 0, 255), "hint": (0, 165, 255)}
        for kind in KINDS:
            for item in self.constraints[kind]:
                a, b = (tuple(int(round(v)) for v in item[k]) for k in ("a", "b"))
                cv2.line(canvas, a, b, colors[kind], 1, cv2.LINE_AA)
        for c in self.result.accepted or []:
            pts = np.round(c.bridge_points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 170, 0), max(2, width), cv2.LINE_AA)
        for event in self.result.report["detailed_events"]:
            x, y = (int(v) for v in event["coordinates"])
            cv2.putText(canvas, event["id"], (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1, cv2.LINE_AA)
        for ep in self.endpoints:
            cv2.circle(canvas, tuple(int(round(v)) for v in ep.position), 3, (255, 120, 0), -1, cv2.LINE_AA)
        for ep in selected:
            cv2.circle(canvas, tuple(int(round(v)) for v in ep.position), 6, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas


def run_window(session: RefineSession, constraints_path: str) -> None:
    window = "refine: f force, h hint, x forbid (two ends selected) | right-click bridge = forbid | u undo | s save | q quit"
    selected: List[Any] = []
    h, w = session.render().shape[:2]
    scale = min(1.0, 1400.0 / w, 900.0 / h)

    def to_image(x: int, y: int) -> Tuple[float, float]:
        return x / scale, y / scale

    def on_mouse(event, x, y, flags, param):
        xy = to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ep = session.endpoint_at(xy, PICK_RADIUS_PX / scale)
            if ep is not None:
                selected[:] = (selected + [ep])[-2:]
        elif event == cv2.EVENT_RBUTTONDOWN:
            bridge = session.bridge_at(xy, PICK_RADIUS_PX / scale)
            if bridge is not None:
                session.add("forbid", bridge.ep_a.position, bridge.ep_b.position)
                selected.clear()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = session.render(selected)
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, session.message[:120], (8, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 200), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("f"), ord("h"), ord("x")) and len(selected) == 2:
            kind = {"f": "force", "h": "hint", "x": "forbid"}[chr(key)]
            session.add(kind, selected[0].position, selected[1].position)
            selected.clear()
        elif key == ord("u"):
            session.undo()
        elif key == ord("s"):
            session.save(constraints_path)
        elif key in (ord("q"), 27):
            session.save(constraints_path)
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            session.save(constraints_path)
            break
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--constraints", default=None,
                        help="JSON file to start from and save to (default: feedback/<image>_constraints.json)")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK_PATH, help="feedback log (JSON Lines); '' to disable")
    args = parser.parse_args()

    path = args.constraints or os.path.join("feedback", os.path.splitext(os.path.basename(args.image))[0] + "_constraints.json")
    start = None
    if os.path.exists(path):
        with open(path) as f:
            start = json.load(f)
    session = RefineSession(args.image, start, args.model, args.feedback or None)
    print(f"{len(session.result.accepted)} bridges drawn; {len(session.endpoints)} free stroke ends. "
          f"Constraints -> {path}")
    run_window(session, path)
    print(f"Saved {path}: " + ", ".join(f"{len(session.constraints[k])} {k}" for k in KINDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
