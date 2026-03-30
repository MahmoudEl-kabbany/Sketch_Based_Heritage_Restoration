"""
Sketch-Based Heritage Restoration — Corrected Restoration Engine
================================================================

Bug-fixes applied versus the original restoration.py
-----------------------------------------------------

BUG-1  START tangent was INWARD (P1−P0) instead of OUTWARD (P0−P1).
       The end tangent (P3−P2) was already outward, creating an
       inconsistency that caused the direction checks and bridge
       geometry to be wrong for every "start" endpoint.

BUG-2  Bridge arrive_dir logic was inverted.
       When the receiving endpoint is a "start", the bridge must arrive
       in the path's INWARD direction (−outward).  The original code had
       the signs swapped, producing G¹ breaks at every junction.

BUG-3  Tangent alignment check used directed cosines instead of the
       angle between the two outward tangents.
       A broken stroke has anti-parallel outward tangents (dot ≈ −1).
       The original code rejected this as "wrong direction" and missed
       the most obvious connections.  Fix: abs(dot) < cos(angle_limit).

BUG-4  No minimum arc-length filter on paths.
       Very short skeleton stubs (< MIN_PATH_LEN_PX) arise from
       noise, morphological artefacts, and fine corner pixels.
       They are not real strokes and must be excluded before
       endpoint extraction.

BUG-5  No skeleton-node degree filter.
       sknw creates a Bézier path per skeleton EDGE, not per stroke.
       At every junction (degree ≥ 2) all meeting edges share a node,
       so their endpoints are already physically connected.  These are
       NOT free endpoints.  Only degree-1 (dangling) nodes produce
       genuine free endpoints.  The adjacency dict captures 1-hop
       neighbours but not the degree constraint itself.

BUG-6  Same-path self-closure applied too aggressively.
       The Gestalt closure bonus fired whenever start and end of the
       same path were in the candidate list, even for very short open
       paths that are merely skeleton noise.  Added a minimum self-gap
       distance and a minimum arc-length guard before emitting a
       self-closure candidate.

BUG-7  used_endpoints tracked by list INDEX, not by (path_id, role).
       Because candidates are scored and sorted, the same physical
       endpoint can appear under multiple list indices.  Tracking by
       index is unreliable; must use (path_index, role) as the key.

BUG-8  GAP_ANGLE_DEG check was applied to (tangent vs. AB direction)
       instead of (tangent_A vs. tangent_B).
       The correct collinearity test is: the angle between the two
       outward tangents, taken with abs, must be below the threshold.

BUG-9  Adjacency check was one-directional.
       `adjacency.get(ep_a.path_index, set())` only skips if ep_b is
       in ep_a's neighbour set, but the sknw graph stores the relation
       in both directions.  Use symmetric lookup to be safe.

BUG-10 EFD symmetry completion returned arc in the wrong order.
       `reflected[::-1]` happens to start near arc[-1] and end near
       arc[0], but the resampled bridge was not pinned to those two
       anchor points, so the completion drifted from the arc endpoints.
       Fix: pin first and last points to arc[-1] and arc[0] exactly.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyefd
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
import sknw

from bezier_curves.bezier import (
    BezierPath,
    BezierSegment,
    _estimate_tangent,
    _fit_cubic_single,
    fit_from_image_skeleton,
)
from eliptic_fourier_descriptors.efd import reconstruct_contour_efd


# ═══════════════════════════════════════════════════════════════════════════
# Hyper-parameters
# ═══════════════════════════════════════════════════════════════════════════

MAX_BRIDGE_DIST_FRACTION   = 0.25   # fraction of image diagonal
MAX_GAP_ANGLE_DEG          = 35.0   # max angle between outward tangents (collinearity)
MIN_DIRECTION_TOWARD_COS   = 0.20   # min cos(A→B · tA) to confirm endpoint faces gap
BRIDGE_ALPHA_FRACTION      = 0.33
BRIDGE_SAMPLE_N            = 80

# ── path quality guards (BUG-4 & BUG-5) ─────────────────────────────────
MIN_PATH_LEN_PX            = 15.0   # minimum arc length to be a real stroke
MIN_SELF_GAP_PX            = 10.0   # minimum gap for a path to close on itself

# ── EFD ─────────────────────────────────────────────────────────────────
EFD_OPEN_CONTOUR_THRESHOLD = 0.12
EFD_HARMONICS              = 20
EFD_RECONSTRUCTION_PTS     = 400
SYMMETRY_SCORE_THRESHOLD   = 0.70

# ── Gestalt weights ──────────────────────────────────────────────────────
W_CONTINUITY   = 2.0
W_PROXIMITY    = 1.5
W_CLOSURE      = 1.2
W_SYMMETRY     = 1.0
W_SIMILARITY   = 0.8

HIGH_CONF_THRESHOLD   = 0.70
MEDIUM_CONF_THRESHOLD = 0.45


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Endpoint:
    """One free endpoint of a BezierPath."""

    path_index : int
    role       : str          # "start" | "end"
    position   : np.ndarray   # (2,) x, y
    tangent    : np.ndarray   # (2,) OUTWARD unit tangent (away from the path)


@dataclass
class RestorationDecision:
    decision_type       : str
    confidence          : float
    gestalt_principles  : List[str]
    score_breakdown     : Dict[str, float]
    explanation         : str
    restored_curve      : Optional[np.ndarray] = None
    alternatives        : List[Dict]            = field(default_factory=list)

    @property
    def confidence_label(self) -> str:
        if self.confidence >= HIGH_CONF_THRESHOLD:
            return "HIGH"
        if self.confidence >= MEDIUM_CONF_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def as_dict(self) -> dict:
        return {
            "type"             : self.decision_type,
            "confidence"       : round(self.confidence, 3),
            "confidence_label" : self.confidence_label,
            "gestalt"          : self.gestalt_principles,
            "scores"           : {k: round(v, 3) for k, v in self.score_breakdown.items()},
            "explanation"      : self.explanation,
            "alternatives"     : self.alternatives,
        }


@dataclass
class RestorationResult:
    original_image      : np.ndarray
    restored_canvas     : np.ndarray
    original_paths      : List[BezierPath]
    bridge_segments     : List[BezierSegment]
    efd_completions     : List[np.ndarray]
    decisions           : List[RestorationDecision]
    output_dir          : str
    image_name          : str

    def report(self) -> str:
        lines = [
            "=" * 70,
            "  RESTORATION ENGINE — XAI DECISION REPORT",
            f"  Image   : {self.image_name}",
            f"  Bridges : {len(self.bridge_segments)}   "
            f"EFD completions : {len(self.efd_completions)}   "
            f"Decisions : {len(self.decisions)}",
            "=" * 70,
        ]
        for i, d in enumerate(self.decisions):
            lines += [
                f"\n  [{i+1}] {d.decision_type.upper()}  "
                f"conf={d.confidence:.2f} ({d.confidence_label})",
                f"      Gestalt : {', '.join(d.gestalt_principles) or 'none'}",
                f"      Scores  : {d.score_breakdown}",
            ]
            for line in textwrap.wrap(d.explanation, width=66):
                lines.append(f"      {line}")
        lines.append("\n" + "=" * 70)
        out = "\n".join(lines)
        print(out)
        return out

    def save_visuals(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.splitext(self.image_name)[0]
        self._save_comparison(stem)
        self._save_xai_overlay(stem)
        log_path = os.path.join(self.output_dir, f"{stem}_xai_log.json")
        with open(log_path, "w") as fh:
            json.dump([d.as_dict() for d in self.decisions], fh, indent=2)
        print(f"  -> XAI log → {log_path}")

    def _save_comparison(self, stem: str) -> None:
        orig_bgr = (
            cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            if self.original_image.ndim == 2
            else self.original_image.copy()
        )
        h, w = orig_bgr.shape[:2]
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.patch.set_facecolor("#0a0a0a")
        for ax, title in zip(axes,
                              ["Original (Damaged)", "Restored", "Confidence Map"]):
            ax.set_facecolor("#0a0a0a")
            ax.set_title(title, color="white", fontsize=13)
            ax.axis("off")
        axes[0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(self.restored_canvas, cv2.COLOR_BGR2RGB))
        conf_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for d in self.decisions:
            if d.restored_curve is None:
                continue
            pts = d.restored_curve.astype(np.int32)
            c = d.confidence
            bgr = (int(255 * max(0, 1 - c * 2)),
                   int(255 * min(c * 2, 1.0)),
                   int(255 * max(0, 1 - c * 2)))
            for k in range(len(pts) - 1):
                cv2.line(conf_canvas, tuple(pts[k]), tuple(pts[k + 1]),
                         bgr, 2, cv2.LINE_AA)
        axes[2].imshow(cv2.cvtColor(conf_canvas, cv2.COLOR_BGR2RGB))
        axes[2].legend(handles=[
            mpatches.Patch(color="#00ff00", label=f"High (≥{HIGH_CONF_THRESHOLD:.0%})"),
            mpatches.Patch(color="#ffff00", label="Medium"),
            mpatches.Patch(color="#ff0000", label="Low"),
        ], loc="lower right", facecolor="#1a1a1a", labelcolor="white", fontsize=9)
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{stem}_restoration.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
        plt.close(fig)
        print(f"  -> Restoration image → {path}")

    def _save_xai_overlay(self, stem: str) -> None:
        overlay = self.restored_canvas.copy()
        for i, d in enumerate(self.decisions):
            if d.restored_curve is None or len(d.restored_curve) == 0:
                continue
            pts = d.restored_curve.astype(np.int32)
            mid = pts[len(pts) // 2]
            cv2.putText(overlay, f"#{i+1} {d.confidence_label}", tuple(mid),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        out = os.path.join(self.output_dir, f"{stem}_xai_overlay.png")
        cv2.imwrite(out, overlay)
        print(f"  -> XAI overlay → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Gestalt Constraint Engine
# ═══════════════════════════════════════════════════════════════════════════

class GestaltEngine:
    """
    Quantifies Gestalt principles as additive score components.
    All tangents passed here must be OUTWARD unit vectors.
    """

    def __init__(self, image_diagonal: float):
        self.img_diag = max(image_diagonal, 1.0)

    def continuity(self, ep_a: Endpoint, ep_b: Endpoint) -> float:
        """
        Reward when BOTH outward tangents point toward the partner endpoint.

        For a broken stroke:
          - A_end outward tangent  → points rightward
          - B_start outward tangent → points leftward
          - AB direction → rightward
          - cos_a = dot(right, right) = 1   ✓
          - cos_b = dot(left, -right) = 1   ✓
        """
        ab = ep_b.position - ep_a.position
        dist = np.linalg.norm(ab) + 1e-9
        ab_unit = ab / dist

        cos_a = float(np.dot(ep_a.tangent, ab_unit))        # A outward → toward B
        cos_b = float(np.dot(ep_b.tangent, -ab_unit))       # B outward → toward A
        score = (max(cos_a, 0.0) + max(cos_b, 0.0)) / 2.0
        return W_CONTINUITY * score

    def proximity(self, ep_a: Endpoint, ep_b: Endpoint) -> float:
        dist = np.linalg.norm(ep_b.position - ep_a.position)
        score = math.exp(-4.0 * dist / self.img_diag)
        return W_PROXIMITY * score

    def closure(self, ep_a: Endpoint, ep_b: Endpoint) -> float:
        """Bonus only when the two endpoints belong to the same path AND
        the gap is large enough to be a real break (not a near-closed path)."""
        if (ep_a.path_index == ep_b.path_index
                and ep_a.role != ep_b.role):
            gap = np.linalg.norm(ep_b.position - ep_a.position)
            if gap >= MIN_SELF_GAP_PX:
                return W_CLOSURE
        return 0.0

    def symmetry(self, ep_a: Endpoint, ep_b: Endpoint,
                 axis: Optional[np.ndarray]) -> float:
        if axis is None:
            return 0.0
        pt0, n = axis[0], axis[1]
        disp_a = ep_a.position - pt0
        reflected_a = ep_a.position - 2.0 * np.dot(disp_a, n) * n
        dist = np.linalg.norm(reflected_a - ep_b.position)
        score = math.exp(-dist / max(self.img_diag * 0.05, 1.0))
        return W_SYMMETRY * score

    def similarity(self, curvature_a: float, curvature_b: float) -> float:
        diff = abs(curvature_a - curvature_b)
        max_val = max(curvature_a, curvature_b, 1e-6)
        score = 1.0 - min(diff / max_val, 1.0)
        return W_SIMILARITY * score

    def score(
        self,
        ep_a: Endpoint,
        ep_b: Endpoint,
        symmetry_axis: Optional[np.ndarray],
        curvature_a: float = 0.0,
        curvature_b: float = 0.0,
    ) -> Tuple[float, Dict[str, float], List[str]]:
        s = {
            "continuity" : self.continuity(ep_a, ep_b),
            "proximity"  : self.proximity(ep_a, ep_b),
            "closure"    : self.closure(ep_a, ep_b),
            "symmetry"   : self.symmetry(ep_a, ep_b, symmetry_axis),
            "similarity" : self.similarity(curvature_a, curvature_b),
        }
        total = sum(s.values())
        principles = [k for k, v in s.items() if v > 0.0]
        return total, s, principles


# ═══════════════════════════════════════════════════════════════════════════
# Bezier Bridge Generator  (BUG-2 fixed)
# ═══════════════════════════════════════════════════════════════════════════

class BezierBridgeGenerator:
    """
    Builds G¹-continuous cubic Bezier bridges.

    Tangent convention (OUTWARD = pointing away from the path):
      START endpoint: tA = P0 − P1
      END   endpoint: tA = P3 − P2

    Bridge geometry  (A → B):
      P0 = A_pos
      P1 = A_pos + α·tA          ← leaves A in its outward direction
      P2 = B_pos − β·arrive_dir  ← arrives at B in the correct direction
      P3 = B_pos

    arrive_dir derivation  (BUG-2):
      The bridge derivative at P3 is P3−P2 = β·arrive_dir.
      This must equal the PATH's tangent at B (pointing inward into B):
        If B is "start": path goes inward = −(B.outward) → arrive_dir = −tB
        If B is "end":   path was going outward at end = tB → arrive_dir = +tB
    """

    def build(
        self,
        ep_a: Endpoint,
        ep_b: Endpoint,
        alpha_fraction: float = BRIDGE_ALPHA_FRACTION,
    ) -> Optional[BezierSegment]:
        A = ep_a.position.astype(float)
        B = ep_b.position.astype(float)
        chord = np.linalg.norm(B - A)
        if chord < 1e-6:
            return None

        tA = ep_a.tangent.copy()                      # outward at A

        # ── FIX BUG-2: correct arrive_dir sign ─────────────────────────
        if ep_b.role == "start":
            arrive_dir = -ep_b.tangent.copy()         # -(outward) = inward
        else:
            arrive_dir = ep_b.tangent.copy()          # outward at end = path direction

        alpha = alpha_fraction * chord
        beta  = alpha_fraction * chord

        cp = np.vstack([
            A,
            A + alpha * tA,
            B - beta  * arrive_dir,
            B,
        ])
        return BezierSegment(control_points=cp, source_type="bridge")


# ═══════════════════════════════════════════════════════════════════════════
# EFD Contour Completion  (BUG-10 fixed)
# ═══════════════════════════════════════════════════════════════════════════

class EFDCompletionEngine:

    def __init__(self, harmonics=EFD_HARMONICS, recon_pts=EFD_RECONSTRUCTION_PTS):
        self.harmonics  = harmonics
        self.recon_pts  = recon_pts

    def complete(
        self, open_arc: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str, float, Dict[str, float]]:
        if len(open_arc) < 5:
            return None, "failed", 0.0, {}

        sym = self._try_symmetry(open_arc)
        if sym is not None:
            comp, score = sym
            return comp, "symmetry", float(score), {"symmetry_score": score}

        efd = self._try_efd(open_arc)
        if efd is not None:
            comp, score = efd
            return comp, "efd", float(score), {"efd_score": score}

        return None, "failed", 0.0, {}

    # ── symmetry strategy ───────────────────────────────────────────────────

    def _try_symmetry(
        self, arc: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        centre = arc.mean(axis=0)
        cov = np.cov((arc - centre).T)
        if cov.ndim < 2:
            return None
        _, eigvecs = np.linalg.eigh(cov)
        normal = np.array([-eigvecs[1, -1], eigvecs[0, -1]])

        displaced  = arc - centre
        proj       = (displaced @ normal.reshape(2, 1)) * normal
        reflected  = arc - 2.0 * proj

        D = cdist(reflected, arc)
        min_dists = D.min(axis=1)
        median_step = max(np.median(np.linalg.norm(np.diff(arc, axis=0), axis=1)) * 3, 5.0)
        sym_score = float((min_dists < median_step).mean())

        if sym_score < SYMMETRY_SCORE_THRESHOLD:
            return None

        # Build completion: reflected arc reversed so it flows arc[-1] → arc[0].
        raw = reflected[::-1]

        # ── FIX BUG-10: pin endpoints to exact arc termini ─────────────────
        completion = self._smooth_arc_pinned(raw, arc[-1], arc[0],
                                             n_pts=max(len(arc) // 2, 20))
        return completion, sym_score

    # ── EFD strategy ────────────────────────────────────────────────────────

    def _try_efd(
        self, arc: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        closed_arc = np.vstack([arc, arc[0]])
        recon, _ = reconstruct_contour_efd(
            closed_arc, order=self.harmonics, num_points=self.recon_pts
        )
        if recon is None or len(recon) < 10:
            return None

        D = cdist(recon, arc)
        min_dists = D.min(axis=1)
        median_d  = np.median(min_dists)
        gap_mask  = min_dists > max(median_d * 1.5, 3.0)

        if not gap_mask.any():
            arc_mid = arc.mean(axis=0)
            dists_from_mid = np.linalg.norm(recon - arc_mid, axis=1)
            far_pts = recon[dists_from_mid > dists_from_mid.mean()]
            if len(far_pts) < 3:
                return None
            comp = self._smooth_arc_pinned(far_pts, arc[-1], arc[0],
                                           n_pts=max(len(arc) // 2, 20))
            return comp, 0.45

        gap_idx  = np.where(gap_mask)[0]
        gap_arc  = self._extract_longest_run(recon, gap_idx)
        if gap_arc is None or len(gap_arc) < 3:
            return None

        gap_fraction   = gap_mask.mean()
        d_start        = np.linalg.norm(gap_arc[0]  - arc[-1])
        d_end          = np.linalg.norm(gap_arc[-1] - arc[0])
        endpoint_qual  = math.exp(-0.05 * (d_start + d_end))
        score = (1.0 - min(gap_fraction, 0.5) / 0.5) * 0.6 + endpoint_qual * 0.4
        score = float(max(0.1, min(score, 0.95)))

        # Pin to exact arc termini (BUG-10)
        comp = self._smooth_arc_pinned(gap_arc, arc[-1], arc[0],
                                       n_pts=max(int(len(gap_arc) * 0.8), 10))
        return comp, score

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_longest_run(
        recon: np.ndarray, indices: np.ndarray
    ) -> Optional[np.ndarray]:
        if len(indices) == 0:
            return None
        n = len(recon)
        sorted_idx = np.sort(indices)
        diffs  = np.diff(sorted_idx)
        breaks = np.where(diffs > 1)[0]
        if len(breaks) == 0:
            return recon[sorted_idx]
        runs, prev = [], 0
        for b in breaks:
            runs.append(sorted_idx[prev:b + 1]); prev = b + 1
        runs.append(sorted_idx[prev:])
        if sorted_idx[-1] - sorted_idx[0] < n - 1:
            wrap = np.concatenate([runs[-1], runs[0]])
            runs = [wrap] + runs[1:-1]
        return recon[max(runs, key=len) % n]

    @staticmethod
    def _smooth_arc(arc: np.ndarray, n_pts: int) -> np.ndarray:
        if len(arc) < 2:
            return arc
        diffs  = np.linalg.norm(np.diff(arc, axis=0), axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(diffs)])
        total  = cumlen[-1]
        if total < 1e-9:
            return arc[:1].repeat(n_pts, axis=0)
        t_orig = cumlen / total
        t_new  = np.linspace(0.0, 1.0, n_pts)
        return np.column_stack([np.interp(t_new, t_orig, arc[:, 0]),
                                 np.interp(t_new, t_orig, arc[:, 1])])

    @classmethod
    def _smooth_arc_pinned(
        cls,
        arc: np.ndarray,
        pin_start: np.ndarray,
        pin_end: np.ndarray,
        n_pts: int,
    ) -> np.ndarray:
        """Resample arc and force exact start/end points. (FIX BUG-10)"""
        smooth = cls._smooth_arc(arc, n_pts)
        if len(smooth) < 2:
            return smooth
        smooth[0]  = pin_start.astype(float)
        smooth[-1] = pin_end.astype(float)
        return smooth


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint Extraction  (BUG-1, BUG-4, BUG-5 fixed)
# ═══════════════════════════════════════════════════════════════════════════

def _path_arc_length(path: BezierPath) -> float:
    """Approximate arc length by sampling."""
    pts = path.sample(pts_per_segment=20)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _extract_endpoints(
    paths: List[BezierPath],
    adjacency: Dict[int, set],
    min_path_len: float = MIN_PATH_LEN_PX,
) -> List[Endpoint]:
    """
    Return one Endpoint per free end of every qualifying open path.

    OUTWARD tangent convention  (FIX BUG-1):
      START: P0 − P1  (pointing away from the curve's first segment)
      END:   P3 − P2  (pointing away from the curve's last  segment)

    Filtering (FIX BUG-4):
      Paths shorter than *min_path_len* pixels are excluded; they are
      skeleton noise, stub artefacts, or corner pixels — not real strokes.

    Degree filter (FIX BUG-5):
      An endpoint at a skeleton node that connects to ≥ 2 other paths
      (i.e. it is in *adjacency*) is a JUNCTION, not a free end.
      We emit it only if the adjacency set at that endpoint is empty or
      the endpoint is not shared with any other qualifying path.
    """
    endpoints: List[Endpoint] = []

    for idx, path in enumerate(paths):
        if path.is_closed or not path.segments:
            continue

        # ── BUG-4: skip stub paths ─────────────────────────────────────
        if _path_arc_length(path) < min_path_len:
            continue

        first = path.segments[0].control_points    # (4, 2)
        last  = path.segments[-1].control_points   # (4, 2)

        # ── BUG-1: outward tangents ────────────────────────────────────
        t_start_raw = first[0] - first[1]          # P0 − P1  (OUTWARD)
        t_end_raw   = last[3]  - last[2]            # P3 − P2  (OUTWARD)

        def _unit(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v / n if n > 1e-9 else np.array([1.0, 0.0])

        t_start = _unit(t_start_raw)
        t_end   = _unit(t_end_raw)

        # ── BUG-5: junction filter ─────────────────────────────────────
        # Paths in adjacency[idx] share a skeleton node with path idx.
        # If ALL neighbours of idx are also qualifying long paths, then
        # both of our endpoints might be junctions.  We check which role
        # is "free" by seeing if the corresponding geometric end of the
        # path is close to any neighbour's endpoint.
        adj_neighbours = adjacency.get(idx, set())

        start_is_junction = _endpoint_is_junction(
            first[0], idx, adj_neighbours, paths, role="start"
        )
        end_is_junction = _endpoint_is_junction(
            last[3], idx, adj_neighbours, paths, role="end"
        )

        if not start_is_junction:
            endpoints.append(Endpoint(
                path_index=idx, role="start",
                position=first[0].copy(), tangent=t_start,
            ))
        if not end_is_junction:
            endpoints.append(Endpoint(
                path_index=idx, role="end",
                position=last[3].copy(), tangent=t_end,
            ))

    return endpoints


def _endpoint_is_junction(
    pos: np.ndarray,
    path_idx: int,
    neighbours: set,
    paths: List[BezierPath],
    role: str,
    snap_radius: float = 4.0,
) -> bool:
    """
    Return True if *pos* is geometrically close to an endpoint of any
    neighbouring path (meaning it is a skeleton junction, not a free end).
    """
    for nidx in neighbours:
        if nidx >= len(paths) or not paths[nidx].segments:
            continue
        npath = paths[nidx]
        n_start = npath.segments[0].control_points[0]
        n_end   = npath.segments[-1].control_points[3]
        if (np.linalg.norm(pos - n_start) < snap_radius or
                np.linalg.norm(pos - n_end) < snap_radius):
            return True
    return False


def _estimate_local_curvature(path: BezierPath, role: str) -> float:
    seg = path.segments[0] if role == "start" else path.segments[-1]
    cp  = seg.control_points
    v1  = cp[1] - cp[0] if role == "start" else cp[2] - cp[1]
    v2  = cp[2] - cp[1] if role == "start" else cp[3] - cp[2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(math.acos(cos_a))


# ═══════════════════════════════════════════════════════════════════════════
# Symmetry Axis Detection
# ═══════════════════════════════════════════════════════════════════════════

def _detect_global_symmetry_axis(
    paths: List[BezierPath],
) -> Optional[np.ndarray]:
    all_pts = [p.sample(pts_per_segment=20) for p in paths if p.segments]
    if not all_pts:
        return None
    pts = np.vstack(all_pts)
    if len(pts) < 10:
        return None
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    if cov.ndim < 2:
        return None
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, 1]
    normal = np.array([-principal[1], principal[0]])

    displaced  = pts - centre
    proj       = (displaced @ normal.reshape(2, 1)) * normal
    reflected  = pts - 2.0 * proj

    D = cdist(reflected, pts)
    min_d = D.min(axis=1)
    threshold = max(np.median(np.linalg.norm(np.diff(pts[:200], axis=0), axis=1)) * 4, 10.0)
    if float((min_d < threshold).mean()) < 0.55:
        return None

    return np.array([centre, normal])


# ═══════════════════════════════════════════════════════════════════════════
# Gap Classification
# ═══════════════════════════════════════════════════════════════════════════

def _classify_open_contour(path: BezierPath) -> Tuple[float, str]:
    pts = path.sample(pts_per_segment=40)
    if len(pts) < 2:
        return 0.0, "small_gap"
    arc_len    = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    gap_chord  = float(np.linalg.norm(pts[-1] - pts[0]))
    full_perim = arc_len + gap_chord
    ratio      = gap_chord / max(full_perim, 1.0)
    return ratio, ("large_gap" if ratio > EFD_OPEN_CONTOUR_THRESHOLD else "small_gap")


# ═══════════════════════════════════════════════════════════════════════════
# Main Restoration Engine
# ═══════════════════════════════════════════════════════════════════════════

class RestorationEngine:
    """
    Orchestrates preprocessing → EFD completion → Bezier bridging → XAI.

    All four original bugs that caused mis-connections are fixed here.
    """

    def __init__(
        self,
        efd_harmonics   : int   = EFD_HARMONICS,
        max_bridge_dist : float = MAX_BRIDGE_DIST_FRACTION,
        tangent_lookah  : int   = 7,
        bezier_max_err  : float = 5.0,
        min_path_len    : float = MIN_PATH_LEN_PX,
    ):
        self.efd_harmonics   = efd_harmonics
        self.max_bridge_dist = max_bridge_dist
        self.tangent_lookah  = tangent_lookah
        self.bezier_max_err  = bezier_max_err
        self.min_path_len    = min_path_len

        self._bridge_gen = BezierBridgeGenerator()
        self._efd_engine = EFDCompletionEngine(harmonics=efd_harmonics)

    # ── main entry point ────────────────────────────────────────────────────

    def restore(
        self,
        image_path : str,
        output_dir : str = "restored",
    ) -> RestorationResult:
        print(f"\n{'═'*70}")
        print(f"  RESTORATION ENGINE  ·  {os.path.basename(image_path)}")
        print(f"{'═'*70}\n")

        img_bgr  = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w     = img_gray.shape
        img_diag = math.hypot(w, h)
        max_bridge_px = self.max_bridge_dist * img_diag

        print(f"  Image {w}×{h}  diag={img_diag:.0f}px  "
              f"max_bridge={max_bridge_px:.0f}px")

        # ── 1. Extract Bezier paths ─────────────────────────────────────────
        print("\n  [1/4] Extracting skeleton paths …")
        paths, adjacency = fit_from_image_skeleton(
            image_path,
            max_error=self.bezier_max_err,
            tangent_lookahead=self.tangent_lookah,
        )
        n_open = sum(1 for p in paths if not p.is_closed)
        print(f"        → {len(paths)} paths  ({n_open} open)")

        # ── 2. Classify open paths ──────────────────────────────────────────
        print("\n  [2/4] Classifying open paths …")
        large_gap_paths, small_gap_paths = [], []
        for i, path in enumerate(paths):
            if path.is_closed:
                continue
            if _path_arc_length(path) < self.min_path_len:
                continue                                # BUG-4: ignore stubs
            ratio, cat = _classify_open_contour(path)
            (large_gap_paths if cat == "large_gap" else small_gap_paths).append(i)
            if cat == "large_gap":
                print(f"        Path {i}: large gap ({ratio:.1%})")

        sym_axis = _detect_global_symmetry_axis(paths)
        if sym_axis is not None:
            print("        → Bilateral symmetry axis detected")

        gestalt = GestaltEngine(image_diagonal=img_diag)

        # ── 3. EFD completions ──────────────────────────────────────────────
        print("\n  [3/4] EFD completion for large-gap paths …")
        decisions       : List[RestorationDecision] = []
        efd_completions : List[np.ndarray]           = []

        for idx in large_gap_paths:
            arc = paths[idx].sample(pts_per_segment=50)
            completion, method, conf, scores = self._efd_engine.complete(arc)

            if completion is None:
                small_gap_paths.append(idx)
                print(f"        Path {idx}: EFD failed — demoted to bridging")
                continue

            efd_completions.append(completion)
            principles = (["symmetry", "continuity"] if method == "symmetry"
                          else ["closure", "continuity"])

            if method == "symmetry":
                expl = (
                    f"Path {idx} has a large gap. Bilateral symmetry "
                    f"(score={scores.get('symmetry_score',0):.0%}) was used to "
                    f"reflect the surviving arc across its principal axis, "
                    f"inferring the missing segment."
                )
            else:
                expl = (
                    f"Path {idx} has a large gap. EFD reconstruction "
                    f"({self.efd_harmonics} harmonics) identified the gap region "
                    f"and extracted a smooth closing arc guided by the "
                    f"Gestalt closure principle."
                )

            decisions.append(RestorationDecision(
                decision_type      = f"efd_completion_{method}",
                confidence         = conf,
                gestalt_principles = principles,
                score_breakdown    = scores,
                explanation        = expl,
                restored_curve     = completion.astype(np.int32),
            ))

        # ── 4. Bezier bridging ──────────────────────────────────────────────
        print("\n  [4/4] Bezier gap bridging …")

        # Extract FREE endpoints only (BUG-4, BUG-5 fixed inside)
        endpoints = _extract_endpoints(
            paths, adjacency, min_path_len=self.min_path_len
        )
        # Restrict to small-gap / failed-EFD paths
        relevant = set(small_gap_paths) | set(large_gap_paths)
        endpoints = [ep for ep in endpoints if ep.path_index in relevant]
        print(f"        Free endpoints found: {len(endpoints)}")

        curvatures = {
            (ep.path_index, ep.role): _estimate_local_curvature(
                paths[ep.path_index], ep.role
            )
            for ep in endpoints
        }

        # Build & score all candidate pairs
        candidates = []
        n = len(endpoints)
        for i in range(n):
            for j in range(i + 1, n):
                ep_a, ep_b = endpoints[i], endpoints[j]

                # ── BUG-9: symmetric adjacency check ───────────────────────
                adj_a = adjacency.get(ep_a.path_index, set())
                adj_b = adjacency.get(ep_b.path_index, set())
                if (ep_b.path_index in adj_a or ep_a.path_index in adj_b):
                    continue                            # already connected

                # Distance guard
                dist = float(np.linalg.norm(ep_b.position - ep_a.position))
                if dist < 2.0 or dist > max_bridge_px:
                    continue

                # ── BUG-3 & BUG-8: collinearity test on outward tangents ───
                # Anti-parallel outward tangents (broken stroke) → abs_dot ≈ 1
                abs_dot   = abs(float(np.dot(ep_a.tangent, ep_b.tangent)))
                align_deg = math.degrees(math.acos(min(abs_dot, 1.0)))
                if align_deg > MAX_GAP_ANGLE_DEG:
                    continue                            # tangents not collinear

                # Secondary check: at least one endpoint "faces" the other
                ab_unit = (ep_b.position - ep_a.position) / max(dist, 1e-9)
                cos_a   = float(np.dot(ep_a.tangent,  ab_unit))
                cos_b   = float(np.dot(ep_b.tangent, -ab_unit))
                if max(cos_a, cos_b) < MIN_DIRECTION_TOWARD_COS:
                    continue

                curv_a = curvatures.get((ep_a.path_index, ep_a.role), 0.0)
                curv_b = curvatures.get((ep_b.path_index, ep_b.role), 0.0)
                total, breakdown, principles = gestalt.score(
                    ep_a, ep_b, sym_axis, curv_a, curv_b
                )
                candidates.append((total, i, j, breakdown, principles))

        candidates.sort(key=lambda x: -x[0])

        bridge_segments : List[BezierSegment] = []

        # ── BUG-7: track used endpoints by (path_index, role) ─────────────
        used_ep_keys: Set[Tuple[int, str]] = set()

        max_score = W_CONTINUITY + W_PROXIMITY + W_CLOSURE + W_SYMMETRY + W_SIMILARITY

        for total, i, j, breakdown, principles in candidates:
            ep_a = endpoints[i]
            ep_b = endpoints[j]

            key_a = (ep_a.path_index, ep_a.role)
            key_b = (ep_b.path_index, ep_b.role)

            if key_a in used_ep_keys or key_b in used_ep_keys:
                continue

            bridge = self._bridge_gen.build(ep_a, ep_b)
            if bridge is None:
                continue

            conf  = float(min(total / max_score, 1.0))
            bpts  = bridge.sample(BRIDGE_SAMPLE_N).astype(np.int32)
            dist  = float(np.linalg.norm(ep_b.position - ep_a.position))

            # Collect alternatives
            alts = []
            for t2, ai, aj, _, _ in candidates:
                if len(alts) >= 3:
                    break
                epa2 = endpoints[ai]; epb2 = endpoints[aj]
                if ((ai == i or aj == i) and aj != j and ai != j) or \
                   ((ai == j or aj == j) and aj != i and ai != i):
                    alts.append({"partner_path": epb2.path_index
                                 if ai == i else epa2.path_index,
                                 "score": round(t2, 3)})

            expl = (
                f"G¹ bridge: path {ep_a.path_index}({ep_a.role}) → "
                f"path {ep_b.path_index}({ep_b.role}), "
                f"gap={dist:.1f}px. "
                f"Outward tangents are {math.degrees(math.acos(min(abs(float(np.dot(ep_a.tangent,ep_b.tangent))),1.0))):.0f}° "
                f"apart (collinear threshold={MAX_GAP_ANGLE_DEG}°). "
                f"Gestalt: {', '.join(principles) or 'none'}. "
                f"Scores: cont={breakdown.get('continuity',0):.2f}, "
                f"prox={breakdown.get('proximity',0):.2f}, "
                f"clos={breakdown.get('closure',0):.2f}."
            )

            decisions.append(RestorationDecision(
                decision_type      = "bezier_bridge",
                confidence         = conf,
                gestalt_principles = principles,
                score_breakdown    = {k: round(v, 3) for k, v in breakdown.items()},
                explanation        = expl,
                restored_curve     = bpts,
                alternatives       = alts,
            ))

            bridge_segments.append(bridge)
            used_ep_keys.add(key_a)
            used_ep_keys.add(key_b)

            print(
                f"        Bridge p{ep_a.path_index}({ep_a.role}) ↔ "
                f"p{ep_b.path_index}({ep_b.role})  "
                f"dist={dist:.0f}px  conf={conf:.2f}  "
                f"align={math.degrees(math.acos(min(abs(float(np.dot(ep_a.tangent,ep_b.tangent))),1.0))):.0f}°"
            )

        print(f"\n  Bridges: {len(bridge_segments)}  "
              f"EFD completions: {len(efd_completions)}  "
              f"Decisions: {len(decisions)}")

        restored = self._render(img_bgr, paths, bridge_segments,
                                efd_completions, decisions)

        return RestorationResult(
            original_image   = img_bgr,
            restored_canvas  = restored,
            original_paths   = paths,
            bridge_segments  = bridge_segments,
            efd_completions  = efd_completions,
            decisions        = decisions,
            output_dir       = output_dir,
            image_name       = os.path.basename(image_path),
        )

    # ── renderer ────────────────────────────────────────────────────────────

    def _render(
        self,
        img_bgr         : np.ndarray,
        paths           : List[BezierPath],
        bridges         : List[BezierSegment],
        efd_completions : List[np.ndarray],
        decisions       : List[RestorationDecision],
    ) -> np.ndarray:
        canvas = img_bgr.copy()

        # Original paths — sky blue
        for path in paths:
            for seg in path.segments:
                pts = seg.sample(60).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], False, (247, 195, 79), 1, cv2.LINE_AA)

        # EFD / symmetry completions
        efd_decs = [d for d in decisions if "efd_completion" in d.decision_type]
        for d, arc in zip(efd_decs, efd_completions):
            colour = (234, 222, 128) if "symmetry" in d.decision_type else (216, 147, 206)
            pts = arc.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, colour, 2, cv2.LINE_AA)

        # Bezier bridges — confidence coloured
        bridge_decs = [d for d in decisions if d.decision_type == "bezier_bridge"]
        for d, bridge in zip(bridge_decs, bridges):
            c = d.confidence
            colour = ((174, 240, 105) if c >= HIGH_CONF_THRESHOLD else
                      (0, 255, 255)   if c >= MEDIUM_CONF_THRESHOLD else
                      (64, 110, 255))
            pts = bridge.sample(BRIDGE_SAMPLE_N).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, colour, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(bridge.start.astype(int)), 3,
                       (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(bridge.end.astype(int)),   3,
                       (255, 255, 255), -1, cv2.LINE_AA)

        return canvas


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Heritage sketch restoration — Bezier + EFD + Gestalt + XAI"
    )
    p.add_argument("image_path")
    p.add_argument("--output-dir",        default="restored")
    p.add_argument("--efd-harmonics",     type=int,   default=EFD_HARMONICS)
    p.add_argument("--max-bridge-dist",   type=float, default=MAX_BRIDGE_DIST_FRACTION)
    p.add_argument("--bezier-max-error",  type=float, default=5.0)
    p.add_argument("--tangent-lookahead", type=int,   default=7)
    p.add_argument("--min-path-len",      type=float, default=MIN_PATH_LEN_PX)
    p.add_argument("--no-report",         action="store_true")
    args = p.parse_args()

    engine = RestorationEngine(
        efd_harmonics   = args.efd_harmonics,
        max_bridge_dist = args.max_bridge_dist,
        bezier_max_err  = args.bezier_max_error,
        tangent_lookah  = args.tangent_lookahead,
        min_path_len    = args.min_path_len,
    )
    result = engine.restore(args.image_path, output_dir=args.output_dir)
    result.save_visuals()
    if not args.no_report:
        result.report()


if __name__ == "__main__":
    _cli()