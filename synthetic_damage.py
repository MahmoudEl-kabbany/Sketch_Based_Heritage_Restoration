"""Synthetic damage with known ground truth: typed gaps on a clean drawing, then degradation.

For every clean line drawing, make_variant() produces three images and a record:

    clean.png    the drawing as given
    gapped.png   the drawing with gaps erased along its strokes, nothing else
    damaged.png  gapped.png after degradation (stains, fading, uneven light, noise, blur, JPEG)
    gaps.json    one entry per gap: type, centre, the two (or more) cut points,
                 the clean centre line across the gap, and its length

Ground truth is computed from (clean, gapped): stains or faded ink would
otherwise count as missing strokes or fail the alignment check. The pipeline
is run on damaged.png.

Gap types, chosen on the clean skeleton graph (sknw):
    straight   a stretch of stroke that turns by under 10 degrees
    curved     one that turns by 15-50 degrees
    corner     one that turns by more than 50 degrees
    junction   a disk erased over a node where three or more strokes meet

Gap lengths and degradation strength depend on the difficulty level
(DIFFICULTY). Everything is deterministic given the seed.

Used by scripts/make_synthetic.py (generate a dataset), scripts/evaluate_synthetic.py
(evaluate on it) and train_scoring.py --extra-data (train on it).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import sknw

from restoration.preprocess import stroke_geometry_of

GAP_TYPES = ("straight", "curved", "corner", "junction")
# Share of the gaps of each type that a variant aims for (types without sites are skipped).
GAP_TYPE_SHARES = {"straight": 0.35, "curved": 0.30, "corner": 0.20, "junction": 0.15}
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class Difficulty:
    gap_widths: Tuple[float, float]   # gap length range, in stroke widths
    fade: float                       # ink darkness kept (1 = unchanged)
    texture: float                    # paper texture amplitude (0-1 of white)
    stains: int                       # number of stains
    stain_darkness: Tuple[float, float]
    illumination: float               # strength of the brightness gradient across the page
    blur_sigma: float                 # Gaussian blur, px
    noise_sigma: float                # Gaussian noise, grey levels
    salt_pepper: float                # share of pixels set to black or white
    jpeg_quality: Optional[int]       # None = no JPEG round trip


DIFFICULTY: Dict[str, Difficulty] = {
    "easy": Difficulty((1.0, 3.0), 1.0, 0.0, 0, (0.0, 0.0), 0.0, 0.0, 2.0, 0.0, None),
    "medium": Difficulty((1.5, 5.0), 0.8, 0.04, 1, (0.10, 0.20), 0.12, 0.5, 5.0, 0.0, 85),
    "hard": Difficulty((2.0, 7.0), 0.6, 0.08, 3, (0.15, 0.35), 0.25, 0.9, 9.0, 0.001, 65),
}


def split_of(drawing: str) -> str:
    """Train / val / test, fixed per source drawing (never per variant), from a hash of its name."""
    value = int(hashlib.sha256(drawing.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, share in SPLITS:
        cumulative += share
        if value <= cumulative:
            return name
    return SPLITS[-1][0]


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


# ═══════════════════════════════════════════════════════════════════════════
# Gap sites on the clean skeleton
# ═══════════════════════════════════════════════════════════════════════════

def _arc(points: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def _at(points: np.ndarray, arc: np.ndarray, s: float) -> np.ndarray:
    return np.array([np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])])


def _stretch(points: np.ndarray, arc: np.ndarray, s0: float, s1: float) -> np.ndarray:
    inside = points[(arc > s0) & (arc < s1)]
    return np.vstack([_at(points, arc, s0), inside, _at(points, arc, s1)])


def _turn_deg(points: np.ndarray, arc: np.ndarray, s: float, h: float) -> float:
    a, b, c = _at(points, arc, s - h), _at(points, arc, s), _at(points, arc, s + h)
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def gap_sites(clean: np.ndarray) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
    """Candidate gap centres of each type on the clean drawing, and its line width."""
    geometry = stroke_geometry_of(clean)
    width = max(geometry.line_width, 1.0)
    graph = sknw.build_sknw(geometry.skeleton.astype(np.uint8))
    sites: Dict[str, List[Dict[str, Any]]] = {t: [] for t in GAP_TYPES}

    step = max(width, 3.0)
    h = max(2.0 * width, 6.0)
    for u, v, data in graph.edges(data=True):
        pts = np.asarray(data.get("pts", []), dtype=np.float64)
        if len(pts) < 4:
            continue
        points = pts[:, ::-1]                       # (row, col) -> (x, y)
        arc = _arc(points)
        length = float(arc[-1])
        margin = h + 2.0 * width                    # keep clear of the edge's nodes
        for s in np.arange(margin, length - margin, step):
            turn = _turn_deg(points, arc, s, h)
            kind = "straight" if turn < 10 else "curved" if 15 <= turn <= 50 else "corner" if turn > 50 else None
            if kind:
                sites[kind].append({"points": points, "arc": arc, "s": float(s), "edge_length": length})
    for node in graph.nodes():
        if graph.degree(node) >= 3:
            centre = np.asarray(graph.nodes[node]["o"], dtype=np.float64)[::-1]
            branches = []
            for nbr in graph.neighbors(node):
                for data in (graph[node][nbr].values() if graph.is_multigraph() else [graph[node][nbr]]):
                    branches.append(np.asarray(data["pts"], dtype=np.float64)[:, ::-1])
            sites["junction"].append({"centre": centre, "branches": branches})
    return sites, width


# ═══════════════════════════════════════════════════════════════════════════
# One damaged variant
# ═══════════════════════════════════════════════════════════════════════════

def _background(clean: np.ndarray) -> int:
    light = clean[clean > 128]
    return int(np.median(light)) if light.size else 255


def _erase_stretch(img: np.ndarray, site: Dict[str, Any], length: float, width: float, bg: int) -> Optional[Dict[str, Any]]:
    points, arc, s = site["points"], site["arc"], site["s"]
    if length > 0.5 * site["edge_length"]:
        return None
    thickness = int(np.ceil(width + 2.0))           # radius >= half the stroke width + 1
    cap = thickness / 2.0                           # cv2's thick lines have round caps
    half_line = max(length / 2.0 - cap, 0.5)
    line = _stretch(points, arc, s - half_line, s + half_line)
    cv2.polylines(img, [np.round(line).astype(np.int32).reshape(-1, 1, 2)], False, int(bg), thickness, cv2.LINE_8)
    across = _stretch(points, arc, s - length / 2.0, s + length / 2.0)
    return {
        "centre": _at(points, arc, s).round(2).tolist(),
        "cut_points": [across[0].round(2).tolist(), across[-1].round(2).tolist()],
        "centre_line": across.round(2).tolist(),
        "length_px": round(float(length), 2),
    }


def _erase_junction(img: np.ndarray, site: Dict[str, Any], length: float, width: float, bg: int) -> Dict[str, Any]:
    centre = site["centre"]
    radius = max(length / 2.0, width / 2.0 + 1.0)
    cv2.circle(img, tuple(int(round(c)) for c in centre), int(np.ceil(radius + width / 2.0 + 1.0)), int(bg), -1)
    cuts = []
    for branch in site["branches"]:
        d = np.linalg.norm(branch - centre, axis=1)
        outside = np.flatnonzero(d >= radius + width / 2.0 + 1.0)
        if len(outside):
            cuts.append(branch[outside[np.argmin(d[outside])]].round(2).tolist())
    return {
        "centre": centre.round(2).tolist(),
        "cut_points": cuts,
        "centre_line": [centre.round(2).tolist()],
        "length_px": round(float(2.0 * radius), 2),
    }


def apply_typed_gaps(
    clean: np.ndarray, difficulty: str, seed: int, gap_count: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
    """Erase typed gaps from *clean*. Returns (gapped image, gap records, line width)."""
    level = DIFFICULTY[difficulty]
    rng = np.random.default_rng(seed)
    sites, width = gap_sites(clean)
    bg = _background(clean)
    if gap_count is None:
        total_sites = sum(len(v) for v in sites.values())
        gap_count = int(np.clip(total_sites * max(width, 3.0) / (40.0 * width), 3, 25)) if total_sites else 0

    gapped = clean.copy()
    records: List[Dict[str, Any]] = []
    centres: List[Tuple[np.ndarray, float]] = []
    types = [t for t in GAP_TYPES if sites[t]]
    if not types:
        return gapped, records, width
    shares = np.array([GAP_TYPE_SHARES[t] for t in types])
    attempts = 0
    while len(records) < gap_count and attempts < 40 * max(gap_count, 1):
        attempts += 1
        kind = types[int(rng.choice(len(types), p=shares / shares.sum()))]
        site = sites[kind][int(rng.integers(len(sites[kind])))]
        length = float(rng.uniform(*level.gap_widths)) * width
        length = max(length, 3.0)
        centre = site["centre"] if kind == "junction" else _at(site["points"], site["arc"], site["s"])
        if any(np.linalg.norm(centre - c) < (length + other) / 2.0 + 4.0 * width for c, other in centres):
            continue
        record = (_erase_junction if kind == "junction" else _erase_stretch)(gapped, site, length, width, bg)
        if record is None:
            continue
        record = {"id": len(records) + 1, "type": kind, **record}
        records.append(record)
        centres.append((np.asarray(record["centre"]), record["length_px"]))
    return gapped, records, width


def degrade(gapped: np.ndarray, difficulty: str, seed: int) -> np.ndarray:
    """Age a clean (gapped) drawing: fading, paper texture, stains, uneven light, blur, noise, JPEG."""
    level = DIFFICULTY[difficulty]
    rng = np.random.default_rng(seed)
    h, w = gapped.shape
    img = gapped.astype(np.float32) / 255.0

    if level.fade < 1.0:
        img = 1.0 - (1.0 - img) * level.fade
    if level.texture > 0:
        texture = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
        texture = (texture - texture.min()) / max(float(np.ptp(texture)), 1e-6)
        img = img - level.texture * texture
    for _ in range(level.stains):
        mask = np.zeros((h, w), np.float32)
        axes = (int(rng.uniform(0.05, 0.25) * min(h, w)), int(rng.uniform(0.05, 0.25) * min(h, w)))
        centre = (int(rng.integers(w)), int(rng.integers(h)))
        cv2.ellipse(mask, centre, axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(axes) / 3.0 + 1.0)
        img = img * (1.0 - float(rng.uniform(*level.stain_darkness)) * mask)
    if level.illumination > 0:
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        plane = np.cos(angle) * xx / max(w, 1) + np.sin(angle) * yy / max(h, 1)
        plane = (plane - plane.min()) / max(float(np.ptp(plane)), 1e-6)
        img = img * (1.0 - level.illumination * plane)
    if level.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), level.blur_sigma)
    if level.noise_sigma > 0:
        img = img + rng.normal(0.0, level.noise_sigma / 255.0, img.shape).astype(np.float32)
    if level.salt_pepper > 0:
        flips = rng.random(img.shape) < level.salt_pepper
        img[flips] = rng.integers(0, 2, int(flips.sum())).astype(np.float32)
    out = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if level.jpeg_quality:
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(level.jpeg_quality)])
        out = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return out


def make_variant(clean: np.ndarray, drawing: str, variant: int, difficulty: str, seed: int = 0) -> Dict[str, Any]:
    """Gapped and damaged images plus the gap records of one variant of a drawing."""
    gap_seed = _seed("gaps", seed, drawing, variant, difficulty)
    gapped, gaps, width = apply_typed_gaps(clean, difficulty, gap_seed)
    damaged = degrade(gapped, difficulty, _seed("degrade", seed, drawing, variant, difficulty))
    return {"gapped": gapped, "damaged": damaged, "gaps": gaps, "line_width": float(width)}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset on disk
# ═══════════════════════════════════════════════════════════════════════════

def write_dataset(
    sources: Sequence[Tuple[str, str]],
    out_dir: str,
    variants: int = 2,
    difficulties: Sequence[str] = tuple(DIFFICULTY),
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Write variants of every (drawing name, clean image path); return and save the manifest."""
    entries = []
    for drawing, path in sources:
        clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if clean is None:
            raise FileNotFoundError(path)
        split = split_of(drawing)
        drawing_dir = os.path.join(out_dir, split, drawing)
        os.makedirs(drawing_dir, exist_ok=True)
        clean_path = os.path.join(drawing_dir, "clean.png")
        cv2.imwrite(clean_path, clean)
        for difficulty in difficulties:
            for v in range(variants):
                result = make_variant(clean, drawing, v, difficulty, seed)
                vdir = os.path.join(drawing_dir, f"{difficulty}_{v}")
                os.makedirs(vdir, exist_ok=True)
                cv2.imwrite(os.path.join(vdir, "gapped.png"), result["gapped"])
                cv2.imwrite(os.path.join(vdir, "damaged.png"), result["damaged"])
                with open(os.path.join(vdir, "gaps.json"), "w") as f:
                    json.dump({"drawing": drawing, "difficulty": difficulty, "variant": v, "seed": seed,
                               "line_width": result["line_width"], "gaps": result["gaps"]}, f, indent=1)
                entries.append({
                    "drawing": drawing, "split": split, "difficulty": difficulty, "variant": v,
                    "clean": os.path.relpath(clean_path, out_dir).replace("\\", "/"),
                    "gapped": os.path.relpath(os.path.join(vdir, "gapped.png"), out_dir).replace("\\", "/"),
                    "damaged": os.path.relpath(os.path.join(vdir, "damaged.png"), out_dir).replace("\\", "/"),
                    "gaps": os.path.relpath(os.path.join(vdir, "gaps.json"), out_dir).replace("\\", "/"),
                    "gap_count": len(result["gaps"]),
                })
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump({"seed": seed, "variants": variants, "difficulties": list(difficulties), "entries": entries}, f, indent=1)
    return entries


def load_dataset(out_dir: str, split: Optional[str] = None, difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
    """Manifest entries of a dataset written by write_dataset, with absolute paths."""
    with open(os.path.join(out_dir, MANIFEST)) as f:
        entries = json.load(f)["entries"]
    out = []
    for e in entries:
        if (split and e["split"] != split) or (difficulty and e["difficulty"] != difficulty):
            continue
        out.append({**e, **{k: os.path.join(out_dir, e[k]) for k in ("clean", "gapped", "damaged", "gaps")}})
    return out
