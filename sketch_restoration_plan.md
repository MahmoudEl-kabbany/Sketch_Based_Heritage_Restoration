# Sketch Restoration: Improvement Plan (final, rebased on `accuracy_overhaul`)

> **Audience:** a Claude Code agent working in Mahmoud's repository, and Mahmoud.
> **Status:** finalised 2026-11-17. The first draft was written against `overhaul_test1` by a reviewer who did not run
> the code. This version is checked against `accuracy_overhaul`, which already did a large part of the first draft.
> Every "observed issue" below is still a hypothesis. Reproduce it before you fix it.

---

## 0. Read this first

### 0.1 Guiding principle

Do **not** replace the logical pipeline with an end-to-end model, and do **not** use an LLM or RAG for the geometry.
The direction is **neuro-symbolic**:

- **Keep what is guaranteed:** graph extraction, candidate generation, Bézier bridge construction, the ASP decision
  engine and its hard constraints, determinism, and the per-bridge explanation report.
- **Learn what is hand-tuned.** The scorer is already fitted (logistic scorecards). Next come preprocessing and context.
- **Add** synthetic training and evaluation data, interactive refinement through ASP constraints, and (Tier B)
  context-aware graph learning and semantic conditioning.

### 0.2 What `accuracy_overhaul` already provides

| Area | Already there | What it means for this plan |
|---|---|---|
| Change check | `check_changes.py` + `check_baseline.json`: unit tests, a CV refit, a held-out benchmark and a verdict | WP-0 adds only an exact snapshot, a determinism test and profiling |
| Ground-truth evaluation | `evaluate_gaps.py`: gap recall, off-drawing line, wrong strokes, harm on undamaged originals, per-gap stage (`endpoints_missing` / `no_candidate_generated` / `candidate_not_selected` / `repaired`) | These stages give the extraction miss rate and the candidate-recall ceiling. WP-4 extends this rather than duplicating it |
| Fitted scorer | `train_scoring.py` → `restoration/scoring_model.json`: smooth and corner scorecards, sign-constrained weights, CV by drawing, `scoring.candidate_features()` | This is "learned scorer v1". WP-5 becomes an ablation |
| ASP utility | `asp_engine.utility()`: log-odds above the threshold, `candidate(Id,A,B,Utility)`, positive candidates only, tie-break toward fewer bridges | First draft's WP-5.3 is done |
| Removed code | gates, veto, `_candidate_effective_utility`, `_merge_connected_paths`, `_gestalt_closure`, the crossing filters, `_sanitize_accepted_candidates` | First draft's WP-1 items 6–10 and 13 and WP-2 items 1, 3, 4 and 5 are obsolete |
| Performance | vectorized turning angles, compiled DTW, caching, optional figures, process-pool batch | WP-2 is reduced to what the profile still shows |
| Phase 6 | off by default (`efd_closure=False`). Report `schema_version` is `"3.0"` | Rename still to do (D2) |

Baseline (32 pairs, 784 true gaps, held-out models): gap recall **87.2%**, off-drawing line **89.4** line widths
(65.5 damaged + 23.9 undamaged), **5** wrong strokes, **23** strokes on undamaged originals, PSNR **26.06 dB**, SSIM **0.983**.

### 0.3 Repository layout

```
bezier_curves/bezier.py                 # skeleton → sknw graph → chains → Schneider fit; stroke_geometry()
eliptic_fourier_descriptors/efd.py      # standalone EFD demo (not used by the pipeline; leave as is)
restoration/
    extraction.py      # Phase 1 (free stroke ends, sizes in stroke radii)
    candidates.py      # Phase 2 (smooth_bridge, corner_bridge, endpoint_pairs)
    scoring.py         # Phase 3 (FEATURES, candidate_features, Scorecard, load_model, held_out_models)
    scoring_model.json # fitted scorecards
    asp_engine.py      # Phase 4 (utility, encode_facts, solve_candidates)
    asp_rules/restoration.lp
    synthesis.py       # Phase 5 (merge_restored_paths)
    efd_closure.py     # Phase 6 (to be renamed, see WP-1)
    pipeline.py        # Phase 7 + restore() / restore_batch() / map_in_processes()
    geometry.py, render_restored_sketch.py
add_gaps.py  train_scoring.py  evaluate_gaps.py  evaluate_restoration.py  evaluate_bridge_vs_original.py
check_changes.py  check_baseline.json  run_batch_test.py  test_restoration.py
docs/Phase_1 … Phase_7, docs/Schneider_Algorithm.md
test_images/          # untracked; 32 damaged/original pairs + difficult_test_cases/
```

### 0.4 Working rules

1. **One work package (WP) per branch/PR**, branched from the previous WP once it is merged. Do them in order.
2. **Behavioural changes go behind a parameter.** Defaults change only after `check_changes.py` shows a gain and
   Mahmoud approves.
3. **How a change is judged:** `python check_changes.py --output-dir <scratch>`. Fewer wrong strokes come first.
   A change is kept if the off-drawing line falls and recall does not, or recall rises and the off-drawing line does not.
   If they move in opposite directions, it is kept only when the off-drawing line falls ≥ 25% for ≤ 2 points of recall.
   **The model must be refitted after any change to extraction, candidates or measurements.** `check_changes.py` does this.
4. **Determinism is a hard requirement.** clingo is single-threaded with `--opt-strategy=usc`. Seed `numpy`,
   `random` and `torch`.
5. **Output safety (mandatory):**
   - Never write to `restoration/outputs/`. It is git-ignored and holds Mahmoud's results, which cannot be recovered.
     Always pass an explicit scratch `output_dir`.
   - Do not run `run_batch_test.py`, because it has no output option.
   - `evaluate_bridge_vs_original.py` writes `bridge_vs_original_metrics.png` into the cwd.
   - Run only one 8-worker benchmark at a time, to avoid running out of memory.
   - Scripts that use a process pool cannot run through `runpy`. Run the script file directly.
   - Key results by full path: `grid_damaged.png` and `star_junction_damaged.png` exist twice.
   - `test_images/` is untracked. Worktrees run with cwd = the main checkout and `PYTHONPATH=<worktree>`.
6. **`.gitignore` ignores `tests/`.** Committed fixtures go in `regression/`, and tests stay at the root as `test_*.py`.
7. **Report schema:** currently `3.0`. Key renames (WP-1) → `4.0`. Additive keys (WP-6) → `4.1`. Note every change in `Phase_7`.
8. **After every WP:** write `docs/experiments/WP-<n>.md` (the before/after `check_changes` table, timings and a short
   interpretation) and update the matching `docs/Phase_*.md`.
9. **Ask Mahmoud before** deleting code, renaming public identifiers or report keys, or changing defaults.
10. Do not commit generated data, model weights or outputs. Ignore `data/`, `models/`, `feedback/` and `*.prof`.

### 0.5 Decisions

| # | Decision | Answer |
|---|---|---|
| D1 | Scope | Follow-up project: Tier A first, then Tier B |
| D2 | Phase 6 | Rename to "Symmetry-guided single-gap closure". No EFD reconstruction (WP-3b dropped) |
| D3 | Clean drawings for synthetic data | Mahmoud's clean originals + a seeded QuickDraw subset |
| D4 | Dependencies | `scikit-learn`, `torch` and `torch_geometric` approved |
| D5 | Interactive UI | JSON constraints file + OpenCV click tool |
| — | Base branch | `accuracy_overhaul` |

### 0.6 Overview

| Tier | WP | Title | Size |
|---|---|---|---|
| A | 0 | Safety net: exact snapshot, determinism test, profiler, scaling probe | S |
| A | 1 | Docs/code consistency + Phase 6 rename | S |
| A | 2 | Performance (profile-driven) | S–M |
| A | 3 | Preprocessing (flagged variants) | M |
| A | 4 | Synthetic damage dataset | M–L |
| A | 5 | Scorer ablation (scorecard vs gradient boosting) | M |
| A | 6 | Interactive refinement through ASP constraints | M |
| B | 3c | Learned stroke-probability map (U-Net) | M |
| B | 7 | GNN context-aware scorer | L |
| B | 8 | Retrieval memory of solved gaps | M |
| B | 9 | Semantic conditioning | L (research) |

---

## WP-0 — Safety net

`check_changes.py` already guards the metrics. Add the parts that prove a refactor changed **nothing**:

1. **Exact snapshot.** `scripts/make_snapshot.py` runs `restore()` on every benchmark image with the shipped
   `scoring_model.json` and `save_images=False`, with outputs in scratch. For each image, it writes
   `regression/snapshot/<relative path>.json` with:
   - the accepted endpoint pairs
   - per final path: segment count, closed/open, control points
   - Phase 6 diagnostics, from a second variant with `efd_closure=True`
2. **`test_regression.py`** (pytest) compares the current run with the snapshot. Topology must match exactly, and
   control points within `1e-6`. `--rebless` (or `REBLESS=1`) rewrites the snapshot. It is skipped when `test_images/` is missing.
3. **Determinism test:** one image restored twice in one process and once in a fresh subprocess. The reports must be
   identical apart from timings.
4. **`scripts/profile_pipeline.py`:** cProfile on the image with the most endpoints. Prints the top 30 functions by
   cumulative time and saves the `.prof` to scratch. Runs with `save_images` on and off.
5. **Scaling probe** (part of the profiler script): per-stage time against endpoint count over all images, from the
   report's stage timings.

Acceptance: all tests pass on the untouched code. `docs/experiments/WP-0.md` holds the snapshot summary, the profile
top list and the scaling table.

---

## WP-1 — Docs/code consistency and the Phase 6 rename

**Verify, then fix the docs.** These items from the first draft still refer to existing code:

| # | Check |
|---|---|
| 1 | `Phase_1`: the junction continuation threshold against `junction_min_alignment` in `bezier.py` |
| 2 | `Phase_1`: whether `max_error` is in pixels or pixels² (`_fit_cubic_single` gets `max_error ** 2`?) |
| 3 | `Phase_1`: the tangent lookahead the pipeline actually uses |
| 4 | `Phase_1`: spur pruning (now in stroke radii, `_prune_skeleton_spurs`) |
| 5 | `Phase_1`: free-end trimming (`_trim_free_ends`) documented |
| 11 | `Phase_6`: Hausdorff direction (symmetric max?) and the real symmetry-rescue threshold |
| 12 | `Phase_7`: `restore_batch()` with `workers > 1` (process pool) |

Items 6–10 and 13 of the first draft refer to removed code. Close them after a `git grep` confirms they are gone.

**Display rename** to "Symmetry-guided single-gap closure":
- `docs/Phase_6_EFD_Closure.md` → `docs/Phase_6_Single_Gap_Closure.md`, with a new title and text, and every link updated
  (README and the other Phase docs).
- `efd_closure.py` module docstring, `_build_efd_event_explanation` text, `"EFD Gap Closure"` → `"Single-Gap Closure"`,
  the console prints, and the `restore()` docstring.

**Identifier rename**, in a separate commit after the snapshot test passes:
- `restoration/single_gap_closure.py`, with `efd_closure.py` kept as a re-export shim.
- `restore()` kwargs `efd_closure`, `efd_gap_threshold`, `efd_validity_check_enabled`, `efd_plausibility_threshold` and
  `efd_min_gap_for_validity_check` → `gap_closure*`. The old names stay as deprecated aliases that emit a `DeprecationWarning`.
- `CHANGE_SOURCES` `"efd_closure"` → `"gap_closure"`. Report source `"efd_gap_closure"` → `"gap_closure"`.
  The keys `efd_closure_phase`, `efd_closure_validity` and `efd_closure_events_logged` get `gap_closure` names.
  `schema_version` becomes `"4.0"`, with the key mapping in `Phase_7`.
- Update every reader: `evaluate_*.py`, `train_scoring.py`, `check_changes.py`, README.

**`candidate_id` (ask first):** check whether `_change_events` (which uses `_segment_key`) can lose a bridge's candidate
after `_reverse_segments` during merging. Only if it does, add `candidate_id: Optional[int] = None` to `BezierSegment`.
Set it in `synthesize_bridges` and copy it on reverse.

Acceptance: the docs match the code, the snapshot test passes, and every item is checked off in the PR description.

---

## WP-2 — Performance

Fix only what the WP-0 profile shows. The outputs must stay snapshot-identical.
- `synthesis.merge_restored_paths.component_with` scans linearly for each accepted bridge. Replace it with a
  `token → component id` dict that is updated on merge and orientation.
- `efd_closure._detect_symmetry` (24 axes × 2 Hausdorff on ≤ 500 points): coarse 200-point pass, then refine the best 3
  axes. **This can change results**, so it goes behind a flag. Low priority, since Phase 6 is off by default.
- Verify that the sknw numba cache works (time a second fresh process start).
- Report compute time and I/O time separately.

Acceptance: `docs/experiments/WP-2.md` with before/after per-stage timings and the scaling table.

---

## WP-3 — Preprocessing

`restoration/preprocess.py`, built from the existing `bezier.stroke_geometry()`. The skeleton, the radius map and
`dist_map` must come from **one** mask.

```python
@dataclass
class PreprocessConfig:
    threshold: str = "otsu"          # "otsu" | "sauvola" | "wolf" | "adaptive_gaussian"
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    denoise: str = "none"            # "none" | "nlmeans" | "median_gray"
    blur: str = "none"               # "none" | "median_gray"  (today: no blur at all)
    close_kernel: int = 0
    min_component_px: int = 0
    skeleton: str = "skimage"        # "skimage" | "skimage_lee"
    junction_aware_pruning: bool = False

def preprocess(img_gray, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:   # (mask, skeleton, radius_map)
```

`PreprocessConfig()` must reproduce today's behaviour exactly (snapshot test). Hypotheses, each tested as a variant:
- H1: (first draft: "medianBlur on the binary mask cuts strokes"). Moot on this branch: `stroke_geometry()` does not blur, it removes specks below 0.5 line widths². Test instead whether a light gray-level median *before* Otsu helps on noisy scans.
- H2: Sauvola/Wolf beats Otsu on stained or unevenly lit scans.
- H3: nlmeans + adaptive threshold + closing reduces speckle endpoints.
- H4: junction-aware pruning on top of the current radius-based pruning. A spur is pruned only if its junction keeps
  ≥ 2 other long branches.

Measure with `evaluate_gaps` (`endpoints_missing` share, gap recall, off-drawing line) through `check_changes.py`, so
that the model is **refitted for every variant**. Recommend a default. Mahmoud decides.

---

## WP-4 — Synthetic damage dataset

Extend the existing code rather than duplicating it: `add_gaps.apply_gaps` (circle gaps), and
`evaluate_gaps.build_ground_truth` / `label_candidates` (pixel-based ground truth, already used by `train_scoring`).

1. **`scripts/fetch_quickdraw.py`** downloads a seeded subset of QuickDraw ndjson to `data/quickdraw/` (ignored) and
   rasterizes the strokes at a fixed width.
2. **`scripts/make_synthetic.py`** (seeded) takes the clean originals in `test_images/` plus the QuickDraw subset and
   produces N variants each:
   - typed gap placement on the clean skeleton: straight, curved, corner, junction
   - capsule erasure along the skeleton (radius ≥ half the stroke width + 1)
   - degradations: paper texture, stains, faded ink, uneven light, noise, blur, JPEG, at easy/medium/hard levels
   - output to `data/synthetic/<split>/<drawing>/<variant>/`: `damaged.png`, `clean.png`, `gaps.json` (type and cut points)
3. **Splits** are by source drawing, never by variant. The 32 real pairs stay a separate real-world test set.
4. **`evaluate_gaps.py --dataset <dir>`** reports results per gap type and per difficulty.
5. **`train_scoring.py --extra-data <dir>`** adds the synthetic train split. Report whether it helps on the real held-out set.
6. Commit about 20 small samples to `regression/synthetic_samples/` for tests.

Acceptance: the generator is deterministic for a given seed, and the current pipeline's numbers on the synthetic test
split are in `docs/experiments/WP-4.md`.

---

## WP-5 — Scorer ablation

The fitted scorecard is v1. Add `restore(..., scorer="scorecard" | "gbt", model_path=None)`. `"gbt"` is an sklearn
`HistGradientBoostingClassifier` on the same `candidate_features()`, fitted by `train_scoring.py --model gbt` with CV by
drawing and class weights. It feeds the same threshold → log-odds → `utility()` path. The default stays `"scorecard"`.

Ablation (same candidates, same ASP): scorecard vs GBT (vs GNN after WP-7), on the real held-out set and the synthetic
test split. Report gap recall, off-drawing line, wrong strokes, harm on undamaged originals and PSNR/SSIM, plus the
`no_candidate_generated` ceiling.

---

## WP-6 — Interactive refinement

**Constraint format.** Endpoints are referred to by position, because candidate ids are not stable:

```json
{
  "image": "ankh_damaged.png",
  "forbid": [ {"a": [412.0, 96.0], "b": [450.5, 98.0]} ],
  "force":  [ {"a": [100.0, 210.0], "b": [130.0, 214.0]} ],
  "hint":   [ {"a": [300.0, 50.0],  "b": [360.0, 40.0]} ],
  "tolerance_px": 8.0
}
```

Each position is resolved to the nearest endpoint within the tolerance. If none is found, fail loudly.

**ASP** (`restoration.lp`, on the existing `candidate/4`):

```prolog
forbid_pair(-1,-1).  force_pair(-1,-1).
:- accept(Id), candidate(Id, A, B, _), forbid_pair(A, B).
:- accept(Id), candidate(Id, B, A, _), forbid_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, A, B, _), force_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, B, A, _), force_pair(A, B).
:- force_pair(A, B), A >= 0, not forced_satisfied(A, B).
```

- `encode_facts` always emits forced and hinted candidates, even with utility ≤ 0. Today only positive ones are emitted.
- A missing pair gets a synthesized candidate from `candidates.smooth_bridge` (scenario `"user_forced"`).
- A hint gets a large utility bonus instead of a hard constraint.
- Forced candidates are exempt from any post-solve filter in `pipeline.restore`.

**API and tool:**
- `restore(..., user_constraints: Optional[dict | path] = None)`.
- `scripts/refine.py image.png` opens an OpenCV window on the labelled overlay. Click two endpoints for hint or force,
  click a bridge label to forbid it. It re-solves, redisplays and saves the JSON.
- Every action plus the affected candidates' features is appended to `feedback/feedback.jsonl`.
  `train_scoring.py --include-feedback` uses them as labels (forbid → 0, force/hint → 1).
- The report gets a `user_constraints` key (`schema_version` `4.1`).

Acceptance: unit tests for forbid, force and hint on a small synthetic image. Forcing changes the global solution where
expected. With no constraints, the output equals the snapshot.

---

## WP-3c (Tier B) — Learned stroke-probability map

A small U-Net (torch) trained on the WP-4 pairs (degraded → clean stroke mask). Threshold its output, then skeletonize,
as a `PreprocessConfig(threshold="unet")` variant. The mean probability along each bridge becomes a new measurement in
`scoring.FEATURES`, and the model is refitted.

## WP-7 (Tier B) — GNN context-aware scorer

- **`restoration/graph_features.py`:**
  - nodes: endpoints, junctions, corners, optional stroke samples
  - node features: position, tangent sin/cos, curvature, stroke radius, type one-hot, optional patch embedding
  - typed edges: `stroke`, `candidate` (with `candidate_features()` as attributes), `parallel_to`, `symmetric_with`
    (from `_detect_symmetry`), `similar_curvature`, `same_region`
- **Model:** R-GCN / `HeteroData` in PyG with 3–4 layers and an edge MLP on `candidate` edges → p. Weighted BCE,
  trained on WP-4 data + feedback. `torch.use_deterministic_algorithms(True)`, everything seeded.
- **Inference:** `scorer="gnn"`, through the same threshold → utility path. ASP and synthesis do not change.
- **Optional head:** regress the bridge handle lengths.
- **Ablation:** a new row in the WP-5 table, plus a relation-type ablation.

## WP-8 (Tier B) — Retrieval memory of solved gaps

Case-based retrieval, not LLM RAG. Standardized feature vectors (or GNN edge embeddings) from the synthetic train set
and `feedback.jsonl` go into `sklearn.neighbors.NearestNeighbors`. The k-NN label mean and the distance-weighted vote
become extra measurements. Adding a corrected sketch just appends its rows. Evaluate the gain on held-out real sketches
as the bank grows.

## WP-9 (Tier B, research) — Semantic conditioning

A global sketch embedding (QuickDraw-pretrained encoder or a vision-language encoder) conditions the GNN (FiLM or
concatenation). Heritage sketches are out of distribution, so consider a small domain vocabulary (bilateral figures,
cartouches, vessels, borders). Reading list (verify before citing): Sketch-BERT (CVPR 2020), Sketch-RNN (Ha & Eck 2017),
and line-art cleanup work. An LLM is allowed only as an interface (natural language → WP-6 constraints, or explanation
text), never for the geometry.

---

## Per-PR checklist

- [ ] `check_changes.py` verdict is not "worse" (or the trade-off is approved)
- [ ] `test_regression.py` passes, or the differences are intended and listed
- [ ] Determinism test passes
- [ ] `docs/experiments/WP-<n>.md` written
- [ ] Relevant `docs/Phase_*.md` updated
- [ ] New parameters documented. Defaults unchanged unless Mahmoud approved
- [ ] No generated data, weights or outputs committed
