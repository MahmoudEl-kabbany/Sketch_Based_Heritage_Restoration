# Sketch-Based Heritage Restoration

A computational pipeline for automatically restoring damaged heritage sketches. It infers missing lines by geometric reasoning rather than by guessing pixels: strokes are extracted as Bézier curves, every plausible reconnection of two stroke ends is scored by a transparent, fitted scorecard, and Answer Set Programming (ASP) selects the most plausible set of reconnections. Every line it draws comes with the measurements that argued for and against it.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Restore a Single Image](#restore-a-single-image)
  - [Batch Restoration](#batch-restoration)
  - [Damage Simulation](#damage-simulation)
  - [Quantitative Evaluation](#quantitative-evaluation)
  - [Testing a Change](#testing-a-change)
  - [Fitting the Scorer](#fitting-the-scorer)
- [Pipeline Phases in Detail](#pipeline-phases-in-detail)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Heritage sketches often suffer from physical deterioration. This project provides a fully automatic, end-to-end restoration pipeline that:

1. **Extracts** the strokes of a damaged raster image as Bézier curves and finds the places where a stroke stops.
2. **Proposes** every connection between two such stroke ends that lie within reach of each other.
3. **Scores** each proposal with the probability that it redraws a missing stroke, from geometric measurements and weights fitted to drawings whose originals are known.
4. **Selects** the most plausible set of connections that fit together via Answer Set Programming.
5. **Merges** the selected bridges into the drawing's paths.
6. Optionally **closes** remaining single-gap contours using symmetry (off by default).
7. **Outputs** labeled visualizations, overlay images, and a JSON report that explains every bridge drawn and every stroke end left open.

---

## Features

- **Skeleton-based Bézier path extraction** — Converts raster sketches to structured cubic Bézier paths using Schneider's fitting algorithm over morphological skeletons. All sizes are multiples of the stroke width measured in the image, so thin and thick drawings are treated alike.
- **Stroke ends from the skeleton graph** — Only places where a stroke really stops are candidates for repair; path ends that sit on a junction are connected, not open. End directions come from an arc fitted to the stroke behind the end.
- **Permissive candidate generation** — Every pair of nearby free stroke ends is proposed, as a smooth G1 bridge and, where the two strokes' directions meet, as a corner. No hand-set gates.
- **Fitted, transparent scoring** — Scale-free measurements (gap length in stroke widths, aim, sideways offset, crossed strokes, MDTW, MJerk, ...) enter a logistic scorecard with one weight per measurement. The weights are fitted, cross-validated by drawing, and stored as plain numbers in a JSON file.
- **ASP-based global optimization** — Clingo selects the set of bridges with the highest total log-odds in which every stroke end is used at most once. Its answer is final.
- **Explanations** — For each bridge: its probability, the measurements for and against it, and the alternatives it beat. For each stroke end left open: its best candidate and why it was not drawn.
- **Evaluation against ground truth** — A gap-level benchmark (which missing strokes were redrawn, how much line was drawn where the original has none), next to PSNR/SSIM.
- **Symmetry-guided single-gap closure (optional)** — Detects nearly-closed shapes and closes them by symmetry-based mirroring or curvature-aware arcs.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Damaged Sketch Image (Raster)                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                    Phase 1  Extraction
                           ▼
              ┌─────────────────────────┐
              │Skeleton → Graph → Bézier│
              │Free stroke ends:        │
              │ direction, curvature,   │
              │ stroke width            │
              └────────────┬────────────┘
                           │
                   Phase 2  Candidate Generation
                           ▼
              ┌─────────────────────────┐
              │ Pairs of nearby ends    │
              │ Smooth bridge           │
              │ Corner bridge           │
              └────────────┬────────────┘
                           │
                    Phase 3  Scoring
                           ▼
              ┌─────────────────────────┐
              │ Scale-free measurements │
              │ (incl. MDTW, MJerk)     │
              │ Fitted scorecards       │
              │ → probability           │
              └────────────┬────────────┘
                           │
                   Phase 4   ASP Decision
                           ▼
              ┌─────────────────────────┐
              │ Clingo optimization     │
              │ One bridge per end      │
              │ Highest total log-odds  │
              └────────────┬────────────┘
                           │
                   Phase 5   Synthesis
                           ▼
              ┌─────────────────────────┐
              │ Path merging            │
              │ Closure detection       │
              └────────────┬────────────┘
                           │
                   Phase 6   Single-Gap Closure (optional)
                           ▼
              ┌─────────────────────────┐
              │ Symmetry detection      │
              │ Mirror-based closure    │
              │ Plausibility validation │
              └────────────┬────────────┘
                           │
                   Phase 7   Output
                           ▼
              ┌─────────────────────────┐
              │ Labeled overlay image   │
              │ Side-by-side comparison │
              │ JSON report with        │
              │ explanations            │
              └─────────────────────────┘
```

---

## Project Structure

**Dataset Setup:** Test images are not included in the repository. Please use your own images for testing, or download the sample dataset from:
[Google Drive Dataset](https://drive.google.com/drive/folders/1HUMwwl-PoMKwqtJaFARf6tktfA22Sm9w?usp=sharing)

Once downloaded, extract the images and place them in the `test_images/` directory at the root of the project. Make sure the structure looks exactly like this:
- `test_images/difficult_test_cases/`
- `test_images/difficult_test_cases_original/`

This structure is required for the default evaluation scripts to run correctly.

```
Sketch_Based_Heritage_Restoration/
│
├── restoration/                    # Core restoration engine
│   ├── __init__.py                 # Package entry point & API reference
│   ├── pipeline.py                 # Phase 7 — Pipeline orchestrator (main API)
│   ├── preprocess.py               # Phase 1 — Binarization, stroke mask, skeleton (with experimental variants)
│   ├── extraction.py               # Phase 1 — Skeleton → Bézier extraction
│   ├── candidates.py               # Phase 2 — Candidate bridges between free stroke ends
│   ├── scoring.py                  # Phase 3 — Measurements and fitted scorecards
│   ├── scoring_model.json          # The fitted scorecards (written by train_scoring.py)
│   ├── asp_engine.py               # Phase 4 — ASP fact encoding, Clingo solver
│   ├── synthesis.py                # Phase 5 — Merging bridges into paths
│   ├── single_gap_closure.py       # Phase 6 — Symmetry-guided single-gap closure (optional)
│   ├── efd_closure.py              # Deprecated alias of single_gap_closure.py
│   ├── geometry.py                 # Geometry helpers shared by the phases
│   ├── render_restored_sketch.py   # Draws restoration bridges onto the damaged image for evaluation
│   └── asp_rules/
│       └── restoration.lp          # Answer Set Programming logic rules
│
├── bezier_curves/
│   └── bezier.py                   # Bézier curve data structures & Schneider fitting
│
├── eliptic_fourier_descriptors/
│   └── efd.py                      # Standalone EFD demo (not used by the pipeline)
│
├── add_gaps.py                     # Damage simulation tool (gaps, cracks, scrubs)
├── synthetic_damage.py             # Typed gaps + degradation with ground truth (synthetic dataset)
├── check_changes.py                # Tests a code change against the recorded baseline
├── train_scoring.py                # Fits the scorecards; cross-validated report
├── evaluate_gaps.py                # Gap-level benchmark against ground truth
├── evaluate_restoration.py         # PSNR/SSIM evaluation framework
├── evaluate_bridge_vs_original.py  # Fluidity metrics (MDTW/MJerk) vs ground truth
├── run_batch_test.py               # Batch restoration runner
├── test_restoration.py             # Tests (run with: python -m pytest)
├── test_regression.py              # Output identical to the snapshot; determinism
├── regression_snapshot.py          # Records / compares the exact-output snapshot
├── regression/snapshot/            # The recorded snapshot (one JSON per benchmark image)
├── scripts/profile_pipeline.py     # Per-stage timings vs. drawing size; cProfile
├── scripts/compare_preprocessing.py # Benchmarks preprocessing variants, each with a refitted scorer
├── scripts/fetch_quickdraw.py      # Seeded Quick, Draw! subset, rasterized (data/quickdraw/)
├── scripts/make_synthetic.py       # Builds the synthetic damage dataset (data/synthetic/)
├── scripts/evaluate_synthetic.py   # Gap benchmark on it, by difficulty, gap type and source
└── regression/synthetic_samples/   # 21 small committed samples (Quick, Draw!, CC BY 4.0)
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- A **C/C++ compiler** (required for building `clingo` and `dtaidistance` via `pip`). Depending on your OS:
  - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select the "Desktop development with C++" workload).
  - **macOS:** Run `xcode-select --install` in your terminal.
  - **Linux (Ubuntu/Debian):** Run `sudo apt install build-essential cmake`.

> **💡 Tip for Conda Users (Recommended):** Installing `clingo` from source via `pip` can be error-prone on some systems. If you use Conda, it is highly recommended to install `clingo` using Conda instead:
> `conda install -c potassco clingo`

### Setup

```bash
# Clone the repository
git clone https://github.com/MahmoudEl-kabbany/Sketch_Based_Heritage_Restoration.git
cd Sketch_Based_Heritage_Restoration

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

Alternatively, you can install the core dependencies directly via pip:

```bash
pip install opencv-python numpy scipy scikit-image matplotlib pyefd clingo dtaidistance sknw
```

---

## Usage

### Restore a Single Image

```python
from restoration.pipeline import restore

result = restore("path/to/damaged_sketch.png")

# Access results
print(f"Original paths:  {len(result.original_paths)}")
print(f"Restored paths:  {len(result.restored_paths)}")
print(f"Bridges created: {len(result.bridges)}")
```

Output files (visualizations + JSON report) are saved to `restoration/outputs/` by default.

### Batch Restoration

```python
from restoration.pipeline import restore_batch

results = restore_batch([
    "sketch1_damaged.png",
    "sketch2_damaged.png",
    "sketch3_damaged.png",
], workers=4)   # optional: restore several images at once, in separate processes
```

Or use the batch runner script, which restores the test dataset in parallel:

```bash
python run_batch_test.py              # uses most CPU cores (at most 8 workers)
python run_batch_test.py --workers 1  # one image at a time
```

Results do not depend on the number of workers. Because the workers are
separate processes, a script that calls `restore_batch(..., workers=N)` must
keep its top-level code under `if __name__ == "__main__":`.

### Damage Simulation

Create synthetically damaged versions of clean sketches for testing:

```bash
# Apply 20 circular gaps to a sketch
python add_gaps.py input.png output.png --num 20 --type circle

# Apply crack-style damage
python add_gaps.py input.png output.png --num 15 --type crack --min_size 8 --max_size 20

# Process an entire directory
python add_gaps.py input_dir/ output_dir/ --type scrub --seed 42
```

**Gap types:** `circle`, `crack`, `scrub`, `box`

### Quantitative Evaluation

Evaluate restoration quality with PSNR and SSIM against ground truth originals:

```bash
# Evaluate all matched pairs
python evaluate_restoration.py

# Evaluate specific images
python evaluate_restoration.py --filter ankh --limit 5

# Custom output directory
python evaluate_restoration.py --output-dir results/

# Also write the overlay figures for every image (skipped by default to save time)
python evaluate_restoration.py --save-images

# One image at a time instead of in parallel
python evaluate_restoration.py --workers 1
```

This produces:
- Per-image PSNR/SSIM metrics (before and after restoration)
- Aggregate statistics (mean, median, min, max)
- CSV and JSON exports
- Comparison bar charts

PSNR and SSIM barely move when a sketch gains or loses a stroke. The gap-level benchmark asks what matters for restoration: which of the truly missing stroke pieces were redrawn, how much line was drawn where the original has none, and what the pipeline draws on an undamaged drawing.

```bash
python evaluate_gaps.py                       # all matched pairs
python evaluate_gaps.py --filter ankh
python evaluate_gaps.py --output-dir results/
```

For bridge fluidity analysis (MDTW/MJerk vs ground truth segments):

```bash
python evaluate_bridge_vs_original.py --limit 10
```

All evaluation scripts process image pairs in parallel; `--workers N` sets how many at once.

### Testing a Change

After changing the code, one command says whether the change helped:

```bash
python check_changes.py                   # about 2 minutes
python check_changes.py --save-baseline   # record the current results as the new baseline
```

It runs the unit tests, refits the scorecards without each drawing, restores every test drawing with a model that has not seen it, and compares the results with `check_baseline.json`: gaps repaired, line drawn off the drawing, wrong strokes, strokes drawn on undamaged originals, PSNR and SSIM, and which drawings changed. It ends with a verdict. Fewer wrong strokes come first: a change that draws less line off the drawing but repairs fewer gaps is kept only when the line off the drawing falls by at least 25% for at most 2 points of gap recall.

Nothing is written to `restoration/outputs/` or to the shipped model; files go to a folder in the system's temp directory (`--output-dir` to choose another). If the change is kept, refit the shipped model with `python train_scoring.py` and record the new baseline.

A refactor that should change nothing is checked more strictly, against an exact snapshot of the output (which stroke ends were bridged, and every final path's segments and control points, with and without Phase 6) on all 64 benchmark images:

```bash
python regression_snapshot.py             # about 30 seconds; lists any image whose output changed
python regression_snapshot.py --rebless   # after an intended change: record the new output
python -m pytest test_regression.py       # the same check, plus a determinism test
python scripts/profile_pipeline.py        # per-stage time vs. number of stroke ends, and a cProfile
```

### Synthetic Damage Dataset

Unlimited labelled data with known gaps: each clean drawing gets typed gaps (straight, curved, corner, junction) erased along its strokes, then degradation at three levels (easy, medium, hard: fading, paper texture, stains, uneven light, blur, noise, JPEG). Ground truth comes from the gapped image before degradation.

```bash
python scripts/fetch_quickdraw.py        # 500 Quick, Draw! drawings (CC BY 4.0) -> data/quickdraw/
python scripts/make_synthetic.py         # benchmark originals + Quick, Draw! -> data/synthetic/ (train/val/test by drawing)
python scripts/evaluate_synthetic.py --split test --models cv_models/   # recall by difficulty, gap type and source
python train_scoring.py --extra-data data/synthetic                    # also train on its train split
python check_changes.py --extra-data data/synthetic                    # does that help on the real benchmark?
```

`data/` is git-ignored; everything is deterministic for a seed.

### Fitting the Scorer

The scorecards in `restoration/scoring_model.json` were fitted with:

```bash
python train_scoring.py                    # fit on all drawings, print a cross-validated report, write the model
python train_scoring.py --cv-dir models/   # also write, for every drawing, a model fitted without it
```

Candidates are generated on damaged versions of every drawing in `test_images/` and labelled from the undamaged originals; see [docs/Phase_3_Scoring.md](docs/Phase_3_Scoring.md). To measure the pipeline on drawings its model has not seen, pass the held-out models to the evaluation scripts:

```bash
python evaluate_gaps.py --models models/
python evaluate_restoration.py --models models/
```

By default both scripts train as the shipped model was trained: benchmark + synthetic easy/medium, threshold tuned on the synthetic val split. Alternatives, each judged on held-out drawings by `check_changes.py` (see [docs/experiments/WP-5.md](docs/experiments/WP-5.md)):

```bash
python check_changes.py --model gbt                        # gradient-boosted trees instead of the scorecard
python check_changes.py --no-extra-data                    # benchmark drawings only, threshold 2/3 (before WP-5)
python check_changes.py --fixed-threshold                  # synthetic training, but threshold 2/3
python check_changes.py --dataset-cache cache/             # reuse labelled candidates between runs
```

A model file of kind `gbt` is used like any other: `restore(path, model_path="models/gbt.json")`.

---

## Pipeline Phases in Detail

Each phase has its own document in [`docs/`](docs/).

### Phase 1 — Extraction

Converts the raster input into structured geometric representations:
- **Binarization and stroke measurement** — Otsu threshold; the distance transform gives the stroke radius everywhere
- **Skeletonization** via `scikit-image`, **graph construction** using `sknw`
- **Spur pruning** — terminal branches that do not leave the stroke they hang off, or run into a corner of its outline
- **Chain traversal** with junction continuation; every chain end is marked *free* or *on a junction*
- **End trimming** by 1.5 stroke radii, where the skeleton leaves the centre line
- **Schneider fitting** to produce cubic Bézier paths
- **Free stroke ends** — direction and signed curvature from an arc fitted to the stroke behind the end

### Phase 2 — Candidate Generation

Proposes bridges between free stroke ends, without judging them:
- **Pairs** — each end with its 6 nearest partners within 30% of the image diagonal; the two ends of one stroke always
- **Smooth bridge** — one G1-continuous cubic, stabilized against hooks
- **Corner bridge** — two straight segments through the meeting point of the two tangent rays, where they meet

### Phase 3 — Scoring

Estimates for each candidate the probability that it redraws a missing stroke:
- **Measurements** free of scale: gap length in stroke widths, aim of both ends, sideways offset, existing stroke crossed, nearest-partner relations, **MDTW** and **MJerk** of the bridge with its context; for corners the turn, detour, balance and straightness of the strokes
- **Scorecards** — one logistic regression per bridge shape, one weight per measurement, each weight held to the direction its meaning allows
- **Fitted, not tuned** — weights come from `train_scoring.py`, cross-validated by drawing
- **Trained on real and synthetic damage** — the benchmark drawings plus the easy and medium levels of the synthetic dataset
- **Threshold** — chosen on the synthetic validation split (0.55): as many repairs as possible without more line drawn off the drawing

### Phase 4 — ASP Decision

Encodes the candidates above the threshold as Answer Set Programming facts and solves for the best set:
- **Endpoint occupancy** — each stroke end used at most once
- **Objective** — highest total log-odds above the threshold; fewer bridges as the tie-break
- **Deterministic solve** — one program per image, single-threaded, so the same input always gives the same bridges
- **Final** — nothing is removed after the solve

### Phase 5 — Synthesis

Puts the accepted bridges into the drawing:
- **Path merging** — concatenates each bridge with the two paths it joins, reversing a path where needed
- **Closure detection** for paths whose ends now meet

### Phase 6 — Symmetry-Guided Single-Gap Closure (optional)

Closes remaining open contours with a single gap; run with `restore(..., gap_closure=True)`:
- **Gap qualification** — gap/perimeter ratio threshold (default ≤ 30%)
- **Symmetry detection** — Hausdorff-based reflection symmetry across 24 axes
- **Mirror-based closure** — reflects the opposite contour arc to fill the gap
- **Curvature-aware arc** fallback and **plausibility validation**

Off by default: on the test drawings it repaired a few more gaps but drew more line off the drawing than that was worth (see [docs/Phase_6_Single_Gap_Closure.md](docs/Phase_6_Single_Gap_Closure.md)).

### Phase 7 — Output

Generates comprehensive outputs:
- **Labeled overlay** — original image with green restoration bridges and `R1`, `R2`, ... labels
- **Unlabeled overlay** — clean restoration-only overlay
- **Side-by-side comparison** — original, extracted paths, and restored paths
- **JSON report** — per-phase timing, the scorecards used, and for every bridge its probability, the measurements for and against it and the alternatives it beat; for every stroke end left open, its best candidate and the reason it was not drawn

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Gap recall** | Share of the true gaps (stroke present in the original, missing in the damaged image) whose centre line is at least 60% covered, within one line width, by restoration strokes |
| **Off-drawing line** | Length of restoration stroke lying farther than one line width from any stroke of the original, in line widths; on the damaged images and on the undamaged originals |
| **Wrong strokes** | Restoration strokes with less than half their length on the original drawing |
| **PSNR** | Peak Signal-to-Noise Ratio between restored and original sketch |
| **SSIM** | Structural Similarity Index for perceptual quality assessment |
| **MDTW** | Modified Dynamic Time Warping — measures directional deviation of bridges from a straight baseline |
| **MJerk** | Mean Squared Jerk — quantifies smoothness of bridge curves (lower = smoother) |

---

## Results

32 drawings, 784 true gaps. Each drawing is restored with scorecards fitted **without it** (5-fold cross-validation by drawing), so these are results on unseen drawings.

| | Hand-tuned pipeline | Fitted scorecards | + synthetic training, tuned threshold (current) |
|---|---|---|---|
| Gap recall | 77.7% | **87.2%** | 86.3% |
| Line drawn off the drawing (damaged + undamaged images) | 1,010 line widths | 89 line widths | **39 line widths** |
| Wrong strokes (damaged + undamaged images) | 44 | 5 | **3** |
| Strokes drawn on the 32 undamaged originals | 307 | **23** | 26 |
| Repaired gaps: distance from the true centre line (median) | 0.12 line widths | 0.12 line widths | 0.11 line widths |
| Mean PSNR (damaged images: 22.26 dB) | 24.53 dB | 26.06 dB | **26.10 dB** |

The current model (adopted in WP-5, see [docs/experiments/WP-5.md](docs/experiments/WP-5.md)) is also trained on the easy and medium levels of the synthetic dataset, and its threshold (0.55) is chosen on that dataset's validation split. It more than halves the line drawn off the drawing for 0.9 points of recall. The middle column, and the sensitivity table below, are for the fitted scorecards before that change.

Recall by kind of gap: 89.8% for a gap in one stroke (625 gaps), 83.1% for a gap over a junction (118), 61.0% where a stroke end was erased (41).

The "hand-tuned pipeline" is this repository before candidate gates, score adjustments and thresholds were replaced by measurement and fitting; the steps in between are in the commit history.

**How much the results depend on the constants.** The extraction and candidate constants are set from geometry, not fitted. Setting any one of them clearly lower or higher (and refitting the scorecards) gives:

| Constant | Default | Changed to | Gap recall | Off-drawing line (line widths) |
|---|---|---|---|---|
| (all at their defaults) | | | 87.2% | 89 |
| `SPUR_RADIUS_FACTOR` | 1.5 | 1.0 / 2.0 | 87.4% / 87.0% | 93 / 87 |
| `CORNER_TIP_RADIUS_RATIO` | 0.5 | 0.35 / 0.65 | 87.9% / 86.9% | 83 / 86 |
| `CORNER_BRANCH_MAX_RADII` | 3 | 2 / 4 | 88.6% / 84.8% | 105 / 79 |
| `END_TRIM_RADII` | 1.5 | 1.0 / 2.0 | 87.1% / 86.0% | 89 / 71 |
| `END_TRIM_MIN_PX` | 3 | 1 / 5 | 87.5% / 86.6% | 94 / 78 |
| `DIRECTION_WINDOW_WIDTHS` | 6 | 4 / 8 | 87.0% / 87.1% | 78 / 86 |
| `ARC_FIT_RADII` | 0.5 | 0.25 / 1.0 | 87.2% / 86.6% | 88 / 84 |
| `PARTNERS_PER_ENDPOINT` | 6 | 4 / 8 | 87.1% / 87.1% | 80 / 81 |
| `CORNER_MAX_LEG_GAPS` | 2 | 1.5 / 3 | 87.2% / 87.5% | 93 / 86 |

Across all of these, gap recall stays between 84.8% and 88.6% and the off-drawing line between 71 and 105 line widths.

The acceptance threshold is the one deliberate choice. At 0.5 (even odds) gap recall is 88.8% with 173 line widths off the drawing; at 0.8 it is 82.8% with 24.

**What limits it.** The provided damage is cut out with straight edges, and 32 drawings are few; cross-validation guards against memorising drawings, not against a different kind of damage. A stroke end that should be extended to meet another stroke, and a stroke end that was erased, cannot be repaired by joining two ends. Some of the bridges drawn on undamaged originals are across gaps the artist left, which look like damage.

---

## Configuration

Key parameters of the `restore()` function:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `restoration/scoring_model.json` | The fitted scorecards and the acceptance threshold |
| `gap_closure` | `False` | Also run Phase 6 (symmetry-guided single-gap closure) |
| `gap_closure_threshold` | `0.30` | Maximum gap/perimeter ratio for single-gap closure |
| `gap_closure_validity_check` | `True` | Enable the plausibility check for single-gap closure |
| `gap_closure_plausibility_threshold` | `0.50` | Minimum plausibility score for single-gap closure |
| `gap_closure_min_gap_for_validity_check` | `3.0` | Bypass plausibility check for very small gaps (pixels) |
| `asp_timeout_s` | `30.0` | Maximum ASP solver time (seconds) |
| `save_images` | `True` | Write the two overlays and the comparison figure (the JSON report is always written) |

The Phase 6 arguments were called `efd_*` before report schema 4.0; the old names still work, with a `DeprecationWarning`.

The acceptance threshold lives in the model file (`"threshold"`, 0.55 in the shipped model, chosen on the synthetic validation split; `train_scoring.py --fixed-threshold` uses 2/3 instead). Lowering it repairs more gaps and draws more wrong strokes; `train_scoring.py` prints the trade-off.

`train_scoring.py` and `check_changes.py` train as the shipped model was trained, which needs the synthetic dataset in `data/synthetic/` (`python scripts/fetch_quickdraw.py`, then `python scripts/make_synthetic.py`). `--no-extra-data` trains on the benchmark drawings only, as before WP-5.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image I/O, morphology, contour detection |
| `numpy` | Numerical computation |
| `scipy` | KD-trees, spatial distance, Hausdorff, splines; fitting the scorecards |
| `scikit-image` | Skeletonization, PSNR/SSIM metrics |
| `matplotlib` | Visualization and plotting |
| `pyefd` | Elliptic Fourier Descriptor computation (standalone `efd.py` demo only) |
| `clingo` | Answer Set Programming solver |
| `dtaidistance` | Dynamic Time Warping distance |
| `sknw` | Skeleton-to-graph conversion |

---

## License

This project is developed as part of academic research. Please see the repository for license details.
