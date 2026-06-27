# Sketch-Based Heritage Restoration

A computational pipeline for automatically restoring damaged heritage sketches. The system detects gaps, cracks, and missing segments in raster sketch images and reconstructs plausible connections using a combination of geometric reasoning, Bézier curve fitting, Elliptic Fourier Descriptors (EFD), and Answer Set Programming (ASP).

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
- [Pipeline Phases in Detail](#pipeline-phases-in-detail)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Heritage sketches often suffer from physical deterioration. This project provides a fully automatic, end-to-end restoration pipeline that:

1. **Extracts** skeletal structure and Bézier curve representations from damaged raster images.
2. **Identifies** broken endpoints and generates candidate connections using geometric heuristics.
3. **Scores** candidates with curvature-aware, DTW-based, and Gestalt-inspired metrics.
4. **Selects** an optimal, conflict-free set of restorations via Answer Set Programming.
5. **Synthesizes** smooth G1-continuous Bézier bridge curves.
6. **Closes** remaining single-gap contours using Elliptic Fourier Descriptors with symmetry detection.
7. **Outputs** labeled visualizations, overlay images, and detailed JSON reports.

---

## Features

- **Skeleton-based Bézier path extraction** — Converts raster sketches to structured cubic Bézier paths using Schneider's fitting algorithm over morphological skeletons.
- **Multi-scenario candidate generation** — Supports good-continuation, extension-intersection (linear and curved extrapolation), and self-closure scenarios with KD-tree acceleration.
- **Multi-metric scoring** — Combines Gestalt proximity, continuation, and closure heuristics with DTW distance and mean-squared jerk for smooth, natural bridge selection.
- **ASP-based global optimization** — Uses Clingo to solve for the globally optimal set of non-conflicting bridge connections, with endpoint occupancy constraints and utility maximization.
- **EFD gap closure** — Detects nearly-closed shapes and applies symmetry-based mirroring or curvature-aware arc interpolation to close single-gap contours.
- **Semantic plausibility validation** — Gates EFD closure with tangent alignment, curvature coherence, and bilateral continuation checks to prevent false closures.
- **Bridge collision avoidance** — Post-filters bridges that cross existing image lines using distance maps and path body proximity checks.
- **Detailed diagnostics** — Every restoration produces a structured JSON report with per-phase timing, candidate statistics, change explanations, and labeled overlay visualizations.
- **Quantitative evaluation** — Built-in PSNR/SSIM evaluation framework with paired original/damaged test sets and CSV/JSON export.

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
              │Endpoints + Tangents     │
              │  EFD Coefficients       |
              │  Distance Map           |
              └────────────┬────────────┘
                           │
                   Phase 2  Candidate Generation
                           ▼
              ┌─────────────────────────┐
              │ KD-tree endpoint search │
              │ Continuation tests      │
              │ Extension intersection  │
              │ Self-closure detection  │
              └────────────┬────────────┘
                           │
                    Phase 3  Scoring
                           ▼
              ┌─────────────────────────┐
              │ MDTW + MJerk metrics    │
              │ Gestalt heuristics      │
              │ Shape affinity          │
              │ Self-closure priority   │
              └────────────┬────────────┘
                           │
                   Phase 4   ASP Decision
                           ▼
              ┌─────────────────────────┐
              │ Clingo optimization     │
              │ Endpoint occupancy      │
              │ Nested shape exclusion  │
              │ Image mask filtering    │
              └────────────┬────────────┘
                           │
                   Phase 5   Synthesis
                           ▼
              ┌─────────────────────────┐
              │ G1 Bézier bridges       │
              │ Intersection bridges    │
              │ Path merging            │
              └────────────┬────────────┘
                           │
                   Phase 6   EFD Gap Closure
                           ▼
              ┌─────────────────────────┐
              │ Symmetry detection      │
              │ Mirror-based closure    │
              │ Curvature-aware arcs    │
              │ Plausibility validation │
              └────────────┬────────────┘
                           │
                   Phase 7   Output
                           ▼
              ┌─────────────────────────┐
              │ Labeled overlay image   │
              │ Side-by-side comparison │
              │ JSON report             │
              └─────────────────────────┘
```

---

## Project Structure

**Dataset Setup:** Test images are not included in the repository. Please use your own images for testing, or download the sample dataset from:
[Google Drive Dataset](https://drive.google.com/drive/folders/1HUMwwl-PoMKwqtJaFARf6tktfA22Sm9w?usp=sharing)

Once downloaded, extract the images and place them in a `test_images/` directory at the root of the project so that the default scripts run correctly.

place original images in test_images/difficult_test_cases_original and damaged images in test_images/difficult_test_cases


```
Sketch_Based_Heritage_Restoration/
│
├── restoration/                    # Core restoration engine
│   ├── __init__.py                 # Package entry point & API reference
│   ├── pipeline.py                 # Phase 7 — Pipeline orchestrator (main API)
│   ├── extraction.py               # Phase 1 — Skeleton → Bézier extraction
│   ├── candidates.py               # Phase 2 — Connection candidate generation
│   ├── scoring.py                  # Phase 3 — Multi-metric scoring (MDTW, MJerk, Gestalt)
│   ├── asp_engine.py               # Phase 4 — ASP fact encoding, Clingo solver
│   ├── synthesis.py                # Phase 5 — G1 bridge construction & path merging
│   ├── efd_closure.py              # Phase 6 — EFD single-gap closure
│   ├── render_restored_sketch.py   # Binary sketch renderer for evaluation
│   └── asp_rules/
│       └── restoration.lp          # Answer Set Programming logic rules
│
├── bezier_curves/
│   └── bezier.py                   # Bézier curve data structures & Schneider fitting
│
├── eliptic_fourier_descriptors/
│   └── efd.py                      # EFD extraction, reconstruction & visualization
│
├── add_gaps.py                     # Damage simulation tool (gaps, cracks, scrubs)
├── evaluate_restoration.py         # PSNR/SSIM evaluation framework
├── evaluate_bridge_vs_original.py  # Fluidity metrics (MDTW/MJerk) vs ground truth
├── run_batch_test.py               # Batch restoration runner
│
├── test_add_gaps.py                # Tests for gap simulation
├── test_bezier.py                  # Tests for Bézier curve fitting
├── test_efd.py                     # Tests for EFD extraction
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
])
```

Or use the batch runner script:

```bash
python run_batch_test.py
```

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
```

This produces:
- Per-image PSNR/SSIM metrics (before and after restoration)
- Aggregate statistics (mean, median, min, max)
- CSV and JSON exports
- Comparison bar charts

For bridge fluidity analysis (MDTW/MJerk vs ground truth segments):

```bash
python evaluate_bridge_vs_original.py --limit 10
```

---

## Pipeline Phases in Detail

### Phase 1 — Extraction

Converts the raster input into structured geometric representations:
- **Skeletonization** via `scikit-image` to extract medial axes
- **Graph construction** using `sknw` for skeleton topology
- **Chain traversal** with junction-continuation heuristics
- **Schneider fitting** to produce cubic Bézier paths from polylines
- **Endpoint tangent estimation** using multi-scale PCA with confidence scoring
- **EFD coefficient extraction** for contour-level shape descriptors
- **Distance map** computation for collision detection

### Phase 2 — Candidate Generation

Generates bridge candidates between open endpoints:
- **KD-tree spatial search** with two-tier radius (Tier 1: 15% diagonal, Tier 2: 30%)
- **Good-continuation test** — tangent ray proximity to partner endpoint
- **Extension-intersection test** — linear and curved extrapolation convergence
- **Self-closure detection** — identifies endpoints belonging to the same path
- **Direction gates** — conservative bilateral alignment and misalignment filtering
- **Bridge-body crossing filter** — rejects candidates that pass through existing paths

### Phase 3 — Scoring

Multi-factor scoring of each candidate:
- **Proximity** — normalized distance within search radius
- **Continuation** — cosine alignment of tangent with bridge direction
- **Closure** — loop completion bonus
- **MDTW** — Dynamic Time Warping deviation from a straight baseline
- **MJerk** — Mean squared jerk (3rd derivative) for smoothness
- **Shape affinity** — bounding box overlap for fragment reassembly
- **Self-closure priority** — boosted scoring for same-path closures

### Phase 4 — ASP Decision

Encodes the scored candidates as Answer Set Programming facts and solves for the optimal set:
- **Endpoint occupancy constraints** — each endpoint used at most once
- **Self-connection exclusion** — prevents non-closure same-path connections
- **Nested shape exclusion** — avoids cross-connecting concentric shapes
- **Utility maximization** — bi-level optimization (candidate utility + path connectivity)
- **Component partitioning** — splits into independent subproblems for scalability
- **Image mask post-filtering** — removes bridges crossing original image lines

### Phase 5 — Synthesis

Constructs the final Bézier bridge curves:
- **Straight bridges** for collinear, low-curvature endpoints
- **G1-continuous bridges** with curvature-aware handle lengths
- **Intersection bridges** (linear or curved) through convergence points
- **Control point stabilization** to prevent hook/backtracking artifacts
- **Path merging** — concatenates bridge segments into restored paths with closure detection

### Phase 6 — EFD Gap Closure

Addresses remaining open contours with a single gap:
- **Gap qualification** — gap/perimeter ratio threshold (default ≤ 30%)
- **Symmetry detection** — Hausdorff-based reflection symmetry across 24 axes
- **Mirror-based closure** — reflects the opposite contour arc to fill the gap
- **Curvature-aware arc** fallback — single cubic Bézier preserving endpoint tangents
- **Plausibility validation** — tangent continuation, bilateral alignment, and curvature coherence checks

### Phase 7 — Output

Generates comprehensive outputs:
- **Labeled overlay** — original image with green restoration bridges and `R1`, `R2`, ... labels
- **Unlabeled overlay** — clean restoration-only overlay
- **Side-by-side comparison** — original, extracted paths, and restored paths
- **JSON report** — full structured report with per-phase timing, candidate statistics, selection details, and per-event explanations

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio between restored and original sketch |
| **SSIM** | Structural Similarity Index for perceptual quality assessment |
| **MDTW** | Modified Dynamic Time Warping — measures directional deviation of bridges from a straight baseline |
| **MJerk** | Mean Squared Jerk — quantifies smoothness of bridge curves (lower = smoother) |

---

## Configuration

Key parameters of the `restore()` function:

| Parameter | Default | Description |
|---|---|---|
| `lookahead_fraction` | `0.15` | Endpoint search radius as fraction of image diagonal |
| `max_candidates_per_endpoint` | `5` | Maximum connection candidates per endpoint |
| `efd_gap_threshold` | `0.30` | Maximum gap/perimeter ratio for EFD closure |
| `efd_validity_check_enabled` | `True` | Enable semantic plausibility gate for EFD closure |
| `efd_plausibility_threshold` | `0.50` | Minimum plausibility score for EFD closure |
| `efd_min_gap_for_validity_check` | `3.0` | Bypass plausibility check for very small gaps (pixels) |
| `asp_timeout_s` | `30.0` | Maximum ASP solver time per component (seconds) |

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Image I/O, morphology, contour detection |
| `numpy` | Numerical computation |
| `scipy` | KD-trees, spatial distance, Hausdorff |
| `scikit-image` | Skeletonization, PSNR/SSIM metrics |
| `matplotlib` | Visualization and plotting |
| `pyefd` | Elliptic Fourier Descriptor computation |
| `clingo` | Answer Set Programming solver |
| `dtaidistance` | Dynamic Time Warping distance |
| `sknw` | Skeleton-to-graph conversion |

---

## License

This project is developed as part of academic research. Please see the repository for license details.
