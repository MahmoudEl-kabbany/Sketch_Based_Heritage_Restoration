# WP-8 — Retrieval memory of solved gaps

Branch `wp8-retrieval-memory` (from `wp-tjoins`). **Experiment only; not integrated into the pipeline. Defaults unchanged.**

## Question

Can the system "learn from previous sketches without retraining", by remembering labelled gaps and consulting the most
similar ones for every new candidate (case-based retrieval, not an LLM)?

## Method (`scripts/memory_experiment.py`)

- **Memory bank:** labelled candidate bridges with their scorecard-scale measurements, one nearest-neighbour index per
  bridge shape (scikit-learn, standardized measurements). This is the plan's preferred embedding, since the GNN of
  WP-7 was not adopted.
- **Memory vote:** the distance-weighted share of correct labels among a candidate's 15 nearest neighbours.
- **Stacking:** `log-odds = scorecard log-odds + a · logit(vote) + b`, with `a ≥ 0` fitted by maximum likelihood.
  The scorecard is unchanged, and the memory can only correct it.
- **No leakage:**
  - The folds are those of `check_changes.py`.
  - A training candidate's vote comes from a bank without its own drawing: training drawings are split into 5
    groups, and each group is voted on by the others.
  - Held-out drawings are never in the bank.
  - Each fold's threshold is chosen on the synthetic val split by the shipped rule.
- **Scoring:** the held-out *provided* images (the real damage), by ASP selection at the fold threshold
  (candidate-level ground truth, `train_scoring.sweep`).
- **Growth:** with the stacking fixed, each fold's bank is limited to the synthetic drawings plus N real benchmark
  drawings.

## Results

| fold | stacking weight a | shift b | threshold (scorecard → memory) | repaired | off-drawing (lw) |
|---|---|---|---|---|---|
| 0 | 0.26 | −0.02 | 0.550 → 0.450 | 158 → 159 / 200 | 5.2 → 8.1 |
| 1 | 0.22 | +0.01 | 0.550 → 0.525 | 178 → 178 / 200 | 0.4 → 4.2 |
| 2 | 0.24 | −0.01 | 0.550 → 0.500 | 110 → 111 / 135 | 7.9 → 7.9 |
| 3 | 0.26 | +0.00 | 0.575 → 0.550 | 75 → 74 / 80 | 6.3 → 9.8 |
| 4 | 0.25 | −0.00 | 0.550 → 0.550 | 151 → 151 / 169 | 2.9 → 2.9 |

| held-out real pairs | recall % | off-drawing (lw) | wrong | bridges |
|---|---|---|---|---|
| scorecard (shipped) | 85.7 | **22.8** | 1 | 783 |
| + memory vote | 85.8 | 33.0 | 1 | 784 |

| bank = synthetic + N real drawings | recall % | off-drawing (lw) |
|---|---|---|
| N = 0 | 85.8 | 31.6 |
| N = 6 | 85.8 | 31.8 |
| N = 12 | 85.7 | 27.1 |
| all training drawings (about 25) | 85.8 | 33.0 |

(Candidate-level selection, so the absolute numbers differ slightly from `check_changes.py`'s full-pipeline ones;
the comparison between rows is like for like.)

## Interpretation

- **The memory vote carries signal but adds nothing.** The fitted weight is positive and stable (0.22–0.26). The vote
  is computed from the same measurements the scorecard already weighs, and on these 11–13 sign-constrained
  measurements a nearest-neighbour vote has little to add. What it does add mainly lowers the tuned thresholds and
  draws more off-drawing line (+10 line widths for +0.1 points of recall).
- **More solved real drawings in the bank do not help.** There is no trend from N = 0 to all training drawings, so
  the "learns without retraining" claim does not hold with these measurements.
- Retrieval would need an embedding that sees what the measurements miss: for example image patches of the gap and
  its surroundings, the evidence identified for T-joins (ink fragments, junction bumps), or WP-9's semantic context.
  It was not integrated into the pipeline, and `scripts/memory_experiment.py` keeps the experiment reproducible.
