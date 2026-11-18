# WP-7 — Context-aware graph scorer

Branch `wp7-gnn-scorer` (from `wp6-interactive-refinement`). **Not adopted; defaults unchanged.** The shipped
scorecard still gives 86.3% recall and 39.3 line widths off the drawing, and `check_changes.py` reports UNCHANGED.

## What was added

- `restoration/graph_features.py`: one graph per drawing, as NumPy arrays.
  - Nodes: free stroke ends, junctions, and samples every 4 line widths along each stroke (11 features each).
  - Relations: `stroke`, `near`, `parallel`, `candidate`, `symmetric` (via Phase 6's symmetry detector).
  - Built in 20–200 ms per drawing.
- `restoration/gnn.py`: a relational GCN in **plain PyTorch**. PyTorch Geometric is not needed: the message passing is
  `index_add` with a mean per relation and a residual connection.
  - 3 layers, 32 units. Each candidate is classified from `[h_a + h_b, h_a·h_b, measurements, shape]`.
  - Seeded, with `torch.use_deterministic_algorithms(True)`; a test checks that two fits are identical.
- `train_scoring.py` / `check_changes.py`: `--model gnn`, `--gnn-relations` (subset, or `none` for no context) and
  `--gnn-epochs`. Labelled images carry their graph when needed (separate cache).
  - Model file: `"kind": "gnn"` + `.pt`. `restore(..., model_path=...)` uses it like any other.
- The threshold search grid now starts at 0.05 instead of 0.30. With the old grid, the GNN's chosen threshold sat on
  the floor (0.30) in every fold, with a large unused off-drawing budget. The shipped scorecard's thresholds
  (0.55–0.575) are unaffected, and `check_changes.py` stays UNCHANGED.
- Test `test_graph_model_trains_repeatably_and_restores`.
- `torch` added to `requirements.txt`, needed only for `--model gnn`.

## Results on the 32 real pairs (held out by drawing; same training data and threshold rule as the shipped model)

| scorer | separation | recall % | off-drawing (lw) | wrong strokes | on undamaged | PSNR | verdict |
|---|---|---|---|---|---|---|---|
| **scorecard (shipped)** | 98.4% | **86.3** | **39.3** | 3 | **26** | **26.10** | — |
| GNN, all relations (grid from 0.30) | 98.8% | 83.2 | 71.6 | 3 | 30 | 25.93 | worse |
| GNN, all relations | **98.8%** | 85.0 | 83.9 | 3 | 42 | 25.97 | worse |
| same network, no relations (`none`) | 98.8% | 84.6 | 96.2 | 5 | 46 | 26.03 | worse |

The thresholds chosen per fold are 0.10–0.275 for the GNN, against 0.55–0.575 for the scorecard. Its probabilities
are lower and less stable across folds.

## Synthetic test split (89 unseen drawings), easy + medium levels: recall by gap type

| scorer | corner | curved | junction | straight | wrong strokes | off-drawing (lw) |
|---|---|---|---|---|---|---|
| scorecard, benchmark only (before WP-5) | 83.5 | 93.8 | **71.8** | 96.4 | 15 | 296 |
| scorecard (shipped) | 84.3 | 93.8 | 69.8 | 96.1 | 13 | **261** |
| GNN, all relations | **94.1** | **96.8** | 61.8 | **97.0** | 17 | 383 |
| same network, no relations | 91.4 | **96.8** | 60.9 | **97.0** | **8** | 265 |

At the hard level both networks drop to about 51% recall (scorecard: 71%). That level is not in their training data,
and its binarization noise (WP-4) is out of distribution for them.

## Interpretation

1. **The neural scorers rank better but select worse on real damage.** Separation rises to 98.8%, but on the 32 real
   pairs they draw two to two and a half times the off-drawing line of the scorecard for less recall. On synthetic
   drawings they are clearly better at corners (+10 points) and slightly better at curved and straight gaps. They do
   not transfer to the provided real damage. The scorecard's eleven sign-constrained weights cannot overfit the
   synthetic generator's look; a network can.
2. **Graph context adds little.** Against the same network without message passing, context gains +2.7 points on
   synthetic corners, +0.9 on junctions, and +0.4 recall with 13% less off-drawing line on the real pairs. It also
   costs 118 more line widths off the drawing on synthetic data. A full per-relation ablation (removing `symmetric`,
   `parallel`, ... one at a time) was not run: with context worth this little, each 25-minute run would only
   apportion a small effect.
3. **Junctions are not solved by per-candidate context.** Junction recall *falls* with the networks (62% against 70%),
   and about 30% of junction gaps have the right candidate but do not get it selected. At an erased junction three or
   four stubs must be paired as a whole. Per-candidate probabilities, even context-aware ones, are summed
   independently by the ASP objective. A model that scores a pairing of all the stubs of one gap jointly, or ASP rules
   that prefer the straightest continuation through a junction, would address the actual decision. That would be a
   follow-up work package.

## Recommendation

Keep the shipped scorecard. Keep `--model gnn` as an experimental option (and the corner gain as a lead). WP-8
(retrieval memory) and WP-9 (semantic conditioning) build on the GNN embedding in the plan. Given this result, WP-8
should use the scorecard features instead, and the junction-level model above is the more promising next step.
