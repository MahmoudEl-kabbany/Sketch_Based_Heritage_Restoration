# Junction gaps: why they fail (diagnosis before a junction-pairing model)

Branch `wp-junctions` (from `wp7-gnn-scorer`). No pipeline change. `scripts/diagnose_junctions.py` reproduces
everything below.

## Question

WP-7 left junction gaps as the weakest gap type (about 63–70% repaired on the synthetic test split, against 96% for
straight gaps). The proposed fix was a model that pairs all the stroke ends of a junction jointly, instead of
scoring each candidate bridge alone. That only helps if junction gaps fail because the solver pairs the ends
*wrongly*. The diagnostic checks this first.

## Method

Synthetic val split, easy + medium levels: 328 variants, 328 junction gaps, the shipped model. Every gap that is not
repaired gets one reason:

| reason | meaning | fixed by joint pairing? |
|---|---|---|
| no single bridge covers it | no candidate covers ≥ 60% of the gap (it needs two bridges, or a stroke end joined to the middle of a stroke) | no |
| covering bridge below threshold | a candidate would repair it, but the scorer rejects it | no |
| covering bridge lost a conflict | the covering bridge is above the threshold, but the solver chose a rival sharing one of its ends | **yes** |

## Results

| stubs entering the gap | gaps | repaired | below threshold | lost a conflict | no single bridge |
|---|---|---|---|---|---|
| 2 | 70 | 66% | 24% | 3% | 7% |
| 3 | 176 | 84% | 12% | 3% | 2% |
| 4 | 73 | 85% | 7% | 4% | 4% |
| all failures | | | **67%** | **16%** | 17% |

The under-scored covering bridges are not near misses: their median probability is **0.04** (threshold 0.55).
Against the drawn bridges that repair a gap, their median measurements are:

| measurement | under-scored junction bridge | drawn, repairing bridge |
|---|---|---|
| gap length (stroke widths) | 12.5 | 4.0 |
| aim of the worse end (°) | 110 | 9 |
| aim of the better end (°) | 40 | 2 |
| sideways offset (widths) | 7.7 | 0.6 |
| MDTW | 79 | 3.8 |
| corner turn (°, corner bridges) | 122 | 46 |
| corner detour | 0.79 | 0.07 |

## Interpretation

- **Joint pairing would fix at most about 3% of junction gaps** (the "lost a conflict" cases). That is about 0.5 points
  of overall recall, so it is not worth a new model.
- **Most junction failures are geometric extremes.** An erased junction leaves a long gap (three times the usual)
  whose stubs meet at a sharp angle, often with one end pointing away from the other. Everywhere else in the data,
  that geometry means a wrong stroke, so every scorer (scorecard, boosted trees, GNN) rejects it with high confidence.
  Accepting such bridges would also accept the much more frequent wrong ones that look the same.
- **The remaining 17% are structural:** they need two bridges or a T-join (a stroke end extended onto the middle of
  another stroke). End-to-end bridges cannot repair them; the README names this limitation.
- **Worst case: 2-stub junction gaps** (one branch erased completely) are repaired only 66% of the time. The correct
  repair there is a sharp corner, which is ambiguous by nature.

## What would help (not done)

1. **T-join candidates:** bridges from a free end to the nearest point of another stroke. This addresses the
   structural 17% and is also the README's known limitation for real drawings. It changes Phases 2, 4 and 5 (a bridge
   would no longer join two ends).
2. **A junction-aware measurement** given to the scorecard (for example "number of other free ends within the gap
   length"), refitted. It would let the scorer learn that long, sharply turning bridges are plausible *when several
   ends meet at one place*. It could recover some of the under-scored 67%, but risks more wrong strokes. Judge it with
   `check_changes.py`.

## Follow-up: a junction-aware measurement (tried, not kept)

Branch `wp-junction-measurement`. Mahmoud chose option 2 above. A measurement was added to every scorecard:
`ends_nearby`, the number of other free stroke ends within half the gap plus two stroke widths of the bridge's
middle (0 at an ordinary gap, 1–2 at an erased junction). It entered as `log2(1 + x)`, sign unconstrained, and was
refitted with the shipped training (synthetic easy/medium, tuned threshold).

| | shipped | + `ends_nearby` |
|---|---|---|
| real pairs: recall % | 86.3 | 86.7 |
| real pairs: off-drawing (lw) | **39.3** | 53.5 |
| real pairs: wrong strokes | **3** | 5 |
| real pairs: strokes on undamaged | **26** | 29 |
| synthetic test, easy+medium: recall % | 89.6 | 90.2 |
| synthetic test, easy+medium: junction recall % | 69.8 | 70.4 |
| synthetic test, easy+medium: corner recall % | 84.3 | 86.3 |
| synthetic test, easy+medium: wrong strokes / off-drawing | **13 / 261** | 16 / 292 |

`check_changes.py` verdict: trade-off, **not kept** (+0.4 points of recall for 14 more line widths off the drawing).

The fit gave `ends_nearby` a **negative** weight (−0.19). Over all candidates, more ends nearby means a bridge is
*less* likely to be right: at a crowded spot the wrong candidates outnumber the right one. An additive measurement
cannot say "long, sharply turning bridges are plausible at junctions". That needs an interaction (e.g. aim or gap
length *given* a junction), on a lever worth about 0.5 points of overall recall.

The measurement was reverted (it would otherwise change cached labels and the GBT/GNN inputs for no gain). The
commit history keeps it.

**Conclusion on junctions:** on per-bridge scoring, junctions are close to what end-to-end bridges can do. The
remaining structural part needs T-join candidates (a free end joined to the middle of another stroke).
