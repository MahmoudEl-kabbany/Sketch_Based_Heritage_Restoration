# Image evidence inside gaps ("image patches")

Branch `wp-image-patches` (from `wp8-retrieval-memory`). **Tried, not kept; the code was reverted.** Defaults are unchanged.

## Question

The T-join, junction and memory results all pointed at the same limit: the scorer sees only geometry. Could the image
itself, inside and around a candidate bridge, tell a missing stroke from a stroke that genuinely stops?

## Check first: what evidence exists in this data (`scripts/diagnose_image_evidence.py`)

Four grey-level statistics along each candidate bridge (without 1.5 line widths at either end), relative to the paper
in a window 4 line widths around it. AUC of each alone (0.5 = no signal, below 0.5 = signal in the inverted direction):

| statistic | real pairs (8,719 candidates) | synthetic val (28,814) | among candidates the scorecard is unsure about (real / synthetic) |
|---|---|---|---|
| median darkness along the bridge (faint residual ink) | 0.395 | 0.416 | 0.452 / 0.523 |
| darkest point along the bridge | **0.156** | **0.153** | 0.368 / 0.441 |
| grey-level variation along the bridge | **0.168** | **0.253** | 0.369 / 0.423 |
| share brighter than the paper (a clean white fill) | 0.500 | 0.485 | 0.500 / 0.509 |

- **No erasure artefact.** Gaps are not detectably whiter or cleaner than paper (share brighter ≈ 0.5), so an image
  model could not cheat on how the damage was made. But there is also **no faint ink left inside the gaps**: the damage
  in both datasets is a clean erasure.
- **The signal that exists** is wrong bridges running over other strokes (darkest point, variation). The binarized
  `crossed_ink` measurement already captures most of it. Some remains among unsure candidates (AUC about 0.37 on
  real pairs) because the grey image also sees faint and anti-aliased ink. A patch CNN would mostly relearn this, with
  more room for artefacts, so the evidence was added as two measurements.

## The measurements (tried)

- `faint_ink`: darkness of the darkest point inside the bridge, relative to the paper (direction −1).
- `gap_texture`: grey-level variation along the bridge, relative to the paper's own (`log2(1 + x)`, direction −1).

Both were added to the smooth and corner scorecards and refitted with the shipped training. The weights were negative
in every fold (`faint_ink` −0.09 to −0.48 on smooth bridges; `gap_texture` −0.11 to −0.40 smooth and −0.78 to −0.93
on corners), next to `crossed_ink` (−2.1 to −2.3).

| | shipped | + `faint_ink`, `gap_texture` |
|---|---|---|
| real pairs: recall % | **86.3** | 85.8 |
| real pairs: off-drawing (lw) | **39.3** | 46.2 |
| real pairs: wrong strokes | **3** | 5 |
| synthetic test: recall % | **86.8** | 86.3 |
| synthetic test off-drawing, easy / medium / hard (lw) | 123 / 138 / 176 | **111 / 122 / 149** |

`check_changes.py` verdict: **worse** on the real pairs.

## Interpretation

- On synthetic drawings the grey-level evidence does what it should: 10–15% less off-drawing line at every level, for
  half a point of recall. On the real pairs it makes things worse. This is the same split as the boosted trees (WP-5)
  and the graph network (WP-7): evidence that separates candidates on the synthetic generator's images does not carry
  over to the provided real damage.
- In these datasets, damage is a clean erasure, so the only image evidence is ink a bridge would run over, and the
  binarized `crossed_ink` already holds most of that. Real heritage damage (abrasion, flaking, faded rather than
  missing ink) may leave traces inside gaps. An image-based scorer needs real damaged scans with ground truth to be
  trained and judged on; neither benchmark has such damage.
