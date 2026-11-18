# WP-6 — Interactive refinement

Branch `wp6-interactive-refinement` (from `wp5-scorer-ablation`). **Without constraints the output is unchanged:**
the snapshot is identical for 64/64 images, and `check_changes.py` reports UNCHANGED (86.3% recall, 39.3 line widths
off the drawing).

## What was added

| Part | Where | What it does |
|---|---|---|
| Constraint format | `restoration/constraints.py` | `forbid` / `force` / `hint` pairs of stroke ends **by position**, plus `tolerance_px` (default 8), as a dict or JSON file. Each position resolves to the nearest free end; no end within tolerance raises `ConstraintError`. Two forces on one end, or a pair both forced and forbidden, are rejected. |
| Missing candidates | `add_missing_candidates` | A forced or hinted pair the pipeline never proposed gets a smooth candidate, added before scoring, so it has real measurements and a probability. |
| ASP | `asp_rules/restoration.lp`, `asp_engine.encode_facts` | `forbid_pair/2` and `force_pair/2` facts, with grounding dummies. Integrity constraints cover both orders of the ends. Forced and hinted candidates are always encoded. A hinted candidate is worth `max(utility, 0) + HINT_BONUS` (100 units = 10 log-odds). If no answer satisfies the forces, that is an error, not an empty restoration. |
| API | `restore(..., user_constraints=dict \| path)` | Re-solves the whole drawing. `RestorationResult` now also carries `extraction`, `candidates` and `accepted`. |
| Report | schema **4.1** | `user_constraints` (null without constraints): one row per constraint with resolved ends, candidates, `synthesized`, `bridged` and `held`. Bridge events of forced or hinted pairs have `"user"`. |
| Click tool | `scripts/refine.py` | OpenCV window: click two ends, then `f`/`h`/`x`; right-click a bridge to forbid it; `u` undo, `s` save, `q` quit. It re-solves after every change, and a constraint that cannot be satisfied is reported and not applied. The session logic (`RefineSession`) has no GUI dependency and is unit-tested. |
| Feedback | `restoration/feedback.py`, `train_scoring.py --include-feedback` | Every action is appended to `feedback/feedback.jsonl` (git-ignored) with the affected candidates' scorecard-scale measurements: forbid → label 0 for every candidate of the pair; force/hint → label 1 for the bridge drawn. A later action on the same pair replaces an earlier one. In training, each corrected drawing becomes a pseudo-image of kind `extra_feedback`. |

The post-solve filters the first draft mentioned (`_sanitize_accepted_candidates`, the Phase 4.5 mask filter) no longer
exist on this branch. Nothing after the solver drops an accepted bridge, so forced bridges need no exemption.

## Tests

- `test_user_constraints_forbid_force_and_hint` (synthetic drawing):
  - Forbidding the default bridge removes it.
  - Forcing one end onto another stroke takes that end from the gap it used to close (global re-optimization).
  - Hinting the same unlikely pair is enough to draw it.
  - Unresolvable positions and contradictory constraints raise `ConstraintError`.
  - The report rows say the constraints held.
- `test_refine_session_logs_feedback_that_trains`: the click-tool session runs forbid, force, a rejected second force
  on the same end, undo and save. The feedback log reads back with labels 0/1 and in the shape `train_scoring` expects.
- `train_scoring.py --include-feedback` was smoke-tested end to end, writing to scratch; the shipped model was not touched.
- The OpenCV window itself (`run_window`) was not run in this session, since it needs a display and a person
  clicking. To try it: `python scripts/refine.py test_images/difficult_test_cases/ankh_damaged.png`.

## Limits

- A hint is a strong preference, not a guarantee. It loses to a force on one of its ends, or to displaced bridges
  together worth more than 10 log-odds.
- Feedback rows carry no gap ground truth. They train the scorer but do not appear in the gap-level selection report
  (their `gap_count` is 0).
- With a training subset this small that the feedback "drawing" shares the only fold (e.g. `--filter ankh`), one fold
  has nothing to train on. With the full benchmark this cannot happen.
