# Phase 4 — ASP Decision Engine

> **Source file:** [`restoration/asp_engine.py`](../restoration/asp_engine.py)  
> **Rules:** [`restoration/asp_rules/restoration.lp`](../restoration/asp_rules/restoration.lp)

---

## 1. Purpose

Phase 3 says how likely each candidate bridge is on its own. Candidates compete: a stroke end can be continued by one bridge only. Phase 4 chooses the **most plausible set of bridges that fit together**, with an Answer Set Programming solver (clingo).

The solver's answer is final. Nothing is removed from it afterwards; an earlier version deleted bridges after the solve, which left their stroke ends with no chance to take their next-best partner.

---

## 2. Pipeline Overview

```
Scored candidates
    │
    ├─► Keep the candidates above the acceptance threshold (positive utility)
    ├─► Encode them as ASP facts
    ├─► Solve: maximize total utility, each stroke end used at most once
    └─► Decode accepted ids back to ConnectionCandidate objects
```

---

## 3. Utility

```python
utility = round(candidate.score * 10)
```

`score` is the candidate's log-odds above the acceptance threshold (Phase 3). The utility of a set of bridges is the sum of their utilities, so the solver maximizes the summed log-odds of the bridges it draws.

- A candidate at or below the threshold has no positive utility. Accepting it could only lower the total, so it is not passed to the solver at all. This keeps the program small: a few dozen candidates per image.
- There are no other terms. The reward for connecting both ends of a path, the bonuses for closures and the penalties by scenario that this phase used to add by hand are gone; what they tried to express is now a fitted measurement in Phase 3 or was measured and found not to help (section 6).

---

## 4. ASP Rules (`restoration.lp`)

Facts, one per candidate:

```prolog
candidate(Id, EndpointA, EndpointB, Utility).
```

Rules:

```prolog
% Choice: accept or reject each candidate
{ accept(Id) } :- candidate(Id, _, _, _).

% A stroke end can be continued by at most one bridge
uses_endpoint(Id, E) :- candidate(Id, E, _, _).
uses_endpoint(Id, E) :- candidate(Id, _, E, _).
:- uses_endpoint(_, E), 2 { accept(Id) : uses_endpoint(Id, E) }.

% The most plausible set of bridges
#maximize { Utility@1, Id : accept(Id), candidate(Id, _, _, Utility) }.

% Among otherwise equal answers, prefer fewer bridges
#minimize { 1@0, Id : accept(Id) }.
```

The two shapes of bridge between the same pair of ends share both ends, so at most one of them is accepted: the more probable one.

### 4.1 User constraints (interactive refinement)

A user can forbid, force or hint bridges (`restore(..., user_constraints=...)`, `scripts/refine.py`; see `restoration/constraints.py`). The constraints name stroke ends by position; each position is resolved to the nearest free end, and it is an error if none is within `tolerance_px`. They reach the solver as facts:

```prolog
forbid_pair(EndpointA, EndpointB).
force_pair(EndpointA, EndpointB).
```

and these rules, which do nothing without such facts (the `-1` dummies keep the predicates defined):

```prolog
forbid_pair(-1, -1).
force_pair(-1, -1).
:- accept(Id), candidate(Id, A, B, _), forbid_pair(A, B).
:- accept(Id), candidate(Id, B, A, _), forbid_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, A, B, _), force_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, B, A, _), force_pair(A, B).
:- force_pair(A, B), A >= 0, not forced_satisfied(A, B).
```

- **Forced and hinted pairs are always encoded**, whatever their utility. If the pipeline proposed no candidate for such a pair, a smooth one is synthesized before scoring.
- **A hinted candidate is worth at least `HINT_BONUS` = 100 units** (10 log-odds): its own utility if positive, plus the bonus. It is drawn unless a force needs one of its ends, or the bridges it displaces are together worth more.
- **The whole drawing is re-optimized.** Forbidding a bridge frees its ends for other partners. Forcing one takes its ends away from their former partners.
- **Inconsistent constraints are rejected before solving:** two forced bridges sharing an end, or a pair both forced and forbidden.

Without constraints the facts are exactly those of before, and the answers are identical (exact snapshot).

---

## 5. Deterministic Solving

```python
["--models=1", "--opt-mode=optN", "--opt-strategy=usc"]
```

- One program per image, on a single thread: the same candidates always give the same bridges. With several threads the choice among equally good answers depends on which thread finishes first.
- Core-guided optimization (`usc`) proves the optimum in milliseconds on these programs.
- `asp_timeout_s` (default 30 s) bounds the solve; it is not reached in practice.

---

## 6. Measured and Left Out

Each of these was tried on the benchmark (README) and kept only if it reduced the line drawn off the drawing without losing repaired gaps, or the reverse.

| Idea | Effect | Decision |
|---|---|---|
| Accepted bridges may not cross each other | −1.4 points of gap recall, no fewer wrong strokes: it forbids the repair of an erased crossing, which needs two bridges that cross | left out |
| Reward for a path with both ends bridged | +0.5 to +1.1 points of gap recall, 33% to 80% more line off the drawing | left out |
| Image-mask veto after the solve | replaced by the `crossed_ink` measurement, which the solver sees | removed |

---

## 7. Entry Points

```python
def solve_candidates(
    candidates: List[ConnectionCandidate],
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
    timeout_s: float = 30.0,
    constraints: Optional[ResolvedConstraints] = None,   # forbid / force / hint, see 4.1
) -> Tuple[List[int], Dict[str, Any]]:      # accepted ids, solver statistics

def decode_solution(accepted_ids, candidates) -> List[ConnectionCandidate]
```

---

## 8. Dependencies

| Library | Usage |
|---|---|
| `clingo` | ASP grounding and solving |
