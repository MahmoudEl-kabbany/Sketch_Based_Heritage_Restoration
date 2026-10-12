# Phase 4 — ASP Decision Engine

> **Source file:** [`restoration/asp_engine.py`](../restoration/asp_engine.py)  
> **ASP Rules:** [`restoration/asp_rules/restoration.lp`](../restoration/asp_rules/restoration.lp)

---

## 1. Purpose

Phase 4 frames the candidate selection process as an Answer Set Programming (ASP) problem. Instead of greedily picking connections, it globally optimizes the entire connection set simultaneously. 

It guarantees that:
- No endpoint is connected more than once (degree constraint)
- The chosen set of bridges maximizes the overall "utility" (sum of candidate scores with context-aware adjustments)
- Cross-shape false positive connections are mathematically suppressed
- The same candidates always produce the same set of bridges (the solve is deterministic)

---

## 2. Pipeline Overview

```
List[ConnectionCandidate]
    │
    ├─► Prune strictly dominated candidates (reduce search space)
    │
    ├─► Encode candidates to ASP facts (strings)
    │       └─► Compute effective utility per candidate
    │
    ├─► Call Clingo once for the whole image
    │
    └─► Decode accepted candidate IDs
            └─► List[ConnectionCandidate] (accepted only)
```

---

## 3. Encoding to ASP Facts

The system converts Python objects into logic programming facts. 

### 3.1 Core Facts

For each candidate, the following facts are generated:
```prolog
% candidate(CandidateID, PathIndexA, EndA, PathIndexB, EndB, ScaledScore).
% End maps: 0 = start, 1 = end. Score is int(float_score * 100).
candidate(42, 0, 1, 3, 0, 85).

% Maps CandidateID to unique endpoint tokens (stable IDs).
candidate_endpoint(42, TokenA, TokenB).

% Indicates this candidate closes a loop on a single path.
candidate_self_closure(42).
```

### 3.2 Effective Utility

To keep the ASP logic simple, many of the soft constraints and context adjustments are pre-calculated in Python and bundled into a single `candidate_utility` fact.

```python
def _candidate_effective_utility(candidate, path_strength):
    utility = int(candidate.score * 100)
    
    # Closure reward (+3 for strong closure, +2 for same-path)
    if closure_flag: utility += 3
    if same_path_closure: utility += 2
    
    # Cross-shape suppression
    # If a path has strong self-closure options (bucket 1 or 2), 
    # cross-path candidates touching it lose utility (-2 or -4).
    if bucket == 1: utility -= 2
    elif bucket == 2: utility -= 4
        
    # Weak continuation penalty
    if continuation_int < 40: utility -= 2
        
    # Weak extension penalty
    if extension_quality < 45: utility -= 3
        
    # Tier 2 penalty
    if tier == 2: utility -= 1
        
    return utility
```

Generated fact:
```prolog
candidate_utility(42, 87).
```

---

## 4. ASP Rules (`restoration.lp`)

The core logic is handled by `clingo` using the generated facts.

### 4.1 Choice Rule
```prolog
% We can choose to accept any candidate
{ accept(Id) } :- candidate(Id, _, _, _, _, _).
```

### 4.2 Hard Constraints
```prolog
% Extract all endpoints used by candidates
uses_endpoint(Id, E) :- candidate_endpoint(Id, E, _).
uses_endpoint(Id, E) :- candidate_endpoint(Id, _, E).
endpoint_key(E) :- uses_endpoint(_, E).

% No endpoint can be used by more than one accepted candidate
:- endpoint_key(E), 2 { accept(Id) : uses_endpoint(Id, E) }.

% Prevent trivial 0-length loops (connecting a path to itself if it's not a self_closure)
:- accept(Id), candidate(Id, P, _, P, _, _), not candidate_self_closure(Id).
```

### 4.3 Objective Function
```prolog
% 1st priority: maximize the sum of utilities of all accepted candidates
#maximize { Utility@2, Id : accept(Id), candidate_utility(Id, Utility) }.

% 2nd priority: prefer answers that connect both ends of more paths
#maximize { 8@1, P : path_fully_connected(P) }.

% 3rd priority (tie-break): prefer fewer bridges
#minimize { 1@0, Id : accept(Id) }.
```

A bridge with utility 0 that completes no path has nothing in its favour, so the tie-break leaves it out.

---

## 5. Optimization & Performance

### 5.1 Strict Dominance Pruning
If two candidates use the *exact same two endpoints*, only the one with the highest utility is kept. The other is permanently dropped before solving. (Exception: same-path closures are never pruned, as they influence soft suppression logic).

### 5.2 One Program per Image
All candidates of an image are solved together. Splitting them into independent endpoint-conflict components is not valid here: the reward for connecting both ends of a path (`path_fully_connected`) links the two endpoints of that path, and they can fall into different components.

### 5.3 Deterministic Solver Settings
`clingo` is run on a single thread with `--opt-mode=optN --opt-strategy=usc`. Core-guided optimization proves the optimum in milliseconds on these programs (about 0.4 s for the whole 32-image test set). With several threads the answer picked among equally good ones depends on which thread finishes first, so results differed between runs of the same input.

---

## 6. Decoding

After `clingo` finds the optimal model, it outputs `accept(Id)` atoms. These integer IDs are mapped back to the original `ConnectionCandidate` objects in Python.

```python
def decode_solution(accepted_ids, candidates):
    id_set = set(accepted_ids)
    return [c for c in candidates if c.id in id_set]
```

---

## 7. Entry Point

```python
def solve_candidates(
    candidates: List[ConnectionCandidate],
    rules_path: str = RULES_PATH,
    max_solutions: int = 1,
    timeout_s: float = 30.0,
) -> Tuple[List[int], Dict[str, Any]]:
```

---

## 8. Dependencies

| Library | Usage |
|---|---|
| `clingo` | ASP Grounding and Solving Engine |
