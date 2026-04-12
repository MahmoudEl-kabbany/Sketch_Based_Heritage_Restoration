% ═══════════════════════════════════════════════════════════════════════════
% Heritage Sketch Restoration — ASP Rules
% ═══════════════════════════════════════════════════════════════════════════
%
% Facts supplied by Python (asp_engine.py):
%   endpoint(PathIdx, End, X, Y).
%   endpoint_id(EndpointId, PathIdx, End).
%   candidate(Id, PathA, EndA, PathB, EndB, Score).
%   candidate_endpoint(Id, EndpointAId, EndpointBId).
%   gestalt_continuation(Id, ContScore).   % 0..100
%   gestalt_closure(Id, Flag).             % 0 or 1
%   tier(Id, Tier).                        % 1 or 2

% --- Choice: accept or reject each candidate ---
{ accept(Id) } :- candidate(Id, _, _, _, _, _).

% --- Endpoint usage helper ---
uses_endpoint(Id, E) :- candidate_endpoint(Id, E, _).
uses_endpoint(Id, E) :- candidate_endpoint(Id, _, E).

% --- Strong endpoint occupancy by endpoint identity ---
:- accept(Id1), accept(Id2), Id1 != Id2,
   uses_endpoint(Id1, E), uses_endpoint(Id2, E).

% --- Constraint: each (path, end) participates in at most one connection ---
% Source side
:- accept(Id1), accept(Id2), Id1 != Id2,
   candidate(Id1, P, E, _, _, _),
   candidate(Id2, P, E, _, _, _).

% Target side
:- accept(Id1), accept(Id2), Id1 != Id2,
   candidate(Id1, _, _, P, E, _),
   candidate(Id2, _, _, P, E, _).

% Cross: source of one = target of another
:- accept(Id1), accept(Id2), Id1 != Id2,
   candidate(Id1, P, E, _, _, _),
   candidate(Id2, _, _, P, E, _).

% --- Constraint: no self-connections (same path both sides) ---
:- accept(Id), candidate(Id, P, _, P, _, _).

% --- Optimization: maximize total score ---
#maximize { Score, Id : accept(Id), candidate(Id, _, _, _, _, Score) }.

% --- Soft: favour closure-completing connections ---
:~ accept(Id), gestalt_closure(Id, 1). [-3, Id]

% --- Soft: penalize weak directional continuation ---
:~ accept(Id), gestalt_continuation(Id, C), C < 40. [2, Id]

% --- Soft: penalize Tier 2 candidates (prefer Tier 1 when both viable) ---
:~ accept(Id), tier(Id, 2). [1, Id]

% --- Show directive ---
#show accept/1.
