% ═══════════════════════════════════════════════════════════════════════════
% Heritage Sketch Restoration — ASP Rules
% ═══════════════════════════════════════════════════════════════════════════
%
% Facts supplied by Python (asp_engine.py):
%   candidate(Id, PathA, EndA, PathB, EndB, Score).
%   candidate_endpoint(Id, EndpointAId, EndpointBId).
%   candidate_self_closure(Id).
%   candidate_utility(Id, Utility).

% --- Choice: accept or reject each candidate ---
{ accept(Id) } :- candidate(Id, _, _, _, _, _).

% --- Endpoint usage helper ---
uses_endpoint(Id, E) :- candidate_endpoint(Id, E, _).
uses_endpoint(Id, E) :- candidate_endpoint(Id, _, E).

% Keep optional predicates in the grounded signature when no matching facts
% are present in an instance.
candidate_self_closure(-1).

% --- Strong endpoint occupancy by endpoint identity ---
endpoint_key(E) :- uses_endpoint(_, E).
:- endpoint_key(E), 2 { accept(Id) : uses_endpoint(Id, E) }.

% --- Constraint: no self-connections (same path both sides) ---
:- accept(Id), candidate(Id, P, _, P, _, _), not candidate_self_closure(Id).

% --- Optimization: maximize precomputed utility ---
#maximize { Utility, Id : accept(Id), candidate_utility(Id, Utility) }.

% --- Show directive ---
#show accept/1.
