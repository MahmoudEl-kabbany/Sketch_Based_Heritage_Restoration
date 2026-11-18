% ═══════════════════════════════════════════════════════════════════════════
% Heritage Sketch Restoration — ASP Rules
% ═══════════════════════════════════════════════════════════════════════════
%
% Facts supplied by Python (asp_engine.py):
%   candidate(Id, EndpointA, EndpointB, Utility).
%       A bridge between two free stroke ends. Utility is its log-odds of
%       being a missing stroke, above the acceptance threshold (scaled to an
%       integer). Only candidates with positive utility are supplied, and
%       those whose two ends the user forced or hinted (a hinted one carries a
%       utility bonus).
%   forbid_pair(EndpointA, EndpointB).   no bridge may join these two ends
%   force_pair(EndpointA, EndpointB).    a bridge must join these two ends
%       (restoration/constraints.py; both orders of the ends are meant).

% --- Choice: accept or reject each candidate ---
{ accept(Id) } :- candidate(Id, _, _, _).

% --- A stroke end can be continued by at most one bridge ---
uses_endpoint(Id, E) :- candidate(Id, E, _, _).
uses_endpoint(Id, E) :- candidate(Id, _, E, _).
:- uses_endpoint(_, E), 2 { accept(Id) : uses_endpoint(Id, E) }.

% --- User constraints (none by default; the dummies keep the predicates defined) ---
forbid_pair(-1, -1).
force_pair(-1, -1).
:- accept(Id), candidate(Id, A, B, _), forbid_pair(A, B).
:- accept(Id), candidate(Id, B, A, _), forbid_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, A, B, _), force_pair(A, B).
forced_satisfied(A, B) :- accept(Id), candidate(Id, B, A, _), force_pair(A, B).
:- force_pair(A, B), A >= 0, not forced_satisfied(A, B).

% --- Optimization: the most plausible set of bridges ---
#maximize { Utility@1, Id : accept(Id), candidate(Id, _, _, Utility) }.

% --- Tie-break: among otherwise equal answers, prefer fewer bridges ---
#minimize { 1@0, Id : accept(Id) }.

% --- Show directive ---
#show accept/1.
