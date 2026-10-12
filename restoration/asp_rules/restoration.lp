% ═══════════════════════════════════════════════════════════════════════════
% Heritage Sketch Restoration — ASP Rules
% ═══════════════════════════════════════════════════════════════════════════
%
% Facts supplied by Python (asp_engine.py):
%   candidate(Id, EndpointA, EndpointB, Utility).
%       A bridge between two free stroke ends. Utility is its log-odds of
%       being a missing stroke, above the acceptance threshold (scaled to an
%       integer). Only candidates with positive utility are supplied.

% --- Choice: accept or reject each candidate ---
{ accept(Id) } :- candidate(Id, _, _, _).

% --- A stroke end can be continued by at most one bridge ---
uses_endpoint(Id, E) :- candidate(Id, E, _, _).
uses_endpoint(Id, E) :- candidate(Id, _, E, _).
:- uses_endpoint(_, E), 2 { accept(Id) : uses_endpoint(Id, E) }.

% --- Optimization: the most plausible set of bridges ---
#maximize { Utility@1, Id : accept(Id), candidate(Id, _, _, Utility) }.

% --- Tie-break: among otherwise equal answers, prefer fewer bridges ---
#minimize { 1@0, Id : accept(Id) }.

% --- Show directive ---
#show accept/1.
