% ═══════════════════════════════════════════════════════════════════════
% Gestalt-Informed Restoration Rules  (ASP / clingo)
% ═══════════════════════════════════════════════════════════════════════
%
% Facts are injected by serialize_features_to_asp().
% All confidence values are integers in [0, 100].
%
% Action atoms produced:
%   extend_curve(A, B, EA, EB, Conf)   — bridge gap between paths A,B
%   complete_contour(P, Conf)          — EFD-close open contour P
%   mirror_element(P, Axis)            — reflect path P across Axis
%
% ─── Domain declarations (from injected facts) ───────────────────────
% path(P).
% open_path(P).
% observed_closed(P).
% continues(A, B, EA, EB, Conf).
% closure(P, Conf).
% proximity_group(A, B).
% similar_shape(A, B).
% symmetry_axis(CX, CY, Angle).
% pragnanz(P, Circ).
% gap_distance(A, B, Dist).
% continuation_angle(A, B, Angle).
% segment_type(P, Seg, Type).
% max_gap_limit(V).   max_angle_limit(V).
% min_continue_conf(V).   min_closure_conf(V).

% ─── Good Continuation ───────────────────────────────────────────────
% A continuation fact with sufficient confidence becomes an action.
{ extend_curve(A, B, EA, EB, Conf) } :-
    continues(A, B, EA, EB, Conf),
    min_continue_conf(MinConf),
    Conf >= MinConf.

% ─── Closure (EFD-based contour completion) ──────────────────────────
{ complete_contour(P, Conf) } :-
    closure(P, Conf),
    min_closure_conf(MinConf),
    Conf >= MinConf,
    not observed_closed(P).

% ─── Proximity boost ─────────────────────────────────────────────────
% If two paths are in a proximity group and a continuation exists,
% boost the preference for connecting them.
proximity_boost(A, B, 10) :-
    proximity_group(A, B),
    continues(A, B, _, _, _).

% ─── Symmetry-driven restoration ─────────────────────────────────────
{ mirror_element(P, Angle) } :-
    symmetry_axis(_, _, Angle),
    open_path(P).

% ─── Pragnanz (regularity) ───────────────────────────────────────────
% Highly regular shapes prefer closure over bridging.
prefer_closure(P) :-
    pragnanz(P, Circ),
    Circ > 70,
    closure(P, _).

% ─── Similar shape linking ───────────────────────────────────────────
% If two shapes are similar, connecting their fragments is encouraged.
shape_link(A, B) :- similar_shape(A, B).

% ─── Mutual exclusion: each endpoint participates in at most 1 action ─
% An endpoint (path, side) can only be extended once.
:- extend_curve(A, _, EA, _, _), extend_curve(A, _, EA, _, _),
   #count{ B, EB : extend_curve(A, B, EA, EB, _) } > 1.

:- extend_curve(_, B, _, EB, _), extend_curve(_, B, _, EB, _),
   #count{ A, EA : extend_curve(A, B, EA, EB, _) } > 1.

% Don't both close and extend the same path (closure is sufficient)
:- complete_contour(P, _), extend_curve(P, _, _, _, _).
:- complete_contour(P, _), extend_curve(_, P, _, _, _).

% Don't self-bridge (path A to itself via extend_curve)
:- extend_curve(A, A, _, _, _).

% ─── Optimisation: maximise total confidence of applied actions ──────
#maximize {
    Conf, extend, A, B : extend_curve(A, B, _, _, Conf) ;
    Conf, close, P     : complete_contour(P, Conf) ;
    Boost, prox, A, B  : proximity_boost(A, B, Boost),
                          extend_curve(A, B, _, _, _)
}.

% ─── Show only action atoms ──────────────────────────────────────────
#show extend_curve/5.
#show complete_contour/2.
#show mirror_element/2.
