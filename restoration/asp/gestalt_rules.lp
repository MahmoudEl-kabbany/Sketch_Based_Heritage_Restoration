%% ═══════════════════════════════════════════════════════════════════════════
%% Gestalt Inference Rules  (R-3)
%% ═══════════════════════════════════════════════════════════════════════════
%%
%% Predicate conventions (integer-scaled × 100):
%%   closure(ContourID, Confidence, Persistence)
%%   continues(PathA, PathB, Gap, Angle, Confidence)
%%   symmetric(GroupID, Axis, Residual, Confidence)
%%   proximity_group(GroupID, EpsG, Size)
%%   member(ElementID, GroupID)
%%   similar_shape(PathA, PathB, Distance, Confidence)
%%   pragnanz(LoopID, Interpretation, Confidence)
%%
%% Derived action atoms:
%%   complete_contour/1, extend_curve/2, mirror_element/2,
%%   replicate_motif/3, group_elements/2, flag_similar_missing/2,
%%   simplify_interpretation/1
%% ═══════════════════════════════════════════════════════════════════════════

%% --- Closure → complete a contour that is not already observed closed ------
complete_contour(C) :-
    closure(C, Conf, _),
    Conf > 75,
    not observed_closed(C).

%% --- Good Continuation → extend a curve between two endpoints ---------------
extend_curve(A, B) :-
    continues(A, B, Gap, Angle, Conf),
    Gap < 2000,        % 20.0 px scaled ×100
    Angle < 1500,      % 15.0 deg scaled ×100
    Conf > 80.         % 0.80 scaled ×100

%% --- Symmetry → mirror an element across axis --------------------------------
mirror_element(E, Axis) :-
    symmetric(G, Axis, Res, Conf),
    member(E, G),
    Res < 500,         % 5.0 px scaled ×100
    Conf > 85,         % 0.85 scaled ×100
    not observed_at_mirror(E, Axis).

%% --- Periodicity → replicate a motif at an expected position -----------------
replicate_motif(M, Pos, T) :-
    periodic_pattern(M, T, Axis),
    expected_position(M, T, Pos),
    not observed_at(M, Pos).

%% --- Proximity → group nearby elements together ------------------------------
group_elements(E1, E2) :-
    proximity_group(G, _, _),
    member(E1, G),
    member(E2, G),
    E1 != E2.

%% --- Similarity → flag a pair where one may be missing -----------------------
flag_similar_missing(A, B) :-
    similar_shape(A, B, D, C),
    C > 85,            % confidence > 0.85
    D < 15,            % distance   < 0.15
    not both_observed(A, B).

%% --- Pragnanz → simplify an interpretation -----------------------------------
simplify_interpretation(L) :-
    pragnanz(L, Interp, Conf),
    Conf > 80.

%% ═══════════════════════════════════════════════════════════════════════════
%% Optimisation — minimise total cost (negated confidence of chosen actions)
%% ═══════════════════════════════════════════════════════════════════════════

#minimize {
    100 - Conf, complete_contour, C : complete_contour(C), closure(C, Conf, _);
    100 - Conf, extend_curve, A, B  : extend_curve(A, B), continues(A, B, _, _, Conf);
    100 - Conf, mirror, E, Axis     : mirror_element(E, Axis), symmetric(G, Axis, _, Conf), member(E, G);
    100 - C,    similar, A, B       : flag_similar_missing(A, B), similar_shape(A, B, _, C)
}.
