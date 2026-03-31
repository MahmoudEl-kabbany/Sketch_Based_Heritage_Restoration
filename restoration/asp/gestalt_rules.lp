%% ═══════════════════════════════════════════════════════════════════════════
%% Gestalt Inference Rules  (R-3)
%% ═══════════════════════════════════════════════════════════════════════════
%%
%% Predicate conventions (integer-scaled × 100):
%%   closure(ContourID, Confidence, Persistence)
%%   continues(PathA, PathB, EndpointA, EndpointB, Gap, Angle, Confidence)
%%     EndpointA/B: 0 = start, 1 = end
%%   symmetric(GroupID, Axis, Residual, Confidence)
%%   proximity_group(GroupID, EpsG, Size)
%%   member(ElementID, GroupID)
%%   similar_shape(PathA, PathB, Distance, Confidence)
%%   pragnanz(LoopID, Interpretation, Confidence)
%%
%% Derived action atoms:
%%   complete_contour/2, extend_curve/4, mirror_element/2,
%%   replicate_motif/3, group_elements/2, flag_similar_missing/2,
%%   simplify_interpretation/1
%% ═══════════════════════════════════════════════════════════════════════════

%% --- Closure → complete a contour that is not already observed closed ------
complete_contour(C, Conf) :-
    closure(C, Conf, _),
    Conf > 45,
    not observed_closed(C).

%% --- Good Continuation → extend a curve between two endpoints ---------------
%% Now carries endpoint labels (0=start, 1=end) so bridge_curves uses correct ends
extend_curve(A, B, EA, EB, Conf) :-
    continues(A, B, EA, EB, Gap, Angle, Conf),
    A != B,
    Gap < 5000,
    Angle < 3000,
    Conf > 35.

%% --- Mutual exclusion: an endpoint can only be used by one extend_curve ------
:- extend_curve(A, _, EA, _, _), extend_curve(A, _, EA, _, _),
   #count { B,EB : extend_curve(A, B, EA, EB, _) } > 1.

:- extend_curve(_, B, _, EB, _), extend_curve(_, B, _, EB, _),
   #count { A,EA : extend_curve(A, B, EA, EB, _) } > 1.

%% --- Symmetry → mirror an element across axis --------------------------------
mirror_element(E, Axis) :-
    symmetric(G, Axis, Res, Conf),
    member(E, G),
    Res < 1000,
    Conf > 55,
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
    C > 55,
    D < 50,
    not both_observed(A, B).

%% --- Pragnanz → simplify an interpretation -----------------------------------
simplify_interpretation(L) :-
    pragnanz(L, Interp, Conf),
    Conf > 60.

%% ═══════════════════════════════════════════════════════════════════════════
%% Optimisation — minimise total cost (negated confidence of chosen actions)
%% ═══════════════════════════════════════════════════════════════════════════

#minimize {
    100 - Conf, complete_contour, C : complete_contour(C, Conf), closure(C, Conf, _);
    100 - Conf, extend_curve, A, B, EA, EB : extend_curve(A, B, EA, EB, Conf), continues(A, B, EA, EB, _, _, Conf);
    100 - Conf, mirror, E, Axis     : mirror_element(E, Axis), symmetric(G, Axis, _, Conf), member(E, G);
    100 - C,    similar, A, B       : flag_similar_missing(A, B), similar_shape(A, B, _, C)
}.
