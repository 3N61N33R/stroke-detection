% ==============================================================================
% STROKE DETECTION – LOGIC LAYER (BE-FAST)
% ==============================================================================
% DeepProbLog program combining neural facial analysis with symbolic
% clinical rules for stroke risk assessment.
% ==============================================================================


% ------------------------------------------------------------------------------
% 0. BASE INITIALIZATION
% ------------------------------------------------------------------------------
% Ensures predicates exist even when symptoms are absent.

gender(null_patient, null).
speech_issue(null_patient).
arm_weakness(null_patient).
vision_change(null_patient).
dizziness(null_patient).
history_recent_tia(null_patient).
history_prior_stroke(null_patient).
new_symptom(null_patient).
check_face(null_patient, null).


% ------------------------------------------------------------------------------
% 1. NEURAL INTERFACE (Facial Droop)
% ------------------------------------------------------------------------------

nn(droop_classifier, [Image], State, [normal, droop]) ::
    check_face(Image, State).

facial_droop_detected(Person, NeutralImg, SmileImg) :-
    belongs_to(NeutralImg, Person),
    belongs_to(SmileImg, Person),
    check_face(NeutralImg, normal),
    check_face(SmileImg, droop).

facial_droop_detected(Person, NeutralImg, SmileImg) :-
    belongs_to(NeutralImg, Person),
    belongs_to(SmileImg, Person),
    check_face(NeutralImg, droop),
    check_face(SmileImg, droop).


% ------------------------------------------------------------------------------
% 2. SYMPTOM WEIGHTING (FAST CORE)
% ------------------------------------------------------------------------------

0.56::speech_risk(P) :- gender(P, female), speech_issue(P).
0.42::speech_risk(P) :- gender(P, male),   speech_issue(P).

0.89::arm_risk(P) :- arm_weakness(P).

fast_positive(P) :-
    facial_droop_detected(P, _, _).

fast_positive(P) :-
    speech_risk(P).

fast_positive(P) :-
    arm_risk(P).


% ------------------------------------------------------------------------------
% 3. STROKE PROBABILITY
% ------------------------------------------------------------------------------

% Neural + reported symptoms
0.73::stroke_probability(P) :-
    facial_droop_detected(P, _, _),
    (speech_risk(P) ; arm_risk(P)).

% Reported symptoms only
0.56::stroke_probability(P) :-
    \+ facial_droop_detected(P, _, _),
    (speech_risk(P) ; arm_risk(P)).

% Neural signal only
0.60::stroke_probability(P) :-
    facial_droop_detected(P, _, _),
    \+ speech_risk(P),
    \+ arm_risk(P).


% ------------------------------------------------------------------------------
% 4. BE (Balance & Eyes)
% ------------------------------------------------------------------------------

0.20::hidden_stroke_risk(P) :-
    \+ stroke_probability(P),
    dizziness(P).

0.527::hidden_stroke_risk(P) :-
    \+ stroke_probability(P),
    vision_change(P).


% ------------------------------------------------------------------------------
% 5. RISK MODIFIERS
% ------------------------------------------------------------------------------

0.10::recurrence_boost(P) :-
    history_recent_tia(P).

0.14::is_mimic(P) :-
    history_prior_stroke(P),
    \+ new_symptom(P).


% ------------------------------------------------------------------------------
% 6. CLINICAL DECISION
% ------------------------------------------------------------------------------

urgent_call_911(P) :-
    stroke_probability(P),
    fast_positive(P),
    \+ is_mimic(P).

urgent_call_911(P) :-
    stroke_probability(P),
    recurrence_boost(P),
    \+ is_mimic(P).

seek_urgent_care(P) :-
    stroke_probability(P),
    \+ urgent_call_911(P),
    \+ is_mimic(P).

seek_urgent_care(P) :-
    hidden_stroke_risk(P),
    \+ is_mimic(P).

seek_urgent_care(P) :-
    recurrence_boost(P),
    \+ urgent_call_911(P),
    \+ is_mimic(P).

consider_evaluation(P) :-
    hidden_stroke_risk(P),
    is_mimic(P).

consider_evaluation(P) :-
    is_mimic(P),
    \+ stroke_probability(P),
    \+ hidden_stroke_risk(P).


% ------------------------------------------------------------------------------
% 7. UX RISK CATEGORIES
% ------------------------------------------------------------------------------

risk_category(critical, P) :-
    urgent_call_911(P).

risk_category(high, P) :-
    seek_urgent_care(P),
    \+ urgent_call_911(P).

risk_category(moderate, P) :-
    consider_evaluation(P),
    \+ seek_urgent_care(P),
    \+ urgent_call_911(P).

risk_category(low, P) :-
    \+ urgent_call_911(P),
    \+ seek_urgent_care(P),
    \+ consider_evaluation(P).
