% ==============================================================================
% Module: Neuro-Symbolic Stroke Detection Logic (BE-FAST)
% Description:
%   This DeepProbLog program serves as the inferential logic layer for stroke
%   assessment. It integrates continuous probabilistic outputs from a visual 
%   Convolutional Neural Network (CNN) with discrete, symbolically encoded 
%   clinical guidelines (BE-FAST criteria). It evaluates patient symptoms 
%   and visual data to compute the overarching probability of a stroke event.
% ==============================================================================

% ------------------------------------------------------------------------------
% 0. BASE INITIALIZATION
% ------------------------------------------------------------------------------
% Ensures predicates exist in the graph even when symptoms are absent.

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
% The NeutralImg serves as a baseline reference. By comparing the resting state 
% to the active state (SmileImg), the CNN isolates acute facial droop from 
% naturally occurring baseline asymmetry.

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

0.56::speech_deficit(P) :- gender(P, female), speech_issue(P).
0.42::speech_deficit(P) :- gender(P, male),   speech_issue(P).

0.89::arm_deficit(P) :- arm_weakness(P).

fast_positive(P) :-
    facial_droop_detected(P, _, _).

fast_positive(P) :-
    speech_deficit(P).

fast_positive(P) :-
    arm_deficit(P).


% ------------------------------------------------------------------------------
% 3. STROKE DETECTION
% ------------------------------------------------------------------------------

% Neural + reported symptoms
0.73::stroke(P) :-
    facial_droop_detected(P, _, _),
    (speech_deficit(P) ; arm_deficit(P)).

% Reported symptoms only
0.56::stroke(P) :-
    \+ facial_droop_detected(P, _, _),
    (speech_deficit(P) ; arm_deficit(P)).

% Neural signal only
0.60::stroke(P) :-
    facial_droop_detected(P, _, _),
    \+ speech_deficit(P),
    \+ arm_deficit(P).


% ------------------------------------------------------------------------------
% 4. BE (Balance & Eyes) - ATYPICAL PRESENTATIONS
% ------------------------------------------------------------------------------

0.20::atypical_stroke(P) :-
    \+ stroke(P),
    dizziness(P).

0.527::atypical_stroke(P) :-
    \+ stroke(P),
    vision_change(P).


% ------------------------------------------------------------------------------
% 5. RECURRENCE & MIMIC MODIFIERS
% ------------------------------------------------------------------------------

0.10::recurrence_boost(P) :-
    history_recent_tia(P).

0.14::is_mimic(P) :-
    history_prior_stroke(P),
    \+ new_symptom(P).