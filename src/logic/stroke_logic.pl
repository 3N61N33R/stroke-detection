% ==============================================================================
% STROKE DETECTION - LOGIC LAYER (BE-FAST COMPLIANT)
% ==============================================================================
% This file defines the Probabilistic Logic Program (DeepProbLog) for stroke risk
% assessment. It integrates neural observations (facial droop) with symbolic
% clinical rules based on the BE-FAST protocol.
% ==============================================================================

% ------------------------------------------------------------------------------
% 0. BASE CASES (PREDICATE INITIALIZATION)
% ------------------------------------------------------------------------------
% Define base cases to ensure predicates exist in the knowledge base.
% This prevents runtime errors when specific symptoms are absent for a patient.
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
% 1. NEURAL INTERFACE (The "F" in FAST)
% ------------------------------------------------------------------------------
% Rules connecting neural network outputs (check_face/2) to logical symptoms.
% Facial droop is confirmed if the smile deviates from the neutral state or
% if both states register as droop.

% nn(face_net, [Img], State, [normal,droop]) :: check_face(Img, State).

% Face states per patient
neutral_face(P, NeutralImg).
smile_face(P, SmileImg).

% Dynamic droop: neutral normal, smile droop
facial_droop_detected(P) :-
    neutral_face(P, NeutralImg),
    smile_face(P, SmileImg),
    check_face(NeutralImg, normal),
    check_face(SmileImg, droop).

% Static droop: both state droop
facial_droop_detected(P) :-
    neutral_face(P, NeutralImg),
    smile_face(P, SmileImg),
    check_face(NeutralImg, droop),
    check_face(SmileImg, droop).

% ------------------------------------------------------------------------------
% 2. SYMPTOM LOGIC (The "A", "S", and "T" in FAST)
% These rules define the presence of speech issues, arm weakness, and vision changes.
% ------------------------------------------------------------------------------

speech_positive(P) :- speech_issue(P).
arm_positive(P) :- arm_weakness(P).
vision_positive(P) :- vision_change(P).
dizziness_positive(P) :- dizziness(P).

fast_positive(P) :- facial_droop_detected(P).
fast_positive(P) :- speech_positive(P).
fast_positive(P) :- arm_positive(P).

% ------------------------------------------------------------------------------
% 3. STROKE PROBABILITY (CORE FAST LOGIC)
% ------------------------------------------------------------------------------

% High Confidence: Neural vision confirmation + User reported symptoms.
0.73::stroke(P) :-
    facial_droop_detected(P),
    speech_positive(P).

0.73::stroke(P) :-
    facial_droop_detected(P),
    arm_positive(P).

% Moderate Confidence: User reported symptoms only (No visual confirmation).
0.56::stroke(P) :-
    \+ facial_droop_detected(P),
    speech_positive(P).

0.56::stroke(P) :-
    \+ facial_droop_detected(P),
    arm_positive(P).

% Visual Only: Camera detects droop, but user reports no other symptoms.
0.60::stroke(P) :-
    facial_droop_detected(P),
    \+ speech_positive(P),
    \+ arm_positive(P).

% ------------------------------------------------------------------------------
% 4. MISSED SYMPTOMS (The "BE" in BE-FAST)
% ------------------------------------------------------------------------------
% These rules capture posterior circulation strokes often missed by standard FAST.

% Balance (Dizziness)
0.20::hidden_stroke(P) :-
    dizziness(P),
    \+ fast_positive(P).

% Eyes (Vision Change)
0.527::hidden_stroke(P) :-
    vision_change(P),
    \+ fast_positive(P).

% ------------------------------------------------------------------------------
% 5. RISK MODIFIERS (PATIENT HISTORY)
% ------------------------------------------------------------------------------
% TIA (Transient Ischemic Attack) is a significant precursor to stroke.
0.10::recurrence_boost(P) :- history_recent_tia(P).

% Stroke Mimic Logic:
% A prior stroke is treated as a mimic (false positive) only if no NEW symptoms
% have appeared, suggesting residual effects rather than an acute event.
0.14::is_mimic(P) :- history_prior_stroke(P), \+ new_symptom(P).

% ------------------------------------------------------------------------------
% 6. CLINICAL DECISION OUTPUTS
% ------------------------------------------------------------------------------
% Hierarchy of action based on calculated risks and mimics.

% CRITICAL: Call 911 immediately.
urgent_call_911(P) :-
    stroke(P),
    fast_positive(P),
    \+ is_mimic(P).

urgent_call_911(P) :-
    stroke(P),
    recurrence_boost(P),
    \+ is_mimic(P).

% URGENT: Seek medical care ASAP (Lower probability or hidden symptoms).
seek_urgent_care(P) :-
    stroke(P),
    \+ urgent_call_911(P),
    \+ is_mimic(P).

seek_urgent_care(P) :-
    hidden_stroke(P),
    \+ is_mimic(P).

seek_urgent_care(P) :-
    recurrence_boost(P),
    \+ urgent_call_911(P),
    \+ is_mimic(P).

% EVALUATE: Symptoms present but likely a mimic (e.g., old stroke).
consider_evaluation(P) :-
    hidden_stroke(P),
    is_mimic(P).

consider_evaluation(P) :-
    is_mimic(P),
    \+ stroke(P),
    \+ hidden_stroke(P).

% ------------------------------------------------------------------------------
% 7. UX HELPERS (RISK CATEGORIZATION)
% ------------------------------------------------------------------------------
% Maps clinical decisions to user-facing risk categories.

risk_category(critical, P) :- urgent_call_911(P).
risk_category(high, P)     :- seek_urgent_care(P), \+ urgent_call_911(P).
risk_category(moderate, P) :- consider_evaluation(P), \+ seek_urgent_care(P), \+ urgent_call_911(P).

% Low risk default category if no urgent symptoms or mimics are detected.
risk_category(low, P) :-
    \+ urgent_call_911(P),
    \+ seek_urgent_care(P),
    \+ consider_evaluation(P).

risk_category(low, P) :-
    is_mimic(P),
    \+ stroke(P),
    \+ hidden_stroke(P),
    \+ recurrence_boost(P).