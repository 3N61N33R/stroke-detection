% ==============================================================================
% STROKE DETECTION - MERGED VERSION
% ==============================================================================

% ==============================================================================
% SECTION 1: NEURAL PERCEPTION (The "Eyes")
% ==============================================================================
% We use a Neural Predicate to classify TWO images: Neutral and Smile.
% This detects dynamic asymmetry (Mild Droop) vs static asymmetry (Severe Droop).

nn(stroke_resnet, [Img], [normal, droop]) :: check_face(Img).

% RULE: Facial Droop is DETECTED if:
% 1. Neutral face is Normal BUT Smile has Droop (Mild/Action Droop).
% 2. Neutral face has Droop AND Smile has Droop (Severe/Static Droop).
facial_droop_detected(NeutralImg, SmileImg) :-
    check_face(NeutralImg, normal),
    check_face(SmileImg, droop).

facial_droop_detected(NeutralImg, SmileImg) :-
    check_face(NeutralImg, droop),
    check_face(SmileImg, droop).

% ==============================================================================
% SECTION 2: GENDER & SYMPTOM BIAS (The "Demographics")
% ==============================================================================
% ==============================================================================
% SPEECH DIFFICULTY - QUESTION BASED
% ==============================================================================
% User reports speech difficulty via questionnaire.
% Women are significantly more likely to present with speech issues than men.
% Source: Berglund et al. (2014) [cite_start][cite: 562].

% IF Female AND Speech Issues -> 56% Probability weight for this symptom.
0.56::speech_risk(P) :- gender(P, female), speech_issue(P).

% IF Male AND Speech Issues -> 42% Probability weight for this symptom.
0.42::speech_risk(P) :- gender(P, male), speech_issue(P).

% ==============================================================================
% ARM WEAKNESS - QUESTION BASED
% ==============================================================================
% User reports arm weakness via questionnaire.

% Base weight for Arm Weakness when reported (No significant gender difference).
% Source: Claus et al. (2024) [cite_start][cite: 118].
0.89::arm_risk(P) :- arm_weakness(P).

% ==============================================================================
% SECTION 3: CORE STROKE PROBABILITY (The "FAST" Logic)
% ==============================================================================
% FAST is positive if AT LEAST ONE core symptom is present
fast_positive(P) :-
    facial_droop_detected(P_Neutral, P_Smile).

fast_positive(P) :-
    speech_risk(P).

fast_positive(P) :-
    arm_risk(P).

% We determine confidence based on WHO is reporting (Camera vs. User).
% SCENARIO A: HIGH CONFIDENCE (Camera Confirmed)
% If the Neural Net sees a droop AND the user reports symptoms.
[cite_start]% We use the 73% PPV found in Ambulance/On-Scene settings[cite: 538, 558].
0.73::stroke_probability(P) :-
    fast_positive(P),
    facial_droop_detected(P_Neutral, P_Smile),
    (speech_risk(P) ; arm_risk(P)).

% SCENARIO B: MODERATE CONFIDENCE (User Report Only)
% If the Camera is normal (or unsure) but user reports FAST symptoms.
[cite_start]% We use the 56% PPV found in Dispatcher/Phone settings[cite: 538, 554].
0.56::stroke_probability(P) :-
    fast_positive(P),
    \+ facial_droop_detected(P_Neutral, P_Smile),
    (speech_risk(P) ; arm_risk(P)).

% SCENARIO C: VISUAL ONLY
% Camera sees droop, but user denies other symptoms.
% We assign 60% (average of specificity) but flag it for further evaluation.
0.60::stroke_probability(P) :-
    fast_positive(P),
    facial_droop_detected(P_Neutral, P_Smile),
    \+ speech_risk(P),
    \+ arm_risk(P).

% ==============================================================================
% SECTION 4: THE "MISSED" SYMPTOMS (BE-FAST Expansion)
% ==============================================================================
% Standard FAST misses ~30% of minor strokes. We check for Eyes & Balance.
% Source: Claus et al. (2024) [cite_start][cite: 191].

% VISUALS: 52.7% of FAST-negative strokes had visual symptoms.
0.527::hidden_stroke_risk(P) :-
    \+ stroke_probability(P), % FAST was negative
    vision_change(P).

% BALANCE: 19.5% of FAST-negative strokes had dizziness.
% Note: Specificity is low (lots of false positives), so weight is low (20%).
0.20::hidden_stroke_risk(P) :-
    \+ stroke_probability(P),
    dizziness(P).

% ==============================================================================
% SECTION 5: RISK MODIFIERS (History & Mimics)
% ==============================================================================

% RECURRENCE RISK:
[cite_start]% 10% risk of stroke within 1 week of a TIA[cite: 38].
0.10::recurrence_boost(P) :- history_recent_tia(P).

% STROKE MIMICS (Reducing the Probability):
[cite_start]% 1. Seizures: 21% of false positives[cite: 857].
0.21::is_mimic(P) :- history_seizures(P).

[cite_start]% 2. Old Stroke: 14% of false positives are residual deficits[cite: 871].
0.14::is_mimic(P) :- history_prior_stroke(P), \+ new_symptom(P).

% ==============================================================================
% SECTION 6: CLINICAL DECISION OUTPUTS
% ==============================================================================
% Granular action levels for better UX guidance

% CRITICAL: Immediate 911 call required
urgent_call_911(P) :-
    stroke_probability(P),
    fast_positive(P),
    \+ is_mimic(P).

urgent_call_911(P) :-
    stroke_probability(P),
    recurrence_boost(P),
    \+ is_mimic(P).

% URGENT: Seek immediate medical attention (ER visit)
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

% MONITOR: Consider medical evaluation (less urgent)
consider_evaluation(P) :-
    hidden_stroke_risk(P),
    is_mimic(P).  % Even if mimic, visual/balance symptoms need checking

consider_evaluation(P) :-
    is_mimic(P),
    \+ stroke_probability(P),
    \+ hidden_stroke_risk(P).


% ==============================================================================
% SECTION 7: EXPLANATION HELPERS
% ==============================================================================

% List all detected symptoms for display
symptom_list(List, P) :-
    findall(Symptom,
        (   (facial_droop_detected(_, _), Symptom = 'Facial asymmetry')
        ;   (arm_risk(P), Symptom = 'Arm weakness')
        ;   (speech_risk(P), Symptom = 'Speech difficulty')
        ;   (vision_change(P), Symptom = 'Visual disturbance')
        ;   (dizziness(P), Symptom = 'Dizziness/Vertigo')
        ),
        List
    ).

% Risk category for color-coding in UI
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
    is_mimic(P),
    \+ stroke_probability(P),
    \+ hidden_stroke_risk(P),
    \+ recurrence_boost(P).