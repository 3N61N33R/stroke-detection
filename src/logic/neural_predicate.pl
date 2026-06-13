% ==============================================================================
% FILE: src/logic/neural_predicate.pl
% ==============================================================================

% Neural predicate: maps CNN output to facial palsy classification.
nn(droop_classifier, [Image], Classification, [no, yes]) ::
    analyze_face(Image, Classification).


% Convert CNN facial analysis into the symbolic symptom used by the
% Claus stroke prediction model.
facial_palsy :-
    image(ImageID),
    analyze_face(ImageID, yes).


% Import diagnostic model generated from Claus et al. symptom statistics.
% :- consult('../../modelling_resources/stroke_prediction_model_Claus_simplified.pl').