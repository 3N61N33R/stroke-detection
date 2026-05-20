0.002 :: stroke_or_TIA. 
0.078 :: weakness.
0.063 :: sensory.
0.047 :: visual. 
0.005 :: facial_palsy.
0.005 :: speech.

fast :- weakness.
fast :- facial_palsy.
fast :- speech.
e_fast :- fast.
e_fast :- sensory.
e_fast :- visual. 

(409/900) :: tia; (254/900) :: minor; (237/900) :: major :- stroke_or_TIA.

(36/409) :: facial_palsy :- tia. 
(176/409) :: speech :- tia.
(96/409) :: weakness :- tia.
(69/409) :: weakness :- tia. 
(100/409) :: visual :- tia.

(47/254) :: facial_palsy :- minor. 
(99/254) :: speech :- minor.
(126/254) :: weakness :- minor.
(89/254) :: weakness :- minor.
(68/254) :: sensory :- minor. 
(46/254) :: visual :- minor.

(126/237) :: facial_palsy :- major. 
(167/237) :: speech :- major.
(211/237) :: weakness :- major.
(183/237) :: sensory :- major. 
(62/237) :: visual :- major.

