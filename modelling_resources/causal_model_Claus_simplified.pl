0.002 :: stroke_or_tia. 
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

weakness :- arm_weakness.
weakness :- leg_weakness.
0.23 :: facial_palsy :- stroke_or_tia. 
0.49 :: speech :- stroke_or_tia.
0.48 :: arm_weakness :- stroke_or_tia.
0.38 :: leg_weakness :- stroke_or_tia.
0.23 :: sensory :- stroke_or_tia. 
0.23 :: visual :- stroke_or_tia.
