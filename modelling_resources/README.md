# Modelling resources

This folder contains resources and scripts we used for creating the DeepProbLog models.

Overall, there are four models for predicting stroke as such and two models for predicting TIAs, minor and major strokes separately.

## Prediction of type of stroke 
We first consider the more detailed models:
There are two models derived from Claus _et al._, a simplified model derived directly from Table 2, and a model taking account of the correlations between different symptoms derived from Figure 2. 


### Simplified model

The simplified model was encoded as a causal ProbLog model directly in `detailed_causal_model_Claus_simplified.pl`.
The stroke prediction logic is derived from this model by querying all possible combinations of evidences for querying all of `tia`. `minor` and `major`. 
The code for this can be found in `Compute_detailed_probabilities.sh`. 

### Full model

The model taking full account of correlations between symptoms was created by first manually transcribing Figure 2 into a list, `Claus_combinations_detailed.txt`. 
Then, a short awk script `summarise_combinations.sh` is used to aggregate combinations which do not differ in the symptoms covered here (saved as `Claus_detailed_summary.txt`).
We then use another awk script, `make_causal_stroke_logic.sh`, to convert this information into a logic program with annotated disjunctions. 
After completing this script with the null model taken from `detailed_causal_model_Claus_simplified.pl`, we then continue using  `Compute_detailed_probabilities.sh` as above. 
  
## Stroke prediction as such

Three models can be derived from Claus _et al._, with progressively more account taken of interactions between symptoms. 
* The simplified model is derived from Table 2, treating all symptoms as independent when conditioned on stroke;
* A more fine-grained model can be obtained by using the simplified model for the three types of stroke and then querying this for stroke as such. This gives a more accurate outcome since we only assume indepndence conditioned on the exact type of stroke. 
* Another model can be obtained by following the procedure for the full model above; here, it is irrelevant whether we include the information on TIA, minor and major stroke, as symptom combinations are captured explicitly. 

We have also applied ProbFOIL+ to compress this final model, leading to one with significantly fewer diagnostic rules. 

### Simplified model 

The direct encoding can be found in `causal_model_Claus_simplified.pl`, to which we can apply `Compute_stroke_probabilities.sh`, the analogue of  `Compute_detailed_probabilities.sh` for predicting only stroke.

### Divided model

The script  `Compute_stroke_probabilities.sh` can also be applied to  `detailed_causal_model_Claus_simplified.pl`, giving a more fine-grained model. 

### Full model 

The full model for stroke prediction from the full combination data can be obtained directly by applying  `Compute_stroke_probabilities.sh` to the full causal model predicting the types of stroke calcaulated earlier. 

### ProbFOIL+ model

To test whether this model could be effectively compressed, we generated probabilistic examples corresponding to the different combinations of symptoms (see the script `make_probfoil_data.sh`) and then used ProbFOIL+ to generate a much shorter rule set approximating the original theory. The corresponding rules are contained in `stroke_prediction_model_Claus_probfoil.pl`. 

