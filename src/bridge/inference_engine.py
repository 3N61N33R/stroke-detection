"""
Neuro-Symbolic Inference Engine
===============================

Bridges the PyTorch convolutional neural network with the DeepProbLog 
logic engine. It evaluates patient symptoms and visual data against 
the simplified Claus et al. diagnostic model.
"""

import os
import sys
import torch
import torch.nn.functional as F

from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from problog.logic import Term

# Ensure absolute imports work from the repository root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.networks import facial_net


class SoftmaxInferenceWrapper(torch.nn.Module):
    """
    Wraps the baseline CNN during inference.
    DeepProbLog requires normalized probabilities (0.0 to 1.0) to map to logic states.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        probs = F.softmax(logits, dim=1)
        return probs.squeeze(0)


class StrokeInferenceEngine:
    def __init__(self, model_filename="stroke_mvp.pth", 
                 logic_filename="neural_predicate.pl", 
                 literature_filename="stroke_prediction_model_Claus_simplified.pl"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Resolve Paths
        model_path = os.path.join(ROOT_DIR, "models", model_filename)
        logic_path = os.path.join(ROOT_DIR, "src", "logic", logic_filename)
        literature_path = os.path.join(ROOT_DIR, "modelling_resources", literature_filename)

        # 2. Load Static Logic Files
        if not os.path.exists(logic_path):
            raise FileNotFoundError(f"Neural predicate file not found at: {logic_path}")
        with open(logic_path, "r") as f:
            self.base_logic = f.read()

        if not os.path.exists(literature_path):
            raise FileNotFoundError(f"Literature file not found at: {literature_path}")
        with open(literature_path, "r") as f:
            self.literature_logic = f.read()

        # 3. Initialize and Wrap CNN
        raw_cnn = facial_net.get_model()
        if os.path.exists(model_path):
            raw_cnn.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: {model_filename} not found. Utilizing untrained weights.")
        
        raw_cnn.eval()
        self.wrapped_cnn = SoftmaxInferenceWrapper(raw_cnn).to(self.device)
        
        # 4. Register Network with DeepProbLog
        self.droop_net = Network(self.wrapped_cnn, "droop_classifier", batching=False)
        self.droop_net.eval()

    def analyze_patient(self, image_tensor, user_symptoms: dict) -> dict:
        """
        Evaluates a patient profile using the simplified Claus model.
        """
        vocab_map = {
            "limb_weakness": "weakness",
            "speech_difficulties": "speech",
            "sensory_issues": "sensory",
            "vision_changes": "visual"
        }

        # 1. Compile Patient Evidence
        # Prolog strictly enforces the Closed World Assumption. Every symptom evaluated 
        # by the model must be explicitly declared to prevent compilation crashes.
        dynamic_evidence = "\n\n% --- PATIENT EVIDENCE ---\n"
        dynamic_evidence += "image(tensor(live_camera(live_cam_01))).\n"

        active_ui_symptoms = []
        for ui_key, prolog_key in vocab_map.items():
            state = user_symptoms.get(ui_key, "unknown")
            if state == "yes":
                dynamic_evidence += f"1.0::{prolog_key}.\n"
                active_ui_symptoms.append(ui_key)
            else:
                # Safely declare absent/unknown symptoms as 0.0 to prevent crash
                dynamic_evidence += f"0.0::{prolog_key}.\n"

        # 2. Instantiate Model
        full_program = self.literature_logic + "\n" + self.base_logic + dynamic_evidence
        
        patient_model = Model(full_program, [self.droop_net], load=False)
        patient_model.set_engine(ExactEngine(patient_model))

        # 3. Bind Tensor Source
        tensor_store = {(Term("live_cam_01"),): image_tensor.to(self.device)}
        patient_model.add_tensor_source("live_camera", tensor_store)

        # 4. Define Queries (Stroke Risk + CNN Evaluation for XAI)
        target_queries = [
            Query(Term("stroke_or_tia")),
            Query(Term("facial_palsy"))
        ]

        # 5. Solve
        with torch.no_grad():
            answers = patient_model.solve(target_queries)

        # 6. Extract Probabilities
        final_probability = 0.0
        cnn_detected_droop = False

        for ans in answers:
            for term, prob in ans.result.items():
                term_str = str(term)
                if term_str == "stroke_or_tia":
                    final_probability = float(prob)
                elif term_str == "facial_palsy" and float(prob) > 0.5:
                    cnn_detected_droop = True

        # 7. Triage & Explainability
        category, decision = self._categorize_risk(final_probability)
        reasoning_text = self._generate_xai_reasoning(active_ui_symptoms, cnn_detected_droop)

        return {
            "stroke_prob": final_probability,
            "risk_category": category,
            "clinical_decision": decision,
            "xai_reasoning": reasoning_text,
            "logged_symptoms": {k: v for k, v in user_symptoms.items() if v != "unknown"}
        }

    def _categorize_risk(self, probability: float):
        if probability >= 0.70:
            return "critical", "Urgent: Multiple predictive symptoms detected. Seek emergency care."
        elif probability >= 0.30:
            return "high", "Warning: Elevated risk detected. Consult a medical professional."
        elif probability > 0.05:
            return "moderate", "Consider Medical Evaluation."
        else:
            return "low", "Low acute risk detected based on current inputs."

    def _generate_xai_reasoning(self, active_ui_symptoms: list, cnn_detected_droop: bool) -> str:
        """Generates causal English reasoning directly from active evidence."""
        evidence_terms = [k.replace('_', ' ') for k in active_ui_symptoms]
        
        if cnn_detected_droop:
            evidence_terms.append("facial asymmetry")
            
        if not evidence_terms:
            return "This calculation reflects baseline risk, as no specific symptoms were detected."
            
        if len(evidence_terms) == 1:
            symptoms_str = evidence_terms[0]
        else:
            symptoms_str = ", ".join(evidence_terms[:-1]) + f", and {evidence_terms[-1]}"
            
        return f"Stroke risk increased because {symptoms_str} were detected."
    