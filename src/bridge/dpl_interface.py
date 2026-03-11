"""
DeepProbLog Engine Bridge
=========================

Neuro-symbolic inference interface for the Stroke Detection system.
Handles neural inference, logical evaluation, and clinical risk categorization.
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from problog.logic import Term

from src.networks.facial_net import get_model


class DPLWrapper(torch.nn.Module):
    """
    Wraps the CNN to convert logits to probabilities and strip the batch 
    dimension so DeepProbLog can index the classes correctly.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logits = self.base_model(x)
        probs = F.softmax(logits, dim=-1)
        return probs.squeeze(0)


class StrokeBridge:
    def __init__(self, model_path: str, logic_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(logic_path, "r") as f:
            self.base_logic = f.read()

        raw_cnn = get_model()
        raw_cnn.load_state_dict(torch.load(model_path, map_location=self.device))
        raw_cnn.eval()
        
        self.wrapped_cnn = DPLWrapper(raw_cnn).to(self.device)
        self.droop_net = Network(self.wrapped_cnn, "droop_classifier", batching=False)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def analyze_patient(self, neutral_img, smile_img, patient_data: dict) -> dict:
        neutral_tensor = self.transform(neutral_img).unsqueeze(0).to(self.device)
        smile_tensor = self.transform(smile_img).unsqueeze(0).to(self.device)

        dynamic_logic = self.base_logic + "\n\n% --- DYNAMIC PATIENT FACTS ---\n"
        
        dynamic_logic += "belongs_to(tensor(live_camera(neutral_img)), patient1).\n"
        dynamic_logic += "belongs_to(tensor(live_camera(smile_img)), patient1).\n"
        
        gender = patient_data.get("gender", "male").lower()
        dynamic_logic += f"gender(patient1, {gender}).\n"

        if patient_data.get("speech"): dynamic_logic += "speech_issue(patient1).\n"
        if patient_data.get("arm"): dynamic_logic += "arm_weakness(patient1).\n"
        if patient_data.get("vision"): dynamic_logic += "vision_change(patient1).\n"
        if patient_data.get("dizzy"): dynamic_logic += "dizziness(patient1).\n"
        if patient_data.get("history_recent_tia"): dynamic_logic += "history_recent_tia(patient1).\n"
        if patient_data.get("history_prior_stroke"): dynamic_logic += "history_prior_stroke(patient1).\n"

        patient_model = Model(dynamic_logic, [self.droop_net], load=False)
        patient_model.set_engine(ExactEngine(patient_model))

        live_image_store = {
            (Term("neutral_img"),): neutral_tensor,
            (Term("smile_img"),): smile_tensor
        }
        patient_model.add_tensor_source("live_camera", live_image_store)

        # Updated to match the new, academically precise Prolog terms
        queries = [
            Query(Term("arm_deficit", Term("patient1"))),
            Query(Term("speech_deficit", Term("patient1"))),
            Query(Term("fast_positive", Term("patient1"))),
            Query(Term("atypical_stroke", Term("patient1"))),
            Query(Term("recurrence_boost", Term("patient1"))),
            Query(Term("is_mimic", Term("patient1"))),
            Query(Term("stroke", Term("patient1")))
        ]
        
        # Suppress PyTorch training warnings during inference
        with torch.no_grad():
            answers = patient_model.solve(queries)
        
        debug_results = {}
        for ans in answers:
            for term, prob in ans.result.items():
                debug_results[str(term)] = float(prob)

        # Extract probabilities
        stroke_prob = debug_results.get("stroke(patient1)", 0.0)
        fast_prob = debug_results.get("fast_positive(patient1)", 0.0)
        atypical_prob = debug_results.get("atypical_stroke(patient1)", 0.0)
        recurrence_prob = debug_results.get("recurrence_boost(patient1)", 0.0)
        mimic_prob = debug_results.get("is_mimic(patient1)", 0.0)

        # -------------------------------------------------------------------
        # CLINICAL DECISION & UX RISK CATEGORIZATION
        # -------------------------------------------------------------------
        # Convert probabilistic outputs into boolean triggers for clinical logic
        has_stroke = stroke_prob >= 0.50
        is_fast_pos = fast_prob > 0.0
        is_atypical = atypical_prob > 0.0
        has_recurrence = recurrence_prob > 0.0
        is_mimic = mimic_prob > 0.0

        # Apply the exact clinical rules previously held in the Prolog file
        urgent_911 = (has_stroke and is_fast_pos and not is_mimic) or \
                     (has_stroke and has_recurrence and not is_mimic)
                     
        seek_urgent = (has_stroke and not urgent_911 and not is_mimic) or \
                      (is_atypical and not is_mimic) or \
                      (has_recurrence and not urgent_911 and not is_mimic)
                      
        consider_eval = (is_atypical and is_mimic) or \
                        (is_mimic and not has_stroke and not is_atypical)

        if urgent_911:
            category = "critical"
            decision = "Urgent: Call 911"
        elif seek_urgent:
            category = "high"
            decision = "Seek Urgent Care"
        elif consider_eval:
            category = "moderate"
            decision = "Consider Medical Evaluation"
        else:
            category = "low"
            decision = "Monitor Routine Symptoms"

        # Package XAI contributions for the frontend to explain the math
        contributions = {
            "arm_deficit": debug_results.get("arm_deficit(patient1)", 0.0),
            "speech_deficit": debug_results.get("speech_deficit(patient1)", 0.0),
            "fast_positive": fast_prob,
            "atypical_stroke": atypical_prob,
            "recurrence_boost": recurrence_prob,
            "is_mimic": mimic_prob
        }

        return {
            "stroke_prob": stroke_prob,
            "risk_category": category,
            "clinical_decision": decision,
            "contributions": contributions
        }
    