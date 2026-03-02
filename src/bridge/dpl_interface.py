"""
DeepProbLog Engine Bridge
=========================

Neuro-symbolic inference interface for the Stroke Detection system.
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
    dimension so DeepProbLog can index the classes correctly via val[i].
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # 1. Get raw logits from the CNN (shape: [1, 2])
        logits = self.base_model(x)
        # 2. Convert logits to probabilities 
        probs = F.softmax(logits, dim=-1)
        # 3. Strip the batch dimension to return a 1D tensor (shape: [2])
        return probs.squeeze(0)


class StrokeBridge:
    def __init__(self, model_path: str, logic_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(logic_path, "r") as f:
            self.base_logic = f.read()

        raw_cnn = get_model()
        raw_cnn.load_state_dict(torch.load(model_path, map_location=self.device))
        raw_cnn.eval()
        
        # FIX: Wrap the CNN so it outputs the exact 1D tensor DeepProbLog expects
        self.wrapped_cnn = DPLWrapper(raw_cnn).to(self.device)

        # Register the wrapped model
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

        queries = [
            Query(Term("arm_risk", Term("patient1"))),
            Query(Term("speech_risk", Term("patient1"))),
            Query(Term("fast_positive", Term("patient1"))),
            Query(Term("stroke_probability", Term("patient1")))
        ]
        
        answers = patient_model.solve(queries)
        
        debug_results = {}
        for ans in answers:
            for term, prob in ans.result.items():
                debug_results[str(term)] = float(prob)

        print(f"\n   [Debug] Engine internal evaluation for this patient:")
        for k, v in debug_results.items():
            print(f"      -> {k}: {v*100:.2f}%")

        stroke_prob = debug_results.get("stroke_probability(patient1)", 0.0)

        if stroke_prob >= 0.85: category = "critical"
        elif stroke_prob >= 0.50: category = "high"
        elif stroke_prob >= 0.25: category = "moderate"
        else: category = "low"

        return {
            "stroke_prob": stroke_prob,
            "risk_category": category
        }
    