"""
DeepProbLog Interface Bridge
============================

This module serves as the neuro-symbolic bridge for the Stroke Detection System.
It integrates the output of the PyTorch facial analysis model (Neural) with 
the DeepProbLog logic engine (Symbolic).

Key Features:
- Loads standard SWI-Prolog libraries across Linux/Mac/Windows.
- Injects neural probabilities into a temporary Prolog logic program.
- Executes BE-FAST stroke risk queries using the ExactEngine.

Dependencies:
- torch, torchvision
- deepproblog
- pyswip (via internal ctypes loading)
"""

import ctypes
import os
import sys
import platform
import tempfile
import torch
from torchvision import transforms

# DeepProbLog & Logic Imports
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from problog.logic import Term, Constant

# Internal Imports
from src.networks.facial_net import get_model

# ==============================================================================
# HELPER: CROSS-PLATFORM SWI-PROLOG LOADER
# ==============================================================================
def load_swi_prolog():
    """
    Attempts to manually load the SWI-Prolog shared library.
    This is often required for DeepProbLog to interface with the system-level
    Prolog installation on Linux, Mac, and Windows.
    """
    system = platform.system()
    lib_names = []
    
    if system == "Linux":
        lib_names = [
            '/usr/lib/swi-prolog/lib/x86_64-linux/libswipl.so', 
            'libswipl.so'
        ]
    elif system == "Darwin": # macOS
        lib_names = [
            '/Applications/SWI-Prolog.app/Contents/Frameworks/libswipl.dylib',
            '/opt/homebrew/lib/libswipl.dylib',
            'libswipl.dylib'
        ]
    elif system == "Windows":
        lib_names = [
            r'C:\Program Files\swipl\bin\libswipl.dll',
            'libswipl.dll'
        ]

    for lib_path in lib_names:
        try:
            if os.path.exists(lib_path) or system == "Windows":
                # On Windows, ctypes.cdll.LoadLibrary is often preferred
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL if system != "Windows" else 0)
                
                # Update environment variables for subprocesses if needed
                if system == "Linux" and os.path.exists(lib_path):
                    os.environ['LD_LIBRARY_PATH'] = f"{os.path.dirname(lib_path)}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                
                print(f"✅ System: Linked to SWI-Prolog library ({lib_path}).")
                return
        except Exception:
            continue
            
    # If we reach here, we rely on the system's default PATH/LD_LIBRARY_PATH
    print("ℹ️ System: Relying on default system paths for SWI-Prolog.")

# Execute loader at module level
load_swi_prolog()


# ==============================================================================
# MAIN BRIDGE CLASS
# ==============================================================================
class StrokeBridge:
    """
    Acts as the Neuro-Symbolic interface between the PyTorch Vision Model
    and the DeepProbLog Logic Engine.
    """
    def __init__(self, model_path, logic_path):
        self.device = torch.device("cpu")
        
        # 1. Load Vision Model (PyTorch)
        self.cnn = get_model().to(self.device)
        try:
            self.cnn.load_state_dict(torch.load(model_path, map_location=self.device))
            self.cnn.eval()
            print(f"✅ Bridge: Vision Model loaded successfully.")
        except FileNotFoundError:
            print(f"⚠️ Bridge Warning: Model file not found at {model_path}. Inference will use random weights.")

        # 2. Define Image Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # 3. Load Logic Template
        # We read the logic rules into memory to inject patient data dynamically later.
        if not os.path.exists(logic_path):
            raise FileNotFoundError(f"Logic file not found: {logic_path}")
            
        with open(logic_path, 'r') as f:
            self.logic_template = f.read()

        print("✅ Bridge: DeepProbLog Interface ready.")

    def get_face_probabilities(self, img):
        """
        Runs the CNN on a single image and returns raw probabilities for [Normal, Droop].
        """
        if img is None: return 0.0, 0.0
        
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.cnn(img_t)
        # Assuming output is Softmaxed or raw logits; DeepProbLog expects probabilities.
        # If model outputs logits, apply F.softmax here.
        return output[0][0].item(), output[0][1].item()

    def _get_score(self, answers):
        """
        Helper to safely extract the probability score from DeepProbLog results.
        Handles cases where the engine returns a logical proof (Dict) instead of a float.
        """
        if not answers:
            return 0.0
        
        val = answers[0].result
        
        # Case A: Standard Probability (Float)
        if isinstance(val, (int, float)):
            return float(val)
        
        # Case B: Logical Proof (Dict) -> Implies Deterministic True (1.0)
        if isinstance(val, dict):
            return 1.0
            
        return 0.0

    def analyze_patient(self, neutral_img, smile_img, user_data):
        """
        Performs the full Neuro-Symbolic inference:
        1. Visual Analysis (Neural Network) -> Probabilistic Facts
        2. Symptom Processing (User Data) -> Deterministic Facts
        3. Logic Reasoning (DeepProbLog) -> Stroke Risk Score & Decisions
        """
        
        # ---------------------------------------------------------
        # A. Neural Phase: Extract visual evidence
        # ---------------------------------------------------------
        n_normal, n_droop = self.get_face_probabilities(neutral_img)
        s_normal, s_droop = self.get_face_probabilities(smile_img)
        
        # ---------------------------------------------------------
        # B. Symbolic Phase: Construct the Knowledge Base
        # ---------------------------------------------------------
        patient_id = "patient_x"
        facts = []

        # 1. Inject Neural Probabilities (The "Neuro" part)
        facts.append(f"{n_normal:.4f}::check_face(img_neutral, normal).")
        facts.append(f"{n_droop:.4f}::check_face(img_neutral, droop).")
        facts.append(f"{s_normal:.4f}::check_face(img_smile, normal).")
        facts.append(f"{s_droop:.4f}::check_face(img_smile, droop).")

        # 2. Inject Patient Demographics & Symptoms (The "Symbolic" part)
        if user_data.get('gender') == 'Female': facts.append(f"gender({patient_id}, female).")
        else: facts.append(f"gender({patient_id}, male).")

        if user_data.get('speech'): facts.append(f"speech_issue({patient_id}).")
        if user_data.get('arm'):    facts.append(f"arm_weakness({patient_id}).")
        if user_data.get('vision'): facts.append(f"vision_change({patient_id}).")
        if user_data.get('dizzy'):  facts.append(f"dizziness({patient_id}).")
        
        if user_data.get('history_tia'): facts.append(f"history_recent_tia({patient_id}).")
        
        # BE-FAST Logic for Mimics: Prior Stroke without NEW symptoms
        if user_data.get('prior_stroke'):
            facts.append(f"history_prior_stroke({patient_id}).")
            has_new = (user_data.get('speech') or user_data.get('arm') or 
                       user_data.get('vision') or user_data.get('dizzy'))
            if has_new:
                facts.append(f"new_symptom({patient_id}).")

        # 3. Bridge the Neural Facts to the Patient
        facts.append(f"facial_droop_detected(img_neutral, img_smile).")

        # ---------------------------------------------------------
        # C. Inference Phase: Run the Logic Program
        # ---------------------------------------------------------
        # Merge the static Logic Template with the dynamic Facts
        full_code = self.logic_template + "\n" + "\n".join(facts)
        
        results = {}
        # Create a temporary file for DeepProbLog to read
        fd, temp_path = tempfile.mkstemp(suffix=".pl", text=True)
        
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(full_code)
            
            # Initialize Engine for this specific patient
            model = Model(temp_path, [])
            engine = ExactEngine(model)
            model.set_engine(engine)

            # Define Queries
            pid_term = Constant(patient_id)
            
            # Query 1: Probability of Stroke
            q1 = Query(Term('stroke_probability', pid_term))
            ans1 = model.solve([q1])
            results['stroke_prob'] = self._get_score(ans1)

            # Query 2: Decision - Call 911?
            q2 = Query(Term('urgent_call_911', pid_term))
            ans2 = model.solve([q2])
            results['call_911'] = self._get_score(ans2) > 0.0

            # Query 3: Decision - Seek Urgent Care?
            q3 = Query(Term('seek_urgent_care', pid_term))
            ans3 = model.solve([q3])
            results['urgent_care'] = self._get_score(ans3) > 0.0

            # Query 4: Determine Risk Category (Critical/High/Moderate/Low)
            categories = ['critical', 'high', 'moderate', 'low']
            best_cat = 'low'
            best_score = -1.0
            
            for cat in categories:
                q_cat = Query(Term('risk_category', Constant(cat), pid_term))
                ans_cat = model.solve([q_cat])
                score = self._get_score(ans_cat)
                
                if score > best_score:
                    best_score = score
                    best_cat = cat
            
            results['risk_category'] = best_cat

        except Exception as e:
            print(f"❌ Logic Inference Failed: {e}")
            import traceback
            traceback.print_exc()
            results = {'stroke_prob': 0.0, 'risk_category': 'error'}
        finally:
            # Cleanup temporary logic file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return results
    