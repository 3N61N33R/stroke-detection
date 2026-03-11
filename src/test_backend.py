"""
Backend Integration Test
========================

This script validates the native DeepProbLog neuro-symbolic backend.
It performs the following checks:
1. Verifies that project paths resolve correctly.
2. Initializes the DeepProbLog bridge (PyTorch + Logic Graph).
3. Generates synthetic test images.
4. Executes inference across MULTIPLE patient profiles to verify 
   that symptom probabilities correctly compound (vary).

Usage:
    python src/test_backend.py
"""

import sys
import os
import numpy as np
from PIL import Image

# ==============================================================================
# PATH SETUP
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Ensure these match the actual names in your directory!
MODEL_PATH = os.path.join(ROOT_DIR, "models", "stroke_mvp.pth")
LOGIC_PATH = os.path.join(ROOT_DIR, "src", "logic", "stroke_logic.pl")

# Import the Bridge we just built
from src.bridge.dpl_interface import StrokeBridge


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_dummy_image(height=224, width=224):
    """Creates a synthetic RGB image (random noise)."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ==============================================================================
# MAIN TEST ROUTINE
# ==============================================================================
def main():
    print("🚀 Initializing DeepProbLog Integration Test")
    print(f"📂 Project Root: {ROOT_DIR}\n")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LOGIC_PATH):
        print("❌ ERROR: Model or Logic file not found. Check paths.")
        return

    # 1. INITIALIZE BRIDGE
    try:
        bridge = StrokeBridge(model_path=MODEL_PATH, logic_path=LOGIC_PATH)
        print("✅ Native DeepProbLog bridge initialized successfully.\n")
    except Exception as e:
        print(f"❌ Failed to initialize bridge: {e}")
        return

    # 2. GENERATE SYNTHETIC CAMERA DATA
    print("📸 Generating synthetic neutral and smile images...")
    neutral_img = create_dummy_image()
    smile_img = create_dummy_image()

    # 3. DEFINE TEST PATIENTS (To prove probabilities vary)
    test_cases = [
        {
            "id": "Patient A (Visual Only, No Symptoms)",
            "data": {'gender': 'male', 'speech': False, 'arm': False, 'vision': False}
        },
        {
            "id": "Patient B (Visual + Speech Issue)",
            "data": {'gender': 'male', 'speech': True, 'arm': False, 'vision': False}
        },
        {
            "id": "Patient C (Visual + Speech + Arm Weakness)",
            "data": {'gender': 'male', 'speech': True, 'arm': True, 'vision': False}
        },
        {
            "id": "Patient D (Atypical: Dizziness + Vision Change)",
            "data": {'gender': 'female', 'vision': True, 'dizzy': True}
        },
        {
            "id": "Patient E (Recurrence Risk: Recent TIA)",
            "data": {'gender': 'male', 'speech': True, 'history_recent_tia': True}
        },
        {
            "id": "Patient F (Stroke Mimic: Prior Stroke + No New Symptoms)",
            "data": {'gender': 'female', 'dizzy': True, 'history_prior_stroke': True}
        }
    ]

    # 4. RUN INFERENCE FOR EACH PATIENT
    print("\n🧠 Executing varying symptom tests...\n")
    print("=" * 60)

    for case in test_cases:
        try:
            results = bridge.analyze_patient(
                neutral_img=neutral_img,
                smile_img=smile_img,
                patient_data=case["data"]
            )
            
            prob = results.get("stroke_prob", 0.0) * 100
            category = results.get("risk_category", "N/A").upper()
            decision = results.get("clinical_decision", "N/A")
            contribs = results.get("contributions", {})
            
            print(f"👤 PROFILE   : {case['id']}")
            print(f"   PROBABILITY: {prob:.2f}%")
            print(f"   CATEGORY   : {category}")
            print(f"   DECISION   : {decision}")
            print(f"   XAI RULES TRIGGERED:")
            
            # Print only active contributions
            active_rules = False
            for k, v in contribs.items():
                if v > 0.0:
                    active_rules = True
                    clean_name = k.replace('_', ' ').title()
                    print(f"      -> {clean_name}: {v*100:.2f}%")
            
            if not active_rules:
                print("      -> None")
                
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Inference failed for {case['id']}: {e}")
            print("-" * 60)

    print("\n💡 If DeepProbLog is working correctly, probabilities and decisions")
    print("   will shift dynamically based on the XAI logic pathways.")

if __name__ == "__main__":
    main()
    