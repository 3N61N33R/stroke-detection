"""
Backend Integration Test
========================

This script verifies the full Neuro-Symbolic pipeline without requiring the UI.
It performs the following checks:
1. Verifies that project paths (Models, Logic) are correctly resolved.
2. Initializes the StrokeBridge (loading PyTorch model + SWI-Prolog).
3. Generates dummy synthetic images to simulate camera input.
4. Runs a full inference cycle on a synthetic patient profile.

Usage:
    python src/test_backend.py
"""

import sys
import os
import numpy as np
from PIL import Image

# ==============================================================================
# SETUP PATHS
# ==============================================================================
# Resolve the project root relative to this script (assumed to be in src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Add Root to Sys Path to allow module imports
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Define resource paths
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'stroke_mvp.pth')
LOGIC_PATH = os.path.join(ROOT_DIR, 'src', 'logic', 'stroke_logic.pl')

# Import Bridge after path setup
from src.bridge.dpl_interface import StrokeBridge

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_dummy_image(height=224, width=224):
    """
    Creates a random noise RGB image.
    Used to mock camera input when a real camera is not available.
    """
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def main():
    print(f"🚀 Initializing Backend Integration Test...")
    print(f"📂 Project Root: {ROOT_DIR}")
    
    # ---------------------------------------------------------
    # 1. VALIDATION CHECKS
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  WARNING: Model file not found at {MODEL_PATH}")
        print("    (Inference will proceed with random weights for testing purposes)")
    
    if not os.path.exists(LOGIC_PATH):
        print(f"❌ ERROR: Logic file not found at {LOGIC_PATH}")
        print("    (Cannot proceed without the Prolog knowledge base)")
        return

    # ---------------------------------------------------------
    # 2. LOAD BRIDGE
    # ---------------------------------------------------------
    try:
        bridge = StrokeBridge(model_path=MODEL_PATH, logic_path=LOGIC_PATH)
    except Exception as e:
        print(f"❌ Critical Error initializing StrokeBridge: {e}")
        return

    # ---------------------------------------------------------
    # 3. PREPARE TEST DATA
    # ---------------------------------------------------------
    print("\n📸 Generating synthetic patient data...")
    neutral_img = create_dummy_image()
    smile_img = create_dummy_image()

    # Synthetic Patient Profile:
    # Female, presenting with Speech and Arm issues (Classic FAST symptoms)
    test_patient = {
        'gender': 'Female',         
        'speech': True,             
        'arm': True,                
        'vision': False,
        'dizzy': False,
        'history_tia': False,
        'prior_stroke': False
    }

    print(f"📋 Patient Profile: {test_patient}")

    # ---------------------------------------------------------
    # 4. RUN INFERENCE
    # ---------------------------------------------------------
    print("\n🧠 Executing Neuro-Symbolic Inference...")
    try:
        results = bridge.analyze_patient(neutral_img, smile_img, test_patient)
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. DISPLAY RESULTS
    # ---------------------------------------------------------
    prob = results.get('stroke_prob', 0.0) * 100
    category = results.get('risk_category', 'Unknown').upper()
    call_911 = results.get('call_911', False)

    print("\n" + "="*40)
    print("        DIAGNOSTIC REPORT        ")
    print("="*40)
    print(f"Stroke Probability:   {prob:.2f}%")
    print(f"Risk Category:        {category}")
    print(f"Recommendation:       {'CALL 911 NOW' if call_911 else 'Seek Medical Advice'}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
    