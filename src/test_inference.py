"""
Neuro-Symbolic Inference Test & Benchmark
=========================================

This script validates the DeepProbLog inference engine utilizing 
the simplified Claus et al. diagnostic model. It captures execution 
times to satisfy benchmarking requirements and evaluates the natural-language 
XAI explanations derived directly from active clinical evidence.

Usage:
    python src/test_inference.py
"""

import sys
import os
import time
import torch

# ==============================================================================
# PATH SETUP
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the Engine
from src.bridge.inference_engine import StrokeInferenceEngine

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def create_dummy_tensor():
    """
    Creates a synthetic PyTorch tensor mimicking the output of the 
    preprocessing pipeline [Batch, Channels, Height, Width].
    """
    return torch.randn(1, 3, 224, 224)

# ==============================================================================
# MAIN TEST ROUTINE
# ==============================================================================
def main():
    print("🚀 Initializing Neuro-Symbolic Engine Test")
    print(f"📂 Project Root: {ROOT_DIR}\n")

    # 1. INITIALIZE ENGINE
    print("⚙️ Loading DeepProbLog and PyTorch Models...")
    start_init = time.perf_counter()
    try:
        engine = StrokeInferenceEngine()
        init_time = time.perf_counter() - start_init
        print(f"✅ Engine initialized successfully in {init_time:.4f} seconds.\n")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return

    # 2. GENERATE SYNTHETIC DATA
    print("📸 Generating synthetic camera tensor...")
    dummy_image = create_dummy_tensor()

    # 3. DEFINE TEST PATIENTS
    test_cases = [
        {
            "id": "Patient A (Total Silence / Unknowns)",
            "data": {} 
        },
        {
            "id": "Patient B (Explicit Negatives)",
            "data": {
                "limb_weakness": "no",
                "speech_difficulties": "no",
                "sensory_issues": "no",
                "vision_changes": "no"
            }
        },
        {
            "id": "Patient C (Speech Issue Only)",
            "data": {
                "speech_difficulties": "yes",
                "limb_weakness": "no"
            }
        },
        {
            "id": "Patient D (High Risk: Multiple FAST Symptoms)",
            "data": {
                "limb_weakness": "yes",
                "speech_difficulties": "yes",
                "sensory_issues": "unknown"
            }
        },
        {
            "id": "Patient E (Atypical: Sensory + Vision)",
            "data": {
                "sensory_issues": "yes",
                "vision_changes": "yes",
                "limb_weakness": "no"
            }
        }
    ]

    # 4. RUN INFERENCE FOR EACH PATIENT
    print("\n🧠 Executing probabilistic inference tests against Simplified Claus model...\n")
    print("=" * 75)

    total_inference_time = 0.0
    successful_runs = 0

    for case in test_cases:
        try:
            # --- START BENCHMARK ---
            start_time = time.perf_counter()
            
            results = engine.analyze_patient(
                image_tensor=dummy_image,
                user_symptoms=case["data"]
            )
            
            end_time = time.perf_counter()
            # --- END BENCHMARK ---
            
            exec_time = end_time - start_time
            total_inference_time += exec_time
            successful_runs += 1
            
            prob = results.get("stroke_prob", 0.0) * 100
            category = results.get("risk_category", "N/A").upper()
            decision = results.get("clinical_decision", "N/A")
            reasoning = results.get("xai_reasoning", "No specific reasoning provided.")
            
            print(f"👤 PROFILE    : {case['id']}")
            print(f"⏱️ EXEC TIME  : {exec_time:.4f} seconds")
            print(f"⚠️ ASSESSMENT : {category} ({prob:.1f}% probability of stroke)")
            print(f"🧠 REASONING  : {reasoning}")
            print(f"🏥 DECISION   : {decision}")
            print("-" * 75)
            
        except Exception as e:
            print(f"❌ Inference failed for {case['id']}: {e}")
            print("-" * 75)

    if successful_runs > 0:
        avg_time = total_inference_time / successful_runs
        print("\n📈 Benchmark Summary (Issue #4):")
        print(f"   Total Successful Inferences: {successful_runs}/{len(test_cases)}")
        print(f"   Average Forward Pass Inference Time: {avg_time:.4f} seconds")
    else:
        print("\n❌ Benchmark failed: No successful runs to average.")

if __name__ == "__main__":
    main()
    