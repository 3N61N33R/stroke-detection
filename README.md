# stroke-detection

Overview

A core hybrid architecture for our stroke detection system. It successfully bridges the gap between raw computer vision (Neural) and clinical decision-making (Symbolic) by integrating PyTorch with DeepProbLog.

Key Implementations
Neural Layer (src/networks/)

    Implemented FacialDroopCNN based on the Dua & Sharma (2023) architecture.
    Features a 4-layer convolutional stack with He (Kaiming) initialization to ensure training stability and prevent "Dying ReLU" issues.

Symbolic Layer (src/logic/)

    Developed stroke_logic.pl using Probabilistic Logic.
    Encoded strict BE-FAST clinical protocols, including weights for gender bias and stroke mimics (prior stroke history).
    Implemented "Base Case" safety facts to prevent engine crashes during partial data entry.

Integration Bridge (src/bridge/)

    Created StrokeBridge, a robust interface that manages the data flow between the CNN and the Logic Engine.
    Includes a Cross-Platform Loader that dynamically detects and links SWI-Prolog shared libraries for Linux, macOS, and Windows.

Training & Testing

    train.py: A fully automated pipeline that fetches the annotated facial dataset via kagglehub and exports trained weights to the models/ directory.
    test_backend.py: An integration script to verify the full pipeline without the UI.

Technical Impact

    Explainability: Unlike a standard CNN, this engine provides a reason for its decision (e.g., "Critical" risk due to combined Arm and Speech symptoms).
    Stability: The system now handles "Unknown Clauses" and "Type Mismatches" gracefully, ensuring it won't crash if a user leaves a symptom field blank.

How to Test

    Ensure you have SWI-Prolog installed on your system.
    Run the integration test:

python src/test_backend.py

    You should see the system force-load the Prolog library, run a dummy inference, and output a Diagnostic Report with a specific probability and risk category.
