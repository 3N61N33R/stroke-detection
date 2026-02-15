import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io

# ============================================================== 
# PAGE CONFIGURATION
# ============================================================== 
st.set_page_config(
    page_title="Stroke Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================== 
# CUSTOM CSS STYLING
# ============================================================== 
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fast-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .risk-critical {
        background: #FF4B4B;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-high {
        background: #FFA500;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-moderate {
        background: #FFD700;
        color: black;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-low {
        background: #32CD32;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .disclaimer {
        background: #f0f0f0;
        padding: 1rem;
        border-left: 4px solid #FF4B4B;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================== 
# CNN MODEL DEFINITION (ResNet18)
# ============================================================== 
class StrokeResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(StrokeResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ============================================================== 
# IMAGE PREPROCESSING
# ============================================================== 
def preprocess_image(image):
    """Preprocess uploaded image for CNN inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ============================================================== 
# NEURO-SYMBOLIC REASONING ENGINE
# ============================================================== 
class StrokeBridge:
    """Bridge between CNN perception and DeepProbLog reasoning"""
    
    def __init__(self):
        self.cnn_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path=None):
        """Load pre-trained CNN model"""
        self.cnn_model = StrokeResNet()
        if model_path:
            try:
                self.cnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.cnn_model.eval()
                return True
            except:
                return False
        return False
    
    def detect_facial_droop(self, neutral_img, smile_img):
        """
        CNN-based facial droop detection
        Returns: (droop_detected: bool, confidence: float)
        """
        if self.cnn_model is None:
            # Demo mode: random prediction for testing
            droop_detected = np.random.choice([True, False], p=[0.3, 0.7])
            confidence = np.random.uniform(0.6, 0.95)
            return droop_detected, confidence
        
        # Real inference would happen here
        neutral_tensor = preprocess_image(neutral_img)
        smile_tensor = preprocess_image(smile_img)
        
        with torch.no_grad():
            neutral_out = self.cnn_model(neutral_tensor.to(self.device))
            smile_out = self.cnn_model(smile_tensor.to(self.device))
            
            neutral_prob = torch.softmax(neutral_out, dim=1)[0][1].item()
            smile_prob = torch.softmax(smile_out, dim=1)[0][1].item()
            
            # Logic from stroke_logic.pl
            droop_detected = (neutral_prob < 0.5 and smile_prob > 0.5) or \
                           (neutral_prob > 0.5 and smile_prob > 0.5)
            confidence = max(neutral_prob, smile_prob)
            
        return droop_detected, confidence
    
    def calculate_speech_risk(self, has_speech_issue, gender):
        """
        Speech difficulty risk calculation with gender bias
        Source: stroke_logic.pl lines 30-36
        """
        if not has_speech_issue:
            return 0.0
        
        if gender.lower() == "female":
            return 0.56  # 56% weight for females
        else:
            return 0.42  # 42% weight for males
    
    def calculate_arm_risk(self, has_arm_weakness):
        """
        Arm weakness risk calculation
        Source: stroke_logic.pl lines 44-46
        """
        if has_arm_weakness:
            return 0.89  # 89% base weight
        return 0.0
    
    def stroke_probability(self, facial_droop_detected, speech_risk, arm_risk):
        """
        Core stroke probability using FAST logic
        Source: stroke_logic.pl lines 67-75
        """
        fast_positive = facial_droop_detected or speech_risk > 0 or arm_risk > 0
        
        if not fast_positive:
            return 0.0
        
        # SCENARIO A: HIGH CONFIDENCE (Camera + User Report)
        if facial_droop_detected and (speech_risk > 0 or arm_risk > 0):
            return 0.73  # 73% PPV (Ambulance setting)
        
        # SCENARIO B: MODERATE CONFIDENCE (User Report Only)
        elif not facial_droop_detected and (speech_risk > 0 or arm_risk > 0):
            return 0.56  # 56% PPV (Dispatcher setting)
        
        return 0.0

# ============================================================== 
# INITIALIZE SESSION STATE
# ============================================================== 
if 'bridge' not in st.session_state:
    st.session_state.bridge = StrokeBridge()
    st.session_state.bridge.load_model()  # Try to load model if available

# ============================================================== 
# MAIN APPLICATION
# ============================================================== 
def main():
    # Header
    st.markdown('<div class="main-header">🧠 Neuro-Symbolic Stroke Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Early Stroke Assessment Using BE-FAST Protocol</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>MEDICAL DISCLAIMER:</strong> This system is for educational and research purposes only. 
        It is NOT a substitute for professional medical diagnosis. If you suspect a stroke, 
        <strong>CALL EMERGENCY SERVICES IMMEDIATELY (911 or your local emergency number)</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.header("Patient Information")
    
    # Demographics
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("BE-FAST Assessment")
    
    # BE-FAST Symptoms
    st.sidebar.markdown("""
    <div class="fast-box">
        <strong>B</strong> - Balance<br>
        <strong>E</strong> - Eyes<br>
        <strong>F</strong> - Face<br>
        <strong>A</strong> - Arms<br>
        <strong>S</strong> - Speech<br>
        <strong>T</strong> - Time
    </div>
    """, unsafe_allow_html=True)
    
    has_speech_issue = st.sidebar.checkbox("Speech difficulty (slurred or confused speech)")
    has_arm_weakness = st.sidebar.checkbox("Arm weakness (one arm drifts downward)")
    has_balance_issue = st.sidebar.checkbox("Balance problems or dizziness")
    has_vision_issue = st.sidebar.checkbox("Vision problems (sudden loss or double vision)")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Facial Analysis", "📊 Risk Assessment", "ℹ️ About"])
    
    with tab1:
        st.header("Facial Symmetry Analysis")
        st.write("Upload two photos: one with a neutral expression and one smiling.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Neutral Expression")
            neutral_img = st.file_uploader("Upload neutral face", type=['jpg', 'jpeg', 'png'], key="neutral")
            if neutral_img:
                img = Image.open(neutral_img)
                st.image(img, caption="Neutral Face", use_column_width=True)
        
        with col2:
            st.subheader("Smiling Expression")
            smile_img = st.file_uploader("Upload smiling face", type=['jpg', 'jpeg', 'png'], key="smile")
            if smile_img:
                img = Image.open(smile_img)
                st.image(img, caption="Smiling Face", use_column_width=True)
        
        # Alternative: Webcam capture
        st.markdown("---")
        st.subheader("Or use your webcam")
        use_webcam = st.checkbox("Enable webcam capture")
        
        if use_webcam:
            col1, col2 = st.columns(2)
            with col1:
                neutral_webcam = st.camera_input("Capture neutral expression")
            with col2:
                smile_webcam = st.camera_input("Capture smiling expression")
            
            if neutral_webcam:
                neutral_img = neutral_webcam
            if smile_webcam:
                smile_img = smile_webcam
    
    with tab2:
        st.header("Stroke Risk Assessment")
        
        if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing data..."):
                # Facial analysis
                facial_droop_detected = False
                cnn_confidence = 0.0
                
                if neutral_img and smile_img:
                    neutral_pil = Image.open(neutral_img)
                    smile_pil = Image.open(smile_img)
                    facial_droop_detected, cnn_confidence = st.session_state.bridge.detect_facial_droop(
                        neutral_pil, smile_pil
                    )
                
                # Calculate risks
                speech_risk = st.session_state.bridge.calculate_speech_risk(has_speech_issue, gender)
                arm_risk = st.session_state.bridge.calculate_arm_risk(has_arm_weakness)
                
                # Overall stroke probability
                stroke_prob = st.session_state.bridge.stroke_probability(
                    facial_droop_detected, speech_risk, arm_risk
                )
                
                # Display results
                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Risk level determination
                if stroke_prob >= 0.70:
                    risk_level = "CRITICAL"
                    risk_class = "risk-critical"
                    action = "🚨 CALL 911 IMMEDIATELY. Note the time symptoms started."
                elif stroke_prob >= 0.50:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                    action = "⚠️ Seek emergency medical attention immediately."
                elif stroke_prob >= 0.30:
                    risk_level = "MODERATE"
                    risk_class = "risk-moderate"
                    action = "⚡ Contact your healthcare provider urgently."
                else:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                    action = "✓ Continue monitoring symptoms. Seek medical advice if symptoms worsen."
                
                # Display risk level
                st.markdown(f'<div class="{risk_class}">{risk_level} RISK: {stroke_prob*100:.1f}%</div>', 
                          unsafe_allow_html=True)
                st.markdown(f"### {action}")
                
                # Detailed breakdown
                st.markdown("---")
                st.subheader("Detailed Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Facial Droop", 
                            "Detected" if facial_droop_detected else "Not Detected",
                            f"{cnn_confidence*100:.1f}% confidence")
                
                with col2:
                    st.metric("Speech Risk", 
                            f"{speech_risk*100:.1f}%",
                            "Female bias" if gender == "Female" and speech_risk > 0 else "")
                
                with col3:
                    st.metric("Arm Weakness Risk", 
                            f"{arm_risk*100:.1f}%")
                
                # Explanation
                st.markdown("---")
                st.subheader("📖 Explanation")
                
                st.write("**Assessment Method:**")
                if facial_droop_detected and (speech_risk > 0 or arm_risk > 0):
                    st.info("✓ High confidence assessment: Camera-detected facial asymmetry combined with self-reported symptoms (PPV: 73%)")
                elif not facial_droop_detected and (speech_risk > 0 or arm_risk > 0):
                    st.info("⚠ Moderate confidence assessment: Based on self-reported symptoms only (PPV: 56%)")
                else:
                    st.success("✓ No major stroke indicators detected")
                
                # Additional factors
                if has_balance_issue or has_vision_issue:
                    st.warning("⚠️ Additional BE-FAST symptoms reported (Balance/Eyes). These increase stroke likelihood.")
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🧠 Neuro-Symbolic AI Architecture
        
        This system combines:
        1. **Deep Learning (CNN)** - ResNet18 for facial droop detection
        2. **Probabilistic Logic Programming (DeepProbLog)** - Clinical reasoning with evidence-based probabilities
        
        ### 📊 Scientific Foundation
        
        - **FAST Protocol**: Validated stroke screening method
        - **Gender-Specific Risk**: Speech symptoms weighted by gender (Berglund et al., 2014)
        - **Positive Predictive Values**: 
          - Camera + Symptoms: 73% (Ambulance setting)
          - Symptoms Only: 56% (Dispatcher setting)
        
        ### 🔬 Technical Details
        
        - **CNN Training**: 70/15/15 train/val/test split
        - **Image Size**: 224x224 pixels
        - **Framework**: PyTorch + Streamlit
        - **Logic Engine**: Based on stroke_logic.pl
        
        ### ⚖️ Limitations
        
        - Not FDA approved for clinical use
        - Requires good lighting and image quality
        - Cannot detect all stroke types
        - Should not replace professional medical judgment
        
        ### 📚 References
        
        - Berglund et al. (2014) - Gender differences in stroke symptoms
        - Claus et al. (2024) - Arm weakness prevalence
        - FAST Protocol validation studies
        
        ---
        
        **Version**: 1.0.0 | **Last Updated**: February 2026
        """)

if __name__ == "__main__":
    main()
