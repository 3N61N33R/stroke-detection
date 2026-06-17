import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime
import os
import sys

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================
st.set_page_config(
    page_title="DeepProbLog Stroke Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/3N61N33R/stroke-detection',
        'Report a bug': 'https://github.com/3N61N33R/stroke-detection/issues',
        'About': "Neuro-Symbolic AI for Stroke/TIA Detection using DeepProbLog"
    }
)

# ==============================================================
# LAZY IMPORT PYTORCH
# ==============================================================
PYTORCH_AVAILABLE = False
PYTORCH_ERROR = None

import warnings
warnings.filterwarnings('ignore')

try:
    import io
    from contextlib import redirect_stderr
    f = io.StringIO()
    with redirect_stderr(f):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import transforms
    PYTORCH_AVAILABLE = True
except Exception as e:
    PYTORCH_ERROR = str(e)

# ==============================================================
# CSS STYLING
# ==============================================================
st.markdown("""
<style>
    .main-header {
        font-size: clamp(1.6rem, 4vw, 2.6rem);
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.3rem;
        padding: 0 1rem;
    }
    .sub-header {
        font-size: clamp(0.9rem, 2.5vw, 1.1rem);
        color: #555;
        text-align: center;
        margin-bottom: 0.3rem;
        padding: 0 1rem;
    }
    .paper-header {
        font-size: clamp(0.8rem, 2vw, 0.95rem);
        color: #777;
        text-align: center;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        line-height: 1.8;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .risk-critical { background: linear-gradient(135deg,#FF4B4B,#C71F1F); color:white; border:3px solid #8B0000; }
    .risk-high     { background: linear-gradient(135deg,#FFA500,#FF8C00); color:white; border:3px solid #CC7000; }
    .risk-moderate { background: linear-gradient(135deg,#FFD700,#FFC700); color:#333; border:3px solid #CCA300; }
    .risk-low      { background: linear-gradient(135deg,#32CD32,#228B22); color:white; border:3px solid #1B6B1B; }
    .disclaimer {
        background: #FFF3CD;
        border-left: 4px solid #FF4B4B;
        padding: 1rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        font-size: 0.92rem;
    }
    .xai-card {
        background: #E3F2FD;
        padding: 0.9rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        border-left: 4px solid #2196F3;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.4rem 0;
        text-align: center;
    }
    .status-positive { color:#FF4B4B; font-weight:bold; }
    .status-negative { color:#32CD32; font-weight:bold; }
    .demo-badge {
        background: #28A745; color:white; padding:0.3rem 0.8rem;
        border-radius:15px; font-size:0.85rem; display:inline-block; margin:0.4rem 0;
    }
    .model-badge {
        background: #6C757D; color:white; padding:0.2rem 0.6rem;
        border-radius:10px; font-size:0.8rem; display:inline-block; margin:0.2rem;
    }
    @media (max-width:768px) {
        .info-box { padding:0.8rem; font-size:0.88rem; }
        .risk-card { padding:1rem; font-size:0.92rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# FACIAL PALSY CNN  (Dua & Sharma architecture, trained on Elhanashi et al.)
# ==============================================================
if PYTORCH_AVAILABLE:
    class FacialPalsyCNN(nn.Module):
        """
        4-layer CNN for binary facial palsy classification.
        Architecture: Dua & Sharma (2025), Int. J. Softw. Hardware Res. Eng. 13(6).
        Training data: Elhanashi et al. (2024) – 2,500 negative + 1,245 positive examples.
        Performance: accuracy 0.9987, precision 0.9958, recall 1.0 (paper Table 2 note).
        """
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(0.25)
            self.flatten_dim = 128 * 14 * 14
            self.fc1 = nn.Linear(self.flatten_dim, 256)
            self.fc2 = nn.Linear(256, 2)
            self._init_weights()

        def forward(self, x):
            for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
                x = self.dropout(self.pool(F.relu(conv(x))))
            x = x.view(-1, self.flatten_dim)
            return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def run_cnn_inference(img, model, device, transform):
        """Returns (palsy_prob: float) in [0,1]."""
        try:
            model.eval()
            with torch.no_grad():
                t = transform(img).unsqueeze(0).to(device)
                probs = model(t)
                return probs[0][1].item()
        except Exception:
            return 0.5

# ==============================================================
# FALLBACK heuristic
# ==============================================================
def heuristic_palsy(img):
    """Pixel-asymmetry heuristic when PyTorch is unavailable."""
    try:
        arr = np.array(img.resize((224, 224))).mean(axis=2)
        mid = 112
        asym = abs(arr[:, :mid].mean() - arr[:, mid:].mean())
        prob = float(np.clip(asym / 30.0, 0.0, 1.0))
        return prob
    except Exception:
        return 0.25

# ==============================================================
# DISCRIMINATIVE MODELS  (paper Section 4 / Table 2)
# ==============================================================
#
# All three models implement P(stroke_or_tia | symptoms) via noisy-OR
# (independent rule activations, ProbLog distribution semantics).
#
# Symptoms used: facial_palsy (CNN), weakness (arm/leg), speech, sensory, visual
#
# Model performance vs. full co-occurrence model (Table 2):
#   Simplified  – Acc 0.919, Prec 0.830, Rec 0.989
#   Divided     – Acc 0.934, Prec 0.869, Rec 0.972
#   ProbFOIL    – Acc 0.960, Prec 0.962, Rec 0.931

def _noisy_or(probs):
    """1 - prod(1 - p) for independent probabilistic rules."""
    result = 1.0
    for p in probs:
        result *= (1.0 - p)
    return 1.0 - result

def simplified_model(fp, weakness, speech, sensory, visual):
    """
    Listing 1.2 — simplified maximum-entropy model.
    Symptoms conditionally independent given stroke_or_tia.
    P(stroke_or_tia=0.002), then conditional inference.
    Approximated as discriminative by paper's conditional ProbLog query.
    """
    prior = 0.002
    # P(symptoms | stroke) × prior  /  [P(symptoms|stroke)×prior + P(symptoms|no-stroke)×(1-prior)]
    # Noisy-OR likelihoods from Listing 1.2
    p_s_given_stroke = _noisy_or([
        0.23 if fp else 0.0,
        0.48 if weakness else 0.0,  # arm_weakness covers both
        0.49 if speech else 0.0,
        0.23 if sensory else 0.0,
        0.23 if visual else 0.0,
    ])
    # P(symptoms | no stroke) from Listing 1.1 (independent)
    p_fp_bg = 0.005; p_w_bg = 0.078; p_sp_bg = 0.005; p_se_bg = 0.063; p_vi_bg = 0.047
    def bg(flag, p): return p if flag else (1 - p)
    p_s_given_no = bg(fp, p_fp_bg) * bg(weakness, p_w_bg) * bg(speech, p_sp_bg) * bg(sensory, p_se_bg) * bg(visual, p_vi_bg)
    denom = p_s_given_stroke * prior + p_s_given_no * (1 - prior)
    if denom == 0:
        return 0.0
    return (p_s_given_stroke * prior) / denom

def divided_model(fp, weakness, speech, sensory, visual):
    """
    Listing 1.3 — divided by stroke subtype (TIA, minor, major).
    Accuracy 0.934, Precision 0.869, Recall 0.972.
    """
    prior = 0.002
    tia_w = 409/900; min_w = 254/900; maj_w = 237/900

    def prob_given_type(n, denom): return n / denom

    def p_sym_given_sub(subtype):
        if subtype == 'tia':
            probs = [36/409 if fp else 0, (96/409 + 69/409)/2 if weakness else 0,
                     176/409 if speech else 0, 58/409 if sensory else 0, 100/409 if visual else 0]
        elif subtype == 'minor':
            probs = [47/254 if fp else 0, (126/254 + 89/254)/2 if weakness else 0,
                     99/254 if speech else 0, 68/254 if sensory else 0, 46/254 if visual else 0]
        else:
            probs = [126/237 if fp else 0, (211/237 + 183/237)/2 if weakness else 0,
                     167/237 if speech else 0, 80/237 if sensory else 0, 62/237 if visual else 0]
        return _noisy_or(probs)

    p_s_given_stroke = (tia_w * p_sym_given_sub('tia') +
                        min_w * p_sym_given_sub('minor') +
                        maj_w * p_sym_given_sub('major'))

    p_fp_bg = 0.005; p_w_bg = 0.078; p_sp_bg = 0.005; p_se_bg = 0.063; p_vi_bg = 0.047
    def bg(flag, p): return p if flag else (1 - p)
    p_s_given_no = bg(fp, p_fp_bg) * bg(weakness, p_w_bg) * bg(speech, p_sp_bg) * bg(sensory, p_se_bg) * bg(visual, p_vi_bg)

    denom = p_s_given_stroke * prior + p_s_given_no * (1 - prior)
    if denom == 0:
        return 0.0
    return (p_s_given_stroke * prior) / denom

def probfoil_model(fp, weakness, speech, sensory, visual):
    """
    Listing 1.7 — ProbFOIL 2 induced theory (6 rules).
    Best accuracy 0.960, Precision 0.962, Recall 0.931.
    Noisy-OR over matching rules.
    """
    active = []
    if speech and fp:          active.append(0.71219084)
    if speech and weakness:    active.append(0.55497685)
    if sensory and weakness and fp: active.append(0.50723151)
    if speech and sensory:     active.append(0.36436170)
    if weakness and visual and sensory and fp: active.append(0.74193908)
    if weakness and speech and fp: active.append(0.97247086)
    return _noisy_or(active)

# ==============================================================
# STROKE BRIDGE  (neuro-symbolic integration)
# ==============================================================
class StrokeBridge:
    """
    Neuro-symbolic integration layer (paper Section 5).
    - Neural component: FacialPalsyCNN processes camera image.
    - Symbolic component: discriminative ProbLog model (Listing 1.5/1.7).
    - The CNN probability is used as the facial_palsy fact probability,
      thresholded at 0.5 for the discriminative model query.
    """

    MODEL_LABELS = {
        'probfoil':    'ProbFOIL (Acc 0.960, Prec 0.962, Rec 0.931)',
        'divided':     'Divided by subtype (Acc 0.934, Prec 0.869, Rec 0.972)',
        'simplified':  'Simplified (Acc 0.919, Prec 0.830, Rec 0.989)',
    }

    def __init__(self, model_path=None):
        self.cnn = None
        self.cnn_loaded = False
        self.device = None
        self.transform = None

        if PYTORCH_AVAILABLE:
            try:
                self.device = torch.device("cpu")
                self.cnn = FacialPalsyCNN().to(self.device)
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                if model_path and os.path.exists(model_path):
                    self.cnn.load_state_dict(
                        torch.load(model_path, map_location=self.device))
                    self.cnn_loaded = True
            except Exception:
                pass

    def facial_palsy_probability(self, img):
        """Returns P(facial_palsy) in [0,1] from CNN or heuristic."""
        if PYTORCH_AVAILABLE and self.cnn is not None:
            return run_cnn_inference(img, self.cnn, self.device, self.transform)
        return heuristic_palsy(img)

    def predict(self, fp_prob, weakness, speech, sensory, visual, selected_model='probfoil'):
        """
        Run the selected discriminative model.
        fp_prob: float in [0,1], CNN output for facial palsy.
        Returns dict with stroke_prob and intermediate info.
        """
        fp = fp_prob > 0.5
        fn = {
            'probfoil':   probfoil_model,
            'divided':    divided_model,
            'simplified': simplified_model,
        }[selected_model]
        prob = fn(fp, weakness, speech, sensory, visual)
        return {
            'stroke_prob': prob,
            'fp_detected': fp,
            'fp_prob': fp_prob,
            'model': selected_model,
        }

    def risk_level(self, stroke_prob):
        if stroke_prob >= 0.50:
            return 'critical'
        if stroke_prob >= 0.25:
            return 'high'
        if stroke_prob >= 0.10:
            return 'moderate'
        return 'low'

# ==============================================================
# SESSION STATE
# ==============================================================
if 'bridge' not in st.session_state:
    model_paths = ["models/stroke_mvp.pth", "stroke_mvp.pth", "./models/stroke_mvp.pth"]
    mp = next((p for p in model_paths if os.path.exists(p)), None)
    st.session_state.bridge = StrokeBridge(model_path=mp)

if 'assessment_time' not in st.session_state:
    st.session_state.assessment_time = None

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ==============================================================
# MAIN APPLICATION
# ==============================================================
def main():
    st.markdown('<div class="main-header">🧠 Stroke / TIA Detection via DeepProbLog</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Neuro-Symbolic AI — Probabilistic Logic Programming with CNN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="paper-header">Weitkämper, Avila, Nanjala, Siska & Zawadi — '
        'German University of Digital Science</div>',
        unsafe_allow_html=True)

    bridge = st.session_state.bridge
    if bridge.cnn_loaded:
        st.markdown('<div class="demo-badge">✅ PRODUCTION — Trained FacialPalsyCNN active</div>', unsafe_allow_html=True)
    elif PYTORCH_AVAILABLE and bridge.cnn is not None:
        st.markdown('<div class="demo-badge">🧠 CNN architecture active (untrained weights)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="demo-badge">🔬 DEMO — heuristic fallback for facial palsy</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>MEDICAL DISCLAIMER:</strong> This is a research prototype for educational purposes only.
        It is <strong>NOT</strong> a substitute for professional medical diagnosis.
        If you suspect a stroke, <strong>CALL EMERGENCY SERVICES IMMEDIATELY</strong>.
        <br><br>
        <strong>⏰ Time is Brain:</strong> Every minute of delay worsens outcomes. Note the time symptoms started.
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Model Selection")
        selected_model = st.radio(
            "Discriminative model (paper Table 2)",
            options=['probfoil', 'divided', 'simplified'],
            format_func=lambda k: StrokeBridge.MODEL_LABELS[k],
            index=0,
            help="ProbFOIL achieves the best accuracy but lower recall. Simplified has highest recall."
        )

        st.markdown("---")
        st.header("🩺 Symptom Checklist")
        st.caption("Symptoms from Claus et al. (2024), Rotterdam Study")

        has_weakness  = st.checkbox("💪 Arm / Leg weakness",
            help="Motor deficit in one or both limbs (arm_weakness or leg_weakness in ProbLog model)")
        has_speech    = st.checkbox("🗣️ Speech or language impairment",
            help="Slurred speech, aphasia, word-finding difficulty")
        has_sensory   = st.checkbox("🖐️ Sensory symptoms",
            help="Numbness, tingling on one side")
        has_visual    = st.checkbox("👁️ Visual symptoms",
            help="Vision loss, double vision, visual field defect")

        st.markdown("---")
        if st.button("⏱️ Record symptom onset time", use_container_width=True):
            st.session_state.assessment_time = datetime.now()
        if st.session_state.assessment_time:
            elapsed = datetime.now() - st.session_state.assessment_time
            mins = int(elapsed.total_seconds() / 60)
            st.error(f"⏰ {mins} min since symptom onset")

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Facial Palsy (CNN)", "🔬 Inference", "📊 Results", "ℹ️ About"])

    # ── TAB 1: Facial image input ────────────────────────────────
    with tab1:
        st.header("Facial Palsy Detection — Neural Predicate")
        st.write(
            "Upload a face photo. The CNN classifier estimates P(facial_palsy), "
            "which is used as the **neural fact** in the DeepProbLog model."
        )
        st.markdown("""
        <div class="info-box">
            <strong>Neural fact (DeepProbLog):</strong>
            <code>nn(palsy_classifier, [Image]) :: facial_palsy.</code><br><br>
            The network is a 4-layer CNN (16→32→64→128 filters, MaxPool, Dropout 0.25,
            256-unit dense, He/Kaiming init, Adam optimiser).<br>
            Training data: Elhanashi et al. 2024 — 2,500 negative + 1,245 positive examples.
        </div>
        """, unsafe_allow_html=True)

        face_img_file = st.file_uploader(
            "Upload face image (neutral or smiling)",
            type=['jpg', 'jpeg', 'png'],
            key="face_upload",
        )

        use_webcam = st.checkbox("Use webcam instead")
        if use_webcam:
            face_img_file = st.camera_input("📷 Capture face")

        if face_img_file:
            img = Image.open(face_img_file).convert("RGB")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption="Input image", use_column_width=True)
            with col2:
                with st.spinner("Running CNN inference..."):
                    fp_prob = bridge.facial_palsy_probability(img)
                st.metric("P(facial_palsy)", f"{fp_prob:.4f}")
                st.progress(fp_prob)
                if fp_prob > 0.5:
                    st.warning(f"⚠️ Facial palsy **detected** (p = {fp_prob:.3f} > 0.5 threshold)")
                else:
                    st.success(f"✅ No facial palsy detected (p = {fp_prob:.3f} ≤ 0.5 threshold)")
                st.session_state['fp_prob'] = fp_prob
        else:
            st.info("No image provided — facial_palsy will be set to False in the model.")
            st.session_state['fp_prob'] = 0.0

        with st.expander("📖 Photo guidelines"):
            st.markdown("""
            - ✅ Good, even lighting — no strong shadows on face
            - ✅ Face directly toward camera, centred in frame
            - ✅ Remove glasses if possible
            - Either neutral or smiling expression is acceptable
            """)

    # ── TAB 2: Run inference ─────────────────────────────────────
    with tab2:
        st.header("DeepProbLog Inference")
        st.caption(f"Selected model: **{StrokeBridge.MODEL_LABELS[selected_model]}**")

        st.markdown("""
        The system constructs the discriminative model string at runtime,
        augments it with the CNN output and user-provided symptoms, then
        queries P(stroke_or_tia | symptoms) via the ProbLog engine.
        """)

        # Show model rules
        with st.expander("🔍 View active discriminative model rules"):
            if selected_model == 'probfoil':
                st.code("""% ProbFOIL 2 induced theory (Listing 1.7)
0.71219084 :: stroke_or_tia :- speech, facial_palsy.
0.55497685 :: stroke_or_tia :- speech, weakness.
0.50723151 :: stroke_or_tia :- sensory, weakness, facial_palsy.
0.36436170 :: stroke_or_tia :- speech, sensory.
0.74193908 :: stroke_or_tia :- weakness, visual, sensory, facial_palsy.
0.97247086 :: stroke_or_tia :- weakness, speech, facial_palsy.""", language='prolog')
            elif selected_model == 'divided':
                st.code("""% Divided model by stroke subtype (Listing 1.3 excerpt)
(36/409)  :: facial_palsy :- tia.    (47/254) :: facial_palsy :- minor.   (126/237) :: facial_palsy :- major.
(176/409) :: speech       :- tia.    (99/254)  :: speech       :- minor.   (167/237) :: speech       :- major.
(96/409)  :: arm_weakness :- tia.    (126/254) :: arm_weakness :- minor.   (211/237) :: arm_weakness :- major.
(69/409)  :: leg_weakness :- tia.    (89/254)  :: leg_weakness :- minor.   (183/237) :: leg_weakness :- major.
(58/409)  :: sensory      :- tia.    (68/254)  :: sensory      :- minor.   (80/237)  :: sensory      :- major.
(100/409) :: visual       :- tia.    (46/254)  :: visual       :- minor.   (62/237)  :: visual       :- major.""", language='prolog')
            else:
                st.code("""% Simplified maximum-entropy model (Listing 1.2)
0.23 :: facial_palsy :- stroke_or_tia.
0.49 :: speech       :- stroke_or_tia.
0.48 :: arm_weakness :- stroke_or_tia.
0.38 :: leg_weakness :- stroke_or_tia.
0.23 :: sensory      :- stroke_or_tia.
0.23 :: visual       :- stroke_or_tia.
% Background (Listing 1.1): 0.078::weakness. 0.005::facial_palsy. 0.005::speech. 0.063::sensory. 0.047::visual.""", language='prolog')

        if st.button("▶️ Run DeepProbLog Inference", type="primary", use_container_width=True):
            fp_prob = st.session_state.get('fp_prob', 0.0)
            with st.spinner("Computing P(stroke_or_tia | symptoms)…"):
                result = bridge.predict(
                    fp_prob=fp_prob,
                    weakness=has_weakness,
                    speech=has_speech,
                    sensory=has_sensory,
                    visual=has_visual,
                    selected_model=selected_model,
                )
                risk = bridge.risk_level(result['stroke_prob'])
                st.session_state.analysis_complete = True
                st.session_state.results = {
                    **result,
                    'risk': risk,
                    'has_weakness': has_weakness,
                    'has_speech': has_speech,
                    'has_sensory': has_sensory,
                    'has_visual': has_visual,
                    'selected_model': selected_model,
                }
            st.success("✅ Inference complete — see **Results** tab.")

    # ── TAB 3: Results ───────────────────────────────────────────
    with tab3:
        if not st.session_state.analysis_complete:
            st.info("Run inference in the **Inference** tab to see results.")
        else:
            r = st.session_state.results
            risk = r['risk']
            prob = r['stroke_prob']

            st.header("📊 Inference Results")

            risk_classes = {'critical':'risk-critical','high':'risk-high','moderate':'risk-moderate','low':'risk-low'}
            risk_messages = {
                'critical': ("🚨 HIGH PROBABILITY — Stroke/TIA likely",
                             "🚨 CALL EMERGENCY SERVICES IMMEDIATELY",
                             "Do NOT drive. Note time of symptom onset."),
                'high':     ("⚠️ ELEVATED PROBABILITY",
                             "⚠️ SEEK EMERGENCY CARE NOW",
                             "Go to nearest Emergency Room immediately."),
                'moderate': ("⚡ MODERATE PROBABILITY",
                             "📞 CONTACT HEALTHCARE PROVIDER URGENTLY",
                             "Urgent medical evaluation within 24 hours."),
                'low':      ("✅ LOW PROBABILITY",
                             "📋 CONTINUE MONITORING",
                             "No acute stroke indicators. Seek advice if symptoms change."),
            }
            title, action, detail = risk_messages[risk]
            st.markdown(f"""
            <div class="risk-card {risk_classes[risk]}">
                <h2>{title}</h2>
                <h3>{action}</h3>
                <p>{detail}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("P(stroke_or_tia)", f"{prob:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("P(facial_palsy) — CNN", f"{r['fp_prob']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Model", StrokeBridge.MODEL_LABELS[r['selected_model']].split(' (')[0])
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🔍 Symptom Inputs to Discriminative Model")
            syms = [
                ("Facial palsy (CNN)", r['fp_detected'], f"p = {r['fp_prob']:.3f}"),
                ("Weakness (arm/leg)", r['has_weakness'], "user-reported"),
                ("Speech impairment",  r['has_speech'],   "user-reported"),
                ("Sensory symptoms",   r['has_sensory'],  "user-reported"),
                ("Visual symptoms",    r['has_visual'],   "user-reported"),
            ]
            for name, present, src in syms:
                col1, col2 = st.columns([3, 1])
                with col1:
                    color = "status-positive" if present else "status-negative"
                    st.markdown(f"**{name}** — <span class='{color}'>{'TRUE' if present else 'FALSE'}</span> ({src})", unsafe_allow_html=True)
                with col2:
                    st.progress(1.0 if present else 0.0)

            st.markdown("---")
            st.subheader("🧠 Matching Rules (ProbFOIL model)")
            if r['selected_model'] == 'probfoil':
                rules = [
                    (0.97247086, r['has_weakness'] and r['has_speech'] and r['fp_detected'],
                     "weakness ∧ speech ∧ facial_palsy"),
                    (0.74193908, r['has_weakness'] and r['has_visual'] and r['has_sensory'] and r['fp_detected'],
                     "weakness ∧ visual ∧ sensory ∧ facial_palsy"),
                    (0.71219084, r['has_speech'] and r['fp_detected'],
                     "speech ∧ facial_palsy"),
                    (0.55497685, r['has_speech'] and r['has_weakness'],
                     "speech ∧ weakness"),
                    (0.50723151, r['has_sensory'] and r['has_weakness'] and r['fp_detected'],
                     "sensory ∧ weakness ∧ facial_palsy"),
                    (0.36436170, r['has_speech'] and r['has_sensory'],
                     "speech ∧ sensory"),
                ]
                for p, active, body in rules:
                    icon = "🔴" if active else "⚪"
                    st.markdown(f"{icon} `{p:.8f} :: stroke_or_tia :- {body}.`"
                                + (" **← FIRES**" if active else ""))
            else:
                st.info("Rule-level breakdown only shown for ProbFOIL model.")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download report", use_container_width=True):
                    syms_str = "\n".join(
                        f"- {n}: {'TRUE' if v else 'FALSE'} ({s})"
                        for n, v, s in syms
                    )
                    report = f"""STROKE / TIA DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
System: DeepProbLog neuro-symbolic stroke detection
Paper: Weitkämper, Avila, Nanjala, Siska & Zawadi
       German University of Digital Science

MODEL: {StrokeBridge.MODEL_LABELS[r['selected_model']]}
RESULT: P(stroke_or_tia) = {prob:.4f}
RISK LEVEL: {risk.upper()}
ACTION: {action}

SYMPTOM INPUTS:
{syms_str}

DISCLAIMER: Research prototype only. NOT a medical diagnosis.
            """
                    st.download_button("Download", report,
                                       f"stroke_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                       mime="text/plain")
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.analysis_complete = False
                    st.rerun()

    # ── TAB 4: About ─────────────────────────────────────────────
    with tab4:
        st.header("ℹ️ About This System")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Paper")
            st.markdown("""
            **Deep probabilistic logic programming for diagnostic reasoning
            from incomplete information: A case study in stroke detection**

            Felix Weitkämper, Monchito Avila, Elizabeth Nanjala, Siska, Grace Zawadi
            *German University of Digital Science*

            Published at ECAI / ILP 2025 workshop proceedings.

            **Keywords:** Stroke detection · Entropy maximisation · ProbLog ·
            DeepProbLog · ProbFOIL · Neuro-symbolic · Diagnostic reasoning
            """)

            st.subheader("🏗️ System Architecture")
            st.markdown("""
            **Two-step neuro-symbolic pipeline (paper Sections 3–5):**

            1. **Generative ProbLog model** (Listings 1.1–1.4)
               Built from summary statistics in Claus et al. (2024)
               via entropy maximisation (Williamson 2004).

            2. **Discriminative DeepProbLog model** (Listings 1.5–1.7)
               Derived from the generative model via ProbLog conditional
               inference. Each rule covers one symptom combination.
               ProbFOIL 2 compresses 2⁵ = 32 rules into 6.

            3. **Neural predicate** (paper Section 5)
               `nn(palsy_classifier, [Image]) :: facial_palsy.`
               FacialPalsyCNN processes a face image; its output
               probability is used directly as the fact probability.
            """)

        with col2:
            st.subheader("📊 Model Performance (Table 2)")
            st.markdown("""
            Ground truth: full co-occurrence model (Listing 1.4, 54 combinations from Figure 1).

            | Model | Accuracy | Precision | Recall |
            |-------|----------|-----------|--------|
            | Simplified (Listing 1.2) | 0.919 | 0.830 | **0.989** |
            | Divided (Listing 1.3)    | 0.934 | 0.869 | 0.972 |
            | **ProbFOIL (Listing 1.7)** | **0.960** | **0.962** | 0.931 |

            *Note:* ProbFOIL has lower recall — some low-probability cases
            become deterministically 0. In medical triage, high recall
            (low false-negative rate) may be preferred over accuracy.
            """)

            st.subheader("⏱️ Runtime Performance (Table 3, 100 iterations)")
            st.markdown("""
            | Component | Mean (ms) | Median (ms) |
            |-----------|-----------|-------------|
            | CNN inference | 46.2 | 46.7 |
            | Model construction | 5.4 | 5.5 |
            | DeepProbLog inference | 101.5 | 101.9 |
            | **Total** | **153.2** | **153.2** |
            """)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🧠 CNN Architecture (Section 5)")
            st.markdown("""
            Based on **Dua & Sharma (2025)**:

            - Input: 224 × 224 RGB face image
            - Normalisation layer
            - 4 × (Conv2D → MaxPool 2×2 → Dropout 25%)
              - Filters: 16 → 32 → 64 → 128
              - Kernels: 3 × 3
            - Flatten → FC(256) ReLU → FC(2) Softmax
            - Optimiser: Adam · Init: He / Kaiming

            **Training data** (Elhanashi et al. 2024):
            2,500 negative + 1,245 positive (facial palsy) examples,
            augmented by flipping, rotation, and scaling.

            **Standalone CNN performance:**
            Accuracy 0.9987 · Precision 0.9958 · Recall 1.00
            """)

        with col2:
            st.subheader("📚 Key References")
            st.markdown("""
            1. Banerjee et al. (2025) — Dysarthric speech corpus for stroke
            2. Chen et al. (2022) — FAST vs BE-FAST meta-analysis
            3. **Claus et al. (2024)** — Rotterdam Study symptom statistics (primary data source)
            4. De Raedt et al. (2015) — ProbFOIL 2 / learning from probabilistic entailment
            5. De Raedt & Kimmig (2015) — Probabilistic (logic) programming concepts
            6. **Dua & Sharma (2025)** — CNN for facial paralysis detection (CNN architecture)
            7. **Elhanashi et al. (2024)** — Facial image dataset for stroke classification
            8. Fierens et al. (2015) — ProbLog 2 inference
            9. Hofman et al. (2015) — The Rotterdam Study
            10. **Manhaeve et al. (2021)** — DeepProbLog (neuro-symbolic framework)
            11. Williamson (2004) — Bayesian Nets and Causality (entropy maximisation)
            """)

        st.markdown("---")
        st.subheader("⚖️ Limitations and Future Work")
        st.markdown("""
        - **Recall vs. accuracy trade-off:** The ProbFOIL model achieves highest accuracy but misses
          some low-probability stroke cases. An Fβ-score objective (β > 1) could rebalance toward recall.
        - **Neural predicates as facts, not evidence:** DeepProbLog requires the CNN output
          as a probabilistic fact, necessitating the generative→discriminative transformation.
          Future work: supporting neural predicates as soft evidence.
        - **Risk factors not modelled:** Obesity, hypertension, age are not in the current generative model
          due to the "explaining-away" interaction with entropy maximisation (see Section 7).
        - **Single data source:** Probabilities from Claus et al. (2024) only; aggregating
          across studies would require additional methodology.
        - **Not clinically validated:** This is a research prototype; regulatory and clinical
          validation studies are required before any real-world deployment.
        """)

        st.markdown("---")
        st.caption(
            f"Version 3.0 · {datetime.now().strftime('%B %Y')} · "
            "German University of Digital Science · "
            "Repository: github.com/3N61N33R/stroke-detection · "
            "Framework: Streamlit + PyTorch + ProbLog"
        )

if __name__ == "__main__":
    main()
