import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.theory import Theory
from deepproblog.dataset import DataLoader

# ==========================================
# 1. DEFINE MODEL ARCHITECTURE
# ==========================================
# This matches the 4-layer CNN trained.
class StrokeNet(nn.Module):
    def __init__(self):
        super(StrokeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Input features: 128 channels * 14 * 14 pixels
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 2) # [normal, droop]

    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# ==========================================
# 2. THE BRIDGE CLASS
# ==========================================
class StrokeBridge:
    def __init__(self, model_path='models/stroke_mvp_model.pth', logic_path='src/logic/stroke_logic.pl'):
        self.device = torch.device("cpu") # CPU is sufficient for inference
        
        # A. LOAD PYTORCH MODEL
        self.cnn = StrokeNet().to(self.device)
        try:
            # Load weights (map_location ensures CPU compatibility)
            self.cnn.load_state_dict(torch.load(model_path, map_location=self.device))
            self.cnn.eval()
            print("✅ Bridge: Vision Model Loaded.")
        except FileNotFoundError:
            print(f"⚠️ Bridge Warning: Model not found at {model_path}. Using random weights.")

        # B. DEFINE IMAGE TRANSFORM
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # C. SETUP DEEPPROBLOG NETWORK
        # Matches Prolog: nn(stroke_resnet, [Img], [normal, droop])
        self.net = Network(self.cnn, "stroke_resnet", batching=True)
        self.net.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=0.001)

        # D. LOAD LOGIC ENGINE
        self.logic_path = logic_path
        self.model = Model(self.logic_path, [self.net])
        self.engine = ExactEngine(self.model)
        print("✅ Bridge: Logic Engine Ready.")

    def analyze_patient(self, neutral_img, smile_img, user_data):
        """
        Main Interface.
        Args:
            neutral_img (PIL Image): User's neutral face.
            smile_img (PIL Image): User's smiling face.
            user_data (dict): Dictionary of symptoms.
        Returns:
            dict: Probabilities and Risk Category.
        """
        # 1. PREPARE IMAGES
        # Transform and add batch dimension
        t_neutral = self.transform(neutral_img).unsqueeze(0)
        t_smile = self.transform(smile_img).unsqueeze(0)

        # Register images in DeepProbLog's internal store
        # We give them logical names: 'img_neutral' and 'img_smile'
        self.net.add_tensor_source("img_neutral", t_neutral)
        self.net.add_tensor_source("img_smile", t_smile)

        # 2. BUILD THE QUERY CONTEXT
        # We create a temporary string of facts for this specific patient
        patient_id = "patient_x"
        facts = ""

        # --- A. Demographics ---
        if user_data.get('gender') == 'Female':
            facts += f"gender({patient_id}, female).\n"
        else:
            facts += f"gender({patient_id}, male).\n"

        # --- B. FAST Symptoms ---
        if user_data.get('speech'): facts += f"speech_issue({patient_id}).\n"
        if user_data.get('arm'):    facts += f"arm_weakness({patient_id}).\n"

        # --- C. BE-FAST / Hidden Symptoms ---
        if user_data.get('vision'): facts += f"vision_change({patient_id}).\n"
        if user_data.get('dizzy'):  facts += f"dizziness({patient_id}).\n"

        # --- D. History & Mimics ---
        if user_data.get('history_tia'):    facts += f"history_recent_tia({patient_id}).\n"
        if user_data.get('seizure'):        facts += f"history_seizures({patient_id}).\n"
        
        # Complex Mimic Logic: Old stroke BUT NO new symptoms
        if user_data.get('prior_stroke'):
            facts += f"history_prior_stroke({patient_id}).\n"
            # If they report ANY new symptom, we assert new_symptom(P).
            has_new = (user_data.get('speech') or user_data.get('arm') or 
                       user_data.get('vision') or user_data.get('dizzy'))
            if has_new:
                facts += f"new_symptom({patient_id}).\n"

        # --- E. The Crucial Bridge Rule ---
        # Connect the physical images to the logical patient
        # Prolog Rule: facial_droop_detected(Neutral, Smile)
        facts += f"facial_droop_detected(img_neutral, img_smile).\n"
        
        # 3. EXECUTE QUERIES
        # We need to temporarily add these facts to the database
        db = self.model.database
        
        # Add facts individually
        fact_list = [f for f in facts.split('\n') if f.strip()]
        for f in fact_list:
            db.add_fact(f)

        results = {}
        try:
            # Query 1: Probability of Stroke
            q_prob = f"stroke_probability({patient_id})"
            results['stroke_prob'] = self.engine.query(q_prob)[q_prob]

            # Query 2: Is it Critical? (911)
            q_911 = f"urgent_call_911({patient_id})"
            results['call_911'] = self.engine.query(q_911)[q_911]

            # Query 3: Is it Urgent Care?
            q_urgent = f"seek_urgent_care({patient_id})"
            results['urgent_care'] = self.engine.query(q_urgent)[q_urgent]
            
            # Query 4: Risk Category
            # This is tricky in DPL because it returns a distribution.
            # We check the highest probability category.
            categories = ['critical', 'high', 'moderate', 'low']
            best_cat = 'low'
            best_score = -1.0
            
            for cat in categories:
                q_cat = f"risk_category({cat}, {patient_id})"
                score = self.engine.query(q_cat)[q_cat]
                if score > best_score:
                    best_score = score
                    best_cat = cat
            
            results['risk_category'] = best_cat

        except Exception as e:
            print(f"❌ Logic Inference Failed: {e}")
            results = {'stroke_prob': 0.0, 'risk_category': 'error'}

        # 4. CLEANUP
        # Remove the temporary facts so they don't pollute the next patient's scan
        for f in fact_list:
            db.remove_fact(f)

        return results