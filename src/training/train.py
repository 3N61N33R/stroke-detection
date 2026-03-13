"""
Stroke Detection Model Trainer
==============================

This script handles the end-to-end training pipeline for the Facial Droop CNN.
It performs the following steps:
1. Downloads the specific stroke dataset from Kaggle.
2. Preprocesses images (Resize, Tensor conversion).
3. Trains the ResNet-based model (defined in src.networks.facial_net).
4. Validates performance after each epoch.
5. Saves the trained weights to the project's 'models' directory.

Usage:
    python src/training/train.py
"""

import os
import sys
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ==============================================================================
# SETUP PATHS
# ==============================================================================
# Calculate project root to allow importing modules from 'src'
# Logic: Current File -> src/training -> src -> Project Root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Import the model architecture
from src.networks.facial_net import get_model

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model output path
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "stroke_mvp.pth")
DATASET_HANDLE = "abdussalamelhanashy/annotated-facial-images-for-stroke-classification"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def find_dataset_root(start_path):
    """
    Recursively searches for the directory containing class subfolders
    ('Stroke' and 'NonStroke') required by ImageFolder.
    """
    if not os.path.exists(start_path):
        return None

    for root, dirs, files in os.walk(start_path):
        # Case-insensitive check might be safer, but strict for now matches dataset
        if "Stroke" in dirs and "NonStroke" in dirs:
            return root
    return None


def train_model():
    """
    Main execution routine: Download -> Load -> Train -> Save.
    """
    print(f"🚀 Initializing Training Pipeline on {DEVICE}...")

    # ---------------------------------------------------------
    # 1. DOWNLOAD DATASET
    # ---------------------------------------------------------
    print("📥 Downloading dataset via KaggleHub...")
    try:
        # Note: Requires ~/.kaggle/kaggle.json to be present
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"✅ Download complete at: {path}")
    except Exception as e:
        print(f"❌ Kaggle Download Failed: {e}")
        print("💡 Hint: Ensure you have your API key at ~/.kaggle/kaggle.json")
        return

    # ---------------------------------------------------------
    # 2. LOCATE DATA
    # ---------------------------------------------------------
    dataset_dir = find_dataset_root(path)
    if not dataset_dir:
        raise FileNotFoundError(
            "Could not find 'Stroke' and 'NonStroke' folders in the downloaded dataset."
        )

    print(f"✅ Valid dataset root identified: {dataset_dir}")

    # ---------------------------------------------------------
    # 3. PREPROCESSING & SPLITTING
    # ---------------------------------------------------------
    data_transforms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            # Optional: Add Normalization here if needed for ResNet
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    full_dataset = datasets.ImageFolder(dataset_dir, transform=data_transforms)
    print(f"ℹ️  Classes Detected: {full_dataset.classes}")

    # 80/20 Train-Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"ℹ️  Training Samples: {len(train_data)} | Validation Samples: {len(val_data)}"
    )

    # ---------------------------------------------------------
    # 4. INITIALIZE MODEL
    # ---------------------------------------------------------
    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ---------------------------------------------------------
    # 5. TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # --- TRAINING PHASE ---
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # --- STATISTICS ---
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} ({train_acc:.1f}%) | "
            f"Val Loss: {avg_val_loss:.4f} ({val_acc:.1f}%)"
        )

    # ---------------------------------------------------------
    # 6. SAVE ARTIFACTS
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 Model saved successfully to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
