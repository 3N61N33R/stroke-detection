"""
Stroke Detection Model Trainer
==============================

End-to-end training pipeline for the Facial Droop CNN.
Downloads the dataset, applies preprocessing, trains the model,
and saves the trained weights for inference.

Usage (Default):
    python src/training/train.py
    
Usage (Custom Parameters):
    python src/training/train.py --epochs 15 --lr 0.0005
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import kagglehub

# ==============================================================================
# SETUP PATHS
# ==============================================================================
# Establish root directory to ensure reliable imports regardless of where 
# the script is executed from.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.networks.facial_net import get_model

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def set_seed(seed):
    """
    Locks random number generators across all libraries.
    This guarantees that the 80/20 data split and weight initialization
    are exactly the same every time, ensuring experimental reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset_path(dataset_handle):
    """
    Downloads dataset via KaggleHub and recursively locates the root directory
    that contains the 'Stroke' and 'NonStroke' class folders.
    """
    print("Downloading dataset from Kaggle...")
    base_path = kagglehub.dataset_download(dataset_handle)
    
    for root, dirs, _ in os.walk(base_path):
        if "Stroke" in dirs and "NonStroke" in dirs:
            return root
            
    raise FileNotFoundError("Required class directories ('Stroke', 'NonStroke') not found.")

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def train(args):
    # 1. Initialization
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training pipeline on: {device}")

    # 2. Dataset Preparation
    dataset_dir = get_dataset_path(args.dataset)
    
    # Define transformations: resize for uniform CNN input, convert to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Load dataset and calculate an 80/20 split for training vs. validation
    full_dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Pass the locked seed generator to random_split to prevent data leakage
    generator = torch.Generator().manual_seed(args.seed)
    train_data, val_data = random_split(full_dataset, [train_size, val_size], generator=generator)

    # DataLoaders handle batching and shuffling the data automatically
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    print(f"Classes Detected: {full_dataset.classes}")
    print(f"Training samples: {len(train_data)} | Validation samples: {len(val_data)}")

    # 3. Model & Optimizer Configuration
    model = get_model().to(device)
    
    # CrossEntropyLoss expects raw unscaled logits, which matches our CNN output.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Training Loop
    print("\nBeginning training...")
    for epoch in range(args.epochs):
        
        # Set model to training mode (enables dropout layers)
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear previous gradients to prevent accumulation
            optimizer.zero_grad()
            
            # Forward pass: Predict, calculate error, backpropagate, update weights
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # 5. Validation Loop
        # Set model to evaluation mode (disables dropout layers for consistent testing)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        # torch.no_grad() disables gradient calculation, saving memory and speeding up validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)")

    # 6. Save Final Artifacts
    # Only save the state_dict (weights and biases), not the entire class object
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved successfully to: {args.save_path}")

if __name__ == "__main__":
    # Argparse allows hyperparameters to be modified from the terminal
    # without altering the source code, a standard requirement for academic review.
    parser = argparse.ArgumentParser(description="Train Facial Droop CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="abdussalamelhanashy/annotated-facial-images-for-stroke-classification", help="Kaggle dataset handle")
    parser.add_argument("--save-path", type=str, default=os.path.join(ROOT_DIR, "models", "stroke_mvp.pth"), help="Destination path for trained weights")
    
    args = parser.parse_args()
    train(args)
