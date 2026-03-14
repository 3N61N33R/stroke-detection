"""
Facial Droop CNN Architecture
=============================

This module defines the Convolutional Neural Network (CNN) used to detect
facial asymmetry (droop) in static images.

Architecture Source:
    Based on the methodology described by Dua & Sharma (2023) for stroke detection.
    - 4 Convolutional Blocks (16 -> 32 -> 64 -> 128 filters)
    - Max Pooling & Dropout (0.25) after each block
    - Fully Connected Layers (256 -> 2 classes)
    - Softmax Output (Probability distribution)

Usage:
    from src.networks.facial_net import get_model
    model = get_model()
"""

import torch.nn as nn
import torch.nn.functional as F


class FacialDroopCNN(nn.Module):
    """
    A 4-layer CNN for binary classification of facial droop (Normal vs Stroke).
    """

    def __init__(self):
        super(FacialDroopCNN, self).__init__()

        # ------------------------------------------------------------------
        # 1. CONVOLUTIONAL LAYERS
        # ------------------------------------------------------------------
        # Increasing filter depth: 16 -> 32 -> 64 -> 128
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # ------------------------------------------------------------------
        # 2. SHARED LAYERS
        # ------------------------------------------------------------------
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # ------------------------------------------------------------------
        # 3. DENSE LAYERS
        # ------------------------------------------------------------------
        # Flatten size calculation:
        # Input 224x224 -> Pool 1 (112) -> Pool 2 (56) -> Pool 3 (28) -> Pool 4 (14)
        # Final Feature Map: 128 filters * 14 * 14 pixels
        self.flatten_dim = 128 * 14 * 14

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 2)  # Output: [P(Normal), P(Stroke)]

        # ------------------------------------------------------------------
        # 4. WEIGHT INITIALIZATION
        # ------------------------------------------------------------------
        self._initialize_weights()

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Block 4
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Classifier
        x = x.view(-1, self.flatten_dim)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Output Probabilities
        return F.softmax(x, dim=1)

    def _initialize_weights(self):
        """
        Applies He (Kaiming) Initialization to Conv layers and
        Normal Initialization to Linear layers to prevent 'Dying ReLU'.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================
def get_model():
    """Returns an instance of the FacialDroopCNN."""
    return FacialDroopCNN()
