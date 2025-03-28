import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url


class BasicMLPPredictor(nn.Module):
    """A very basic MLP model for video frame prediction"""
    def __init__(self, input_frames=4, output_frames=4, frame_size=64, hidden_dims=[2048, 4096, 2048]):
        super(BasicMLPPredictor, self).__init__()
        
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.frame_size = frame_size
        
        # Calculate input and output dimensions
        self.input_dim = input_frames * frame_size * frame_size  # Flattened input frames
        self.output_dim = output_frames * frame_size * frame_size  # Flattened output frames
        
        # Create MLP layers
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Sigmoid())  # Use sigmoid to bound output between 0 and 1
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)
        
        # Flatten input to [batch_size, input_dim]
        x_flat = x.view(batch_size, -1)
        
        # Forward through MLP
        y_flat = self.mlp(x_flat)
        
        # Reshape output to [batch_size, output_frames, frame_size, frame_size]
        y = y_flat.view(batch_size, self.output_frames, self.frame_size, self.frame_size)
        
        return y