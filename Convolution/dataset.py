import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url


class MovingMNISTDataset(Dataset):
    def __init__(self, root="./data", train=True, seq_length=4, pred_length=4):
        """
        Simple Moving MNIST dataset
        
        Args:
            root: Directory to store dataset
            train: Whether to use training or test set
            seq_length: Number of input frames
            pred_length: Number of frames to predict
        """
        self.root = root
        self.train = train
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        # Create directory if it doesn't exist
        os.makedirs(self.root, exist_ok=True)
        
        # URL and filename
        self.url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
        self.filename = "mnist_test_seq.npy"
        self.file_path = os.path.join(self.root, self.filename)
        
        # Download the dataset if it doesn't exist
        if not os.path.exists(self.file_path):
            print(f"Downloading Moving MNIST dataset to {self.file_path}...")
            download_url(self.url, self.root, self.filename, None)
            print("Download complete!")
        
        # Load the dataset
        self.data = np.load(self.file_path)
        # Moving MNIST shape: [20, 10000, 64, 64]
        # 20 frames, 10000 samples, 64x64 resolution
        
        # Reshape to [10000, 20, 64, 64]
        self.data = np.transpose(self.data, (1, 0, 2, 3))
        
        # Split into train and test
        if self.train:
            self.data = self.data[:9000]  # Use 9000 samples for training
        else:
            self.data = self.data[9000:]  # Use 1000 samples for testing
        
        print(f"Dataset loaded: {'Training' if train else 'Test'} set with {len(self.data)} sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the sequence
        sequence = self.data[idx]
        
        # Convert to torch tensor and add channel dimension [frame, height, width] -> [frame, channel, height, width]
        sequence = torch.from_numpy(sequence).float().unsqueeze(1) / 255.0
        
        # Split into input and target sequences
        input_frames = sequence[:self.seq_length]
        target_frames = sequence[self.seq_length:self.seq_length + self.pred_length]
        
        return input_frames, target_frames