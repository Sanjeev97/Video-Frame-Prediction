import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from einops import rearrange, repeat


class MovingMNISTDataset(Dataset):
    def __init__(self, root="./data", train=True, seq_length=4, pred_length=4):
        """Simple Moving MNIST dataset"""
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
            print(f"Downloading Moving MNIST dataset...")
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


def visualize_dataset(dataset, num_samples=3):
    """
    Visualize examples from the Moving MNIST dataset
    
    Args:
        dataset: MovingMNISTDataset instance
        num_samples: Number of sequences to visualize
    """
    fig, axes = plt.subplots(num_samples, 8, figsize=(20, 3*num_samples))
    
    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        input_frames, target_frames = dataset[idx]
        
        # Plot input frames
        for j in range(4):
            if num_samples > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            ax.imshow(input_frames[j, 0], cmap='gray')
            ax.set_title(f'Input {j+1}')
            ax.axis('off')
        
        # Plot target frames
        for j in range(4):
            if num_samples > 1:
                ax = axes[i, j+4]
            else:
                ax = axes[j+4]
            ax.imshow(target_frames[j, 0], cmap='gray')
            ax.set_title(f'Target {j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_visualization.png')
    plt.show()
    print("Dataset visualization saved to 'dataset_visualization.png'")
