import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from model import *
from dataset import *

def train_model(train_dataloader, val_dataloader=None, epochs=10, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train the SimpleVideoPredictor model
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        device: Device to train on
    """
    # Create model
    model = SimpleVideoPredictor(
        input_channels=1,     # Grayscale images
        hidden_channels=64,   # Size of hidden features
        kernel_size=3,        # Standard 3x3 kernel
        num_layers=3          # 3 ConvLSTM layers
    )
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (input_frames, target_frames) in enumerate(train_dataloader):
            # Move data to device
            input_frames = input_frames.to(device)
            target_frames = target_frames.to(device)
            
            # Forward pass
            predicted_frames = model(input_frames)
            
            # Calculate loss
            loss = criterion(predicted_frames, target_frames)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} complete, Training Loss: {avg_train_loss:.6f}")
        
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_idx, (input_frames, target_frames) in enumerate(val_dataloader):
                    # Move data to device
                    input_frames = input_frames.to(device)
                    target_frames = target_frames.to(device)
                    
                    # Forward pass
                    predicted_frames = model(input_frames)
                    
                    # Calculate loss
                    loss = criterion(predicted_frames, target_frames)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_simple_model.pth")
                print("New best model saved!")
    
    return model


def predict_future_frames(model, input_frames, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predict future frames using trained model
    
    Args:
        model: Trained SimpleVideoPredictor model
        input_frames: Input sequence [batch, seq_len, channels, height, width]
        device: Device to run prediction on
        
    Returns:
        predicted_frames: Predicted frames [batch, future_frames, channels, height, width]
    """
    # Move data to device
    input_frames = input_frames.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Predict future frames
    with torch.no_grad():
        predicted_frames = model(input_frames)
    
    return predicted_frames


def visualize_prediction(input_frames, target_frames, predicted_frames, save_path=None):
    """
    Visualize input, target, and predicted frames
    
    Args:
        input_frames: Input sequence [batch, seq_len, channels, height, width]
        target_frames: Target sequence [batch, seq_len, channels, height, width]
        predicted_frames: Predicted sequence [batch, seq_len, channels, height, width]
        save_path: Path to save visualization
    """
    # Move tensors to CPU and take first batch
    input_frames = input_frames[0].cpu()
    target_frames = target_frames[0].cpu()
    predicted_frames = predicted_frames[0].cpu()
    
    # Get dimensions
    n_input = input_frames.shape[0]
    n_output = target_frames.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(3, max(n_input, n_output), figsize=(15, 5))
    
    # Plot input frames
    for i in range(n_input):
        axes[0, i].imshow(input_frames[i, 0], cmap='gray')
        axes[0, i].set_title(f'Input t={i+1}')
        axes[0, i].axis('off')
    
    # Plot target frames
    for i in range(n_output):
        axes[1, i].imshow(target_frames[i, 0], cmap='gray')
        axes[1, i].set_title(f'Target t={n_input+i+1}')
        axes[1, i].axis('off')
    
    # Plot predicted frames
    for i in range(n_output):
        axes[2, i].imshow(predicted_frames[i, 0], cmap='gray')
        axes[2, i].set_title(f'Predicted t={n_input+i+1}')
        axes[2, i].axis('off')
    
    # Set empty titles for unused axes
    for i in range(n_output, max(n_input, n_output)):
        if i < n_input:
            axes[1, i].axis('off')
            axes[2, i].axis('off')
        else:
            axes[0, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def run_training(batch_size=32, epochs=10):
    """
    Train the model on Moving MNIST dataset
    
    Args:
        batch_size: Batch size for training
        epochs: Number of training epochs
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MovingMNISTDataset(train=True)
    val_dataset = MovingMNISTDataset(train=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Train model
    model = train_model(train_dataloader, val_dataloader, epochs=epochs, device=device)
    
    return model


def demo_prediction():
    """
    Demonstrate prediction using a pre-trained model
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test dataset and dataloader
    test_dataset = MovingMNISTDataset(train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Create model
    model = SimpleVideoPredictor()
    model = model.to(device)
    
    # Load pre-trained model if available
    model_path = "best_simple_model.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No pre-trained model found. Using random initialization.")
        print("Consider training the model first using the run_training function.")
    
    # Get one sample for demonstration
    input_frames, target_frames = next(iter(test_dataloader))
    
    # Predict future frames
    predicted_frames = predict_future_frames(model, input_frames, device)
    
    # Print shapes
    print(f"Input frames shape: {input_frames.shape}")
    print(f"Target frames shape: {target_frames.shape}")
    print(f"Predicted frames shape: {predicted_frames.shape}")
    
    # Visualize prediction
    visualize_prediction(input_frames, target_frames, predicted_frames, save_path="simple_prediction.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Video Frame Prediction")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='predict',
                      help='Mode: train or predict')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training model...")
        run_training(batch_size=args.batch_size, epochs=args.epochs)
    else:
        print("Running prediction demo...")
        demo_prediction()

# python simple_video_prediction.py --mode train --epochs 10 --batch_size 32
# python simple_video_prediction.py --mode predict        