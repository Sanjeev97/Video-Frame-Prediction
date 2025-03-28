import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from einops import rearrange, repeat
from dataset import *
from model import *

def train_model(model, train_dataloader, val_dataloader=None, epochs=5, 
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train the Vision Transformer model"""
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5
    )
    
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
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
            validation_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best_vit_model.pth")
                print("New best model saved!")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), training_losses, label='Training Loss')
    if val_dataloader is not None:
        plt.plot(range(1, epochs+1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('vit_training_loss.png')
    plt.show()
    print("Training loss plot saved to 'vit_training_loss.png'")
    
    return model


def visualize_predictions(model, dataset, num_samples=4, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize predictions from the model on random samples
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of sequences to visualize
    """
    model = model.to(device)
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 12, figsize=(20, 3*num_samples))
    
    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        input_frames, target_frames = dataset[idx]
        
        # Add batch dimension
        input_frames = input_frames.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            predicted_frames = model(input_frames)
        
        # Move to CPU for visualization
        predicted_frames = predicted_frames.squeeze(0).cpu()
        
        # Plot input frames
        for j in range(4):
            if num_samples > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            ax.imshow(input_frames[0, j, 0].cpu(), cmap='gray')
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
        
        # Plot predicted frames
        for j in range(4):
            if num_samples > 1:
                ax = axes[i, j+8]
            else:
                ax = axes[j+8]
            ax.imshow(predicted_frames[j, 0], cmap='gray')
            ax.set_title(f'Predicted {j+1}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('vit_prediction_visualization.png')
    plt.show()
    print("Prediction visualization saved to 'vit_prediction_visualization.png'")


def visualize_prediction_sequence(model, dataset, sequence_idx=None,
                                 device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize a single prediction sequence with detailed frame-by-frame comparison
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        sequence_idx: Index of sequence to visualize (randomly selected if None)
    """
    model = model.to(device)
    model.eval()
    
    # Get sample (random if not specified)
    if sequence_idx is None:
        sequence_idx = np.random.randint(0, len(dataset))
    
    input_frames, target_frames = dataset[sequence_idx]
    
    # Add batch dimension
    input_frames = input_frames.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        predicted_frames = model(input_frames)
    
    # Move to CPU for visualization
    predicted_frames = predicted_frames.squeeze(0).cpu()
    
    # Plotting
    fig = plt.figure(figsize=(15, 10))
    
    # Add title with sequence information
    plt.suptitle(f"Moving MNIST Sequence {sequence_idx} Prediction (Vision Transformer)", fontsize=16)
    
    # Plot all input frames
    for i in range(4):
        plt.subplot(3, 4, i+1)
        plt.imshow(input_frames[0, i, 0].cpu(), cmap='gray')
        plt.title(f'Input Frame {i+1}')
        plt.axis('off')
    
    # Plot all target frames
    for i in range(4):
        plt.subplot(3, 4, i+5)
        plt.imshow(target_frames[i, 0], cmap='gray')
        plt.title(f'Target Frame {i+1}')
        plt.axis('off')
    
    # Plot all predicted frames
    for i in range(4):
        plt.subplot(3, 4, i+9)
        plt.imshow(predicted_frames[i, 0], cmap='gray')
        plt.title(f'Predicted Frame {i+1}')
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.savefig('vit_sequence_prediction.png')
    plt.show()
    print("Sequence prediction visualization saved to 'vit_sequence_prediction.png'")


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MovingMNISTDataset(train=True)
    val_dataset = MovingMNISTDataset(train=False)
    
    # Visualize some examples from the dataset
    print("Visualizing dataset examples...")
    visualize_dataset(train_dataset, num_samples=3)
    
    # Create model
    model = VideoVisionTransformer(
        image_size=64,        # 64x64 images
        patch_size=8,         # 8x8 patches
        input_frames=4,       # 4 input frames
        output_frames=4,      # 4 output frames
        dim=512,              # Embedding dimension
        depth=6,              # Number of transformer layers
        heads=8,              # Number of attention heads
        mlp_dim=1024,         # Dimension of MLP in transformer
        channels=1,           # Grayscale images
        dim_head=64,          # Dimension per head
        dropout=0.1,          # Dropout rate
        emb_dropout=0.1       # Embedding dropout rate
    )
    
    # Print model details
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters")
    
    # Check if we should train or load a pre-trained model
    model_path = "best_vit_model.pth"
    if not os.path.exists(model_path):
        # Create dataloaders for training
        batch_size = 16  # Lower batch size for ViT due to memory requirements
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train the model
        print("\nTraining model...")
        train_model(model, train_dataloader, val_dataloader, epochs=5, device=device)
    else:
        # Load pre-trained model
        print(f"\nLoading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Visualize predictions on multiple examples
    print("\nVisualizing model predictions on multiple examples...")
    visualize_predictions(model, val_dataset, num_samples=4, device=device)
    
    # Visualize a single example in detail
    print("\nVisualizing a single prediction sequence in detail...")
    visualize_prediction_sequence(model, val_dataset, device=device)


if __name__ == "__main__":
    main()