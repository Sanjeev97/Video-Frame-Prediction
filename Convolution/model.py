import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url

class ConvLSTMCell(nn.Module):
    """
    Basic Convolutional LSTM Cell
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Combined gates for efficiency (input, forget, output, cell gates)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 
            4 * hidden_channels, 
            kernel_size=kernel_size, 
            padding=self.padding
        )
    
    def forward(self, input_tensor, hidden_state):
        """
        Forward pass of ConvLSTM Cell
        
        Args:
            input_tensor: Input feature map [batch, channels, height, width]
            hidden_state: Previous hidden and cell states (h, c)
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        h_cur, c_cur = hidden_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Calculate all gates at once
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Calculate next cell state
        c_next = f * c_cur + i * g
        
        # Calculate next hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """Initialize hidden state and cell state to zeros"""
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class SimpleVideoPredictor(nn.Module):
    """
    Simple model for predicting future video frames using ConvLSTM
    """
    def __init__(self, input_channels=1, hidden_channels=64, kernel_size=3, num_layers=3):
        super(SimpleVideoPredictor, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        # First layer takes in the input frame
        self.cell_list = nn.ModuleList()
        self.cell_list.append(ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        ))
        
        # Additional layers
        for i in range(1, num_layers):
            self.cell_list.append(ConvLSTMCell(
                input_channels=hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size
            ))
        
        # Output layer: convert back to the original number of channels
        self.conv_output = nn.Conv2d(
            hidden_channels, input_channels, kernel_size=1, padding=0, bias=True
        )
    
    def forward(self, input_frames, future_frames=4):
        """
        Forward pass of the model
        
        Args:
            input_frames: Input sequence [batch, seq_len, channels, height, width]
            future_frames: Number of future frames to predict
            
        Returns:
            outputs: Predicted frames [batch, future_frames, channels, height, width]
        """
        # Get dimensions
        batch_size, seq_len, channels, height, width = input_frames.size()
        device = input_frames.device
        
        # Initialize hidden states for all layers
        hidden_states = []
        for i in range(self.num_layers):
            hidden_states.append(self.cell_list[i].init_hidden(batch_size, (height, width)))
        
        # Process input sequence to initialize hidden states
        for t in range(seq_len):
            # Current input frame
            current_input = input_frames[:, t]
            
            # First layer
            h, c = self.cell_list[0](current_input, hidden_states[0])
            hidden_states[0] = (h, c)
            
            # Remaining layers
            for i in range(1, self.num_layers):
                h, c = self.cell_list[i](hidden_states[i-1][0], hidden_states[i])
                hidden_states[i] = (h, c)
        
        # Predict future frames
        outputs = []
        
        # Use the last output as the first input for prediction
        next_input = input_frames[:, -1]
        
        for t in range(future_frames):
            # First layer
            h, c = self.cell_list[0](next_input, hidden_states[0])
            hidden_states[0] = (h, c)
            
            # Remaining layers
            for i in range(1, self.num_layers):
                h, c = self.cell_list[i](hidden_states[i-1][0], hidden_states[i])
                hidden_states[i] = (h, c)
            
            # Generate output frame
            next_input = self.conv_output(hidden_states[-1][0])
            outputs.append(next_input)
        
        # Stack along time dimension [batch, seq_len, channels, height, width]
        outputs = torch.stack(outputs, dim=1)
        
        return outputs