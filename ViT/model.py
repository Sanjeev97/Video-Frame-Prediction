import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from einops import rearrange, repeat

# Custom permute module to replace nn.Permute for compatibility
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return x.permute(*self.dims)


# Custom unflatten module for older PyTorch versions
class Unflatten(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        self.dim = dim
        self.shape = shape
        
    def forward(self, x):
        return x.reshape(-1, *self.shape) if self.dim == 1 else x


# Helper modules for Vision Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x


class VideoVisionTransformer(nn.Module):
    def __init__(
        self, 
        image_size=64, 
        patch_size=8, 
        input_frames=4,
        output_frames=4,
        dim=512, 
        depth=6, 
        heads=8, 
        mlp_dim=1024, 
        channels=1, 
        dim_head=64, 
        dropout=0.1, 
        emb_dropout=0.1
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.num_patches = num_patches
        self.channels = channels
        self.patch_dim = patch_dim
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2, 3)
        )
        
        # Position and frame encodings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.frame_embedding = nn.Parameter(torch.randn(1, input_frames, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer
        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # MLP head for prediction
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim * output_frames)
        )
        
        # Frame reconstruction
        self.to_frame = nn.Linear(patch_dim, image_size * image_size * channels)
    
    def forward(self, x):
        # x: [batch, frames, channels, height, width]
        b, f, c, h, w = x.shape
        
        # Process each frame into patches
        # Each frame goes from [batch, channels, height, width] to [batch, dim, n_patches]
        frame_patches = []
        for i in range(f):
            patches = self.to_patch_embedding(x[:, i])  # [batch, dim, n_patches]
            frame_patches.append(patches.permute(0, 2, 1))  # [batch, n_patches, dim]
        
        # Stack frame patches
        x = torch.stack(frame_patches, dim=1)  # [batch, frames, n_patches, dim]
        
        # Add positional and frame embeddings
        for i in range(f):
            x[:, i] = x[:, i] + self.pos_embedding
        
        # Add frame embedding across patches dimension
        frame_emb = self.frame_embedding.unsqueeze(2)  # [1, frames, 1, dim]
        x = x + frame_emb  # [batch, frames, n_patches, dim]
        
        # Add cls token to each frame
        cls_tokens = repeat(self.cls_token, '1 1 d -> b f 1 d', b=b, f=f)
        x = torch.cat((cls_tokens, x), dim=2)  # [batch, frames, n_patches+1, dim]
        
        # Combine batch and frames for transformer input
        x = rearrange(x, 'b f n d -> b (f n) d')
        
        # Apply transformer
        x = self.dropout(x)
        x = self.transformer(x)
        
        # Extract cls tokens for prediction
        cls_tokens = x[:, :f]  # [batch, frames, dim]
        
        # Predict future frames
        predictions = self.mlp_head(cls_tokens[:, -1])  # [batch, patch_dim * output_frames]
        
        # Reshape to separate output frames and patches
        predictions = predictions.reshape(b, self.output_frames, -1)  # [batch, output_frames, patch_dim]
        
        # Convert each predicted frame to image
        output_frames = []
        for i in range(self.output_frames):
            frame = predictions[:, i]  # [batch, patch_dim]
            
            # Linear projection to get pixels
            pixels = self.to_frame(frame)  # [batch, image_size*image_size*channels]
            
            # Reshape directly to image shape (ensure correct size)
            frame_recon = pixels.reshape(b, self.channels, self.image_size, self.image_size)
            
            # Apply sigmoid for values between 0 and 1
            frame_recon = torch.sigmoid(frame_recon)
            output_frames.append(frame_recon)
            
            # Stack output frames
            output = torch.stack(output_frames, dim=1)  # [batch, frames, channels, height, width]
        
        return output
