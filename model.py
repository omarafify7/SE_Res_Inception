"""
SE-Res-Inception: A Novel Hybrid Architecture
==============================================

This module implements a modified GoogLeNet (Inception V1) with two modern "twists":

1. TWIST 1 (SE): Squeeze-and-Excitation blocks applied to EACH inception branch
   before concatenation, enabling channel-wise attention per branch.

2. TWIST 2 (Residual): Global residual connections around each Inception module
   with 1x1 projection convolutions when channel dimensions mismatch.

Author: Senior Computer Vision Research Engineer
Target: CIFAR-100 (32x32) | Hardware: NVIDIA RTX 5070 Ti with AMP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Applies channel-wise attention by:
    1. Squeeze: Global Average Pooling to get channel descriptor
    2. Excitation: FC -> ReLU -> FC -> Sigmoid to get channel weights
    3. Scale: Multiply input channels by their weights
    
    Args:
        channels (int): Number of input/output channels
        reduction (int): Reduction ratio for the bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        squeeze = self.squeeze(x).view(batch, channels)
        # Excitation: (B, C) -> (B, C)
        excitation = self.excitation(squeeze).view(batch, channels, 1, 1)
        # Scale: (B, C, H, W) * (B, C, 1, 1) -> (B, C, H, W)
        return x * excitation


class InceptionBranch(nn.Module):
    """
    A single branch of the Inception module.
    
    Supports four branch types:
    - '1x1': Single 1x1 convolution
    - '3x3': 1x1 reduce -> 3x3 convolution
    - '5x5': 1x1 reduce -> 5x5 convolution  
    - 'pool': MaxPool3x3 -> 1x1 projection
    
    Each branch includes BatchNorm and ReLU after each convolution.
    
    Args:
        in_channels (int): Input channel count
        out_channels (int): Output channel count for this branch
        branch_type (str): One of '1x1', '3x3', '5x5', 'pool'
        reduce_channels (int): Channels for 1x1 reduction (for 3x3/5x5 branches)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        branch_type: str,
        reduce_channels: int = None
    ):
        super().__init__()
        self.branch_type = branch_type
        
        if branch_type == '1x1':
            self.branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif branch_type == '3x3':
            self.branch = nn.Sequential(
                nn.Conv2d(in_channels, reduce_channels, 1, bias=False),
                nn.BatchNorm2d(reduce_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif branch_type == '5x5':
            # Using two 3x3 convs instead of 5x5 for efficiency (same receptive field)
            self.branch = nn.Sequential(
                nn.Conv2d(in_channels, reduce_channels, 1, bias=False),
                nn.BatchNorm2d(reduce_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_channels, out_channels, 5, padding=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        elif branch_type == 'pool':
            self.branch = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class ResInceptionBlock(nn.Module):
    """
    SE-Res-Inception Block: The Core Module
    
    This implements both "twists" of our architecture:
    
    TWIST 1 (SE): Each of the 4 parallel branches has its own SE block applied
                  BEFORE concatenation. This allows channel attention to be
                  learned independently for each branch's feature type.
    
    TWIST 2 (Residual): A skip connection adds the input to the output.
                        When channel dimensions mismatch, a 1x1 projection
                        aligns the input channels to match the output.
    
    Args:
        in_channels (int): Input channels to the block
        ch1x1 (int): Output channels for 1x1 branch
        ch3x3_reduce (int): Reduction channels for 3x3 branch
        ch3x3 (int): Output channels for 3x3 branch
        ch5x5_reduce (int): Reduction channels for 5x5 branch
        ch5x5 (int): Output channels for 5x5 branch
        pool_proj (int): Output channels for pooling branch projection
        se_reduction (int): SE block reduction ratio (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
        se_reduction: int = 16
    ):
        super().__init__()
        
        # Calculate total output channels (for residual projection)
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        
        # === INCEPTION BRANCHES ===
        self.branch_1x1 = InceptionBranch(in_channels, ch1x1, '1x1')
        self.branch_3x3 = InceptionBranch(in_channels, ch3x3, '3x3', ch3x3_reduce)
        self.branch_5x5 = InceptionBranch(in_channels, ch5x5, '5x5', ch5x5_reduce)
        self.branch_pool = InceptionBranch(in_channels, pool_proj, 'pool')
        
        # === TWIST 1: SE BLOCKS (one per branch, BEFORE concatenation) ===
        # This is the key innovation: we apply channel attention to each
        # branch independently, allowing the network to learn which channels
        # are important within each receptive field size.
        self.se_1x1 = SEBlock(ch1x1, se_reduction)
        self.se_3x3 = SEBlock(ch3x3, se_reduction)
        self.se_5x5 = SEBlock(ch5x5, se_reduction)
        self.se_pool = SEBlock(pool_proj, se_reduction)
        
        # === TWIST 2: RESIDUAL CONNECTION ===
        # Project input channels if they don't match output channels
        if in_channels != self.out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.residual_proj = nn.Identity()
        
        # Final activation after residual addition
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        identity = x
        
        # === PARALLEL BRANCHES ===
        out_1x1 = self.branch_1x1(x)
        out_3x3 = self.branch_3x3(x)
        out_5x5 = self.branch_5x5(x)
        out_pool = self.branch_pool(x)
        
        # === TWIST 1: Apply SE attention to EACH branch individually ===
        # This allows the network to weigh channels differently based on
        # the receptive field that produced them
        out_1x1 = self.se_1x1(out_1x1)
        out_3x3 = self.se_3x3(out_3x3)
        out_5x5 = self.se_5x5(out_5x5)
        out_pool = self.se_pool(out_pool)
        
        # === CONCATENATE along channel dimension ===
        out = torch.cat([out_1x1, out_3x3, out_5x5, out_pool], dim=1)
        
        # === TWIST 2: Add residual connection ===
        # Project identity if channel dimensions mismatch
        identity = self.residual_proj(identity)
        out = out + identity
        out = self.relu(out)
        
        return out


class SEResInception(nn.Module):
    """
    SE-Res-Inception: Full Network Architecture
    
    A modified GoogLeNet with:
    - SE blocks on each inception branch (channel attention per receptive field)
    - Residual connections around each inception block
    - NO auxiliary classifiers (residual connections make them obsolete)
    
    Optimized for CIFAR-100 (32x32 input) with adapted stem layer.
    
    Architecture Flow:
        Input (3, 32, 32)
        -> Stem: Conv 3x3/1 (64 channels, keep spatial)
        -> Inception blocks 1-2 (256 -> 480 channels)
        -> MaxPool (16 -> 8)
        -> Inception blocks 3-4 (480 -> 512 channels)
        -> Inception blocks 5-7 (512 -> 832 channels)
        -> MaxPool (8 -> 4)
        -> Inception blocks 8-9 (832 -> 1024 channels)
        -> Global Average Pool (1024, 1, 1)
        -> Dropout -> FC (num_classes)
    
    Args:
        num_classes (int): Number of output classes (default: 100 for CIFAR-100)
        dropout (float): Dropout probability before final FC (default: 0.4)
    """
    
    def __init__(self, num_classes: int = 100, dropout: float = 0.4):
        super().__init__()
        
        # === STEM LAYER ===
        # Adapted for CIFAR-100 32x32 input (no aggressive downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # === SE-RES-INCEPTION BLOCKS ===
        # Block configurations: (in_ch, 1x1, 3x3r, 3x3, 5x5r, 5x5, pool_proj)
        # Designed to progressively increase channel count while maintaining
        # a balance between different receptive field branches
        
        # Stage 1: 32x32 spatial, 64 -> 256 -> 480 channels
        self.inception1 = ResInceptionBlock(64, 64, 96, 128, 16, 32, 32)    # out: 256
        self.inception2 = ResInceptionBlock(256, 128, 128, 192, 32, 96, 64) # out: 480
        
        # Downsample: 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2: 16x16 spatial, 480 -> 512 -> 512 channels
        self.inception3 = ResInceptionBlock(480, 192, 96, 208, 16, 48, 64)   # out: 512
        self.inception4 = ResInceptionBlock(512, 160, 112, 224, 24, 64, 64)  # out: 512
        
        # Stage 3: 16x16 spatial, 512 -> 528 -> 832 channels
        self.inception5 = ResInceptionBlock(512, 128, 128, 256, 24, 64, 80)  # out: 528 (adjusted from original)
        self.inception6 = ResInceptionBlock(528, 256, 160, 320, 32, 128, 128) # out: 832
        self.inception7 = ResInceptionBlock(832, 256, 160, 320, 32, 128, 128) # out: 832
        
        # Downsample: 16x16 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 4: 8x8 spatial, 832 -> 1024 channels
        self.inception8 = ResInceptionBlock(832, 256, 160, 320, 32, 128, 128)     # out: 832
        self.inception9 = ResInceptionBlock(832, 384, 192, 384, 48, 128, 128)     # out: 1024
        
        # === CLASSIFIER HEAD ===
        # Global Average Pooling -> Dropout -> Fully Connected
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for Conv/Linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)           # (B, 64, 32, 32)
        
        # Stage 1
        x = self.inception1(x)     # (B, 256, 32, 32)
        x = self.inception2(x)     # (B, 480, 32, 32)
        x = self.pool1(x)          # (B, 480, 16, 16)
        
        # Stage 2
        x = self.inception3(x)     # (B, 512, 16, 16)
        x = self.inception4(x)     # (B, 512, 16, 16)
        
        # Stage 3
        x = self.inception5(x)     # (B, 528, 16, 16)
        x = self.inception6(x)     # (B, 832, 16, 16)
        x = self.inception7(x)     # (B, 832, 16, 16)
        x = self.pool2(x)          # (B, 832, 8, 8)
        
        # Stage 4
        x = self.inception8(x)     # (B, 832, 8, 8)
        x = self.inception9(x)     # (B, 1024, 8, 8)
        
        # Classifier
        x = self.global_pool(x)    # (B, 1024, 1, 1)
        x = torch.flatten(x, 1)    # (B, 1024)
        x = self.dropout(x)
        x = self.fc(x)             # (B, num_classes)
        
        return x


def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> None:
    """Print model parameter count and a test forward pass"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{'='*60}")
    print(f"SE-Res-Inception Model Summary")
    print(f"{'='*60}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*60}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input Shape:  {list(x.shape)}")
    print(f"Output Shape: {list(out.shape)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Quick test
    model = SEResInception(num_classes=100)
    get_model_summary(model)
