"""
SE-Res-Inception V2: Improved Generalization Architecture
==========================================================

This module implements a modified GoogLeNet (Inception V1) with modern techniques
to combat overfitting in low-data regimes like CIFAR-100.

ARCHITECTURE CHANGES (V2):

1. SE OPTIMIZATION: Single SE block applied AFTER branch concatenation
   (more efficient than per-branch SE, enables global feature recalibration)

2. REGULARIZATION: Dropout layer after SE block within each Inception module

3. STOCHASTIC DEPTH: Randomly drops residual paths during training
   (similar to DropPath in Vision Transformers)

Author: Senior Computer Vision Research Engineer
Target: CIFAR-100 (32x32) | Hardware: NVIDIA RTX 5070 Ti with AMP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    
    Applies channel-wise attention by:
    1. Squeeze: Global Average Pooling to get channel descriptor
    2. Excitation: FC -> ReLU -> FC -> Sigmoid to get channel weights
    3. Scale: Multiply input channels by their weights
    
    Args:
        channels: Number of input/output channels
        reduction: Reduction ratio for the bottleneck (default: 16)
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 8)  # Ensure minimum of 8 channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
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


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for residual connections.
    
    During training, randomly drops the entire residual branch with probability `drop_prob`,
    forcing the network to rely on the skip connection. This provides strong regularization
    and is widely used in Vision Transformers (ViT, DeiT, Swin).
    
    During inference, the full path is always used (with appropriate scaling).
    
    Args:
        drop_prob: Probability of dropping the path (default: 0.0)
    """
    
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Survival probability
        keep_prob = 1 - self.drop_prob
        
        # Create binary mask: shape (batch_size, 1, 1, 1) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        
        # Scale output during training to maintain expected value
        return x / keep_prob * binary_mask
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


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
        in_channels: Input channel count
        out_channels: Output channel count for this branch
        branch_type: One of '1x1', '3x3', '5x5', 'pool'
        reduce_channels: Channels for 1x1 reduction (for 3x3/5x5 branches)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        branch_type: str,
        reduce_channels: Optional[int] = None
    ) -> None:
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
    SE-Res-Inception Block V2: Improved Core Module
    
    CHANGES FROM V1 (to reduce overfitting):
    
    1. SE OPTIMIZATION: Single SE block applied AFTER concatenation
       - More computationally efficient (1 SE vs 4 SE blocks)
       - Enables global feature recalibration across all branches
       - Applied BEFORE residual addition for cleaner gradients
    
    2. DROPOUT: Applied after SE block for regularization
       - Prevents co-adaptation of features
       - Only active during training
    
    3. STOCHASTIC DEPTH: Randomly drops residual path during training
       - Forces network to learn robust features through skip connection
       - Drop probability can be scheduled (higher for deeper blocks)
    
    Args:
        in_channels: Input channels to the block
        ch1x1: Output channels for 1x1 branch
        ch3x3_reduce: Reduction channels for 3x3 branch
        ch3x3: Output channels for 3x3 branch
        ch5x5_reduce: Reduction channels for 5x5 branch
        ch5x5: Output channels for 5x5 branch
        pool_proj: Output channels for pooling branch projection
        se_reduction: SE block reduction ratio (default: 16)
        dropout_prob: Dropout probability after SE (default: 0.2)
        drop_path_prob: Stochastic depth probability (default: 0.0)
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
        se_reduction: int = 16,
        dropout_prob: float = 0.2,
        drop_path_prob: float = 0.0
    ) -> None:
        super().__init__()
        
        # Calculate total output channels (for residual projection)
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        
        # === INCEPTION BRANCHES ===
        self.branch_1x1 = InceptionBranch(in_channels, ch1x1, '1x1')
        self.branch_3x3 = InceptionBranch(in_channels, ch3x3, '3x3', ch3x3_reduce)
        self.branch_5x5 = InceptionBranch(in_channels, ch5x5, '5x5', ch5x5_reduce)
        self.branch_pool = InceptionBranch(in_channels, pool_proj, 'pool')
        
        # === SE BLOCK (V2: Single block AFTER concatenation) ===
        # This is more efficient and allows global recalibration
        self.se = SEBlock(self.out_channels, se_reduction)
        
        # === DROPOUT (V2: Regularization after SE) ===
        self.dropout = nn.Dropout2d(p=dropout_prob)
        
        # === STOCHASTIC DEPTH (V2: DropPath for residual) ===
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        
        # === RESIDUAL CONNECTION ===
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
        
        # === CONCATENATE along channel dimension ===
        out = torch.cat([out_1x1, out_3x3, out_5x5, out_pool], dim=1)
        
        # === V2: Apply SE attention AFTER concatenation ===
        # Global recalibration across all branch features
        out = self.se(out)
        
        # === V2: Apply Dropout for regularization ===
        out = self.dropout(out)
        
        # === V2: Apply Stochastic Depth (DropPath) ===
        # Randomly drops the entire transformed branch during training
        out = self.drop_path(out)
        
        # === Add residual connection ===
        identity = self.residual_proj(identity)
        out = out + identity
        out = self.relu(out)
        
        return out


class SEResInception(nn.Module):
    """
    SE-Res-Inception V2: Full Network Architecture with Improved Generalization
    
    Changes from V1:
    - Single SE block per inception module (after concatenation)
    - Dropout after SE for regularization
    - Stochastic Depth with linearly increasing drop probability
    - NO auxiliary classifiers (residual connections make them obsolete)
    
    Optimized for CIFAR-100 (32x32 input) with adapted stem layer.
    
    Architecture Flow:
        Input (3, 32, 32)
        -> Stem: Conv 3x3/1 (64 channels, keep spatial)
        -> Inception blocks 1-2 (256 -> 480 channels)
        -> MaxPool (32 -> 16)
        -> Inception blocks 3-4 (480 -> 512 channels)
        -> Inception blocks 5-7 (512 -> 832 channels)
        -> MaxPool (16 -> 8)
        -> Inception blocks 8-9 (832 -> 1024 channels)
        -> Global Average Pool (1024, 1, 1)
        -> Dropout -> FC (num_classes)
    
    Args:
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        dropout: Dropout probability before final FC (default: 0.5)
        block_dropout: Dropout within inception blocks (default: 0.2)
        drop_path_rate: Maximum stochastic depth rate (default: 0.2)
    """
    
    def __init__(
        self, 
        num_classes: int = 100, 
        dropout: float = 0.5,
        block_dropout: float = 0.2,
        drop_path_rate: float = 0.2
    ) -> None:
        super().__init__()
        
        self.num_blocks = 9  # Total inception blocks
        
        # === STEM LAYER ===
        # Adapted for CIFAR-100 32x32 input (no aggressive downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # === STOCHASTIC DEPTH SCHEDULE ===
        # Linear increase from 0 to drop_path_rate across blocks
        # Earlier blocks have lower drop probability (more important features)
        drop_rates = [drop_path_rate * i / (self.num_blocks - 1) for i in range(self.num_blocks)]
        
        # === SE-RES-INCEPTION BLOCKS ===
        # Block configurations: (in_ch, 1x1, 3x3r, 3x3, 5x5r, 5x5, pool_proj)
        
        # Stage 1: 32x32 spatial, 64 -> 256 -> 480 channels
        self.inception1 = ResInceptionBlock(
            64, 64, 96, 128, 16, 32, 32,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[0]
        )  # out: 256
        self.inception2 = ResInceptionBlock(
            256, 128, 128, 192, 32, 96, 64,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[1]
        )  # out: 480
        
        # Downsample: 32x32 -> 16x16
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 2: 16x16 spatial, 480 -> 512 -> 512 channels
        self.inception3 = ResInceptionBlock(
            480, 192, 96, 208, 16, 48, 64,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[2]
        )  # out: 512
        self.inception4 = ResInceptionBlock(
            512, 160, 112, 224, 24, 64, 64,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[3]
        )  # out: 512
        
        # Stage 3: 16x16 spatial, 512 -> 528 -> 832 channels
        self.inception5 = ResInceptionBlock(
            512, 128, 128, 256, 24, 64, 80,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[4]
        )  # out: 528
        self.inception6 = ResInceptionBlock(
            528, 256, 160, 320, 32, 128, 128,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[5]
        )  # out: 832
        self.inception7 = ResInceptionBlock(
            832, 256, 160, 320, 32, 128, 128,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[6]
        )  # out: 832
        
        # Downsample: 16x16 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stage 4: 8x8 spatial, 832 -> 1024 channels
        self.inception8 = ResInceptionBlock(
            832, 256, 160, 320, 32, 128, 128,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[7]
        )  # out: 832
        self.inception9 = ResInceptionBlock(
            832, 384, 192, 384, 48, 128, 128,
            dropout_prob=block_dropout, drop_path_prob=drop_rates[8]
        )  # out: 1024
        
        # === CLASSIFIER HEAD ===
        # Global Average Pooling -> Dropout -> Fully Connected
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
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
    print(f"SE-Res-Inception V2 Model Summary")
    print(f"{'='*60}")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*60}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # Set to eval mode for consistent output
    x = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input Shape:  {list(x.shape)}")
    print(f"Output Shape: {list(out.shape)}")
    print(f"{'='*60}")
    
    # Print regularization info
    print(f"\nRegularization Settings:")
    print(f"  - Block Dropout: {model.inception1.dropout.p}")
    print(f"  - Final Dropout: {model.dropout.p}")
    print(f"  - Stochastic Depth: 0.0 -> {model.inception9.drop_path.drop_prob:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Quick test
    model = SEResInception(num_classes=100)
    get_model_summary(model)
    
    # Test training vs eval mode behavior
    print("\nTesting train/eval mode:")
    model.train()
    x = torch.randn(2, 3, 32, 32)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    out_train = model(x)
    model.eval()
    out_eval = model(x)
    print(f"  Train mode output std: {out_train.std().item():.4f}")
    print(f"  Eval mode output std:  {out_eval.std().item():.4f}")
