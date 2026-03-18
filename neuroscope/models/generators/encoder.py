"""
encoder modules for generator architectures.

this module provides various encoder implementations for image-to-image translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from ..blocks.conv import ConvBlock, DownsampleBlock
from ..blocks.residual import ResidualBlock
from ..attention.self_attention import SelfAttention2d


class ConvEncoder(nn.Module):
    """
    standard convolutional encoder.
    
    progressive downsampling with optional attention.
    
    args:
        in_channels: input channels
        base_channels: base channel count
        n_downsample: number of downsampling layers
        norm_type: normalization type
        activation: activation type
        use_attention: whether to use self-attention
        attention_layers: which layers to add attention to
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_downsample: int = 2,
        norm_type: str = 'instance',
        activation: str = 'relu',
        use_attention: bool = False,
        attention_layers: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.n_downsample = n_downsample
        self.use_attention = use_attention
        self.attention_layers = attention_layers or []
        
        # initial convolution
        self.initial = ConvBlock(
            in_channels, base_channels,
            kernel_size=7, stride=1, padding=3,
            norm_type=norm_type, activation=activation
        )
        
        # downsampling layers
        self.downsample_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleDict()
        
        current_channels = base_channels
        for i in range(n_downsample):
            out_channels = min(current_channels * 2, 512)
            
            self.downsample_layers.append(
                DownsampleBlock(
                    current_channels, out_channels,
                    norm_type=norm_type, activation=activation
                )
            )
            
            if use_attention and i in attention_layers:
                self.attention_modules[f'attn_{i}'] = SelfAttention2d(out_channels)
                
            current_channels = out_channels
            
        self.output_channels = current_channels
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        forward pass.
        
        args:
            x: input tensor [b, c, h, w]
            return_intermediates: whether to return intermediate features
            
        returns:
            final encoded features and optionally intermediate features
        """
        intermediates = []
        
        # initial conv
        x = self.initial(x)
        if return_intermediates:
            intermediates.append(x)
            
        # downsampling
        for i, down in enumerate(self.downsample_layers):
            x = down(x)
            
            if self.use_attention and f'attn_{i}' in self.attention_modules:
                x = self.attention_modules[f'attn_{i}'](x)
                
            if return_intermediates:
                intermediates.append(x)
                
        if return_intermediates:
            return x, intermediates
        else:
            return x, None


class ResidualEncoder(nn.Module):
    """
    residual encoder with skip connections.
    
    uses residual blocks between downsampling stages.
    
    args:
        in_channels: input channels
        base_channels: base channel count
        n_downsample: number of downsampling layers
        n_residual: residual blocks per scale
        norm_type: normalization type
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_downsample: int = 3,
        n_residual: int = 2,
        norm_type: str = 'instance'
    ):
        super().__init__()
        
        self.n_downsample = n_downsample
        
        # initial conv
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # encoder stages
        self.stages = nn.ModuleList()
        
        current_channels = base_channels
        for i in range(n_downsample):
            out_channels = min(current_channels * 2, 512)
            
            stage = nn.ModuleList([
                # residual blocks
                nn.Sequential(*[
                    ResidualBlock(current_channels, norm_type=norm_type)
                    for _ in range(n_residual)
                ]),
                # downsampling
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            ])
            
            self.stages.append(stage)
            current_channels = out_channels
            
        self.output_channels = current_channels
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """forward pass with optional intermediate outputs."""
        intermediates = []
        
        x = self.initial(x)
        if return_intermediates:
            intermediates.append(x)
            
        for residual, downsample in self.stages:
            x = residual(x)
            if return_intermediates:
                intermediates.append(x)
            x = downsample(x)
            if return_intermediates:
                intermediates.append(x)
                
        return x, intermediates if return_intermediates else None


class DenseEncoder(nn.Module):
    """
    dense encoder with dense connections.
    
    features are concatenated across layers for rich representations.
    
    args:
        in_channels: input channels
        base_channels: base channel count
        growth_rate: channels added per dense layer
        n_dense_blocks: number of dense blocks
        n_layers_per_block: layers per dense block
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        growth_rate: int = 32,
        n_dense_blocks: int = 3,
        n_layers_per_block: int = 4
    ):
        super().__init__()
        
        # initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # dense blocks with transitions
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        current_channels = base_channels
        for i in range(n_dense_blocks):
            # dense block
            block = DenseBlock(current_channels, growth_rate, n_layers_per_block)
            self.dense_blocks.append(block)
            
            current_channels = current_channels + growth_rate * n_layers_per_block
            
            # transition (except last)
            if i < n_dense_blocks - 1:
                out_channels = current_channels // 2
                transition = nn.Sequential(
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(current_channels, out_channels, 1),
                    nn.AvgPool2d(2)
                )
                self.transitions.append(transition)
                current_channels = out_channels
                
        self.output_channels = current_channels
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """forward pass."""
        intermediates = []
        
        x = self.initial(x)
        if return_intermediates:
            intermediates.append(x)
            
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if return_intermediates:
                intermediates.append(x)
                
            if i < len(self.transitions):
                x = self.transitions[i](x)
                if return_intermediates:
                    intermediates.append(x)
                    
        return x, intermediates if return_intermediates else None


class DenseBlock(nn.Module):
    """dense block with concatenated features."""
    
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        n_layers: int
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(current_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(current_channels, growth_rate, 3, padding=1)
                )
            )
            current_channels += growth_rate
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
            
        return torch.cat(features, dim=1)


class MultiModalEncoder(nn.Module):
    """
    multi-modal encoder for multi-channel mri.
    
    processes each modality separately then fuses features.
    
    args:
        n_modalities: number of input modalities
        base_channels: base channel count per modality
        fusion_type: how to fuse modalities ('concat', 'attention', 'sum')
        n_downsample: number of downsampling layers
    """
    
    def __init__(
        self,
        n_modalities: int = 4,
        base_channels: int = 32,
        fusion_type: str = 'attention',
        n_downsample: int = 2
    ):
        super().__init__()
        
        self.n_modalities = n_modalities
        self.fusion_type = fusion_type
        
        # separate encoder for each modality
        self.modality_encoders = nn.ModuleList([
            ConvEncoder(
                in_channels=1,
                base_channels=base_channels,
                n_downsample=n_downsample
            ) for _ in range(n_modalities)
        ])
        
        encoder_out_channels = self.modality_encoders[0].output_channels
        
        # fusion module
        if fusion_type == 'concat':
            self.fusion = nn.Conv2d(
                encoder_out_channels * n_modalities,
                encoder_out_channels,
                1
            )
            self.output_channels = encoder_out_channels
            
        elif fusion_type == 'attention':
            self.fusion = ModalityAttentionFusion(
                n_modalities, encoder_out_channels
            )
            self.output_channels = encoder_out_channels
            
        elif fusion_type == 'sum':
            self.fusion = None
            self.output_channels = encoder_out_channels
            
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        forward pass.
        
        args:
            x: input tensor [b, n_modalities, h, w]
            
        returns:
            fused features and optionally per-modality features
        """
        modality_features = []
        all_intermediates = {}
        
        # encode each modality
        for i, encoder in enumerate(self.modality_encoders):
            modality_input = x[:, i:i+1, :, :]
            features, intermediates = encoder(modality_input, return_intermediates)
            modality_features.append(features)
            
            if return_intermediates:
                all_intermediates[f'modality_{i}'] = intermediates
                
        # fuse modalities
        if self.fusion_type == 'concat':
            fused = self.fusion(torch.cat(modality_features, dim=1))
        elif self.fusion_type == 'attention':
            fused = self.fusion(torch.stack(modality_features, dim=1))
        elif self.fusion_type == 'sum':
            fused = sum(modality_features)
            
        if return_intermediates:
            return fused, all_intermediates
        else:
            return fused, None


class ModalityAttentionFusion(nn.Module):
    """attention-based modality fusion."""
    
    def __init__(self, n_modalities: int, channels: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * n_modalities, n_modalities),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [b, n_modalities, c, h, w]
        """
        B, N, C, H, W = x.shape
        
        # compute attention weights
        x_flat = x.view(B, N * C, H, W)
        weights = self.attention(x_flat)  # [b, n]
        
        # weighted sum
        weights = weights.view(B, N, 1, 1, 1)
        fused = (x * weights).sum(dim=1)
        
        return fused


class HierarchicalEncoder(nn.Module):
    """
    hierarchical encoder with multi-scale features.
    
    extracts features at multiple resolutions simultaneously.
    
    args:
        in_channels: input channels
        base_channels: base channel count
        n_scales: number of scales
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_scales: int = 4
    ):
        super().__init__()
        
        self.n_scales = n_scales
        
        # scale-specific encoders
        self.scale_encoders = nn.ModuleList()
        
        for i in range(n_scales):
            scale_channels = base_channels * (2 ** i)
            self.scale_encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, scale_channels, 3, padding=1),
                    nn.InstanceNorm2d(scale_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(scale_channels, scale_channels, 3, padding=1),
                    nn.InstanceNorm2d(scale_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # feature aggregation
        total_channels = sum(base_channels * (2 ** i) for i in range(n_scales))
        self.aggregate = nn.Conv2d(total_channels, base_channels * 4, 1)
        self.output_channels = base_channels * 4
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """forward pass with multi-scale processing."""
        scale_features = []
        
        for i, encoder in enumerate(self.scale_encoders):
            # downsample input to appropriate scale
            if i > 0:
                scale_input = F.avg_pool2d(x, 2 ** i)
            else:
                scale_input = x
                
            # encode at this scale
            features = encoder(scale_input)
            
            # upsample back to original resolution
            if i > 0:
                features = F.interpolate(
                    features, size=x.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                
            scale_features.append(features)
            
        # aggregate multi-scale features
        aggregated = self.aggregate(torch.cat(scale_features, dim=1))
        
        if return_intermediates:
            return aggregated, scale_features
        else:
            return aggregated, None
