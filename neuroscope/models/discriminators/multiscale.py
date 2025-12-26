"""
Multi-Scale Discriminator Architectures.

Discriminators that operate at multiple scales for
multi-resolution adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from .base import BaseDiscriminator, MultiScaleDiscriminatorBase
from .patch import NLayerPatchDiscriminator


class MultiScaleDiscriminator(MultiScaleDiscriminatorBase):
    """
    Multi-Scale Discriminator.
    
    Uses multiple PatchGAN discriminators at different scales.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        n_scales: Number of scales
        n_layers: Layers per discriminator
        norm_type: Normalization type
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_scales: int = 3,
        n_layers: int = 4,
        norm_type: str = 'instance'
    ):
        super().__init__(in_channels, ndf, n_scales)
        
        for i in range(n_scales):
            # Smaller networks for finer scales
            scale_ndf = ndf if i == 0 else ndf // (2 ** i)
            scale_ndf = max(scale_ndf, 32)
            
            self.discriminators.append(
                NLayerPatchDiscriminator(
                    in_channels=in_channels,
                    ndf=scale_ndf,
                    n_layers=n_layers,
                    norm_type=norm_type
                )
            )
            
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[List[torch.Tensor]]]]:
        """
        Forward pass at multiple scales.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            
        Returns:
            List of outputs (and optionally features) from each scale
        """
        outputs = []
        all_features = []
        current_input = x
        
        for i, disc in enumerate(self.discriminators):
            if return_features and hasattr(disc, 'get_all_activations'):
                features = disc.get_all_activations(current_input)
                outputs.append(features[-1])
                all_features.append(features)
            else:
                outputs.append(disc(current_input))
                
            if i < len(self.discriminators) - 1:
                current_input = F.avg_pool2d(current_input, 2)
                
        if return_features:
            return outputs, all_features
        return outputs


class PyramidDiscriminator(BaseDiscriminator):
    """
    Feature Pyramid Discriminator.
    
    Builds a feature pyramid and discriminates at each level.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        n_levels: Number of pyramid levels
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_levels: int = 4
    ):
        super().__init__(in_channels, ndf)
        
        self.n_levels = n_levels
        
        # Feature extractors for each level
        self.feature_extractors = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.discriminators = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(n_levels):
            out_ch = min(ndf * (2 ** i), 512)
            
            # Feature extractor (downsampling)
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            
            # Lateral connection
            self.lateral_convs.append(
                nn.Conv2d(out_ch, ndf * 2, 1)
            )
            
            # Per-level discriminator
            self.discriminators.append(
                nn.Sequential(
                    nn.Conv2d(ndf * 2, ndf * 2, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(ndf * 2, 1, 3, padding=1)
                )
            )
            
            in_ch = out_ch
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning outputs at each pyramid level."""
        # Extract features at each level
        features = []
        current = x
        
        for extractor in self.feature_extractors:
            current = extractor(current)
            features.append(current)
            
        # Build pyramid and discriminate
        outputs = []
        prev_features = None
        
        for i in range(self.n_levels - 1, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            if prev_features is not None:
                # Upsample and add
                upsampled = F.interpolate(
                    prev_features, size=lateral.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                lateral = lateral + upsampled
                
            outputs.append(self.discriminators[i](lateral))
            prev_features = lateral
            
        return outputs[::-1]  # Return in coarse-to-fine order


class SharedEncoderMultiScaleDiscriminator(BaseDiscriminator):
    """
    Multi-Scale Discriminator with Shared Encoder.
    
    Uses a shared encoder and multiple output heads.
    More efficient than separate discriminators.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        n_scales: Number of output scales
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_scales: int = 3
    ):
        super().__init__(in_channels, ndf)
        
        self.n_scales = n_scales
        
        # Shared encoder
        self.encoder = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(5):  # 5 layers in shared encoder
            out_ch = min(ndf * (2 ** i), 512)
            
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch
            
        # Output heads at different scales
        self.output_heads = nn.ModuleList()
        
        # Get channels at each scale
        channels_per_scale = [
            min(ndf * (2 ** i), 512) for i in range(5)
        ]
        
        for i in range(n_scales):
            layer_idx = min(i + 2, 4)  # Start from layer 2
            self.output_heads.append(
                nn.Conv2d(channels_per_scale[layer_idx], 1, 4, padding=1)
            )
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass with shared encoder."""
        outputs = []
        intermediate_features = []
        
        # Encode
        for layer in self.encoder:
            x = layer(x)
            intermediate_features.append(x)
            
        # Get outputs at multiple scales
        for i, head in enumerate(self.output_heads):
            layer_idx = min(i + 2, 4)
            outputs.append(head(intermediate_features[layer_idx]))
            
        return outputs


class AdaptiveMultiScaleDiscriminator(BaseDiscriminator):
    """
    Adaptive Multi-Scale Discriminator.
    
    Dynamically adjusts which scales are most important
    based on training progress.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        n_scales: Number of scales
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_scales: int = 3
    ):
        super().__init__(in_channels, ndf)
        
        self.n_scales = n_scales
        
        # Scale-specific discriminators
        self.discriminators = nn.ModuleList()
        
        for i in range(n_scales):
            self.discriminators.append(
                NLayerPatchDiscriminator(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=3 + i,  # More layers for coarser scales
                    norm_type='instance'
                )
            )
            
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        
    def forward(
        self,
        x: torch.Tensor,
        return_weighted: bool = False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_weighted: If True, return weighted sum of outputs
        """
        outputs = []
        current_input = x
        
        for disc in self.discriminators:
            outputs.append(disc(current_input))
            current_input = F.avg_pool2d(current_input, 2)
            
        if return_weighted:
            # Weighted combination
            weights = F.softmax(self.scale_weights, dim=0)
            weighted_output = 0
            
            for i, (out, w) in enumerate(zip(outputs, weights)):
                # Upsample to original scale
                if i > 0:
                    out = F.interpolate(
                        out, size=outputs[0].shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                weighted_output = weighted_output + w * out
                
            return weighted_output
            
        return outputs


class ProgressiveMultiScaleDiscriminator(BaseDiscriminator):
    """
    Progressive Multi-Scale Discriminator.
    
    For progressive training - starts with coarse scales
    and gradually adds finer scales.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        max_scales: Maximum number of scales
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        max_scales: int = 4
    ):
        super().__init__(in_channels, ndf)
        
        self.max_scales = max_scales
        self.active_scales = 1
        
        # Create all discriminators
        self.discriminators = nn.ModuleList()
        
        for i in range(max_scales):
            n_layers = 3 + i
            self.discriminators.append(
                NLayerPatchDiscriminator(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    norm_type='instance'
                )
            )
            
        # Alpha for blending new scale
        self.register_buffer('alpha', torch.ones(1))
        
    def set_active_scales(self, n_scales: int, alpha: float = 1.0):
        """Set number of active scales and blending alpha."""
        self.active_scales = min(n_scales, self.max_scales)
        self.alpha.fill_(alpha)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass with active scales only."""
        outputs = []
        current_input = x
        
        for i in range(self.active_scales):
            output = self.discriminators[i](current_input)
            
            # Blend newest scale
            if i == self.active_scales - 1 and self.active_scales > 1:
                output = self.alpha * output
                
            outputs.append(output)
            current_input = F.avg_pool2d(current_input, 2)
            
        return outputs


class DualScaleDiscriminator(BaseDiscriminator):
    """
    Dual-Scale Discriminator.
    
    Simple two-scale discriminator: global and local.
    
    Args:
        in_channels: Input channels
        ndf: Base number of filters
        global_layers: Layers for global discriminator
        local_layers: Layers for local discriminator
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        global_layers: int = 5,
        local_layers: int = 3
    ):
        super().__init__(in_channels, ndf)
        
        # Global discriminator (full image)
        self.global_disc = NLayerPatchDiscriminator(
            in_channels=in_channels,
            ndf=ndf,
            n_layers=global_layers,
            norm_type='instance'
        )
        
        # Local discriminator (patches)
        self.local_disc = NLayerPatchDiscriminator(
            in_channels=in_channels,
            ndf=ndf // 2,
            n_layers=local_layers,
            norm_type='instance'
        )
        
    def forward(
        self,
        x: torch.Tensor,
        patch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            patch_size: Size of local patches
            
        Returns:
            Global and local discriminator outputs
        """
        # Global
        global_out = self.global_disc(x)
        
        # Local (random patches)
        B, C, H, W = x.size()
        
        # Sample random patches
        if H > patch_size and W > patch_size:
            h_start = torch.randint(0, H - patch_size, (B,))
            w_start = torch.randint(0, W - patch_size, (B,))
            
            patches = []
            for i in range(B):
                patch = x[i:i+1, :, h_start[i]:h_start[i]+patch_size, w_start[i]:w_start[i]+patch_size]
                patches.append(patch)
            patches = torch.cat(patches, dim=0)
        else:
            patches = x
            
        local_out = self.local_disc(patches)
        
        return global_out, local_out
