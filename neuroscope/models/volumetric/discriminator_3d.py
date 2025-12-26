"""
3D Discriminator Architectures.

Volumetric discriminators for adversarial training including:
- 3D PatchGAN Discriminator
- Multi-scale 3D Discriminator
- Spectral normalized variants
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_3d import SpectralNorm3d, GroupNorm3D


class Discriminator3D(nn.Module):
    """
    3D PatchGAN Discriminator.
    
    Classifies overlapping patches as real or fake for
    more stable adversarial training on volumetric data.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 32,
        n_layers: int = 3,
        norm_type: str = 'instance',
        use_spectral_norm: bool = True,
        use_sigmoid: bool = False
    ):
        super().__init__()
        
        self.use_sigmoid = use_sigmoid
        
        # Build discriminator layers
        layers = []
        
        # First layer (no normalization)
        conv = nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=True)
        if use_spectral_norm:
            conv = SpectralNorm3d(conv)
        layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
        
        # Intermediate layers
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            
            conv = nn.Conv3d(
                ndf * mult_prev, ndf * mult,
                4, 2, 1, bias=False
            )
            if use_spectral_norm:
                conv = SpectralNorm3d(conv)
            
            if norm_type == 'instance':
                norm = nn.InstanceNorm3d(ndf * mult, affine=True)
            elif norm_type == 'batch':
                norm = nn.BatchNorm3d(ndf * mult)
            elif norm_type == 'group':
                norm = GroupNorm3D(ndf * mult)
            else:
                norm = nn.Identity()
            
            layers.extend([conv, norm, nn.LeakyReLU(0.2, inplace=True)])
        
        # Penultimate layer
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        
        conv = nn.Conv3d(
            ndf * mult_prev, ndf * mult,
            4, 1, 1, bias=False
        )
        if use_spectral_norm:
            conv = SpectralNorm3d(conv)
        
        if norm_type == 'instance':
            norm = nn.InstanceNorm3d(ndf * mult, affine=True)
        elif norm_type == 'batch':
            norm = nn.BatchNorm3d(ndf * mult)
        else:
            norm = nn.Identity()
        
        layers.extend([conv, norm, nn.LeakyReLU(0.2, inplace=True)])
        
        # Output layer
        output_conv = nn.Conv3d(ndf * mult, 1, 4, 1, 1)
        if use_spectral_norm:
            output_conv = SpectralNorm3d(output_conv)
        layers.append(output_conv)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


class MultiScaleDiscriminator3D(nn.Module):
    """
    Multi-Scale 3D Discriminator.
    
    Operates at multiple spatial scales to capture both
    fine-grained textures and global structures.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 32,
        n_layers: int = 3,
        n_scales: int = 3,
        norm_type: str = 'instance',
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.n_scales = n_scales
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        for _ in range(n_scales):
            self.discriminators.append(
                Discriminator3D(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=n_layers,
                    norm_type=norm_type,
                    use_spectral_norm=use_spectral_norm
                )
            )
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool3d(
            kernel_size=3, stride=2, padding=1,
            count_include_pad=False
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning predictions at all scales.
        
        Args:
            x: Input volume [B, C, D, H, W]
            
        Returns:
            List of predictions at each scale
        """
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            
            if i < self.n_scales - 1:
                x = self.downsample(x)
        
        return outputs


class NLayerDiscriminator3D(nn.Module):
    """
    N-Layer 3D Discriminator with configurable depth.
    
    More flexible architecture allowing for different
    receptive field sizes.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = 'instance',
        use_spectral_norm: bool = False,
        get_intermediate_features: bool = False
    ):
        super().__init__()
        
        self.get_intermediate_features = get_intermediate_features
        
        # First layer
        sequence = [[
            nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            conv = nn.Conv3d(
                ndf * nf_mult_prev, ndf * nf_mult,
                4, 2, 1, bias=False
            )
            
            if norm_type == 'instance':
                norm = nn.InstanceNorm3d(ndf * nf_mult, affine=True)
            elif norm_type == 'batch':
                norm = nn.BatchNorm3d(ndf * nf_mult)
            else:
                norm = nn.Identity()
            
            sequence.append([
                conv, norm, nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        conv = nn.Conv3d(
            ndf * nf_mult_prev, ndf * nf_mult,
            4, 1, 1, bias=False
        )
        
        if norm_type == 'instance':
            norm = nn.InstanceNorm3d(ndf * nf_mult, affine=True)
        else:
            norm = nn.Identity()
        
        sequence.append([
            conv, norm, nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Output layer
        sequence.append([nn.Conv3d(ndf * nf_mult, 1, 4, 1, 1)])
        
        # Build sequential blocks
        if get_intermediate_features:
            self.blocks = nn.ModuleList()
            for block in sequence:
                self.blocks.append(nn.Sequential(*block))
        else:
            flat_sequence = []
            for block in sequence:
                flat_sequence.extend(block)
            self.model = nn.Sequential(*flat_sequence)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass optionally returning intermediate features.
        
        Args:
            x: Input volume
            
        Returns:
            Final prediction and optional intermediate features
        """
        if self.get_intermediate_features:
            features = []
            for block in self.blocks:
                x = block(x)
                features.append(x)
            return x, features
        else:
            return self.model(x), None


class ProjectionDiscriminator3D(nn.Module):
    """
    Projection Discriminator for 3D conditional generation.
    
    Uses projection of class embeddings for conditional
    adversarial training (e.g., conditioning on tumor type).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 32,
        n_layers: int = 3,
        num_classes: int = 0,
        norm_type: str = 'instance'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build feature extractor
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv3d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            
            layers.extend([
                nn.Conv3d(ndf * mult_prev, ndf * mult, 4, 2, 1, bias=False),
                nn.InstanceNorm3d(ndf * mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final feature layer
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv3d(ndf * mult_prev, ndf * mult, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(ndf * mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.features = nn.Sequential(*layers)
        self.feature_dim = ndf * mult
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Output layer
        self.output = nn.Linear(self.feature_dim, 1)
        
        # Class embedding for projection
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, self.feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional class conditioning.
        
        Args:
            x: Input volume [B, C, D, H, W]
            y: Optional class labels [B]
            
        Returns:
            Discrimination scores
        """
        h = self.features(x)
        h = self.global_pool(h).view(h.size(0), -1)
        
        out = self.output(h)
        
        if y is not None and self.num_classes > 0:
            embed = self.embed(y)
            out = out + (h * embed).sum(dim=1, keepdim=True)
        
        return out


class FeatureMatchingDiscriminator3D(nn.Module):
    """
    3D Discriminator with Feature Matching Support.
    
    Returns intermediate features for feature matching loss
    computation during generator training.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 32,
        n_layers: int = 4,
        norm_type: str = 'instance'
    ):
        super().__init__()
        
        self.n_layers = n_layers
        
        # Build layers individually for feature extraction
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Sequential(
            nn.Conv3d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ))
        
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            
            self.layers.append(nn.Sequential(
                nn.Conv3d(ndf * mult_prev, ndf * mult, 4, 2, 1, bias=False),
                nn.InstanceNorm3d(ndf * mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        
        # Output layer
        self.output = nn.Conv3d(ndf * mult, 1, 4, 1, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning discrimination scores and features.
        
        Args:
            x: Input volume
            return_features: Whether to return intermediate features
            
        Returns:
            Tuple of (output, list of intermediate features)
        """
        features = []
        
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
        
        output = self.output(x)
        
        return output, features
