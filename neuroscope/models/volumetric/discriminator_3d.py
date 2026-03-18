"""
3d discriminator architectures.

volumetric discriminators for adversarial training including:
- 3d patchgan discriminator
- multi-scale 3d discriminator
- spectral normalized variants
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_3d import SpectralNorm3d, GroupNorm3D


class Discriminator3D(nn.Module):
    """
    3d patchgan discriminator.
    
    classifies overlapping patches as real or fake for
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
        
        # build discriminator layers
        layers = []
        
        # first layer (no normalization)
        conv = nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=True)
        if use_spectral_norm:
            conv = SpectralNorm3d(conv)
        layers.extend([conv, nn.LeakyReLU(0.2, inplace=True)])
        
        # intermediate layers
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
        
        # penultimate layer
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
        
        # output layer
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
    multi-scale 3d discriminator.
    
    operates at multiple spatial scales to capture both
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
        
        # create discriminators for each scale
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
        
        # downsampling for multi-scale
        self.downsample = nn.AvgPool3d(
            kernel_size=3, stride=2, padding=1,
            count_include_pad=False
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        forward pass returning predictions at all scales.
        
        args:
            x: input volume [b, c, d, h, w]
            
        returns:
            list of predictions at each scale
        """
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            
            if i < self.n_scales - 1:
                x = self.downsample(x)
        
        return outputs


class NLayerDiscriminator3D(nn.Module):
    """
    n-layer 3d discriminator with configurable depth.
    
    more flexible architecture allowing for different
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
        
        # first layer
        sequence = [[
            nn.Conv3d(in_channels, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # intermediate layers
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
        
        # penultimate layer
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
        
        # output layer
        sequence.append([nn.Conv3d(ndf * nf_mult, 1, 4, 1, 1)])
        
        # build sequential blocks
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
        forward pass optionally returning intermediate features.
        
        args:
            x: input volume
            
        returns:
            final prediction and optional intermediate features
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
    projection discriminator for 3d conditional generation.
    
    uses projection of class embeddings for conditional
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
        
        # build feature extractor
        layers = []
        
        # first layer
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
        
        # final feature layer
        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        layers.extend([
            nn.Conv3d(ndf * mult_prev, ndf * mult, 4, 1, 1, bias=False),
            nn.InstanceNorm3d(ndf * mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.features = nn.Sequential(*layers)
        self.feature_dim = ndf * mult
        
        # global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # output layer
        self.output = nn.Linear(self.feature_dim, 1)
        
        # class embedding for projection
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, self.feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass with optional class conditioning.
        
        args:
            x: input volume [b, c, d, h, w]
            y: optional class labels [b]
            
        returns:
            discrimination scores
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
    3d discriminator with feature matching support.
    
    returns intermediate features for feature matching loss
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
        
        # build layers individually for feature extraction
        self.layers = nn.ModuleList()
        
        # first layer
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
        
        # output layer
        self.output = nn.Conv3d(ndf * mult, 1, 4, 1, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        forward pass returning discrimination scores and features.
        
        args:
            x: input volume
            return_features: whether to return intermediate features
            
        returns:
            tuple of (output, list of intermediate features)
        """
        features = []
        
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
        
        output = self.output(x)
        
        return output, features
