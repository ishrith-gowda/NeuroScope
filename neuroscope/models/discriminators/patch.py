"""
patch discriminator architectures.

patchgan-style discriminators that classify nxn patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .base import BaseDiscriminator, PatchDiscriminator


class NLayerPatchDiscriminator(PatchDiscriminator):
    """
    n-layer patchgan discriminator.
    
    standard patchgan with configurable depth.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_layers: number of conv layers
        norm_type: normalization type ('instance', 'batch', 'none')
        use_sigmoid: whether to apply sigmoid at output
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = 'instance',
        use_sigmoid: bool = False
    ):
        self._in_channels = in_channels
        self._ndf = ndf
        self._n_layers = n_layers
        self._norm_type = norm_type
        self._use_sigmoid = use_sigmoid
        
        super().__init__(in_channels, ndf, n_layers, norm_type)
        
    def _build_network(self) -> nn.Sequential:
        """build the discriminator network."""
        # normalization layer
        if self._norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif self._norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None
            
        layers = []
        
        # first layer (no normalization)
        layers.append(
            nn.Conv2d(self._in_channels, self._ndf, 4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # intermediate layers
        nf_mult = 1
        for n in range(1, self._n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            layers.append(
                nn.Conv2d(
                    self._ndf * nf_mult_prev,
                    self._ndf * nf_mult,
                    4, stride=2, padding=1
                )
            )
            if norm_layer is not None:
                layers.append(norm_layer(self._ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        # second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self._n_layers, 8)
        
        layers.append(
            nn.Conv2d(
                self._ndf * nf_mult_prev,
                self._ndf * nf_mult,
                4, stride=1, padding=1
            )
        )
        if norm_layer is not None:
            layers.append(norm_layer(self._ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # output layer
        layers.append(
            nn.Conv2d(self._ndf * nf_mult, 1, 4, stride=1, padding=1)
        )
        
        if self._use_sigmoid:
            layers.append(nn.Sigmoid())
            
        return nn.Sequential(*layers)


class PixelDiscriminator(BaseDiscriminator):
    """
    1x1 patchgan (pixel) discriminator.
    
    classifies each pixel independently.
    
    args:
        in_channels: input channels
        ndf: number of filters
        norm_type: normalization type
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        norm_type: str = 'instance'
    ):
        super().__init__(in_channels, ndf)
        
        if norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.Identity
            
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 1, stride=1, padding=0),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, stride=1, padding=0)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepPatchDiscriminator(BaseDiscriminator):
    """
    deep patchgan discriminator.
    
    deeper network with more capacity.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_layers: number of layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 5
    ):
        super().__init__(in_channels, ndf)
        
        self.layers = nn.ModuleList()
        
        # first layer
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        
        # hidden layers
        in_ch = ndf
        for i in range(1, n_layers):
            out_ch = min(ndf * (2 ** i), 512)
            stride = 2 if i < n_layers - 1 else 1
            
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch
            
        # output layer
        self.output = nn.Conv2d(in_ch, 1, 4, stride=1, padding=1)
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """forward pass with optional feature return."""
        features = []
        
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
                
        output = self.output(x)
        
        if return_features:
            return output, features
        return output


class ResidualPatchDiscriminator(BaseDiscriminator):
    """
    residual patchgan discriminator.
    
    uses residual connections for better gradient flow.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_blocks: number of residual blocks
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_blocks: int = 4
    ):
        super().__init__(in_channels, ndf)
        
        # initial conv
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # residual blocks with downsampling
        self.blocks = nn.ModuleList()
        in_ch = ndf
        
        for i in range(n_blocks):
            out_ch = min(in_ch * 2, 512)
            self.blocks.append(
                ResidualDownBlock(in_ch, out_ch)
            )
            in_ch = out_ch
            
        # output
        self.output = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, 1, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output(x)


class ResidualDownBlock(nn.Module):
    """residual block with downsampling."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x) + self.shortcut(x))


class DilatedPatchDiscriminator(BaseDiscriminator):
    """
    dilated patchgan discriminator.
    
    uses dilated convolutions for larger receptive field
    without downsampling.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_layers: number of dilated layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 4
    ):
        super().__init__(in_channels, ndf)
        
        layers = []
        
        # first layer (no dilation)
        layers.append(
            nn.Conv2d(in_channels, ndf, 3, stride=1, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # dilated layers
        in_ch = ndf
        for i in range(n_layers):
            out_ch = min(ndf * (2 ** (i + 1)), 512)
            dilation = 2 ** i
            padding = dilation
            
            layers.append(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=padding, dilation=dilation)
            )
            layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_ch = out_ch
            
        # output layer
        layers.append(nn.Conv2d(in_ch, 1, 3, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttentionPatchDiscriminator(BaseDiscriminator):
    """
    patchgan with self-attention.
    
    adds self-attention layers for capturing long-range dependencies.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_layers: number of layers
        attention_layer: which layer to add attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 4,
        attention_layer: int = 2
    ):
        super().__init__(in_channels, ndf)
        
        self.layers = nn.ModuleList()
        self.attention_layer = attention_layer
        
        # build layers
        in_ch = in_channels
        for i in range(n_layers):
            out_ch = min(ndf * (2 ** i), 512)
            
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            
            # add attention at specified layer
            if i == attention_layer:
                self.layers.append(SelfAttention(out_ch))
                
            in_ch = out_ch
            
        # output
        self.output = nn.Conv2d(in_ch, 1, 4, padding=1)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """forward pass with optional attention map return."""
        attention_map = None
        
        for layer in self.layers:
            if isinstance(layer, SelfAttention):
                x, attention_map = layer(x, return_attention=True)
            else:
                x = layer(x)
                
        output = self.output(x)
        
        if return_attention:
            return output, attention_map
        return output


class SelfAttention(nn.Module):
    """self-attention module for discriminator."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        B, C, H, W = x.size()
        
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        out = self.gamma * out + x
        
        if return_attention:
            return out, attention
        return out
