"""
spectral normalization discriminators.

discriminators with spectral normalization for improved
training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from torch.nn.utils import spectral_norm

from .base import BaseDiscriminator


class SpectralNormDiscriminator(BaseDiscriminator):
    """
    spectral normalization discriminator.
    
    patchgan with spectral normalization on all layers.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_layers: number of layers
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 4
    ):
        super().__init__(in_channels, ndf)
        
        layers = []
        
        # first layer
        layers.append(
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1))
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # hidden layers
        in_ch = ndf
        for i in range(1, n_layers):
            out_ch = min(ndf * (2 ** i), 512)
            
            layers.append(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_ch = out_ch
            
        # output layer
        layers.append(
            spectral_norm(nn.Conv2d(in_ch, 1, 4, padding=1))
        )
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SNResNetDiscriminator(BaseDiscriminator):
    """
    spectral normalization resnet discriminator.
    
    resnet-style discriminator with spectral normalization.
    used in biggan and similar architectures.
    
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
        
        # initial convolution
        self.initial = spectral_norm(
            nn.Conv2d(in_channels, ndf, 3, padding=1)
        )
        
        # residual blocks
        self.blocks = nn.ModuleList()
        in_ch = ndf
        
        for i in range(n_blocks):
            out_ch = min(in_ch * 2, 512)
            self.blocks.append(
                SNResBlock(in_ch, out_ch, downsample=True)
            )
            in_ch = out_ch
            
        # output
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(in_ch, 1))
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward pass with optional feature return."""
        x = self.initial(x)
        
        for block in self.blocks:
            x = block(x)
            
        features = x
        output = self.output(x)
        
        if return_features:
            return output, features
        return output


class SNResBlock(nn.Module):
    """spectral normalization residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True
    ):
        super().__init__()
        
        self.downsample = downsample
        
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.shortcut = spectral_norm(
            nn.Conv2d(in_channels, out_channels, 1)
        )
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
            
        return h + self.shortcut(x)


class SNProjectionDiscriminator(BaseDiscriminator):
    """
    spectral normalization projection discriminator.
    
    for conditional gans with class labels.
    uses projection for conditioning (from cgan-pd).
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_classes: number of classes (0 for unconditional)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_classes: int = 0
    ):
        super().__init__(in_channels, ndf)
        
        self.n_classes = n_classes
        
        # feature extractor
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            SNResBlock(ndf, ndf * 2),
            SNResBlock(ndf * 2, ndf * 4),
            SNResBlock(ndf * 4, ndf * 8),
            SNResBlock(ndf * 8, ndf * 8),
            
            nn.ReLU(inplace=True)
        )
        
        # output
        self.output_linear = spectral_norm(nn.Linear(ndf * 8, 1))
        
        # class embedding for projection
        if n_classes > 0:
            self.embed = spectral_norm(nn.Embedding(n_classes, ndf * 8))
            
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: input tensor
            labels: optional class labels for conditional discrimination
        """
        # extract features
        h = self.features(x)
        
        # global sum pooling
        h = h.sum(dim=[2, 3])
        
        # output
        output = self.output_linear(h)
        
        # add projection if labels provided
        if labels is not None and self.n_classes > 0:
            embed = self.embed(labels)
            output = output + (embed * h).sum(dim=1, keepdim=True)
            
        return output


class SNMultiScaleDiscriminator(BaseDiscriminator):
    """
    multi-scale spectral normalization discriminator.
    
    multiple sn discriminators at different scales.
    
    args:
        in_channels: input channels
        ndf: base number of filters
        n_scales: number of scales
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_scales: int = 3
    ):
        super().__init__(in_channels, ndf)
        
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(n_scales):
            self.discriminators.append(
                SpectralNormDiscriminator(
                    in_channels=in_channels,
                    ndf=ndf,
                    n_layers=4 - i  # fewer layers for smaller scales
                )
            )
            
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """forward pass at multiple scales."""
        outputs = []
        current = x
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(current))
            if i < self.n_scales - 1:
                current = F.avg_pool2d(current, 2)
                
        return outputs


class SNSelfAttentionDiscriminator(BaseDiscriminator):
    """
    spectral normalization self-attention discriminator.
    
    combines sn with self-attention (sagan-style).
    
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
        n_layers: int = 5,
        attention_layer: int = 2
    ):
        super().__init__(in_channels, ndf)
        
        self.layers = nn.ModuleList()
        
        in_ch = in_channels
        for i in range(n_layers):
            out_ch = min(ndf * (2 ** i), 512)
            
            self.layers.append(
                SNConvBlock(in_ch, out_ch, stride=2)
            )
            
            # add self-attention at specified layer
            if i == attention_layer:
                self.layers.append(SNSelfAttention(out_ch))
                
            in_ch = out_ch
            
        # output
        self.output = spectral_norm(nn.Conv2d(in_ch, 1, 4, padding=1))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward pass with optional attention map return."""
        attention_map = None
        
        for layer in self.layers:
            if isinstance(layer, SNSelfAttention):
                x, attention_map = layer(x, return_attention=True)
            else:
                x = layer(x)
                
        output = self.output(x)
        
        if return_attention:
            return output, attention_map
        return output


class SNConvBlock(nn.Module):
    """spectral normalization convolutional block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SNSelfAttention(nn.Module):
    """self-attention with spectral normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.query = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(channels, channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(channels, channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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


class SNUNetDiscriminator(BaseDiscriminator):
    """
    u-net discriminator with spectral normalization.
    
    provides both global and per-pixel discrimination.
    
    args:
        in_channels: input channels
        ndf: base number of filters
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64
    ):
        super().__init__(in_channels, ndf)
        
        # encoder
        self.enc1 = SNConvBlock(in_channels, ndf, stride=2)
        self.enc2 = SNConvBlock(ndf, ndf * 2, stride=2)
        self.enc3 = SNConvBlock(ndf * 2, ndf * 4, stride=2)
        self.enc4 = SNConvBlock(ndf * 4, ndf * 8, stride=2)
        
        # decoder
        self.dec4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ndf * 8, ndf * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ndf * 4, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(ndf * 2, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # output
        self.output = spectral_norm(nn.Conv2d(ndf, 1, 3, padding=1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        # encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # decode with skip connections
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return self.output(d1)
