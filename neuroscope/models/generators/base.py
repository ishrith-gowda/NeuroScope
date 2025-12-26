"""
Base Generator Classes.

This module provides abstract base classes and common functionality
for generator architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


class BaseGenerator(nn.Module, ABC):
    """
    Abstract base class for all generators.
    
    Provides common interface and utilities for generator networks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        **kwargs
    ):
        """
        Initialize base generator.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            ngf: Base number of generator filters
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C', H, W]
        """
        pass
        
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization/loss computation.
        
        Override in subclasses for specific implementations.
        """
        return {'output': self.forward(x)}
        
    def init_weights(self, init_type: str = 'normal', gain: float = 0.02):
        """
        Initialize network weights.
        
        Args:
            init_type: Initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
            gain: Scaling factor for initialization
        """
        def init_func(m):
            classname = m.__class__.__name__
            
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f'Initialization {init_type} not implemented')
                    
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                    
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
                
        self.apply(init_func)


class EncoderDecoderGenerator(BaseGenerator):
    """
    Base class for encoder-decoder style generators.
    
    Provides structure for U-Net, ResNet-based, and other encoder-decoder architectures.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        n_downsampling: int = 2,
        n_blocks: int = 6,
        use_skip: bool = True,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, ngf, **kwargs)
        
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks
        self.use_skip = use_skip
        
        self.encoder = None  # Override in subclass
        self.decoder = None  # Override in subclass
        self.bottleneck = None  # Override in subclass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder."""
        # Encode
        enc_features = self.encode(x)
        
        # Bottleneck
        if self.bottleneck is not None:
            bottleneck_out = self.bottleneck(enc_features[-1])
        else:
            bottleneck_out = enc_features[-1]
            
        # Decode
        output = self.decode(bottleneck_out, enc_features if self.use_skip else None)
        
        return output
        
    def encode(self, x: torch.Tensor) -> list:
        """
        Encode input to feature representations.
        
        Returns list of features at each scale.
        """
        raise NotImplementedError
        
    def decode(
        self,
        x: torch.Tensor,
        skip_features: Optional[list] = None
    ) -> torch.Tensor:
        """
        Decode features to output.
        
        Args:
            x: Bottleneck features
            skip_features: Optional skip connection features
        """
        raise NotImplementedError


class ResidualGenerator(BaseGenerator):
    """
    Base class for residual-style generators.
    
    Generates output as input + learned residual.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        learn_residual: bool = True,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, ngf, **kwargs)
        
        self.learn_residual = learn_residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual learning."""
        residual = self.compute_residual(x)
        
        if self.learn_residual and x.size(1) == residual.size(1):
            return x + residual
        else:
            return residual
            
    @abstractmethod
    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual features."""
        pass


class MultiScaleGenerator(BaseGenerator):
    """
    Base class for multi-scale generators.
    
    Generates output at multiple scales for progressive training.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        num_scales: int = 3,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, ngf, **kwargs)
        
        self.num_scales = num_scales
        self.scale_generators = nn.ModuleList()
        
    def forward(
        self,
        x: torch.Tensor,
        return_all_scales: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            return_all_scales: If True, return outputs at all scales
            
        Returns:
            Output at finest scale, or list of outputs at all scales
        """
        outputs = []
        
        for i, generator in enumerate(self.scale_generators):
            if i == 0:
                out = generator(x)
            else:
                # Upsample previous output and combine
                prev_upsampled = nn.functional.interpolate(
                    outputs[-1], scale_factor=2, mode='bilinear', align_corners=False
                )
                out = generator(torch.cat([x, prev_upsampled], dim=1))
                
            outputs.append(out)
            
            # Downsample input for next scale
            x = nn.functional.avg_pool2d(x, 2)
            
        if return_all_scales:
            return outputs
        else:
            return outputs[-1]


class ConditionalGenerator(BaseGenerator):
    """
    Base class for conditional generators.
    
    Conditions generation on additional inputs (labels, embeddings, etc.).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        condition_dim: int = 128,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, ngf, **kwargs)
        
        self.condition_dim = condition_dim
        self.condition_encoder = None  # Override in subclass
        
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with conditioning.
        
        Args:
            x: Input tensor
            condition: Conditioning input (labels, embeddings, etc.)
            
        Returns:
            Conditional output
        """
        if condition is not None and self.condition_encoder is not None:
            condition_embedding = self.condition_encoder(condition)
        else:
            condition_embedding = None
            
        return self.generate(x, condition_embedding)
        
    @abstractmethod
    def generate(
        self,
        x: torch.Tensor,
        condition_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate output with optional condition embedding."""
        pass


class StyleGenerator(BaseGenerator):
    """
    Base class for style-based generators.
    
    Uses learned style codes for generation control.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        style_dim: int = 512,
        n_mlp: int = 8,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, ngf, **kwargs)
        
        self.style_dim = style_dim
        self.n_mlp = n_mlp
        
        # Style mapping network
        self.mapping = self._build_mapping_network()
        
    def _build_mapping_network(self) -> nn.Module:
        """Build style mapping network."""
        layers = []
        
        for i in range(self.n_mlp):
            if i == 0:
                layers.append(nn.Linear(self.style_dim, self.style_dim))
            else:
                layers.append(nn.Linear(self.style_dim, self.style_dim))
                
            layers.append(nn.LeakyReLU(0.2))
            
        return nn.Sequential(*layers)
        
    def map_style(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent z to style w."""
        return self.mapping(z)
        
    def forward(
        self,
        x: torch.Tensor,
        style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            style: Optional style code (latent z)
            
        Returns:
            Style-controlled output
        """
        if style is not None:
            w = self.map_style(style)
        else:
            w = None
            
        return self.synthesize(x, w)
        
    @abstractmethod
    def synthesize(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Synthesize output with style modulation."""
        pass
