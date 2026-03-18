"""
base generator classes.

this module provides abstract base classes and common functionality
for generator architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


class BaseGenerator(nn.Module, ABC):
    """
    abstract base class for all generators.
    
    provides common interface and utilities for generator networks.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ngf: int = 64,
        **kwargs
    ):
        """
        initialize base generator.
        
        args:
            in_channels: number of input channels
            out_channels: number of output channels
            ngf: base number of generator filters
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            output tensor [b, c', h, w]
        """
        pass
        
    def count_parameters(self) -> int:
        """count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        get intermediate feature maps for visualization/loss computation.
        
        override in subclasses for specific implementations.
        """
        return {'output': self.forward(x)}
        
    def init_weights(self, init_type: str = 'normal', gain: float = 0.02):
        """
        initialize network weights.
        
        args:
            init_type: initialization type ('normal', 'xavier', 'kaiming', 'orthogonal')
            gain: scaling factor for initialization
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
    base class for encoder-decoder style generators.
    
    provides structure for u-net, resnet-based, and other encoder-decoder architectures.
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
        
        self.encoder = None  # override in subclass
        self.decoder = None  # override in subclass
        self.bottleneck = None  # override in subclass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through encoder-decoder."""
        # encode
        enc_features = self.encode(x)
        
        # bottleneck
        if self.bottleneck is not None:
            bottleneck_out = self.bottleneck(enc_features[-1])
        else:
            bottleneck_out = enc_features[-1]
            
        # decode
        output = self.decode(bottleneck_out, enc_features if self.use_skip else None)
        
        return output
        
    def encode(self, x: torch.Tensor) -> list:
        """
        encode input to feature representations.
        
        returns list of features at each scale.
        """
        raise NotImplementedError
        
    def decode(
        self,
        x: torch.Tensor,
        skip_features: Optional[list] = None
    ) -> torch.Tensor:
        """
        decode features to output.
        
        args:
            x: bottleneck features
            skip_features: optional skip connection features
        """
        raise NotImplementedError


class ResidualGenerator(BaseGenerator):
    """
    base class for residual-style generators.
    
    generates output as input + learned residual.
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
        """forward pass with residual learning."""
        residual = self.compute_residual(x)
        
        if self.learn_residual and x.size(1) == residual.size(1):
            return x + residual
        else:
            return residual
            
    @abstractmethod
    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """compute residual features."""
        pass


class MultiScaleGenerator(BaseGenerator):
    """
    base class for multi-scale generators.
    
    generates output at multiple scales for progressive training.
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
        forward pass.
        
        args:
            x: input tensor
            return_all_scales: if true, return outputs at all scales
            
        returns:
            output at finest scale, or list of outputs at all scales
        """
        outputs = []
        
        for i, generator in enumerate(self.scale_generators):
            if i == 0:
                out = generator(x)
            else:
                # upsample previous output and combine
                prev_upsampled = nn.functional.interpolate(
                    outputs[-1], scale_factor=2, mode='bilinear', align_corners=False
                )
                out = generator(torch.cat([x, prev_upsampled], dim=1))
                
            outputs.append(out)
            
            # downsample input for next scale
            x = nn.functional.avg_pool2d(x, 2)
            
        if return_all_scales:
            return outputs
        else:
            return outputs[-1]


class ConditionalGenerator(BaseGenerator):
    """
    base class for conditional generators.
    
    conditions generation on additional inputs (labels, embeddings, etc.).
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
        self.condition_encoder = None  # override in subclass
        
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass with conditioning.
        
        args:
            x: input tensor
            condition: conditioning input (labels, embeddings, etc.)
            
        returns:
            conditional output
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
        """generate output with optional condition embedding."""
        pass


class StyleGenerator(BaseGenerator):
    """
    base class for style-based generators.
    
    uses learned style codes for generation control.
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
        
        # style mapping network
        self.mapping = self._build_mapping_network()
        
    def _build_mapping_network(self) -> nn.Module:
        """build style mapping network."""
        layers = []
        
        for i in range(self.n_mlp):
            if i == 0:
                layers.append(nn.Linear(self.style_dim, self.style_dim))
            else:
                layers.append(nn.Linear(self.style_dim, self.style_dim))
                
            layers.append(nn.LeakyReLU(0.2))
            
        return nn.Sequential(*layers)
        
    def map_style(self, z: torch.Tensor) -> torch.Tensor:
        """map latent z to style w."""
        return self.mapping(z)
        
    def forward(
        self,
        x: torch.Tensor,
        style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass.
        
        args:
            x: input tensor
            style: optional style code (latent z)
            
        returns:
            style-controlled output
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
        """synthesize output with style modulation."""
        pass
