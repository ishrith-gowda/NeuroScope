"""
base discriminator classes.

this module provides abstract base classes and common functionality
for discriminator architectures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union


class BaseDiscriminator(nn.Module, ABC):
    """
    abstract base class for all discriminators.
    
    provides common interface and utilities for discriminator networks.
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        **kwargs
    ):
        """
        initialize base discriminator.
        
        args:
            in_channels: number of input channels
            ndf: base number of discriminator filters
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.ndf = ndf
        
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        forward pass.
        
        args:
            x: input tensor [b, c, h, w]
            
        returns:
            discriminator output(s) - can be single tensor, list of tensors,
            or tuple of (output, intermediate features)
        """
        pass
        
    def count_parameters(self) -> int:
        """count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def init_weights(self, init_type: str = 'normal', gain: float = 0.02):
        """initialize network weights."""
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
                    
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
                    
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)
                
        self.apply(init_func)


class PatchDiscriminator(BaseDiscriminator):
    """
    base class for patchgan-style discriminators.
    
    classifies nxn patches of the input as real or fake.
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = 'instance',
        **kwargs
    ):
        super().__init__(in_channels, ndf, **kwargs)
        
        self.n_layers = n_layers
        self.norm_type = norm_type
        
        # build the network
        self.model = self._build_network()
        
    def _build_network(self) -> nn.Sequential:
        """build the discriminator network. override in subclass."""
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass."""
        return self.model(x)
        
    def get_all_activations(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """get activations from all intermediate layers."""
        activations = []
        
        for layer in self.model:
            x = layer(x)
            activations.append(x)
            
        return activations


class MultiScaleDiscriminatorBase(BaseDiscriminator):
    """
    base class for multi-scale discriminators.
    
    uses multiple discriminators at different scales.
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_scales: int = 3,
        **kwargs
    ):
        super().__init__(in_channels, ndf, **kwargs)
        
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()
        
    def forward(
        self,
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        forward pass at multiple scales.
        
        returns list of outputs from each scale.
        """
        outputs = []
        current_input = x
        
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(current_input))
            
            if i < len(self.discriminators) - 1:
                # downsample for next scale
                current_input = nn.functional.avg_pool2d(current_input, 2)
                
        return outputs
        
    def get_all_features(
        self,
        x: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """get features from all scales and layers."""
        all_features = []
        current_input = x
        
        for i, disc in enumerate(self.discriminators):
            if hasattr(disc, 'get_all_activations'):
                features = disc.get_all_activations(current_input)
            else:
                features = [disc(current_input)]
            all_features.append(features)
            
            if i < len(self.discriminators) - 1:
                current_input = nn.functional.avg_pool2d(current_input, 2)
                
        return all_features


class ConditionalDiscriminator(BaseDiscriminator):
    """
    base class for conditional discriminators.
    
    conditions discrimination on additional inputs.
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        condition_channels: int = 0,
        **kwargs
    ):
        super().__init__(in_channels, ndf, **kwargs)
        
        self.condition_channels = condition_channels
        self.total_in_channels = in_channels + condition_channels
        
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass with optional condition.
        
        args:
            x: input tensor
            condition: optional condition tensor (same spatial size as x)
            
        returns:
            discriminator output
        """
        if condition is not None:
            x = torch.cat([x, condition], dim=1)
            
        return self.discriminate(x)
        
    @abstractmethod
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """discriminate the (possibly conditioned) input."""
        pass


class ProjectionDiscriminator(BaseDiscriminator):
    """
    base class for projection discriminators.
    
    uses projection for conditional discrimination (cgan with projection).
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_classes: int = 0,
        **kwargs
    ):
        super().__init__(in_channels, ndf, **kwargs)
        
        self.n_classes = n_classes
        
        # class embedding for projection
        if n_classes > 0:
            self.embed = nn.Embedding(n_classes, ndf * 8)  # adjust based on architecture
            
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        forward pass with optional class labels.
        
        uses projection discriminator formulation if labels provided.
        """
        features = self.extract_features(x)
        output = self.output_layer(features)
        
        if labels is not None and self.n_classes > 0:
            # project class embedding
            embed = self.embed(labels)
            # global sum pooled features
            h = features.sum(dim=[2, 3])
            # inner product
            projection = (embed * h).sum(dim=1, keepdim=True)
            output = output + projection.view(-1, 1, 1, 1)
            
        return output
        
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """extract features from input."""
        pass
        
    @abstractmethod
    def output_layer(self, features: torch.Tensor) -> torch.Tensor:
        """final output layer."""
        pass


class FeatureMatchingDiscriminator(BaseDiscriminator):
    """
    base class for discriminators with feature matching.
    
    returns both output and intermediate features for feature matching loss.
    """
    
    def __init__(
        self,
        in_channels: int,
        ndf: int = 64,
        n_layers: int = 3,
        **kwargs
    ):
        super().__init__(in_channels, ndf, **kwargs)
        
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        forward pass.
        
        args:
            x: input tensor
            return_features: whether to return intermediate features
            
        returns:
            output tensor, and optionally list of intermediate features
        """
        features = []
        
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
                
        if return_features:
            return x, features
        else:
            return x
