"""
optimizer factories and configurations.

provides flexible optimizer creation with proper configuration
for gan training scenarios.
"""

from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class OptimizerConfig:
    """base optimizer configuration."""
    name: str = 'adam'
    lr: float = 2e-4
    weight_decay: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}


@dataclass
class AdamConfig(OptimizerConfig):
    """adam optimizer configuration."""
    name: str = 'adam'
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class AdamWConfig(OptimizerConfig):
    """adamw optimizer configuration."""
    name: str = 'adamw'
    lr: float = 2e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class SGDConfig(OptimizerConfig):
    """sgd optimizer configuration."""
    name: str = 'sgd'
    lr: float = 0.01
    momentum: float = 0.9
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclass
class RMSpropConfig(OptimizerConfig):
    """rmsprop optimizer configuration."""
    name: str = 'rmsprop'
    lr: float = 0.01
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


def create_optimizer(
    params: Iterator[nn.Parameter],
    config: OptimizerConfig = None,
    name: str = 'adam',
    lr: float = 2e-4,
    **kwargs
) -> Optimizer:
    """
    create optimizer from configuration or parameters.
    
    args:
        params: model parameters to optimize
        config: optimizer configuration object
        name: optimizer name (if no config provided)
        lr: learning rate (if no config provided)
        **kwargs: additional optimizer arguments
        
    returns:
        configured optimizer
    """
    if config is not None:
        name = config.name
        lr = config.lr
        config_dict = config.to_dict()
        config_dict.update(kwargs)
        kwargs = config_dict
    
    name = name.lower()
    
    if name == 'adam':
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(kwargs.get('beta1', 0.5), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            amsgrad=kwargs.get('amsgrad', False)
        )
    
    elif name == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01),
            amsgrad=kwargs.get('amsgrad', False)
        )
    
    elif name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            dampening=kwargs.get('dampening', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.0),
            nesterov=kwargs.get('nesterov', False)
        )
    
    elif name == 'rmsprop':
        return torch.optim.RMSprop(
            params,
            lr=lr,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            momentum=kwargs.get('momentum', 0.0),
            centered=kwargs.get('centered', False)
        )
    
    elif name == 'radam':
        return torch.optim.RAdam(
            params,
            lr=lr,
            betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_gan_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    g_config: OptimizerConfig = None,
    d_config: OptimizerConfig = None,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999
) -> Tuple[Optimizer, Optimizer]:
    """
    create optimizers for gan training.
    
    args:
        generator: generator model
        discriminator: discriminator model
        g_config: generator optimizer config
        d_config: discriminator optimizer config
        lr_g: generator learning rate
        lr_d: discriminator learning rate
        beta1: adam beta1
        beta2: adam beta2
        
    returns:
        tuple of (generator_optimizer, discriminator_optimizer)
    """
    if g_config is None:
        g_config = AdamConfig(lr=lr_g, beta1=beta1, beta2=beta2)
    
    if d_config is None:
        d_config = AdamConfig(lr=lr_d, beta1=beta1, beta2=beta2)
    
    opt_g = create_optimizer(generator.parameters(), g_config)
    opt_d = create_optimizer(discriminator.parameters(), d_config)
    
    return opt_g, opt_d


def create_cyclegan_optimizers(
    G_A2B: nn.Module,
    G_B2A: nn.Module,
    D_A: nn.Module,
    D_B: nn.Module,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    beta2: float = 0.999
) -> Tuple[Optimizer, Optimizer]:
    """
    create optimizers for cyclegan training.
    
    combines generator parameters and discriminator parameters
    into joint optimizers.
    
    args:
        g_a2b: a to b generator
        g_b2a: b to a generator
        d_a: domain a discriminator
        d_b: domain b discriminator
        lr_g: generator learning rate
        lr_d: discriminator learning rate
        beta1: adam beta1
        beta2: adam beta2
        
    returns:
        tuple of (generator_optimizer, discriminator_optimizer)
    """
    g_params = list(G_A2B.parameters()) + list(G_B2A.parameters())
    d_params = list(D_A.parameters()) + list(D_B.parameters())
    
    opt_g = torch.optim.Adam(
        g_params,
        lr=lr_g,
        betas=(beta1, beta2)
    )
    
    opt_d = torch.optim.Adam(
        d_params,
        lr=lr_d,
        betas=(beta1, beta2)
    )
    
    return opt_g, opt_d


class TTUR:
    """
    two time-scale update rule for gans.
    
    uses different learning rates for generator and discriminator
    as proposed in "gans trained by a two time-scale update rule
    converge to a local nash equilibrium".
    """
    
    @staticmethod
    def create_optimizers(
        generator: nn.Module,
        discriminator: nn.Module,
        lr_g: float = 1e-4,
        lr_d: float = 4e-4,
        beta1: float = 0.0,
        beta2: float = 0.9
    ) -> Tuple[Optimizer, Optimizer]:
        """
        create ttur optimizers.
        
        args:
            generator: generator model
            discriminator: discriminator model
            lr_g: generator learning rate (smaller)
            lr_d: discriminator learning rate (larger)
            beta1: adam beta1 (typically 0 for ttur)
            beta2: adam beta2
            
        returns:
            tuple of (generator_optimizer, discriminator_optimizer)
        """
        opt_g = torch.optim.Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        
        opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        return opt_g, opt_d


class DifferentialLR:
    """
    differential learning rates for different model parts.
    
    useful for fine-tuning pretrained models where different
    layers should have different learning rates.
    """
    
    @staticmethod
    def create_param_groups(
        model: nn.Module,
        base_lr: float = 1e-4,
        layer_decay: float = 0.65,
        num_groups: int = 4
    ) -> List[Dict[str, Any]]:
        """
        create parameter groups with differential learning rates.
        
        args:
            model: model to create groups for
            base_lr: base learning rate for the last layer
            layer_decay: decay factor for each layer group
            num_groups: number of parameter groups
            
        returns:
            list of parameter groups for optimizer
        """
        # get all named parameters
        named_params = list(model.named_parameters())
        num_params = len(named_params)
        params_per_group = max(1, num_params // num_groups)
        
        param_groups = []
        
        for i in range(num_groups):
            start_idx = i * params_per_group
            end_idx = start_idx + params_per_group if i < num_groups - 1 else num_params
            
            group_params = [p for _, p in named_params[start_idx:end_idx] if p.requires_grad]
            
            if group_params:
                lr = base_lr * (layer_decay ** (num_groups - 1 - i))
                param_groups.append({
                    'params': group_params,
                    'lr': lr
                })
        
        return param_groups
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        base_lr: float = 1e-4,
        layer_decay: float = 0.65,
        optimizer_cls: type = torch.optim.AdamW,
        **kwargs
    ) -> Optimizer:
        """
        create optimizer with differential learning rates.
        
        args:
            model: model to optimize
            base_lr: base learning rate
            layer_decay: decay factor
            optimizer_cls: optimizer class to use
            **kwargs: additional optimizer arguments
            
        returns:
            configured optimizer
        """
        param_groups = DifferentialLR.create_param_groups(
            model, base_lr, layer_decay
        )
        
        return optimizer_cls(param_groups, **kwargs)


class ExponentialMovingAverage:
    """
    exponential moving average of model parameters.
    
    maintains an ema of model parameters for improved stability
    and evaluation performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        """
        args:
            model: model to track
            decay: ema decay rate
            device: device for ema parameters
        """
        self.model = model
        self.decay = decay
        self.device = device
        
        # create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if device is not None:
                    self.shadow[name] = param.data.clone().to(device)
                else:
                    self.shadow[name] = param.data.clone()
    
    def update(self):
        """update ema parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """apply ema parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """return ema state dict."""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """load ema state dict."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
