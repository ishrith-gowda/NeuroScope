"""CycleGAN optimizer implementation.

This module provides specialized optimizers for CycleGAN training,
including separate optimizers for generators and discriminators.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class CycleGANOptimizer:
    """Optimizer for CycleGAN training with separate optimizers for each component."""
    
    def __init__(
        self,
        generators: Dict[str, torch.nn.Module],
        discriminators: Dict[str, torch.nn.Module],
        config: Dict[str, Any]
    ):
        """Initialize CycleGAN optimizer.
        
        Args:
            generators: Dictionary of generator models
            discriminators: Dictionary of discriminator models
            config: Optimizer configuration
        """
        self.generators = generators
        self.discriminators = discriminators
        self.config = config
        
        # Initialize optimizers
        self._init_optimizers()
    
    def _init_optimizers(self):
        """Initialize all optimizers."""
        # Generator optimizers
        self.optimizer_G_A2B = self._create_optimizer(
            self.generators['G_A2B'].parameters(),
            'generator'
        )
        
        self.optimizer_G_B2A = self._create_optimizer(
            self.generators['G_B2A'].parameters(),
            'generator'
        )
        
        # Discriminator optimizers
        self.optimizer_D_A = self._create_optimizer(
            self.discriminators['D_A'].parameters(),
            'discriminator'
        )
        
        self.optimizer_D_B = self._create_optimizer(
            self.discriminators['D_B'].parameters(),
            'discriminator'
        )
        
        # Learning rate schedulers
        self.schedulers = self._create_schedulers()
    
    def _create_optimizer(self, parameters, component_type: str):
        """Create optimizer for given parameters.
        
        Args:
            parameters: Model parameters
            component_type: Type of component ('generator' or 'discriminator')
            
        Returns:
            Optimizer instance
        """
        optimizer_config = self.config.get(f'{component_type}_optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 0.0002)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        betas = optimizer_config.get('betas', (0.5, 0.999))
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                parameters,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _create_schedulers(self):
        """Create learning rate schedulers.
        
        Returns:
            Dictionary of schedulers
        """
        schedulers = {}
        
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return schedulers
        
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 50)
            gamma = scheduler_config.get('gamma', 0.5)
            
            schedulers['G_A2B'] = optim.lr_scheduler.StepLR(
                self.optimizer_G_A2B, step_size=step_size, gamma=gamma
            )
            schedulers['G_B2A'] = optim.lr_scheduler.StepLR(
                self.optimizer_G_B2A, step_size=step_size, gamma=gamma
            )
            schedulers['D_A'] = optim.lr_scheduler.StepLR(
                self.optimizer_D_A, step_size=step_size, gamma=gamma
            )
            schedulers['D_B'] = optim.lr_scheduler.StepLR(
                self.optimizer_D_B, step_size=step_size, gamma=gamma
            )
        
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0.0)
            
            schedulers['G_A2B'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G_A2B, T_max=T_max, eta_min=eta_min
            )
            schedulers['G_B2A'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G_B2A, T_max=T_max, eta_min=eta_min
            )
            schedulers['D_A'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D_A, T_max=T_max, eta_min=eta_min
            )
            schedulers['D_B'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D_B, T_max=T_max, eta_min=eta_min
            )
        
        elif scheduler_type == 'plateau':
            mode = scheduler_config.get('mode', 'min')
            factor = scheduler_config.get('factor', 0.5)
            patience = scheduler_config.get('patience', 10)
            threshold = scheduler_config.get('threshold', 1e-4)
            
            schedulers['G_A2B'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G_A2B,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold
            )
            schedulers['G_B2A'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G_B2A,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold
            )
            schedulers['D_A'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D_A,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold
            )
            schedulers['D_B'] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_D_B,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold
            )
        
        return schedulers
    
    def zero_grad_generators(self):
        """Zero gradients for generator optimizers."""
        self.optimizer_G_A2B.zero_grad()
        self.optimizer_G_B2A.zero_grad()
    
    def zero_grad_discriminator_a(self):
        """Zero gradients for discriminator A optimizer."""
        self.optimizer_D_A.zero_grad()
    
    def zero_grad_discriminator_b(self):
        """Zero gradients for discriminator B optimizer."""
        self.optimizer_D_B.zero_grad()
    
    def step_generators(self):
        """Perform optimization step for generators."""
        self.optimizer_G_A2B.step()
        self.optimizer_G_B2A.step()
    
    def step_discriminator_a(self):
        """Perform optimization step for discriminator A."""
        self.optimizer_D_A.step()
    
    def step_discriminator_b(self):
        """Perform optimization step for discriminator B."""
        self.optimizer_D_B.step()
    
    def step_schedulers(self, metrics: Optional[Dict[str, float]] = None):
        """Step all learning rate schedulers.
        
        Args:
            metrics: Optional metrics for plateau scheduler
        """
        for name, scheduler in self.schedulers.items():
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metrics and name in metrics:
                    scheduler.step(metrics[name])
                else:
                    scheduler.step()
            else:
                scheduler.step()
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates for all optimizers.
        
        Returns:
            Dictionary of learning rates
        """
        return {
            'G_A2B': self.optimizer_G_A2B.param_groups[0]['lr'],
            'G_B2A': self.optimizer_G_B2A.param_groups[0]['lr'],
            'D_A': self.optimizer_D_A.param_groups[0]['lr'],
            'D_B': self.optimizer_D_B.param_groups[0]['lr']
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for all optimizers and schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler states
        """
        state = {
            'optimizers': {
                'G_A2B': self.optimizer_G_A2B.state_dict(),
                'G_B2A': self.optimizer_G_B2A.state_dict(),
                'D_A': self.optimizer_D_A.state_dict(),
                'D_B': self.optimizer_D_B.state_dict()
            }
        }
        
        if self.schedulers:
            state['schedulers'] = {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            }
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary for all optimizers and schedulers.
        
        Args:
            state_dict: Dictionary containing optimizer and scheduler states
        """
        # Load optimizer states
        optimizer_states = state_dict.get('optimizers', {})
        self.optimizer_G_A2B.load_state_dict(optimizer_states.get('G_A2B', {}))
        self.optimizer_G_B2A.load_state_dict(optimizer_states.get('G_B2A', {}))
        self.optimizer_D_A.load_state_dict(optimizer_states.get('D_A', {}))
        self.optimizer_D_B.load_state_dict(optimizer_states.get('D_B', {}))
        
        # Load scheduler states
        scheduler_states = state_dict.get('schedulers', {})
        for name, scheduler in self.schedulers.items():
            if name in scheduler_states:
                scheduler.load_state_dict(scheduler_states[name])