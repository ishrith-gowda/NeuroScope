"""
learning rate schedulers.

comprehensive collection of learning rate scheduling strategies
optimized for gan training.
"""

from typing import Optional, List, Callable
from dataclasses import dataclass
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class SchedulerConfig:
    """configuration for learning rate scheduler."""
    name: str = 'linear'
    total_epochs: int = 200
    decay_epochs: int = 100
    warmup_epochs: int = 0
    min_lr: float = 0.0
    max_lr: Optional[float] = None
    
    # cosine-specific
    num_cycles: float = 0.5
    
    # step-specific
    step_size: int = 50
    gamma: float = 0.5


class LinearDecayScheduler(_LRScheduler):
    """
    linear decay learning rate scheduler.
    
    maintains initial lr for a number of epochs, then linearly
    decays to zero (or min_lr).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        decay_epoch: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            total_epochs: total number of training epochs
            decay_epoch: epoch to start decay
            min_lr: minimum learning rate
            last_epoch: last epoch index
        """
        self.total_epochs = total_epochs
        self.decay_epoch = decay_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.decay_epoch:
            return self.base_lrs
        
        decay_range = self.total_epochs - self.decay_epoch
        current_decay = self.last_epoch - self.decay_epoch
        
        return [
            max(self.min_lr, base_lr * (1 - current_decay / decay_range))
            for base_lr in self.base_lrs
        ]


class LinearWarmupScheduler(_LRScheduler):
    """
    linear warmup learning rate scheduler.
    
    linearly increases lr from 0 to initial lr over warmup period,
    then applies the wrapped scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        scheduler: Optional[_LRScheduler] = None,
        warmup_factor: float = 0.0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            warmup_epochs: number of warmup epochs
            scheduler: scheduler to apply after warmup
            warmup_factor: starting factor (0 = start from 0)
            last_epoch: last epoch index
        """
        self.warmup_epochs = warmup_epochs
        self.scheduler = scheduler
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.warmup_factor + (1 - self.warmup_factor) * (
                self.last_epoch / self.warmup_epochs
            )
            return [base_lr * factor for base_lr in self.base_lrs]
        
        if self.scheduler is not None:
            return self.scheduler.get_lr()
        
        return self.base_lrs
    
    def step(self, epoch=None):
        super().step(epoch)
        if self.scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self.scheduler.step()


class CosineAnnealingWarmup(_LRScheduler):
    """
    cosine annealing with linear warmup.
    
    combines linear warmup with cosine annealing decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            total_epochs: total training epochs
            warmup_epochs: warmup period
            min_lr: minimum learning rate
            num_cycles: number of cosine cycles
            last_epoch: last epoch index
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [self.min_lr + factor * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]
        
        # cosine annealing
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        
        factor = 0.5 * (1 + math.cos(math.pi * self.num_cycles * 2 * progress))
        
        return [self.min_lr + factor * (base_lr - self.min_lr) 
                for base_lr in self.base_lrs]


class PolynomialDecay(_LRScheduler):
    """
    polynomial decay learning rate scheduler.
    
    decays lr following polynomial: lr = base_lr * (1 - progress)^power
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            total_epochs: total training epochs
            power: polynomial power
            min_lr: minimum learning rate
            warmup_epochs: warmup period
            last_epoch: last epoch index
        """
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [self.min_lr + factor * (base_lr - self.min_lr) 
                    for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        
        factor = (1 - progress) ** self.power
        
        return [self.min_lr + factor * (base_lr - self.min_lr) 
                for base_lr in self.base_lrs]


class StepDecay(_LRScheduler):
    """
    step decay learning rate scheduler.
    
    decays lr by gamma every step_size epochs.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 50,
        gamma: float = 0.5,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            step_size: epochs between decays
            gamma: decay factor
            min_lr: minimum learning rate
            last_epoch: last epoch index
        """
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        num_decays = self.last_epoch // self.step_size
        factor = self.gamma ** num_decays
        
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


class ExponentialDecay(_LRScheduler):
    """
    exponential decay learning rate scheduler.
    
    decays lr exponentially: lr = base_lr * gamma^epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.99,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            gamma: decay factor per epoch
            min_lr: minimum learning rate
            last_epoch: last epoch index
        """
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        factor = self.gamma ** self.last_epoch
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


class CyclicScheduler(_LRScheduler):
    """
    cyclic learning rate scheduler.
    
    oscillates lr between min and max values following a triangular
    or other pattern.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            base_lr: minimum learning rate
            max_lr: maximum learning rate
            step_size_up: steps to increase
            step_size_down: steps to decrease
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: decay factor for exp_range
            scale_fn: custom scaling function
            last_epoch: last epoch index
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is not None:
            self.scale_fn = scale_fn
        elif mode == 'triangular':
            self.scale_fn = lambda x: 1.0
        elif mode == 'triangular2':
            self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
        elif mode == 'exp_range':
            self.scale_fn = lambda x: gamma ** (x)
        
        self.cycle_size = self.step_size_up + self.step_size_down
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle = self.last_epoch // self.cycle_size
        x = self.last_epoch % self.cycle_size
        
        if x < self.step_size_up:
            # ascending
            phase = x / self.step_size_up
        else:
            # descending
            phase = 1 - (x - self.step_size_up) / self.step_size_down
        
        scale = self.scale_fn(cycle + 1)
        
        return [
            self.base_lr + (self.max_lr - self.base_lr) * phase * scale
            for _ in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """
    one cycle learning rate policy.
    
    implements the 1cycle policy from "super-convergence"
    paper by leslie smith.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            max_lr: maximum learning rate
            total_steps: total number of training steps
            pct_start: percentage of cycle for warmup
            anneal_strategy: 'cos' or 'linear'
            div_factor: initial lr = max_lr / div_factor
            final_div_factor: final lr = initial_lr / final_div_factor
            last_epoch: last epoch index
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.step_up:
            # warmup phase
            pct = self.last_epoch / self.step_up
            
            if self.anneal_strategy == 'cos':
                factor = (1 - math.cos(math.pi * pct)) / 2
            else:
                factor = pct
            
            lr = self.initial_lr + factor * (self.max_lr - self.initial_lr)
        else:
            # annealing phase
            pct = (self.last_epoch - self.step_up) / max(1, self.step_down)
            
            if self.anneal_strategy == 'cos':
                factor = (1 + math.cos(math.pi * pct)) / 2
            else:
                factor = 1 - pct
            
            lr = self.final_lr + factor * (self.max_lr - self.final_lr)
        
        return [lr for _ in self.base_lrs]


class WarmRestarts(_LRScheduler):
    """
    cosine annealing with warm restarts.
    
    implements sgdr: stochastic gradient descent with warm restarts.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ):
        """
        args:
            optimizer: wrapped optimizer
            t_0: initial period
            t_mult: period multiplier
            eta_min: minimum learning rate
            last_epoch: last epoch index
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        
        self.T_cur = 0
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        factor = 0.5 * (1 + math.cos(math.pi * self.T_cur / self.T_i))
        
        return [
            self.eta_min + factor * (base_lr - self.eta_min)
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.T_cur = self.T_cur + 1
        
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig = None,
    name: str = 'linear',
    total_epochs: int = 200,
    **kwargs
) -> _LRScheduler:
    """
    create scheduler from configuration or parameters.
    
    args:
        optimizer: optimizer to schedule
        config: scheduler configuration
        name: scheduler name
        total_epochs: total training epochs
        **kwargs: additional scheduler arguments
        
    returns:
        configured scheduler
    """
    if config is not None:
        name = config.name
        total_epochs = config.total_epochs
        kwargs.update({
            'decay_epochs': config.decay_epochs,
            'warmup_epochs': config.warmup_epochs,
            'min_lr': config.min_lr,
            'num_cycles': config.num_cycles,
            'step_size': config.step_size,
            'gamma': config.gamma
        })
    
    name = name.lower()
    
    if name == 'linear':
        return LinearDecayScheduler(
            optimizer,
            total_epochs=total_epochs,
            decay_epoch=kwargs.get('decay_epochs', total_epochs // 2),
            min_lr=kwargs.get('min_lr', 0.0)
        )
    
    elif name == 'cosine':
        return CosineAnnealingWarmup(
            optimizer,
            total_epochs=total_epochs,
            warmup_epochs=kwargs.get('warmup_epochs', 0),
            min_lr=kwargs.get('min_lr', 0.0),
            num_cycles=kwargs.get('num_cycles', 0.5)
        )
    
    elif name == 'polynomial':
        return PolynomialDecay(
            optimizer,
            total_epochs=total_epochs,
            power=kwargs.get('power', 0.9),
            min_lr=kwargs.get('min_lr', 0.0),
            warmup_epochs=kwargs.get('warmup_epochs', 0)
        )
    
    elif name == 'step':
        return StepDecay(
            optimizer,
            step_size=kwargs.get('step_size', 50),
            gamma=kwargs.get('gamma', 0.5),
            min_lr=kwargs.get('min_lr', 0.0)
        )
    
    elif name == 'exponential':
        return ExponentialDecay(
            optimizer,
            gamma=kwargs.get('gamma', 0.99),
            min_lr=kwargs.get('min_lr', 0.0)
        )
    
    elif name == 'warmrestarts':
        return WarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('min_lr', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")
