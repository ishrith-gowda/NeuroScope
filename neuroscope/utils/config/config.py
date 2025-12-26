"""
Configuration Management.

YAML-based configuration with schema validation
and hierarchical merging.
"""

from typing import Dict, Any, List, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import json


T = TypeVar('T')


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'sa_cyclegan'
    
    # Generator
    generator_type: str = 'sa_generator'
    generator_channels: int = 64
    generator_blocks: int = 9
    use_attention: bool = True
    attention_heads: int = 8
    
    # Discriminator
    discriminator_type: str = 'multiscale'
    discriminator_scales: int = 3
    discriminator_layers: int = 4
    use_spectral_norm: bool = True
    
    # Input/Output
    in_channels: int = 4
    out_channels: int = 4
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DataConfig:
    """Data configuration."""
    # Datasets
    source_dataset: str = 'brats'
    target_dataset: str = 'upenn'
    data_root: str = './preprocessed'
    
    # Preprocessing
    crop_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    normalize: bool = True
    augment: bool = True
    
    # Loading
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LossConfig:
    """Loss configuration."""
    # Adversarial
    adversarial_type: str = 'lsgan'
    adversarial_weight: float = 1.0
    
    # Cycle consistency
    cycle_weight: float = 10.0
    
    # Identity
    identity_weight: float = 5.0
    
    # Perceptual
    use_perceptual: bool = True
    perceptual_weight: float = 1.0
    perceptual_layers: List[str] = field(
        default_factory=lambda: ['relu1_2', 'relu2_2', 'relu3_4']
    )
    
    # Contrastive (PatchNCE)
    use_contrastive: bool = True
    contrastive_weight: float = 1.0
    nce_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16])
    
    # Tumor preservation
    use_tumor_preservation: bool = True
    tumor_weight: float = 2.0
    
    # SSIM
    use_ssim: bool = True
    ssim_weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    # Generator
    generator_lr: float = 2e-4
    generator_betas: List[float] = field(default_factory=lambda: [0.5, 0.999])
    
    # Discriminator
    discriminator_lr: float = 2e-4
    discriminator_betas: List[float] = field(default_factory=lambda: [0.5, 0.999])
    
    # Weight decay
    weight_decay: float = 0.0
    
    # Scheduler
    scheduler_type: str = 'linear_warmup_cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Duration
    epochs: int = 200
    steps_per_epoch: Optional[int] = None
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10
    keep_last: int = 5
    
    # Validation
    val_every: int = 5
    val_samples: int = 4
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'float16'
    
    # Gradient
    gradient_clip: Optional[float] = 1.0
    accumulation_steps: int = 1
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Logging
    log_every: int = 100
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    # Metrics
    compute_ssim: bool = True
    compute_psnr: bool = True
    compute_fid: bool = True
    compute_lpips: bool = True
    
    # Statistical
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Output
    output_dir: str = './results'
    save_predictions: bool = True
    generate_report: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = 'default'
    description: str = ''
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'model': self.model.to_dict(),
            'data': self.data.to_dict(),
            'loss': self.loss.to_dict(),
            'optimizer': self.optimizer.to_dict(),
            'training': self.training.to_dict(),
            'evaluation': self.evaluation.to_dict()
        }


class ConfigManager:
    """
    Manage configuration loading, validation, and merging.
    
    Supports YAML files with hierarchical inheritance.
    """
    
    def __init__(self, config_dir: Union[str, Path] = None):
        """
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self._loaded_configs: Dict[str, Dict] = {}
    
    def load_yaml(self, path: Union[str, Path]) -> Dict:
        """
        Load YAML configuration file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        import yaml
        
        path = Path(path)
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Handle inheritance
        if 'base' in config:
            base_path = config.pop('base')
            if not Path(base_path).is_absolute():
                base_path = path.parent / base_path
            
            base_config = self.load_yaml(base_path)
            config = self._deep_merge(base_config, config)
        
        return config
    
    def _deep_merge(
        self,
        base: Dict,
        override: Dict
    ) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def dict_to_config(
        self,
        config_dict: Dict,
        config_class: Type[T] = ExperimentConfig
    ) -> T:
        """
        Convert dictionary to config dataclass.
        
        Args:
            config_dict: Configuration dictionary
            config_class: Target config class
            
        Returns:
            Config instance
        """
        if config_class == ExperimentConfig:
            return ExperimentConfig(
                name=config_dict.get('name', 'default'),
                description=config_dict.get('description', ''),
                model=self._dict_to_dataclass(
                    config_dict.get('model', {}), ModelConfig
                ),
                data=self._dict_to_dataclass(
                    config_dict.get('data', {}), DataConfig
                ),
                loss=self._dict_to_dataclass(
                    config_dict.get('loss', {}), LossConfig
                ),
                optimizer=self._dict_to_dataclass(
                    config_dict.get('optimizer', {}), OptimizerConfig
                ),
                training=self._dict_to_dataclass(
                    config_dict.get('training', {}), TrainingConfig
                ),
                evaluation=self._dict_to_dataclass(
                    config_dict.get('evaluation', {}), EvaluationConfig
                )
            )
        
        return self._dict_to_dataclass(config_dict, config_class)
    
    def _dict_to_dataclass(
        self,
        data: Dict,
        cls: Type[T]
    ) -> T:
        """Convert dict to dataclass instance."""
        import dataclasses
        
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        
        return cls(**filtered_data)
    
    def load_config(
        self,
        path: Union[str, Path]
    ) -> ExperimentConfig:
        """
        Load and parse configuration file.
        
        Args:
            path: Path to config file
            
        Returns:
            ExperimentConfig instance
        """
        config_dict = self.load_yaml(path)
        return self.dict_to_config(config_dict)
    
    def save_config(
        self,
        config: Union[ExperimentConfig, Dict],
        path: Union[str, Path],
        format: str = 'yaml'
    ):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Output path
            format: 'yaml' or 'json'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(config, 'to_dict'):
            config = config.to_dict()
        
        if format == 'yaml':
            import yaml
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def validate_config(
        self,
        config: ExperimentConfig
    ) -> List[str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Model validation
        if config.model.generator_blocks < 1:
            errors.append("generator_blocks must be >= 1")
        
        if config.model.in_channels < 1:
            errors.append("in_channels must be >= 1")
        
        # Data validation
        if config.data.batch_size < 1:
            errors.append("batch_size must be >= 1")
        
        split_sum = (
            config.data.train_ratio + 
            config.data.val_ratio + 
            config.data.test_ratio
        )
        if abs(split_sum - 1.0) > 0.01:
            errors.append(f"Split ratios must sum to 1.0, got {split_sum}")
        
        # Training validation
        if config.training.epochs < 1:
            errors.append("epochs must be >= 1")
        
        if config.training.save_every < 1:
            errors.append("save_every must be >= 1")
        
        # Loss validation
        if config.loss.cycle_weight < 0:
            errors.append("cycle_weight must be >= 0")
        
        return errors


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Convenience function to load config."""
    manager = ConfigManager()
    return manager.load_config(path)


def save_config(
    config: Union[ExperimentConfig, Dict],
    path: Union[str, Path]
):
    """Convenience function to save config."""
    manager = ConfigManager()
    manager.save_config(config, path)
