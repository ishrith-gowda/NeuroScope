"""Core configuration module for neuroscope."""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class BaseConfig:
    """Base configuration class for neuroscope."""
    
    def __init__(self):
        """Initialize base configuration."""
        # Find project root (parent directory of neuroscope package)
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        # Default paths
        self.paths = {
            "data_dir": self.project_root / "data",
            "preprocessed_dir": self.project_root / "preprocessed",
            "preprocessed_registered_dir": self.project_root / "preprocessed_registered",
            "checkpoints_dir": self.project_root / "checkpoints",
            "templates_dir": self.project_root / "templates",
            "figures_dir": self.project_root / "figures",
            "results_dir": self.project_root / "results",
            "logs_dir": self.project_root / "logs",
            "mlruns_dir": self.project_root / "mlruns",
        }
        
        # Default preprocessing parameters
        self.preprocessing = {
            "n4_bias_correction": True,
            "skull_stripping": True,
            "intensity_normalization": "percentile",
            "registration": False,
            "isotropic_resampling": True,
            "resampling_spacing": [1.0, 1.0, 1.0],
            "normalization_percentiles": [1, 99],
        }
        
        # Default model parameters
        self.model = {
            "dimensions": 3,  # 2D or 3D
            "generator_features": 64,
            "discriminator_features": 64,
            "num_residual_blocks": 9,
            "dropout_rate": 0.0,
            "use_instance_norm": True,
        }
        
        # Default training parameters
        self.training = {
            "batch_size": 4,
            "learning_rate": 0.0002,
            "beta1": 0.5,
            "beta2": 0.999,
            "lambda_cycle": 10.0,
            "lambda_identity": 0.5,
            "epochs": 50,
            "save_interval": 10,
            "patch_size": [96, 96, 96],
            "samples_per_volume": 8,
        }
        
        # Default evaluation parameters
        self.evaluation = {
            "metrics": ["ssim", "psnr", "mse", "mae"],
            "sample_count": 10,
            "batch_size": 4,
        }
        
        # Initialize from environment variables
        self._init_from_env()
    
    def _init_from_env(self):
        """Initialize configuration from environment variables."""
        # Override paths from environment variables
        for key in self.paths:
            env_key = f"NEUROSCOPE_{key.upper()}"
            if env_key in os.environ:
                self.paths[key] = Path(os.environ[env_key])
        
        # Create directories if they don't exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(getattr(self, section), dict):
                getattr(self, section).update(values)
            else:
                setattr(self, section, values)
    
    def from_file(self, config_path: Path):
        """Load configuration from file."""
        import json
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        self.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        # Convert attributes to dictionary
        for attr in dir(self):
            if not attr.startswith("_") and attr != "to_dict" and attr != "from_file" and attr != "update":
                value = getattr(self, attr)
                
                # Convert Path objects to strings
                if isinstance(value, dict):
                    config_dict[attr] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
                else:
                    config_dict[attr] = str(value) if isinstance(value, Path) else value
        
        return config_dict
    
    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file."""
        import json
        
        if config_path is None:
            config_path = self.project_root / "configs" / "config.json"
        
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Create default configuration
config = BaseConfig()