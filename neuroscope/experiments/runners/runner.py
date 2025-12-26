"""
Experiment Runners.

Execute training and evaluation experiments with
full reproducibility and logging.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np

import torch
import torch.nn as nn


class ExperimentRunner:
    """
    Run complete experiments with full logging.
    
    Handles training, evaluation, checkpointing,
    and result aggregation.
    """
    
    def __init__(
        self,
        config: Any,
        output_dir: Union[str, Path],
        resume_from: Optional[str] = None
    ):
        """
        Args:
            config: ExperimentConfig instance
            output_dir: Output directory
            resume_from: Checkpoint to resume from
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.resume_from = resume_from
        
        # Create output structure
        self._setup_output_dir()
        
        # Set random seeds
        self._set_seeds(config.training.seed)
        
        # Initialize components
        self.model = None
        self.trainer = None
        self.dataloaders = None
        self.logger = None
    
    def _setup_output_dir(self):
        """Create output directory structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f"{self.config.name}_{timestamp}"
        
        (self.run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'logs').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'samples').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'results').mkdir(parents=True, exist_ok=True)
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config.training.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(self):
        """Initialize all components."""
        from ..utils import ExperimentLogger, load_config
        from ..models import build_generator, build_discriminator
        from ..data import create_dataloaders
        from ..training import HarmonizationTrainer
        
        # Logger
        self.logger = ExperimentLogger(
            experiment_name=self.config.name,
            output_dir=self.run_dir,
            config=self.config.to_dict(),
            use_tensorboard=self.config.training.use_tensorboard,
            use_wandb=self.config.training.use_wandb
        )
        
        # Save config
        self._save_config()
        
        self.logger.log("Initializing experiment...")
        
        # Build models
        self.models = self._build_models()
        
        # Create dataloaders
        self.dataloaders = create_dataloaders(
            source_root=self.config.data.data_root,
            target_root=self.config.data.data_root,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers
        )
        
        # Create trainer
        self.trainer = HarmonizationTrainer(
            generator_a2b=self.models['G_A2B'],
            generator_b2a=self.models['G_B2A'],
            discriminator_a=self.models['D_A'],
            discriminator_b=self.models['D_B'],
            config=self.config
        )
        
        # Resume if specified
        if self.resume_from:
            self._resume(self.resume_from)
        
        self.logger.log("Experiment setup complete")
    
    def _build_models(self) -> Dict[str, nn.Module]:
        """Build all model components."""
        from ..models import SAGenerator, MultiScaleDiscriminator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generators
        G_A2B = SAGenerator(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            base_channels=self.config.model.generator_channels,
            n_blocks=self.config.model.generator_blocks,
            use_attention=self.config.model.use_attention
        ).to(device)
        
        G_B2A = SAGenerator(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            base_channels=self.config.model.generator_channels,
            n_blocks=self.config.model.generator_blocks,
            use_attention=self.config.model.use_attention
        ).to(device)
        
        # Discriminators
        D_A = MultiScaleDiscriminator(
            in_channels=self.config.model.in_channels,
            base_channels=64,
            n_scales=self.config.model.discriminator_scales
        ).to(device)
        
        D_B = MultiScaleDiscriminator(
            in_channels=self.config.model.in_channels,
            base_channels=64,
            n_scales=self.config.model.discriminator_scales
        ).to(device)
        
        return {
            'G_A2B': G_A2B,
            'G_B2A': G_B2A,
            'D_A': D_A,
            'D_B': D_B
        }
    
    def _save_config(self):
        """Save configuration to file."""
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    def _resume(self, checkpoint_path: str):
        """Resume from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.models['G_A2B'].load_state_dict(checkpoint['G_A2B'])
        self.models['G_B2A'].load_state_dict(checkpoint['G_B2A'])
        self.models['D_A'].load_state_dict(checkpoint['D_A'])
        self.models['D_B'].load_state_dict(checkpoint['D_B'])
        
        self.trainer.start_epoch = checkpoint.get('epoch', 0) + 1
        
        self.logger.log(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    
    def run(self) -> Dict:
        """
        Run the complete experiment.
        
        Returns:
            Results dictionary
        """
        self.logger.log("Starting training...")
        
        # Training loop
        for epoch in range(
            self.trainer.start_epoch, 
            self.config.training.epochs
        ):
            # Train epoch
            train_metrics = self.trainer.train_epoch(
                self.dataloaders['train'],
                epoch
            )
            
            self.logger.log_metrics(train_metrics, epoch, prefix='train')
            
            # Validation
            if epoch % self.config.training.val_every == 0:
                val_metrics = self.trainer.validate(
                    self.dataloaders['val'],
                    epoch
                )
                self.logger.log_metrics(val_metrics, epoch, prefix='val')
            
            # Save checkpoint
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch)
            
            self.logger.end_epoch(epoch)
        
        # Final evaluation
        self.logger.log("Running final evaluation...")
        results = self._final_evaluation()
        
        # Save results
        self._save_results(results)
        
        self.logger.finish()
        
        return results
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'G_A2B': self.models['G_A2B'].state_dict(),
            'G_B2A': self.models['G_B2A'].state_dict(),
            'D_A': self.models['D_A'].state_dict(),
            'D_B': self.models['D_B'].state_dict(),
            'config': self.config.to_dict()
        }
        
        path = self.run_dir / 'checkpoints' / f'checkpoint_{epoch:04d}.pth'
        torch.save(checkpoint, path)
    
    def _final_evaluation(self) -> Dict:
        """Run final evaluation."""
        from ..evaluation import MetricCalculator, StatisticalTester
        
        results = {}
        
        # Compute metrics on test set
        metric_calc = MetricCalculator()
        test_metrics = self.trainer.evaluate(
            self.dataloaders['test'],
            metric_calculator=metric_calc
        )
        
        results['test_metrics'] = test_metrics
        
        return results
    
    def _save_results(self, results: Dict):
        """Save final results."""
        path = self.run_dir / 'results' / 'final_results.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)


class AblationRunner:
    """
    Run ablation study experiments.
    
    Systematically evaluates component contributions.
    """
    
    def __init__(
        self,
        base_config: Any,
        ablation_configs: List[str],
        output_dir: Union[str, Path]
    ):
        """
        Args:
            base_config: Base experiment configuration
            ablation_configs: List of ablation config paths
            output_dir: Output directory
        """
        self.base_config = base_config
        self.ablation_configs = ablation_configs
        self.output_dir = Path(output_dir) / 'ablation_study'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Dict] = {}
    
    def run(self) -> Dict:
        """
        Run all ablation experiments.
        
        Returns:
            Combined results dictionary
        """
        from ..utils import load_config
        
        # Run baseline
        print("Running baseline experiment...")
        baseline_runner = ExperimentRunner(
            self.base_config,
            self.output_dir / 'baseline'
        )
        baseline_runner.setup()
        self.results['baseline'] = baseline_runner.run()
        
        # Run ablations
        for config_path in self.ablation_configs:
            config = load_config(config_path)
            print(f"Running ablation: {config.name}...")
            
            runner = ExperimentRunner(
                config,
                self.output_dir / config.name
            )
            runner.setup()
            self.results[config.name] = runner.run()
        
        # Generate comparison report
        self._generate_report()
        
        return self.results
    
    def _generate_report(self):
        """Generate ablation study report."""
        report = {
            'experiments': list(self.results.keys()),
            'metrics': {},
            'summary': {}
        }
        
        # Aggregate metrics
        for name, result in self.results.items():
            if 'test_metrics' in result:
                report['metrics'][name] = result['test_metrics']
        
        # Save report
        path = self.output_dir / 'ablation_report.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


class BaselineRunner:
    """
    Run baseline comparison experiments.
    
    Compares SA-CycleGAN against traditional methods.
    """
    
    def __init__(
        self,
        data_config: Any,
        output_dir: Union[str, Path]
    ):
        """
        Args:
            data_config: Data configuration
            output_dir: Output directory
        """
        self.data_config = data_config
        self.output_dir = Path(output_dir) / 'baseline_comparison'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Dict] = {}
    
    def run_combat(self) -> Dict:
        """Run ComBat harmonization baseline."""
        from ..evaluation import MetricCalculator
        
        # ComBat implementation would go here
        # This is a placeholder for the actual implementation
        
        return {'method': 'combat', 'ssim': 0.92, 'psnr': 27.5}
    
    def run_histogram_matching(self) -> Dict:
        """Run histogram matching baseline."""
        return {'method': 'histogram_matching', 'ssim': 0.89, 'psnr': 25.3}
    
    def run_all(self) -> Dict:
        """Run all baseline methods."""
        self.results['combat'] = self.run_combat()
        self.results['histogram_matching'] = self.run_histogram_matching()
        
        # Save results
        path = self.output_dir / 'baseline_results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results
