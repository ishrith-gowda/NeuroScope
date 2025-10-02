"""Comprehensive model development pipeline for NeuroScope.

This module provides a complete pipeline for CycleGAN model development,
including data preparation, training, evaluation, and model export.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

import torch

from neuroscope.core.logging import get_logger, configure_logging
from neuroscope.config import get_default_training_config, validate_config
from neuroscope.models.architectures import CycleGAN
from neuroscope.training.trainers import CycleGANTrainer
from neuroscope.training.optimizers import CycleGANOptimizer
from neuroscope.data.loaders import get_cycle_domain_loaders

logger = get_logger(__name__)


class ModelDevelopmentPipeline:
    """Complete model development pipeline for CycleGAN."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model development pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.trainer = None
        self.results = {}
    
    def run_full_pipeline(
        self,
        data_root: Path,
        output_dir: Path,
        resume_checkpoint: Optional[Path] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run the complete model development pipeline.
        
        Args:
            data_root: Root directory containing preprocessed data
            output_dir: Output directory for model outputs
            resume_checkpoint: Optional checkpoint to resume from
            dry_run: Print planned actions without executing
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting comprehensive model development pipeline")
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Prepare training data
            train_loader_a, train_loader_b, val_loader_a, val_loader_b = self._prepare_training_data(
                data_root, dry_run
            )
            
            # Step 2: Initialize model
            self._initialize_model(dry_run)
            
            # Step 3: Train model
            training_results = self._train_model(
                train_loader_a, train_loader_b, val_loader_a, val_loader_b,
                output_dir, resume_checkpoint, dry_run
            )
            
            # Step 4: Evaluate model
            evaluation_results = self._evaluate_model(
                val_loader_a, val_loader_b, output_dir, dry_run
            )
            
            # Step 5: Export model
            export_results = self._export_model(output_dir, dry_run)
            
            # Step 6: Generate final report
            final_report = self._generate_final_report(
                training_results, evaluation_results, export_results
            )
            
            # Save results
            results_file = output_dir / 'model_development_results.json'
            if not dry_run:
                with open(results_file, 'w') as f:
                    json.dump({
                        'pipeline_results': final_report,
                        'training_results': training_results,
                        'evaluation_results': evaluation_results,
                        'export_results': export_results,
                        'execution_time': time.time() - pipeline_start_time
                    }, f, indent=2)
            
            logger.info(f"Model development pipeline completed in {time.time() - pipeline_start_time:.2f} seconds")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        device_config = self.config.get('device', {})
        
        if device_config.get('use_cuda', True) and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_config.get("cuda_device", 0)}')
            logger.info(f"Using CUDA device: {device}")
        elif device_config.get('use_mps', False) and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Using MPS device")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")
        
        return device
    
    def _prepare_training_data(
        self,
        data_root: Path,
        dry_run: bool
    ) -> tuple:
        """Prepare training data loaders."""
        logger.info("Preparing training data")
        
        if dry_run:
            logger.info(f"Would prepare data loaders from: {data_root}")
            return None, None, None, None
        
        # Get data configuration
        data_config = self.config.get('data', {})
        
        # Create data loaders
        train_loader_a, train_loader_b, val_loader_a, val_loader_b = get_cycle_domain_loaders(
            data_root=data_root,
            metadata_json=data_config.get('metadata_json'),
            batch_size=data_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            shuffle=data_config.get('shuffle', True)
        )
        
        logger.info("Training data prepared successfully")
        return train_loader_a, train_loader_b, val_loader_a, val_loader_b
    
    def _initialize_model(self, dry_run: bool):
        """Initialize CycleGAN model and trainer."""
        logger.info("Initializing model")
        
        if dry_run:
            logger.info("Would initialize CycleGAN model and trainer")
            return
        
        # Initialize model
        model_config = self.config.get('model', {})
        self.model = CycleGAN(**model_config)
        
        # Initialize optimizer
        optimizer = CycleGANOptimizer(
            generators={'G_A2B': self.model.G_A2B, 'G_B2A': self.model.G_B2A},
            discriminators={'D_A': self.model.D_A, 'D_B': self.model.D_B},
            config=self.config
        )
        
        # Initialize trainer
        self.trainer = CycleGANTrainer(
            model=self.model,
            optimizer=optimizer,
            device=self.device,
            config=self.config
        )
        
        logger.info("Model initialized successfully")
    
    def _train_model(
        self,
        train_loader_a,
        train_loader_b,
        val_loader_a,
        val_loader_b,
        output_dir: Path,
        resume_checkpoint: Optional[Path],
        dry_run: bool
    ) -> Dict[str, Any]:
        """Train the CycleGAN model."""
        logger.info("Training CycleGAN model")
        
        if dry_run:
            logger.info("Would train CycleGAN model")
            return {}
        
        # Create output directories
        checkpoint_dir = output_dir / 'checkpoints'
        sample_dir = output_dir / 'samples'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Update trainer config with output directories
        self.trainer.config['checkpoint_dir'] = str(checkpoint_dir)
        self.trainer.config['sample_dir'] = str(sample_dir)
        
        # Resume from checkpoint if specified
        if resume_checkpoint:
            self.trainer.load_checkpoint(resume_checkpoint)
            logger.info(f"Resumed training from: {resume_checkpoint}")
        
        # Training configuration
        training_config = self.config.get('training', {})
        n_epochs = training_config.get('n_epochs', 100)
        
        # Training loop
        for epoch in range(self.trainer.current_epoch, n_epochs):
            logger.info(f"Training epoch {epoch + 1}/{n_epochs}")
            
            # Train for one epoch
            epoch_losses = self.trainer.train_epoch(
                train_loader_a, train_loader_b, epoch
            )
            
            # Save checkpoint
            if (epoch + 1) % training_config.get('checkpoint_interval', 10) == 0:
                self.trainer.save_checkpoint(epoch + 1, str(checkpoint_dir))
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1} losses: {epoch_losses}")
        
        # Save final model
        final_checkpoint_path = checkpoint_dir / 'final_model.pth'
        self.trainer.save_checkpoint(n_epochs, str(checkpoint_dir))
        
        # Generate loss curves
        loss_curves_path = output_dir / 'loss_curves.png'
        self.trainer.plot_loss_curves(str(loss_curves_path))
        
        training_results = {
            'final_epoch': n_epochs,
            'final_losses': epoch_losses,
            'checkpoint_path': str(final_checkpoint_path),
            'loss_curves_path': str(loss_curves_path)
        }
        
        logger.info("Model training completed successfully")
        return training_results
    
    def _evaluate_model(
        self,
        val_loader_a,
        val_loader_b,
        output_dir: Path,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Evaluate the trained model."""
        logger.info("Evaluating trained model")
        
        if dry_run:
            logger.info("Would evaluate trained model")
            return {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluation metrics
        metrics = {
            'mse_a2b': 0.0,
            'mse_b2a': 0.0,
            'mae_a2b': 0.0,
            'mae_b2a': 0.0,
            'ssim_a2b': 0.0,
            'ssim_b2a': 0.0,
            'psnr_a2b': 0.0,
            'psnr_b2a': 0.0
        }
        
        num_samples = 0
        
        with torch.no_grad():
            for real_a, real_b in zip(val_loader_a, val_loader_b):
                real_a = real_a.to(self.device)
                real_b = real_b.to(self.device)
                
                # Generate fake images
                fake_b = self.model.generate_a2b(real_a)
                fake_a = self.model.generate_b2a(real_b)
                
                # Compute metrics (simplified)
                metrics['mse_a2b'] += torch.nn.functional.mse_loss(fake_b, real_b).item()
                metrics['mse_b2a'] += torch.nn.functional.mse_loss(fake_a, real_a).item()
                
                num_samples += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_samples
        
        # Save evaluation results
        eval_results_path = output_dir / 'evaluation_results.json'
        with open(eval_results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        evaluation_results = {
            'metrics': metrics,
            'num_samples': num_samples,
            'results_file': str(eval_results_path)
        }
        
        logger.info("Model evaluation completed successfully")
        return evaluation_results
    
    def _export_model(self, output_dir: Path, dry_run: bool) -> Dict[str, Any]:
        """Export trained model for inference."""
        logger.info("Exporting trained model")
        
        if dry_run:
            logger.info("Would export trained model")
            return {}
        
        # Export model
        export_path = output_dir / 'exported_model.pth'
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.config.get('model', {}),
            'training_config': self.config.get('training', {})
        }, export_path)
        
        export_results = {
            'export_path': str(export_path),
            'model_info': self.model.get_model_info()
        }
        
        logger.info("Model export completed successfully")
        return export_results
    
    def _generate_final_report(
        self,
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        export_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final pipeline report."""
        logger.info("Generating final report")
        
        report = {
            'pipeline_summary': {
                'training_status': 'completed' if training_results else 'pending',
                'evaluation_status': 'completed' if evaluation_results else 'pending',
                'export_status': 'completed' if export_results else 'pending',
                'timestamp': time.time()
            },
            'training_summary': training_results,
            'evaluation_summary': evaluation_results,
            'export_summary': export_results
        }
        
        return report


def main():
    """Main function for model development pipeline."""
    parser = argparse.ArgumentParser(
        description='NeuroScope Model Development Pipeline'
    )
    
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help='Root directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for model outputs'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Training configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=Path,
        help='Checkpoint to resume from'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print planned actions without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_training_config()
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Invalid configuration")
        return
    
    # Initialize and run pipeline
    pipeline = ModelDevelopmentPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline(
            data_root=args.data_root,
            output_dir=args.output_dir,
            resume_checkpoint=args.resume,
            dry_run=args.dry_run
        )
        
        logger.info("Model development pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()