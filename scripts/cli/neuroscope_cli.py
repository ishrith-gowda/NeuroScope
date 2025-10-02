"""Main CLI interface for NeuroScope.

This module provides a comprehensive command-line interface for all
NeuroScope functionality including preprocessing, training, and evaluation.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from neuroscope.core.logging import get_logger, configure_logging
from neuroscope.config import (
    get_default_training_config,
    get_default_preprocessing_config,
    get_default_evaluation_config,
    validate_config
)

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='neuroscope',
        description='NeuroScope: Domain-aware standardization of multimodal glioma MRI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  neuroscope preprocess --input-dir /path/to/raw --output-dir /path/to/processed
  
  # Train CycleGAN model
  neuroscope train --config config.json --data-root /path/to/data
  
  # Evaluate model
  neuroscope evaluate --model-path /path/to/model --data-path /path/to/test/data
  
  # Run full pipeline
  neuroscope pipeline --input-dir /path/to/raw --output-dir /path/to/results
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--log-dir',
        type=Path,
        help='Directory for log files'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    # Preprocessing command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess medical imaging data'
    )
    _add_preprocess_args(preprocess_parser)
    
    # Training command
    train_parser = subparsers.add_parser(
        'train',
        help='Train CycleGAN model'
    )
    _add_train_args(train_parser)
    
    # Evaluation command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained model'
    )
    _add_evaluate_args(eval_parser)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Run complete preprocessing and training pipeline'
    )
    _add_pipeline_args(pipeline_parser)
    
    # Configuration command
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management'
    )
    _add_config_args(config_parser)
    
    return parser


def _add_preprocess_args(parser: argparse.ArgumentParser):
    """Add preprocessing arguments to parser."""
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Preprocessing configuration file'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=Path,
        help='Metadata file path'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of existing files'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print planned actions without executing'
    )


def _add_train_args(parser: argparse.ArgumentParser):
    """Add training arguments to parser."""
    parser.add_argument(
        '--data-root',
        type=Path,
        required=True,
        help='Root directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Training configuration file'
    )
    
    parser.add_argument(
        '--metadata-json',
        type=Path,
        help='Metadata JSON file'
    )
    
    parser.add_argument(
        '--n-epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        help='Directory for model checkpoints'
    )
    
    parser.add_argument(
        '--resume',
        type=Path,
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'mps'],
        help='Training device'
    )


def _add_evaluate_args(parser: argparse.ArgumentParser):
    """Add evaluation arguments to parser."""
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--data-path',
        type=Path,
        required=True,
        help='Path to test data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['mse', 'mae', 'ssim', 'psnr'],
        help='Evaluation metrics to compute'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Evaluation batch size'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )


def _add_pipeline_args(parser: argparse.ArgumentParser):
    """Add pipeline arguments to parser."""
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory containing raw data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Pipeline configuration file'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step'
    )


def _add_config_args(parser: argparse.ArgumentParser):
    """Add configuration arguments to parser."""
    parser.add_argument(
        '--generate',
        choices=['training', 'preprocessing', 'evaluation', 'all'],
        help='Generate default configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for generated configuration'
    )
    
    parser.add_argument(
        '--validate',
        type=Path,
        help='Validate configuration file'
    )


def preprocess_command(args: argparse.Namespace):
    """Handle preprocessing command."""
    from neuroscope.preprocessing.normalization import VolumePreprocessor
    
    logger.info("Starting preprocessing pipeline")
    
    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = get_default_preprocessing_config()
    
    # Update config with command line arguments
    if args.max_workers:
        config['parallel_processing']['max_workers'] = args.max_workers
    
    # Initialize preprocessor
    preprocessor = VolumePreprocessor(
        preprocessing_steps=config.get('preprocessing_steps', [])
    )
    
    # Run preprocessing
    if args.dry_run:
        logger.info("Dry run mode - no files will be processed")
        logger.info(f"Would process files from: {args.input_dir}")
        logger.info(f"Would save results to: {args.output_dir}")
    else:
        results = preprocessor.batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern="*.nii.gz"
        )
        
        logger.info(f"Preprocessing completed. Processed {len(results)} files")


def train_command(args: argparse.Namespace):
    """Handle training command."""
    from neuroscope.training.trainers import CycleGANTrainer
    from neuroscope.models.architectures import CycleGAN
    from neuroscope.training.optimizers import CycleGANOptimizer
    
    logger.info("Starting CycleGAN training")
    
    # Load configuration
    if args.config:
        config = load_config_file(args.config)
    else:
        config = get_default_training_config()
    
    # Update config with command line arguments
    if args.n_epochs:
        config['training']['n_epochs'] = args.n_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['generator_optimizer']['lr'] = args.lr
        config['discriminator_optimizer']['lr'] = args.lr
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Invalid configuration")
        sys.exit(1)
    
    # Initialize model
    model = CycleGAN(**config['model'])
    
    # Initialize optimizer
    optimizer = CycleGANOptimizer(
        generators={'G_A2B': model.G_A2B, 'G_B2A': model.G_B2A},
        discriminators={'D_A': model.D_A, 'D_B': model.D_B},
        config=config
    )
    
    # Initialize trainer
    trainer = CycleGANTrainer(model, optimizer, device='cuda', config=config)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    logger.info("Training configuration validated and model initialized")


def evaluate_command(args: argparse.Namespace):
    """Handle evaluation command."""
    logger.info("Starting model evaluation")
    
    # Load model
    model = torch.load(args.model_path, map_location='cpu')
    
    logger.info(f"Loaded model from: {args.model_path}")
    logger.info(f"Evaluating on data from: {args.data_path}")


def pipeline_command(args: argparse.Namespace):
    """Handle pipeline command."""
    logger.info("Starting complete NeuroScope pipeline")
    
    # Run preprocessing
    if not args.skip_preprocessing:
        logger.info("Running preprocessing step")
        preprocess_args = argparse.Namespace(
            input_dir=args.input_dir,
            output_dir=args.output_dir / 'preprocessed',
            config=args.config,
            max_workers=4,
            force=False,
            dry_run=False
        )
        preprocess_command(preprocess_args)
    
    # Run training
    if not args.skip_training:
        logger.info("Running training step")
        train_args = argparse.Namespace(
            data_root=args.output_dir / 'preprocessed',
            config=args.config,
            n_epochs=100,
            batch_size=8,
            lr=0.0002,
            checkpoint_dir=args.output_dir / 'checkpoints',
            resume=None,
            device='cuda'
        )
        train_command(train_args)
    
    # Run evaluation
    if not args.skip_evaluation:
        logger.info("Running evaluation step")
        eval_args = argparse.Namespace(
            model_path=args.output_dir / 'checkpoints' / 'best_model.pth',
            data_path=args.output_dir / 'preprocessed',
            output_dir=args.output_dir / 'evaluation',
            metrics=['mse', 'mae', 'ssim', 'psnr'],
            batch_size=8,
            visualize=True
        )
        evaluate_command(eval_args)
    
    logger.info("Pipeline completed successfully")


def config_command(args: argparse.Namespace):
    """Handle configuration command."""
    import json
    
    if args.generate:
        logger.info(f"Generating {args.generate} configuration")
        
        if args.generate == 'training':
            config = get_default_training_config()
        elif args.generate == 'preprocessing':
            config = get_default_preprocessing_config()
        elif args.generate == 'evaluation':
            config = get_default_evaluation_config()
        elif args.generate == 'all':
            config = {
                'training': get_default_training_config(),
                'preprocessing': get_default_preprocessing_config(),
                'evaluation': get_default_evaluation_config()
            }
        
        # Save configuration
        output_file = args.output or f"{args.generate}_config.json"
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {output_file}")
    
    elif args.validate:
        logger.info(f"Validating configuration: {args.validate}")
        
        try:
            config = load_config_file(args.validate)
            if validate_config(config):
                logger.info("Configuration is valid")
            else:
                logger.error("Configuration validation failed")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            sys.exit(1)


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(args, 'log_level', 'INFO')
    configure_logging(
        level=getattr(logging, log_level),
        log_dir=getattr(args, 'log_dir', None)
    )
    
    # Route to appropriate command handler
    command_handlers = {
        'preprocess': preprocess_command,
        'train': train_command,
        'evaluate': evaluate_command,
        'pipeline': pipeline_command,
        'config': config_command
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        try:
            handler(args)
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()