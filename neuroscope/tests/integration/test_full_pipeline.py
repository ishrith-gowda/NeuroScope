"""Integration tests for complete NeuroScope pipeline.

This module provides comprehensive integration tests for the entire
NeuroScope pipeline including preprocessing, training, and evaluation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import json

from neuroscope.core.logging import configure_logging
from neuroscope.config import get_default_training_config, get_default_preprocessing_config
from neuroscope.preprocessing.normalization import VolumePreprocessor
from neuroscope.models.architectures import CycleGAN
from neuroscope.training.trainers import CycleGANTrainer
from neuroscope.training.optimizers import CycleGANOptimizer


class TestFullPipeline:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """Create sample data for testing."""
        # Create sample volumes
        volume_shape = (32, 32, 32)
        
        # Domain A volumes (BraTS-like)
        domain_a_dir = temp_dir / 'domain_a'
        domain_a_dir.mkdir(parents=True)
        
        for i in range(5):
            volume = np.random.rand(*volume_shape) * 100 + 50
            np.save(domain_a_dir / f'volume_{i:03d}.npy', volume)
        
        # Domain B volumes (UPenn-like)
        domain_b_dir = temp_dir / 'domain_b'
        domain_b_dir.mkdir(parents=True)
        
        for i in range(5):
            volume = np.random.rand(*volume_shape) * 80 + 40
            np.save(domain_b_dir / f'volume_{i:03d}.npy', volume)
        
        # Create metadata
        metadata = {
            'dataset_info': {
                'name': 'Test Dataset',
                'modalities': ['T1', 'T1ce', 'T2', 'FLAIR'],
                'total_subjects': 5
            },
            'subjects': {}
        }
        
        for i in range(5):
            metadata['subjects'][f'subject_{i:03d}'] = {
                'domain': 'A' if i < 3 else 'B',
                'split': 'train' if i < 4 else 'val',
                'files': {
                    'T1': str(domain_a_dir / f'volume_{i:03d}.npy') if i < 3 else str(domain_b_dir / f'volume_{i:03d}.npy')
                }
            }
        
        metadata_file = temp_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'domain_a_dir': domain_a_dir,
            'domain_b_dir': domain_b_dir,
            'metadata_file': metadata_file,
            'metadata': metadata
        }
    
    def test_preprocessing_pipeline(self, temp_dir, sample_data):
        """Test complete preprocessing pipeline."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Get preprocessing configuration
        config = get_default_preprocessing_config()
        
        # Initialize preprocessor
        preprocessor = VolumePreprocessor([
            ('min_max_normalization', {'target_range': (0, 1)}),
            ('percentile_normalization', {'low_percentile': 1.0, 'high_percentile': 99.0})
        ])
        
        # Process volumes
        output_dir = temp_dir / 'preprocessed'
        results = preprocessor.batch_process(
            input_dir=sample_data['domain_a_dir'],
            output_dir=output_dir,
            file_pattern="*.npy"
        )
        
        # Verify results
        assert len(results) == 5
        assert output_dir.exists()
        
        # Check processed files
        processed_files = list(output_dir.glob('*.npy'))
        assert len(processed_files) == 5
        
        # Verify file contents
        for file_path in processed_files:
            volume = np.load(file_path)
            assert volume.shape == (32, 32, 32)
            assert np.min(volume) >= 0.0
            assert np.max(volume) <= 1.0
    
    def test_model_initialization(self):
        """Test model initialization and basic functionality."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,  # Smaller for testing
            discriminator_channels=32,
            n_residual_blocks=3,  # Fewer blocks for testing
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # Test model info
        model_info = model.get_model_info()
        assert 'total_parameters' in model_info
        assert model_info['total_parameters'] > 0
        
        # Test forward pass
        batch_size = 2
        real_a = torch.randn(batch_size, 4, 64, 64)
        real_b = torch.randn(batch_size, 4, 64, 64)
        
        outputs = model(real_a, real_b)
        
        # Verify outputs
        assert 'fake_a' in outputs
        assert 'fake_b' in outputs
        assert 'rec_a' in outputs
        assert 'rec_b' in outputs
        assert 'id_a' in outputs
        assert 'id_b' in outputs
        
        # Check output shapes
        assert outputs['fake_a'].shape == real_a.shape
        assert outputs['fake_b'].shape == real_b.shape
        assert outputs['rec_a'].shape == real_a.shape
        assert outputs['rec_b'].shape == real_b.shape
    
    def test_training_setup(self):
        """Test training setup and basic training step."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # Initialize optimizer
        optimizer = CycleGANOptimizer(
            generators={'G_A2B': model.G_A2B, 'G_B2A': model.G_B2A},
            discriminators={'D_A': model.D_A, 'D_B': model.D_B},
            config=get_default_training_config()
        )
        
        # Initialize trainer
        trainer = CycleGANTrainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),  # Use CPU for testing
            config=get_default_training_config()
        )
        
        # Create dummy data loaders
        class DummyDataLoader:
            def __init__(self, batch_size=2):
                self.batch_size = batch_size
            
            def __iter__(self):
                for _ in range(3):  # 3 batches
                    yield torch.randn(self.batch_size, 4, 64, 64)
            
            def __len__(self):
                return 3
        
        train_loader_a = DummyDataLoader()
        train_loader_b = DummyDataLoader()
        
        # Test training step
        epoch_losses = trainer.train_epoch(train_loader_a, train_loader_b, epoch=0)
        
        # Verify losses
        assert 'G_A2B' in epoch_losses
        assert 'G_B2A' in epoch_losses
        assert 'D_A' in epoch_losses
        assert 'D_B' in epoch_losses
        assert 'cycle_A' in epoch_losses
        assert 'cycle_B' in epoch_losses
        assert 'identity_A' in epoch_losses
        assert 'identity_B' in epoch_losses
        
        # Check that losses are finite
        for loss_name, loss_value in epoch_losses.items():
            assert torch.isfinite(torch.tensor(loss_value)), f"Loss {loss_name} is not finite: {loss_value}"
    
    def test_loss_computation(self):
        """Test loss computation functionality."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # Create test data
        batch_size = 2
        real_a = torch.randn(batch_size, 4, 64, 64)
        real_b = torch.randn(batch_size, 4, 64, 64)
        
        # Forward pass
        outputs = model(real_a, real_b)
        
        # Test generator losses
        g_losses = model.compute_generator_losses(
            real_a, real_b,
            outputs['fake_a'], outputs['fake_b'],
            outputs['rec_a'], outputs['rec_b'],
            outputs['id_a'], outputs['id_b']
        )
        
        # Verify generator losses
        assert 'G_A2B' in g_losses
        assert 'G_B2A' in g_losses
        assert 'total' in g_losses
        
        # Test discriminator losses
        d_losses = model.compute_discriminator_losses(
            real_a, real_b,
            outputs['fake_a'], outputs['fake_b']
        )
        
        # Verify discriminator losses
        assert 'D_A' in d_losses
        assert 'D_B' in d_losses
        assert 'total' in d_losses
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = get_default_training_config()
        from neuroscope.config import validate_config
        assert validate_config(valid_config)
        
        # Test invalid configuration
        invalid_config = {'model': {}}  # Missing required keys
        assert not validate_config(invalid_config)
    
    def test_model_export_import(self, temp_dir):
        """Test model export and import functionality."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # Export model
        export_path = temp_dir / 'exported_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_channels': 4,
                'output_channels': 4,
                'generator_channels': 32,
                'discriminator_channels': 32,
                'n_residual_blocks': 3,
                'lambda_cycle': 10.0,
                'lambda_identity': 5.0
            }
        }, export_path)
        
        # Import model
        checkpoint = torch.load(export_path, map_location='cpu')
        new_model = CycleGAN(**checkpoint['model_config'])
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test that models produce same output
        test_input = torch.randn(1, 4, 64, 64)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model.generate_a2b(test_input)
            output2 = new_model.generate_a2b(test_input)
        
        # Verify outputs are similar (allowing for small numerical differences)
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Configure logging
        configure_logging(level='WARNING')
        
        # Test invalid model configuration
        with pytest.raises(Exception):
            model = CycleGAN(
                input_channels=0,  # Invalid
                output_channels=4,
                generator_channels=32,
                discriminator_channels=32,
                n_residual_blocks=3
            )
        
        # Test invalid preprocessing step
        preprocessor = VolumePreprocessor([
            ('invalid_step', {})  # Invalid step name
        ])
        
        test_volume = np.random.rand(32, 32, 32)
        
        # Should not raise exception, but should log warning
        processed = preprocessor.preprocess(test_volume)
        assert processed.shape == test_volume.shape


if __name__ == '__main__':
    pytest.main([__file__])