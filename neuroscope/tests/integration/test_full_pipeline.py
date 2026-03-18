"""integration tests for complete neuroscope pipeline.

this module provides comprehensive integration tests for the entire
neuroscope pipeline including preprocessing, training, and evaluation.
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
    """integration tests for complete pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """create sample data for testing."""
        # create sample volumes
        volume_shape = (32, 32, 32)
        
        # domain a volumes (brats-like)
        domain_a_dir = temp_dir / 'domain_a'
        domain_a_dir.mkdir(parents=True)
        
        for i in range(5):
            volume = np.random.rand(*volume_shape) * 100 + 50
            np.save(domain_a_dir / f'volume_{i:03d}.npy', volume)
        
        # domain b volumes (upenn-like)
        domain_b_dir = temp_dir / 'domain_b'
        domain_b_dir.mkdir(parents=True)
        
        for i in range(5):
            volume = np.random.rand(*volume_shape) * 80 + 40
            np.save(domain_b_dir / f'volume_{i:03d}.npy', volume)
        
        # create metadata
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
        """test complete preprocessing pipeline."""
        # configure logging
        configure_logging(level='WARNING')
        
        # get preprocessing configuration
        config = get_default_preprocessing_config()
        
        # initialize preprocessor
        preprocessor = VolumePreprocessor([
            ('min_max_normalization', {'target_range': (0, 1)}),
            ('percentile_normalization', {'low_percentile': 1.0, 'high_percentile': 99.0})
        ])
        
        # process volumes
        output_dir = temp_dir / 'preprocessed'
        results = preprocessor.batch_process(
            input_dir=sample_data['domain_a_dir'],
            output_dir=output_dir,
            file_pattern="*.npy"
        )
        
        # verify results
        assert len(results) == 5
        assert output_dir.exists()
        
        # check processed files
        processed_files = list(output_dir.glob('*.npy'))
        assert len(processed_files) == 5
        
        # verify file contents
        for file_path in processed_files:
            volume = np.load(file_path)
            assert volume.shape == (32, 32, 32)
            assert np.min(volume) >= 0.0
            assert np.max(volume) <= 1.0
    
    def test_model_initialization(self):
        """test model initialization and basic functionality."""
        # configure logging
        configure_logging(level='WARNING')
        
        # initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,  # smaller for testing
            discriminator_channels=32,
            n_residual_blocks=3,  # fewer blocks for testing
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # test model info
        model_info = model.get_model_info()
        assert 'total_parameters' in model_info
        assert model_info['total_parameters'] > 0
        
        # test forward pass
        batch_size = 2
        real_a = torch.randn(batch_size, 4, 64, 64)
        real_b = torch.randn(batch_size, 4, 64, 64)
        
        outputs = model(real_a, real_b)
        
        # verify outputs
        assert 'fake_a' in outputs
        assert 'fake_b' in outputs
        assert 'rec_a' in outputs
        assert 'rec_b' in outputs
        assert 'id_a' in outputs
        assert 'id_b' in outputs
        
        # check output shapes
        assert outputs['fake_a'].shape == real_a.shape
        assert outputs['fake_b'].shape == real_b.shape
        assert outputs['rec_a'].shape == real_a.shape
        assert outputs['rec_b'].shape == real_b.shape
    
    def test_training_setup(self):
        """test training setup and basic training step."""
        # configure logging
        configure_logging(level='WARNING')
        
        # initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # initialize optimizer
        optimizer = CycleGANOptimizer(
            generators={'G_A2B': model.G_A2B, 'G_B2A': model.G_B2A},
            discriminators={'D_A': model.D_A, 'D_B': model.D_B},
            config=get_default_training_config()
        )
        
        # initialize trainer
        trainer = CycleGANTrainer(
            model=model,
            optimizer=optimizer,
            device=torch.device('cpu'),  # use cpu for testing
            config=get_default_training_config()
        )
        
        # create dummy data loaders
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
        
        # test training step
        epoch_losses = trainer.train_epoch(train_loader_a, train_loader_b, epoch=0)
        
        # verify losses
        assert 'G_A2B' in epoch_losses
        assert 'G_B2A' in epoch_losses
        assert 'D_A' in epoch_losses
        assert 'D_B' in epoch_losses
        assert 'cycle_A' in epoch_losses
        assert 'cycle_B' in epoch_losses
        assert 'identity_A' in epoch_losses
        assert 'identity_B' in epoch_losses
        
        # check that losses are finite
        for loss_name, loss_value in epoch_losses.items():
            assert torch.isfinite(torch.tensor(loss_value)), f"Loss {loss_name} is not finite: {loss_value}"
    
    def test_loss_computation(self):
        """test loss computation functionality."""
        # configure logging
        configure_logging(level='WARNING')
        
        # initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # create test data
        batch_size = 2
        real_a = torch.randn(batch_size, 4, 64, 64)
        real_b = torch.randn(batch_size, 4, 64, 64)
        
        # forward pass
        outputs = model(real_a, real_b)
        
        # test generator losses
        g_losses = model.compute_generator_losses(
            real_a, real_b,
            outputs['fake_a'], outputs['fake_b'],
            outputs['rec_a'], outputs['rec_b'],
            outputs['id_a'], outputs['id_b']
        )
        
        # verify generator losses
        assert 'G_A2B' in g_losses
        assert 'G_B2A' in g_losses
        assert 'total' in g_losses
        
        # test discriminator losses
        d_losses = model.compute_discriminator_losses(
            real_a, real_b,
            outputs['fake_a'], outputs['fake_b']
        )
        
        # verify discriminator losses
        assert 'D_A' in d_losses
        assert 'D_B' in d_losses
        assert 'total' in d_losses
    
    def test_configuration_validation(self):
        """test configuration validation."""
        # test valid configuration
        valid_config = get_default_training_config()
        from neuroscope.config import validate_config
        assert validate_config(valid_config)
        
        # test invalid configuration
        invalid_config = {'model': {}}  # missing required keys
        assert not validate_config(invalid_config)
    
    def test_model_export_import(self, temp_dir):
        """test model export and import functionality."""
        # configure logging
        configure_logging(level='WARNING')
        
        # initialize model
        model = CycleGAN(
            input_channels=4,
            output_channels=4,
            generator_channels=32,
            discriminator_channels=32,
            n_residual_blocks=3,
            lambda_cycle=10.0,
            lambda_identity=5.0
        )
        
        # export model
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
        
        # import model
        checkpoint = torch.load(export_path, map_location='cpu')
        new_model = CycleGAN(**checkpoint['model_config'])
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # test that models produce same output
        test_input = torch.randn(1, 4, 64, 64)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model.generate_a2b(test_input)
            output2 = new_model.generate_a2b(test_input)
        
        # verify outputs are similar (allowing for small numerical differences)
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_error_handling(self):
        """test error handling in various scenarios."""
        # configure logging
        configure_logging(level='WARNING')
        
        # test invalid model configuration
        with pytest.raises(Exception):
            model = CycleGAN(
                input_channels=0,  # invalid
                output_channels=4,
                generator_channels=32,
                discriminator_channels=32,
                n_residual_blocks=3
            )
        
        # test invalid preprocessing step
        preprocessor = VolumePreprocessor([
            ('invalid_step', {})  # invalid step name
        ])
        
        test_volume = np.random.rand(32, 32, 32)
        
        # should not raise exception, but should log warning
        processed = preprocessor.preprocess(test_volume)
        assert processed.shape == test_volume.shape


if __name__ == '__main__':
    pytest.main([__file__])