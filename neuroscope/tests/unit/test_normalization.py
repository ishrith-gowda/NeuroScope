"""unit tests for normalization modules.

this module provides comprehensive unit tests for the normalization
and preprocessing functionality.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from neuroscope.preprocessing.normalization import (
    VolumeNormalization,
    DataAugmentation,
    VolumePreprocessor
)


class TestVolumeNormalization:
    """test cases for volumenormalization class."""
    
    def test_min_max_normalization_basic(self):
        """test basic min-max normalization."""
        # create test volume
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        # normalize
        normalized = VolumeNormalization.min_max_normalization(volume)
        
        # check range
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        assert np.isclose(np.min(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.max(normalized), 1.0, atol=1e-6)
    
    def test_min_max_normalization_with_mask(self):
        """test min-max normalization with mask."""
        # create test volume and mask
        volume = np.random.rand(32, 32, 32) * 100 + 50
        mask = np.ones_like(volume)
        mask[:16] = 0  # zero out first half
        
        # normalize
        normalized = VolumeNormalization.min_max_normalization(volume, mask=mask)
        
        # check that normalization is based on masked region
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_min_max_normalization_custom_range(self):
        """test min-max normalization with custom target range."""
        volume = np.random.rand(16, 16, 16) * 100
        target_range = (-1, 1)
        
        normalized = VolumeNormalization.min_max_normalization(
            volume, target_range=target_range
        )
        
        assert np.min(normalized) >= target_range[0]
        assert np.max(normalized) <= target_range[1]
    
    def test_z_score_normalization(self):
        """test z-score normalization."""
        volume = np.random.randn(32, 32, 32) * 10 + 100
        
        normalized = VolumeNormalization.z_score_normalization(volume)
        
        # check that mean is approximately 0 and std is approximately 1
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-6)
    
    def test_percentile_normalization(self):
        """test percentile normalization."""
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        normalized = VolumeNormalization.percentile_normalization(
            volume, low_percentile=1.0, high_percentile=99.0
        )
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_histogram_equalization(self):
        """test histogram equalization."""
        # create volume with non-uniform distribution
        volume = np.zeros((32, 32, 32))
        volume[:16] = 0.2
        volume[16:] = 0.8
        
        equalized = VolumeNormalization.histogram_equalization(volume)
        
        # check that equalization improves distribution
        assert np.min(equalized) >= 0.0
        assert np.max(equalized) <= 1.0
    
    def test_white_stripe_normalization(self):
        """test white stripe normalization."""
        # create volume with white matter-like distribution
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        normalized = VolumeNormalization.white_stripe_normalization(volume)
        
        # check that normalization preserves structure
        assert normalized.shape == volume.shape
        assert not np.allclose(normalized, volume)  # should be different


class TestDataAugmentation:
    """test cases for dataaugmentation class."""
    
    def test_random_flip(self):
        """test random flip augmentation."""
        volume = np.random.rand(32, 32, 32)
        
        flipped = DataAugmentation.random_flip(volume, axes=[0], p=1.0)
        
        # check that flip occurred
        assert np.array_equal(flipped, np.flip(volume, axis=0))
    
    def test_random_crop(self):
        """test random crop augmentation."""
        volume = np.random.rand(64, 64, 64)
        crop_size = (32, 32, 32)
        
        cropped = DataAugmentation.random_crop(volume, crop_size)
        
        assert cropped.shape == crop_size
    
    def test_random_intensity_shift(self):
        """test random intensity shift augmentation."""
        volume = np.random.rand(32, 32, 32) * 100
        shift_range = (0.1, 0.2)
        
        shifted = DataAugmentation.random_intensity_shift(volume, shift_range)
        
        # check that shift was applied
        assert not np.allclose(shifted, volume)
        assert shifted.shape == volume.shape
    
    def test_random_intensity_scale(self):
        """test random intensity scale augmentation."""
        volume = np.random.rand(32, 32, 32) * 100
        scale_range = (0.8, 1.2)
        
        scaled = DataAugmentation.random_intensity_scale(volume, scale_range)
        
        # check that scaling was applied
        assert not np.allclose(scaled, volume)
        assert scaled.shape == volume.shape
    
    def test_random_noise_gaussian(self):
        """test gaussian noise augmentation."""
        volume = np.random.rand(32, 32, 32)
        
        noisy = DataAugmentation.random_noise(
            volume, noise_type='gaussian', params={'std_dev': 0.01}
        )
        
        # check that noise was added
        assert not np.allclose(noisy, volume)
        assert noisy.shape == volume.shape


class TestVolumePreprocessor:
    """test cases for volumepreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """test preprocessor initialization."""
        preprocessor = VolumePreprocessor()
        
        assert preprocessor.preprocessing_steps == []
    
    def test_add_step(self):
        """test adding preprocessing steps."""
        preprocessor = VolumePreprocessor()
        
        preprocessor.add_step('min_max_normalization', {'target_range': (0, 1)})
        
        assert len(preprocessor.preprocessing_steps) == 1
        assert preprocessor.preprocessing_steps[0][0] == 'min_max_normalization'
    
    def test_preprocess_pipeline(self):
        """test preprocessing pipeline."""
        # create test volume
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        # initialize preprocessor with normalization step
        preprocessor = VolumePreprocessor([
            ('min_max_normalization', {'target_range': (0, 1)})
        ])
        
        # process volume
        processed = preprocessor.preprocess(volume)
        
        # check results
        assert processed.shape == volume.shape
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
    
    def test_preprocess_with_mask(self):
        """test preprocessing with mask."""
        volume = np.random.rand(32, 32, 32) * 100 + 50
        mask = np.ones_like(volume)
        mask[:16] = 0
        
        preprocessor = VolumePreprocessor([
            ('min_max_normalization', {'target_range': (0, 1)})
        ])
        
        processed = preprocessor.preprocess(volume, mask=mask)
        
        assert processed.shape == volume.shape
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0


class TestIntegration:
    """integration tests for normalization modules."""
    
    def test_full_preprocessing_pipeline(self):
        """test complete preprocessing pipeline."""
        # create test volume
        volume = np.random.rand(64, 64, 64) * 100 + 50
        
        # define preprocessing steps
        steps = [
            ('min_max_normalization', {'target_range': (0, 1)}),
            ('percentile_normalization', {'low_percentile': 1.0, 'high_percentile': 99.0})
        ]
        
        # initialize preprocessor
        preprocessor = VolumePreprocessor(steps)
        
        # process volume
        processed = preprocessor.preprocess(volume)
        
        # check results
        assert processed.shape == volume.shape
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
    
    def test_augmentation_pipeline(self):
        """test data augmentation pipeline."""
        volume = np.random.rand(32, 32, 32) * 100
        
        # apply multiple augmentations
        augmented = DataAugmentation.random_flip(volume, p=1.0)
        augmented = DataAugmentation.random_intensity_shift(
            augmented, shift_range=(0.1, 0.2)
        )
        augmented = DataAugmentation.random_intensity_scale(
            augmented, scale_range=(0.8, 1.2)
        )
        
        # check that augmentations were applied
        assert augmented.shape == volume.shape
        assert not np.allclose(augmented, volume)


if __name__ == '__main__':
    pytest.main([__file__])