"""Unit tests for normalization modules.

This module provides comprehensive unit tests for the normalization
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
    """Test cases for VolumeNormalization class."""
    
    def test_min_max_normalization_basic(self):
        """Test basic min-max normalization."""
        # Create test volume
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        # Normalize
        normalized = VolumeNormalization.min_max_normalization(volume)
        
        # Check range
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        assert np.isclose(np.min(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.max(normalized), 1.0, atol=1e-6)
    
    def test_min_max_normalization_with_mask(self):
        """Test min-max normalization with mask."""
        # Create test volume and mask
        volume = np.random.rand(32, 32, 32) * 100 + 50
        mask = np.ones_like(volume)
        mask[:16] = 0  # Zero out first half
        
        # Normalize
        normalized = VolumeNormalization.min_max_normalization(volume, mask=mask)
        
        # Check that normalization is based on masked region
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_min_max_normalization_custom_range(self):
        """Test min-max normalization with custom target range."""
        volume = np.random.rand(16, 16, 16) * 100
        target_range = (-1, 1)
        
        normalized = VolumeNormalization.min_max_normalization(
            volume, target_range=target_range
        )
        
        assert np.min(normalized) >= target_range[0]
        assert np.max(normalized) <= target_range[1]
    
    def test_z_score_normalization(self):
        """Test z-score normalization."""
        volume = np.random.randn(32, 32, 32) * 10 + 100
        
        normalized = VolumeNormalization.z_score_normalization(volume)
        
        # Check that mean is approximately 0 and std is approximately 1
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-6)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-6)
    
    def test_percentile_normalization(self):
        """Test percentile normalization."""
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        normalized = VolumeNormalization.percentile_normalization(
            volume, low_percentile=1.0, high_percentile=99.0
        )
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_histogram_equalization(self):
        """Test histogram equalization."""
        # Create volume with non-uniform distribution
        volume = np.zeros((32, 32, 32))
        volume[:16] = 0.2
        volume[16:] = 0.8
        
        equalized = VolumeNormalization.histogram_equalization(volume)
        
        # Check that equalization improves distribution
        assert np.min(equalized) >= 0.0
        assert np.max(equalized) <= 1.0
    
    def test_white_stripe_normalization(self):
        """Test white stripe normalization."""
        # Create volume with white matter-like distribution
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        normalized = VolumeNormalization.white_stripe_normalization(volume)
        
        # Check that normalization preserves structure
        assert normalized.shape == volume.shape
        assert not np.allclose(normalized, volume)  # Should be different


class TestDataAugmentation:
    """Test cases for DataAugmentation class."""
    
    def test_random_flip(self):
        """Test random flip augmentation."""
        volume = np.random.rand(32, 32, 32)
        
        flipped = DataAugmentation.random_flip(volume, axes=[0], p=1.0)
        
        # Check that flip occurred
        assert np.array_equal(flipped, np.flip(volume, axis=0))
    
    def test_random_crop(self):
        """Test random crop augmentation."""
        volume = np.random.rand(64, 64, 64)
        crop_size = (32, 32, 32)
        
        cropped = DataAugmentation.random_crop(volume, crop_size)
        
        assert cropped.shape == crop_size
    
    def test_random_intensity_shift(self):
        """Test random intensity shift augmentation."""
        volume = np.random.rand(32, 32, 32) * 100
        shift_range = (0.1, 0.2)
        
        shifted = DataAugmentation.random_intensity_shift(volume, shift_range)
        
        # Check that shift was applied
        assert not np.allclose(shifted, volume)
        assert shifted.shape == volume.shape
    
    def test_random_intensity_scale(self):
        """Test random intensity scale augmentation."""
        volume = np.random.rand(32, 32, 32) * 100
        scale_range = (0.8, 1.2)
        
        scaled = DataAugmentation.random_intensity_scale(volume, scale_range)
        
        # Check that scaling was applied
        assert not np.allclose(scaled, volume)
        assert scaled.shape == volume.shape
    
    def test_random_noise_gaussian(self):
        """Test Gaussian noise augmentation."""
        volume = np.random.rand(32, 32, 32)
        
        noisy = DataAugmentation.random_noise(
            volume, noise_type='gaussian', params={'std_dev': 0.01}
        )
        
        # Check that noise was added
        assert not np.allclose(noisy, volume)
        assert noisy.shape == volume.shape


class TestVolumePreprocessor:
    """Test cases for VolumePreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = VolumePreprocessor()
        
        assert preprocessor.preprocessing_steps == []
    
    def test_add_step(self):
        """Test adding preprocessing steps."""
        preprocessor = VolumePreprocessor()
        
        preprocessor.add_step('min_max_normalization', {'target_range': (0, 1)})
        
        assert len(preprocessor.preprocessing_steps) == 1
        assert preprocessor.preprocessing_steps[0][0] == 'min_max_normalization'
    
    def test_preprocess_pipeline(self):
        """Test preprocessing pipeline."""
        # Create test volume
        volume = np.random.rand(32, 32, 32) * 100 + 50
        
        # Initialize preprocessor with normalization step
        preprocessor = VolumePreprocessor([
            ('min_max_normalization', {'target_range': (0, 1)})
        ])
        
        # Process volume
        processed = preprocessor.preprocess(volume)
        
        # Check results
        assert processed.shape == volume.shape
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
    
    def test_preprocess_with_mask(self):
        """Test preprocessing with mask."""
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
    """Integration tests for normalization modules."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test volume
        volume = np.random.rand(64, 64, 64) * 100 + 50
        
        # Define preprocessing steps
        steps = [
            ('min_max_normalization', {'target_range': (0, 1)}),
            ('percentile_normalization', {'low_percentile': 1.0, 'high_percentile': 99.0})
        ]
        
        # Initialize preprocessor
        preprocessor = VolumePreprocessor(steps)
        
        # Process volume
        processed = preprocessor.preprocess(volume)
        
        # Check results
        assert processed.shape == volume.shape
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
    
    def test_augmentation_pipeline(self):
        """Test data augmentation pipeline."""
        volume = np.random.rand(32, 32, 32) * 100
        
        # Apply multiple augmentations
        augmented = DataAugmentation.random_flip(volume, p=1.0)
        augmented = DataAugmentation.random_intensity_shift(
            augmented, shift_range=(0.1, 0.2)
        )
        augmented = DataAugmentation.random_intensity_scale(
            augmented, scale_range=(0.8, 1.2)
        )
        
        # Check that augmentations were applied
        assert augmented.shape == volume.shape
        assert not np.allclose(augmented, volume)


if __name__ == '__main__':
    pytest.main([__file__])