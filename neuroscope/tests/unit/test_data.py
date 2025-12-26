"""
Data Pipeline Tests.

Unit tests for datasets, transforms, and data loading.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestTransforms:
    """Test data transforms."""
    
    def test_normalize_transform(self):
        """Test normalization transform."""
        from ..data.transforms import NormalizeIntensity
        
        transform = NormalizeIntensity()
        x = np.random.randn(4, 64, 64, 64).astype(np.float32) * 100
        
        y = transform(x)
        
        # Should be normalized per channel
        for c in range(4):
            assert abs(y[c].mean()) < 1.0
            assert y[c].std() < 5.0
    
    def test_random_crop_shape(self):
        """Test random crop output shape."""
        from ..data.transforms import RandomCrop3D
        
        transform = RandomCrop3D(size=(32, 32, 32))
        x = np.random.randn(4, 64, 64, 64).astype(np.float32)
        
        y = transform(x)
        
        assert y.shape == (4, 32, 32, 32)
    
    def test_random_flip(self):
        """Test random flip is consistent across channels."""
        from ..data.transforms import RandomFlip3D
        
        transform = RandomFlip3D(p=1.0, axis=0)
        x = np.random.randn(4, 16, 16, 16).astype(np.float32)
        
        y = transform(x)
        
        # All channels should be flipped consistently
        assert np.allclose(y[0], np.flip(x[0], axis=0))
        assert np.allclose(y[1], np.flip(x[1], axis=0))
    
    def test_compose_transforms(self):
        """Test transform composition."""
        from ..data.transforms import Compose, NormalizeIntensity, ToTensor
        
        transforms = Compose([
            NormalizeIntensity(),
            ToTensor()
        ])
        
        x = np.random.randn(4, 32, 32, 32).astype(np.float32)
        y = transforms(x)
        
        assert isinstance(y, torch.Tensor)
        assert y.shape == (4, 32, 32, 32)
    
    def test_intensity_augmentation(self):
        """Test intensity augmentation."""
        from ..data.transforms import IntensityAugmentation
        
        transform = IntensityAugmentation(
            brightness_range=(-0.1, 0.1),
            contrast_range=(0.9, 1.1)
        )
        
        x = np.random.randn(4, 32, 32, 32).astype(np.float32)
        y = transform(x)
        
        # Shape should be preserved
        assert y.shape == x.shape
        
        # Values should be different (with high probability)
        assert not np.allclose(x, y)


class TestDatasets:
    """Test dataset classes."""
    
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Create mock data directory."""
        import nibabel as nib
        
        data_dir = tmp_path / "mock_data"
        data_dir.mkdir()
        
        # Create mock NIfTI files
        for i in range(5):
            subject_dir = data_dir / f"subject_{i:03d}"
            subject_dir.mkdir()
            
            for modality in ['t1', 't1ce', 't2', 'flair']:
                data = np.random.randn(64, 64, 64).astype(np.float32)
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, subject_dir / f"{modality}.nii.gz")
        
        return data_dir
    
    def test_dataset_length(self, mock_data_dir):
        """Test dataset length."""
        from ..data.datasets import MRIDataset
        
        dataset = MRIDataset(
            root_dir=str(mock_data_dir),
            modalities=['t1', 't1ce', 't2', 'flair']
        )
        
        assert len(dataset) == 5
    
    def test_dataset_getitem(self, mock_data_dir):
        """Test dataset __getitem__."""
        from ..data.datasets import MRIDataset
        
        dataset = MRIDataset(
            root_dir=str(mock_data_dir),
            modalities=['t1', 't1ce', 't2', 'flair']
        )
        
        sample = dataset[0]
        
        assert 'image' in sample
        assert sample['image'].shape[0] == 4  # 4 modalities
    
    def test_paired_dataset(self, mock_data_dir):
        """Test paired dataset for unpaired training."""
        from ..data.datasets import UnpairedMRIDataset
        
        dataset = UnpairedMRIDataset(
            source_dir=str(mock_data_dir),
            target_dir=str(mock_data_dir)
        )
        
        sample = dataset[0]
        
        assert 'source' in sample
        assert 'target' in sample


class TestDataLoaders:
    """Test data loaders."""
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        from ..data.loaders import create_dataloaders
        
        # This will use mock data
        # In real tests, would need actual data path
        pass  # Placeholder for actual implementation
    
    def test_batch_collation(self):
        """Test batch collation."""
        from ..data.loaders import collate_fn
        
        batch = [
            {'image': torch.randn(4, 32, 32, 32), 'id': 'sub1'},
            {'image': torch.randn(4, 32, 32, 32), 'id': 'sub2'},
        ]
        
        collated = collate_fn(batch)
        
        assert collated['image'].shape == (2, 4, 32, 32, 32)
        assert len(collated['id']) == 2


class TestSamplers:
    """Test data samplers."""
    
    def test_balanced_sampler(self):
        """Test balanced batch sampler."""
        from ..data.samplers import BalancedBatchSampler
        
        # Create mock dataset indices
        source_indices = list(range(100))
        target_indices = list(range(100, 200))
        
        sampler = BalancedBatchSampler(
            source_indices=source_indices,
            target_indices=target_indices,
            batch_size=4
        )
        
        batch = next(iter(sampler))
        
        # Should have equal source and target
        source_count = sum(1 for i in batch if i < 100)
        target_count = sum(1 for i in batch if i >= 100)
        
        assert source_count == target_count


class TestDataPipeline:
    """Integration tests for data pipeline."""
    
    @pytest.mark.integration
    def test_full_pipeline(self, mock_data_dir):
        """Test complete data pipeline."""
        from ..data.datasets import MRIDataset
        from ..data.transforms import Compose, NormalizeIntensity, ToTensor
        from torch.utils.data import DataLoader
        
        transforms = Compose([
            NormalizeIntensity(),
            ToTensor()
        ])
        
        dataset = MRIDataset(
            root_dir=str(mock_data_dir),
            modalities=['t1', 't1ce', 't2', 'flair'],
            transform=transforms
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        assert batch['image'].shape[0] == 2
        assert batch['image'].dtype == torch.float32
