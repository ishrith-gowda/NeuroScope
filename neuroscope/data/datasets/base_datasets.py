"""
Dataset Classes for Medical Image Analysis.

Provides comprehensive dataset implementations for multi-modal
MRI data from BraTS, UPenn-GBM, and custom sources.
"""

from typing import List, Optional, Dict, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    root_dir: str
    modalities: List[str] = field(default_factory=lambda: ['t1', 't1ce', 't2', 'flair'])
    include_segmentation: bool = False
    slice_axis: int = 2  # 0=sagittal, 1=coronal, 2=axial
    slice_range: Optional[Tuple[int, int]] = None
    cache_data: bool = False
    preload: bool = False
    
    def __post_init__(self):
        self.root_dir = Path(self.root_dir)


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    Provides common functionality for medical image datasets.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        
        # Validate root directory
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
        
        # Initialize sample list
        self.samples = self._load_samples()
    
    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load list of samples. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_item(self, index: int) -> Dict[str, Any]:
        """Load a single item. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self._load_item(index)
        
        if self.transform is not None:
            item = self.transform(item)
        
        return item
    
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Get metadata for a sample without loading the full data."""
        return self.samples[index]


class PairedDataset(BaseDataset):
    """
    Dataset for paired image-to-image translation.
    
    Each sample contains corresponding source and target images.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        source_subdir: str = 'source',
        target_subdir: str = 'target',
        transform: Optional[Callable] = None,
        extensions: List[str] = None
    ):
        self.source_subdir = source_subdir
        self.target_subdir = target_subdir
        self.extensions = extensions or ['.nii', '.nii.gz', '.npy', '.npz']
        super().__init__(root_dir, transform)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        source_dir = self.root_dir / self.source_subdir
        target_dir = self.root_dir / self.target_subdir
        
        samples = []
        
        if source_dir.exists():
            for source_path in sorted(source_dir.iterdir()):
                if any(source_path.name.endswith(ext) for ext in self.extensions):
                    # Find corresponding target
                    target_path = target_dir / source_path.name
                    if target_path.exists():
                        samples.append({
                            'source': source_path,
                            'target': target_path,
                            'id': source_path.stem
                        })
        
        return samples
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        
        source = self._load_image(sample['source'])
        target = self._load_image(sample['target'])
        
        return {
            'source': source,
            'target': target,
            'id': sample['id']
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load image from various formats."""
        path = Path(path)
        
        if path.suffix in ['.npy']:
            return np.load(path)
        elif path.suffix in ['.npz']:
            return np.load(path)['arr_0']
        elif HAS_NIBABEL and path.suffix in ['.nii', '.gz']:
            return nib.load(str(path)).get_fdata()
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


class UnpairedDataset(BaseDataset):
    """
    Dataset for unpaired image-to-image translation.
    
    Source and target domains are loaded independently.
    """
    
    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        source_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: List[str] = None
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.extensions = extensions or ['.nii', '.nii.gz', '.npy', '.npz']
        
        # Use source_dir as root for base class
        super().__init__(source_dir, transform)
        
        # Load target samples separately
        self.target_samples = self._load_domain_samples(self.target_dir)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        return self._load_domain_samples(self.source_dir)
    
    def _load_domain_samples(self, domain_dir: Path) -> List[Dict[str, Any]]:
        samples = []
        
        if domain_dir.exists():
            for path in sorted(domain_dir.iterdir()):
                if any(path.name.endswith(ext) for ext in self.extensions):
                    samples.append({
                        'path': path,
                        'id': path.stem
                    })
        
        return samples
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        source_sample = self.samples[index]
        
        # Random target selection for unpaired training
        target_idx = random.randint(0, len(self.target_samples) - 1)
        target_sample = self.target_samples[target_idx]
        
        source = self._load_image(source_sample['path'])
        target = self._load_image(target_sample['path'])
        
        if self.source_transform:
            source = self.source_transform(source)
        if self.target_transform:
            target = self.target_transform(target)
        
        return {
            'source': source,
            'target': target,
            'source_id': source_sample['id'],
            'target_id': target_sample['id']
        }
    
    def _load_image(self, path: Path) -> np.ndarray:
        path = Path(path)
        
        if path.suffix in ['.npy']:
            return np.load(path)
        elif path.suffix in ['.npz']:
            return np.load(path)['arr_0']
        elif HAS_NIBABEL and path.suffix in ['.nii', '.gz']:
            return nib.load(str(path)).get_fdata()
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")


class BraTSDataset(BaseDataset):
    """
    BraTS (Brain Tumor Segmentation) Challenge Dataset.
    
    Supports BraTS 2020/2021 format with T1, T1ce, T2, FLAIR modalities.
    """
    
    MODALITY_SUFFIXES = {
        't1': '_t1.nii.gz',
        't1ce': '_t1ce.nii.gz',
        't2': '_t2.nii.gz',
        'flair': '_flair.nii.gz',
        'seg': '_seg.nii.gz'
    }
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: List[str] = None,
        include_segmentation: bool = False,
        transform: Optional[Callable] = None,
        slice_axis: int = 2,
        slice_range: Optional[Tuple[int, int]] = None,
        preload: bool = False,
        cache_slices: bool = False
    ):
        self.modalities = modalities or ['t1', 't1ce', 't2', 'flair']
        self.include_segmentation = include_segmentation
        self.slice_axis = slice_axis
        self.slice_range = slice_range or (30, 130)  # Default to central slices
        self.preload = preload
        self.cache_slices = cache_slices
        
        self._cache = {}
        
        super().__init__(root_dir, transform)
        
        if self.preload:
            self._preload_data()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        # Find all subject directories
        for subject_dir in sorted(self.root_dir.iterdir()):
            if subject_dir.is_dir() and subject_dir.name.startswith('BraTS'):
                subject_id = subject_dir.name
                
                # Verify all modalities exist
                modality_paths = {}
                valid = True
                
                for modality in self.modalities:
                    suffix = self.MODALITY_SUFFIXES.get(modality, f'_{modality}.nii.gz')
                    path = subject_dir / f"{subject_id}{suffix}"
                    
                    if path.exists():
                        modality_paths[modality] = path
                    else:
                        valid = False
                        break
                
                if valid:
                    # Add segmentation if requested
                    if self.include_segmentation:
                        seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
                        if seg_path.exists():
                            modality_paths['seg'] = seg_path
                    
                    # Create sample entries for each slice
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'slice_idx': slice_idx
                        })
        
        return samples
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        subject_id = sample['subject_id']
        slice_idx = sample['slice_idx']
        
        # Check cache
        cache_key = f"{subject_id}_{slice_idx}"
        if self.cache_slices and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load modalities
        modality_data = []
        for modality in self.modalities:
            path = sample['modality_paths'][modality]
            
            if HAS_NIBABEL:
                volume = nib.load(str(path)).get_fdata()
            else:
                raise RuntimeError("nibabel is required for loading NIfTI files")
            
            # Extract slice
            if self.slice_axis == 0:
                slice_data = volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = volume[:, slice_idx, :]
            else:
                slice_data = volume[:, :, slice_idx]
            
            modality_data.append(slice_data)
        
        # Stack modalities: [C, H, W]
        image = np.stack(modality_data, axis=0).astype(np.float32)
        
        item = {
            'image': torch.from_numpy(image),
            'subject_id': subject_id,
            'slice_idx': slice_idx
        }
        
        # Load segmentation if requested
        if self.include_segmentation and 'seg' in sample['modality_paths']:
            seg_path = sample['modality_paths']['seg']
            seg_volume = nib.load(str(seg_path)).get_fdata()
            
            if self.slice_axis == 0:
                seg_slice = seg_volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                seg_slice = seg_volume[:, slice_idx, :]
            else:
                seg_slice = seg_volume[:, :, slice_idx]
            
            item['segmentation'] = torch.from_numpy(seg_slice.astype(np.int64))
        
        if self.cache_slices:
            self._cache[cache_key] = item
        
        return item
    
    def _preload_data(self):
        """Preload all data into memory."""
        print(f"Preloading {len(self)} samples...")
        for i in range(len(self)):
            _ = self._load_item(i)
    
    def get_subject_ids(self) -> List[str]:
        """Get list of unique subject IDs."""
        return list(set(s['subject_id'] for s in self.samples))


class UPennGBMDataset(BaseDataset):
    """
    University of Pennsylvania GBM Dataset.
    
    Large-scale glioblastoma imaging dataset with multiple modalities.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: List[str] = None,
        transform: Optional[Callable] = None,
        slice_axis: int = 2,
        slice_range: Optional[Tuple[int, int]] = None,
        n4_corrected: bool = True
    ):
        self.modalities = modalities or ['t1', 't1ce', 't2', 'flair']
        self.slice_axis = slice_axis
        self.slice_range = slice_range or (30, 130)
        self.n4_corrected = n4_corrected
        
        super().__init__(root_dir, transform)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for subject_dir in sorted(self.root_dir.iterdir()):
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                modality_paths = {}
                valid = True
                
                for modality in self.modalities:
                    # Try different naming conventions
                    possible_names = [
                        f"{subject_id}_{modality}.nii.gz",
                        f"{modality}.nii.gz",
                        f"{subject_id}_{modality.upper()}.nii.gz"
                    ]
                    
                    found = False
                    for name in possible_names:
                        path = subject_dir / name
                        if path.exists():
                            modality_paths[modality] = path
                            found = True
                            break
                    
                    if not found:
                        valid = False
                        break
                
                if valid:
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'slice_idx': slice_idx
                        })
        
        return samples
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        
        modality_data = []
        for modality in self.modalities:
            path = sample['modality_paths'][modality]
            
            if HAS_NIBABEL:
                volume = nib.load(str(path)).get_fdata()
            else:
                raise RuntimeError("nibabel required")
            
            slice_idx = sample['slice_idx']
            if self.slice_axis == 0:
                slice_data = volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = volume[:, slice_idx, :]
            else:
                slice_data = volume[:, :, slice_idx]
            
            modality_data.append(slice_data)
        
        image = np.stack(modality_data, axis=0).astype(np.float32)
        
        return {
            'image': torch.from_numpy(image),
            'subject_id': sample['subject_id'],
            'slice_idx': sample['slice_idx'],
            'domain': 'upenn'
        }


class MultiModalMRIDataset(BaseDataset):
    """
    Generic multi-modal MRI dataset.
    
    Supports flexible directory structures and modality configurations.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modality_dirs: Optional[Dict[str, str]] = None,
        transform: Optional[Callable] = None,
        file_pattern: str = '*.npy',
        return_paths: bool = False
    ):
        self.modality_dirs = modality_dirs or {
            't1': 't1',
            't1ce': 't1ce',
            't2': 't2',
            'flair': 'flair'
        }
        self.file_pattern = file_pattern
        self.return_paths = return_paths
        
        super().__init__(root_dir, transform)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        # Get files from first modality
        first_modality = list(self.modality_dirs.keys())[0]
        first_dir = self.root_dir / self.modality_dirs[first_modality]
        
        if first_dir.exists():
            for path in sorted(first_dir.glob(self.file_pattern)):
                sample_id = path.stem
                modality_paths = {first_modality: path}
                
                # Find other modalities
                valid = True
                for modality, subdir in self.modality_dirs.items():
                    if modality != first_modality:
                        mod_path = self.root_dir / subdir / f"{sample_id}{path.suffix}"
                        if mod_path.exists():
                            modality_paths[modality] = mod_path
                        else:
                            valid = False
                            break
                
                if valid:
                    samples.append({
                        'id': sample_id,
                        'modality_paths': modality_paths
                    })
        
        return samples
    
    def _load_item(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        
        modality_data = []
        for modality in self.modality_dirs.keys():
            path = sample['modality_paths'][modality]
            data = np.load(path)
            modality_data.append(data)
        
        image = np.stack(modality_data, axis=0).astype(np.float32)
        
        item = {
            'image': torch.from_numpy(image),
            'id': sample['id']
        }
        
        if self.return_paths:
            item['paths'] = sample['modality_paths']
        
        return item


class DomainAdaptationDataset(Dataset):
    """
    Dataset for domain adaptation between two image domains.
    
    Wraps source and target datasets for training.
    """
    
    def __init__(
        self,
        source_dataset: Dataset,
        target_dataset: Dataset,
        paired: bool = False
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.paired = paired
        
        self.source_len = len(source_dataset)
        self.target_len = len(target_dataset)
    
    def __len__(self) -> int:
        return max(self.source_len, self.target_len)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        source_idx = index % self.source_len
        
        if self.paired:
            target_idx = source_idx % self.target_len
        else:
            target_idx = random.randint(0, self.target_len - 1)
        
        source_item = self.source_dataset[source_idx]
        target_item = self.target_dataset[target_idx]
        
        return {
            'source': source_item,
            'target': target_item
        }


class CycleGANDataset(Dataset):
    """
    Dataset specifically designed for CycleGAN training.
    
    Handles unpaired domain translation with proper sampling.
    """
    
    def __init__(
        self,
        domain_a_dataset: Dataset,
        domain_b_dataset: Dataset,
        transform_a: Optional[Callable] = None,
        transform_b: Optional[Callable] = None
    ):
        self.domain_a = domain_a_dataset
        self.domain_b = domain_b_dataset
        self.transform_a = transform_a
        self.transform_b = transform_b
        
        self.len_a = len(domain_a_dataset)
        self.len_b = len(domain_b_dataset)
    
    def __len__(self) -> int:
        return max(self.len_a, self.len_b)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        idx_a = index % self.len_a
        idx_b = random.randint(0, self.len_b - 1)
        
        item_a = self.domain_a[idx_a]
        item_b = self.domain_b[idx_b]
        
        # Extract image tensors
        if isinstance(item_a, dict):
            real_a = item_a.get('image', item_a.get('source'))
        else:
            real_a = item_a
        
        if isinstance(item_b, dict):
            real_b = item_b.get('image', item_b.get('target'))
        else:
            real_b = item_b
        
        # Apply transforms
        if self.transform_a:
            real_a = self.transform_a(real_a)
        if self.transform_b:
            real_b = self.transform_b(real_b)
        
        return {
            'real_A': real_a,
            'real_B': real_b,
            'idx_A': idx_a,
            'idx_B': idx_b
        }


def create_dataset(
    dataset_type: str,
    root_dir: Union[str, Path],
    **kwargs
) -> Dataset:
    """
    Factory function to create dataset by type.
    
    Args:
        dataset_type: Type of dataset ('brats', 'upenn', 'paired', 'unpaired')
        root_dir: Root directory for data
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dataset instance
    """
    datasets = {
        'brats': BraTSDataset,
        'upenn': UPennGBMDataset,
        'paired': PairedDataset,
        'unpaired': UnpairedDataset,
        'multimodal': MultiModalMRIDataset,
    }
    
    if dataset_type.lower() not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return datasets[dataset_type.lower()](root_dir, **kwargs)


def get_dataset_stats(dataset: Dataset, num_samples: int = 100) -> Dict[str, float]:
    """
    Compute dataset statistics (mean, std) from samples.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use
        
    Returns:
        Dictionary with mean and std per channel
    """
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    all_data = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, dict):
            data = item.get('image', item.get('source'))
        else:
            data = item
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        all_data.append(data)
    
    stacked = np.stack(all_data, axis=0)  # [N, C, H, W]
    
    mean = stacked.mean(axis=(0, 2, 3))
    std = stacked.std(axis=(0, 2, 3))
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'min': float(stacked.min()),
        'max': float(stacked.max())
    }
