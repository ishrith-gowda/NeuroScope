"""Dataset class for CycleGAN training."""

import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class DomainSliceDataset(Dataset):
    """Dataset for loading 2D slices from 3D MRI volumes for a specific domain.
    
    This dataset loads 2D slices from 3D MRI volumes for CycleGAN training.
    It can randomly sample slices from each volume or use a fixed slice index.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        domain_paths: List[str],
        slice_dim: int = 2,
        transform: Optional[Callable] = None,
        slices_per_volume: int = 5,
        seed: int = 42,
        cache_mode: str = "none",
    ):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory of dataset.
            domain_paths: List of relative paths to domain volumes.
            slice_dim: Dimension to extract slices from (0=sagittal, 1=coronal, 2=axial).
            transform: Optional transform to apply to slices.
            slices_per_volume: Number of slices to sample from each volume.
            seed: Random seed for reproducibility.
            cache_mode: Caching mode ("none", "slices", "volumes").
        """
        self.root_dir = Path(root_dir)
        self.domain_paths = domain_paths
        self.slice_dim = slice_dim
        self.transform = transform
        self.slices_per_volume = slices_per_volume
        self.seed = seed
        self.cache_mode = cache_mode.lower()
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # List of full paths to volumes
        self.volume_paths = [self.root_dir / path for path in domain_paths]
        
        # Validate paths
        valid_paths = []
        for path in self.volume_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Path not found: {path}")
        
        self.volume_paths = valid_paths
        logger.info(f"Found {len(self.volume_paths)} valid volume paths")
        
        # Initialize cache
        self.cache = {}
        
        # Pre-load volumes or slices if caching is enabled
        if self.cache_mode != "none":
            self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache with volumes or slices."""
        if self.cache_mode == "volumes":
            logger.info("Caching entire volumes...")
            for path in self.volume_paths:
                self.cache[str(path)] = self._load_volume(path)
            
        elif self.cache_mode == "slices":
            logger.info("Caching sampled slices...")
            # Sample slices for each volume
            for path in self.volume_paths:
                volume = self._load_volume(path)
                # Get slice indices
                slice_indices = self._get_slice_indices(volume)
                # Cache slices
                self.cache[str(path)] = [
                    self._extract_slice(volume, idx) for idx in slice_indices
                ]
    
    def _load_volume(self, path: Path) -> torch.Tensor:
        """Load a volume from disk.
        
        Args:
            path: Path to volume file.
            
        Returns:
            Volume tensor.
        """
        # Load volume using appropriate library based on file extension
        if path.suffix in [".nii", ".nii.gz"]:
            try:
                import nibabel as nib
                volume = nib.load(path).get_fdata()
                return torch.from_numpy(volume).float()
            except ImportError:
                logger.error("nibabel not found, cannot load NIfTI files")
                raise
        elif path.suffix in [".npy"]:
            return torch.from_numpy(np.load(path)).float()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _get_slice_indices(self, volume: torch.Tensor) -> List[int]:
        """Get random slice indices.
        
        Args:
            volume: Volume tensor.
            
        Returns:
            List of slice indices.
        """
        # Get volume dimensions
        depth = volume.shape[self.slice_dim]
        
        # Generate random slice indices
        if depth <= self.slices_per_volume:
            # If volume has fewer slices than requested, use all slices
            return list(range(depth))
        else:
            # Otherwise, randomly sample slices
            return sorted(random.sample(range(depth), self.slices_per_volume))
    
    def _extract_slice(self, volume: torch.Tensor, idx: int) -> torch.Tensor:
        """Extract a slice from a volume.
        
        Args:
            volume: Volume tensor.
            idx: Slice index.
            
        Returns:
            Slice tensor.
        """
        # Extract slice based on slice dimension
        if self.slice_dim == 0:
            slice_tensor = volume[idx, :, :].clone()
        elif self.slice_dim == 1:
            slice_tensor = volume[:, idx, :].clone()
        else:  # self.slice_dim == 2
            slice_tensor = volume[:, :, idx].clone()
        
        # Ensure slice has 3 dimensions (add channel dimension)
        if slice_tensor.dim() == 2:
            slice_tensor = slice_tensor.unsqueeze(0)
        
        return slice_tensor
    
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of slices in dataset.
        """
        if self.cache_mode == "slices":
            # Count all cached slices
            return sum(len(slices) for slices in self.cache.values())
        else:
            # Estimate based on number of volumes and slices per volume
            return len(self.volume_paths) * self.slices_per_volume
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item at index.
        
        Args:
            idx: Index of item.
            
        Returns:
            Slice tensor.
        """
        if self.cache_mode == "slices":
            # Determine which volume and which slice within that volume
            volume_idx = 0
            remaining_idx = idx
            
            while volume_idx < len(self.volume_paths):
                path = str(self.volume_paths[volume_idx])
                slices = self.cache[path]
                
                if remaining_idx < len(slices):
                    # Found the right volume and slice
                    slice_tensor = slices[remaining_idx]
                    break
                
                remaining_idx -= len(slices)
                volume_idx += 1
            
            if volume_idx >= len(self.volume_paths):
                raise IndexError(f"Index {idx} out of range")
        else:
            # Determine which volume to use
            volume_idx = idx // self.slices_per_volume
            slice_idx_within_volume = idx % self.slices_per_volume
            
            if volume_idx >= len(self.volume_paths):
                # Handle edge case: wrap around to beginning of dataset
                volume_idx = volume_idx % len(self.volume_paths)
            
            path = self.volume_paths[volume_idx]
            
            # Load volume or get from cache
            if self.cache_mode == "volumes" and str(path) in self.cache:
                volume = self.cache[str(path)]
            else:
                volume = self._load_volume(path)
            
            # Get slice indices for this volume
            slice_indices = self._get_slice_indices(volume)
            
            # Get slice index within the selected volume
            if slice_idx_within_volume >= len(slice_indices):
                # Handle edge case: use the first slice index
                slice_idx = slice_indices[0]
            else:
                slice_idx = slice_indices[slice_idx_within_volume]
            
            # Extract slice
            slice_tensor = self._extract_slice(volume, slice_idx)
        
        # Apply transform if provided
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        
        return slice_tensor


def get_cycle_domain_loaders(
    preprocessed_dir: Union[str, Path],
    metadata_json: Union[str, Path],
    batch_size: int = 4,
    num_workers: int = 4,
    slices_per_subject: int = 5,
    seed: int = 42,
    cache_mode: str = "none",
) -> Dict[str, DataLoader]:
    """Get data loaders for CycleGAN training.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data.
        metadata_json: Path to metadata JSON file.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        slices_per_subject: Number of slices to sample per subject.
        seed: Random seed for reproducibility.
        cache_mode: Caching mode ("none", "slices", "volumes").
        
    Returns:
        Dictionary of data loaders.
    """
    import json
    
    # Load metadata
    with open(metadata_json, "r") as f:
        metadata = json.load(f)
    
    # Get domain paths
    domain_a_paths = metadata.get("domain_a", [])
    domain_b_paths = metadata.get("domain_b", [])
    
    # Validate domain paths
    if not domain_a_paths:
        logger.error("No paths found for domain A")
    if not domain_b_paths:
        logger.error("No paths found for domain B")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - 0.5) * 2.0)  # Scale from [0, 1] to [-1, 1]
    ])
    
    # Create datasets
    train_dataset_A = DomainSliceDataset(
        root_dir=preprocessed_dir,
        domain_paths=domain_a_paths,
        transform=transform,
        slices_per_volume=slices_per_subject,
        seed=seed,
        cache_mode=cache_mode,
    )
    
    train_dataset_B = DomainSliceDataset(
        root_dir=preprocessed_dir,
        domain_paths=domain_b_paths,
        transform=transform,
        slices_per_volume=slices_per_subject,
        seed=seed,
        cache_mode=cache_mode,
    )
    
    # Create data loaders
    train_loader_A = DataLoader(
        train_dataset_A,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    train_loader_B = DataLoader(
        train_dataset_B,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return {
        "train_A": train_loader_A,
        "train_B": train_loader_B,
    }