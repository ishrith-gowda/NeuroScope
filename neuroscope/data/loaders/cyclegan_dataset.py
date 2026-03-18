"""dataset class for cyclegan training."""

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
    """dataset for loading 2d slices from 3d mri volumes for a specific domain.
    
    this dataset loads 2d slices from 3d mri volumes for cyclegan training.
    it can randomly sample slices from each volume or use a fixed slice index.
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
        """initialize dataset.
        
        args:
            root_dir: root directory of dataset.
            domain_paths: list of relative paths to domain volumes.
            slice_dim: dimension to extract slices from (0=sagittal, 1=coronal, 2=axial).
            transform: optional transform to apply to slices.
            slices_per_volume: number of slices to sample from each volume.
            seed: random seed for reproducibility.
            cache_mode: caching mode ("none", "slices", "volumes").
        """
        self.root_dir = Path(root_dir)
        self.domain_paths = domain_paths
        self.slice_dim = slice_dim
        self.transform = transform
        self.slices_per_volume = slices_per_volume
        self.seed = seed
        self.cache_mode = cache_mode.lower()
        
        # set random seed for reproducibility
        random.seed(seed)
        
        # list of full paths to volumes
        self.volume_paths = [self.root_dir / path for path in domain_paths]
        
        # validate paths
        valid_paths = []
        for path in self.volume_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Path not found: {path}")
        
        self.volume_paths = valid_paths
        logger.info(f"Found {len(self.volume_paths)} valid volume paths")
        
        # initialize cache
        self.cache = {}
        
        # pre-load volumes or slices if caching is enabled
        if self.cache_mode != "none":
            self._initialize_cache()
    
    def _initialize_cache(self):
        """initialize cache with volumes or slices."""
        if self.cache_mode == "volumes":
            logger.info("Caching entire volumes...")
            for path in self.volume_paths:
                self.cache[str(path)] = self._load_volume(path)
            
        elif self.cache_mode == "slices":
            logger.info("Caching sampled slices...")
            # sample slices for each volume
            for path in self.volume_paths:
                volume = self._load_volume(path)
                # get slice indices
                slice_indices = self._get_slice_indices(volume)
                # cache slices
                self.cache[str(path)] = [
                    self._extract_slice(volume, idx) for idx in slice_indices
                ]
    
    def _load_volume(self, path: Path) -> torch.Tensor:
        """load a volume from disk.
        
        args:
            path: path to volume file.
            
        returns:
            volume tensor.
        """
        # load volume using appropriate library based on file extension
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
        """get random slice indices.
        
        args:
            volume: volume tensor.
            
        returns:
            list of slice indices.
        """
        # get volume dimensions
        depth = volume.shape[self.slice_dim]
        
        # generate random slice indices
        if depth <= self.slices_per_volume:
            # if volume has fewer slices than requested, use all slices
            return list(range(depth))
        else:
            # otherwise, randomly sample slices
            return sorted(random.sample(range(depth), self.slices_per_volume))
    
    def _extract_slice(self, volume: torch.Tensor, idx: int) -> torch.Tensor:
        """extract a slice from a volume.
        
        args:
            volume: volume tensor.
            idx: slice index.
            
        returns:
            slice tensor.
        """
        # extract slice based on slice dimension
        if self.slice_dim == 0:
            slice_tensor = volume[idx, :, :].clone()
        elif self.slice_dim == 1:
            slice_tensor = volume[:, idx, :].clone()
        else:  # self.slice_dim == 2
            slice_tensor = volume[:, :, idx].clone()
        
        # ensure slice has 3 dimensions (add channel dimension)
        if slice_tensor.dim() == 2:
            slice_tensor = slice_tensor.unsqueeze(0)
        
        return slice_tensor
    
    def __len__(self) -> int:
        """get dataset length.
        
        returns:
            number of slices in dataset.
        """
        if self.cache_mode == "slices":
            # count all cached slices
            return sum(len(slices) for slices in self.cache.values())
        else:
            # estimate based on number of volumes and slices per volume
            return len(self.volume_paths) * self.slices_per_volume
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """get item at index.
        
        args:
            idx: index of item.
            
        returns:
            slice tensor.
        """
        if self.cache_mode == "slices":
            # determine which volume and which slice within that volume
            volume_idx = 0
            remaining_idx = idx
            
            while volume_idx < len(self.volume_paths):
                path = str(self.volume_paths[volume_idx])
                slices = self.cache[path]
                
                if remaining_idx < len(slices):
                    # found the right volume and slice
                    slice_tensor = slices[remaining_idx]
                    break
                
                remaining_idx -= len(slices)
                volume_idx += 1
            
            if volume_idx >= len(self.volume_paths):
                raise IndexError(f"Index {idx} out of range")
        else:
            # determine which volume to use
            volume_idx = idx // self.slices_per_volume
            slice_idx_within_volume = idx % self.slices_per_volume
            
            if volume_idx >= len(self.volume_paths):
                # handle edge case: wrap around to beginning of dataset
                volume_idx = volume_idx % len(self.volume_paths)
            
            path = self.volume_paths[volume_idx]
            
            # load volume or get from cache
            if self.cache_mode == "volumes" and str(path) in self.cache:
                volume = self.cache[str(path)]
            else:
                volume = self._load_volume(path)
            
            # get slice indices for this volume
            slice_indices = self._get_slice_indices(volume)
            
            # get slice index within the selected volume
            if slice_idx_within_volume >= len(slice_indices):
                # handle edge case: use the first slice index
                slice_idx = slice_indices[0]
            else:
                slice_idx = slice_indices[slice_idx_within_volume]
            
            # extract slice
            slice_tensor = self._extract_slice(volume, slice_idx)
        
        # apply transform if provided
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
    """get data loaders for cyclegan training.
    
    args:
        preprocessed_dir: directory containing preprocessed data.
        metadata_json: path to metadata json file.
        batch_size: batch size.
        num_workers: number of workers for data loading.
        slices_per_subject: number of slices to sample per subject.
        seed: random seed for reproducibility.
        cache_mode: caching mode ("none", "slices", "volumes").
        
    returns:
        dictionary of data loaders.
    """
    import json
    
    # load metadata
    with open(metadata_json, "r") as f:
        metadata = json.load(f)
    
    # get domain paths
    domain_a_paths = metadata.get("domain_a", [])
    domain_b_paths = metadata.get("domain_b", [])
    
    # validate domain paths
    if not domain_a_paths:
        logger.error("No paths found for domain A")
    if not domain_b_paths:
        logger.error("No paths found for domain B")
    
    # create transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda x: (x - 0.5) * 2.0)  # scale from [0, 1] to [-1, 1]
    ])
    
    # create datasets
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
    
    # create data loaders
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