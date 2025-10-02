"""3D MRI transforms for neuroscope."""

import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

try:
    import torchio as tio
except ImportError:
    tio = None


class MRITransforms:
    """3D MRI transforms for data augmentation and preprocessing."""
    
    @staticmethod
    def get_training_transforms(
        normalize: bool = True,
        augment: bool = True,
        p_noise: float = 0.3,
        p_bias_field: float = 0.3,
        p_spatial: float = 0.3,
    ) -> Callable:
        """Get transforms for training.
        
        Args:
            normalize: Whether to normalize the data.
            augment: Whether to apply data augmentation.
            p_noise: Probability of applying random noise.
            p_bias_field: Probability of applying random bias field.
            p_spatial: Probability of applying spatial transformations.
            
        Returns:
            Composition of transforms.
        """
        if tio is None:
            raise ImportError("TorchIO is required for MRI transforms. Install with 'pip install torchio'.")
        
        transforms_list = []
        
        # Always convert to canonical orientation
        transforms_list.append(tio.ToCanonical())
        
        # Normalization
        if normalize:
            transforms_list.append(tio.ZNormalization())
        
        # Data augmentation
        if augment:
            # Random noise
            if p_noise > 0:
                transforms_list.append(tio.RandomNoise(std=(0.01, 0.1), p=p_noise))
            
            # Random bias field
            if p_bias_field > 0:
                transforms_list.append(tio.RandomBiasField(p=p_bias_field))
            
            # Spatial transformations
            if p_spatial > 0:
                transforms_list.append(
                    tio.RandomAffine(
                        scales=(0.9, 1.1),
                        degrees=10,
                        translation=5,
                        p=p_spatial,
                    )
                )
        
        return tio.Compose(transforms_list)
    
    @staticmethod
    def get_validation_transforms(normalize: bool = True) -> Callable:
        """Get transforms for validation.
        
        Args:
            normalize: Whether to normalize the data.
            
        Returns:
            Composition of transforms.
        """
        if tio is None:
            raise ImportError("TorchIO is required for MRI transforms. Install with 'pip install torchio'.")
        
        transforms_list = []
        
        # Always convert to canonical orientation
        transforms_list.append(tio.ToCanonical())
        
        # Normalization
        if normalize:
            transforms_list.append(tio.ZNormalization())
        
        return tio.Compose(transforms_list)


class SliceExtractor:
    """Extract 2D slices from 3D volumes."""
    
    @staticmethod
    def random_slice(volume: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """Extract a random 2D slice from a 3D volume.
        
        Args:
            volume: 3D volume tensor of shape [C, D, H, W].
            dim: Dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            
        Returns:
            2D slice tensor of shape [C, H, W].
        """
        # Ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # Get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # Choose random slice index
        idx = random.randint(0, depth - 1)
        
        # Extract slice
        if dim == 0:  # Sagittal
            slice_tensor = volume[:, :, idx, :]
        elif dim == 1:  # Coronal
            slice_tensor = volume[:, :, :, idx]
        else:  # dim == 2, Axial
            slice_tensor = volume[:, idx, :, :]
        
        return slice_tensor
    
    @staticmethod
    def center_slice(volume: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """Extract the center 2D slice from a 3D volume.
        
        Args:
            volume: 3D volume tensor of shape [C, D, H, W].
            dim: Dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            
        Returns:
            2D slice tensor of shape [C, H, W].
        """
        # Ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # Get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # Choose center slice index
        idx = depth // 2
        
        # Extract slice
        if dim == 0:  # Sagittal
            slice_tensor = volume[:, :, idx, :]
        elif dim == 1:  # Coronal
            slice_tensor = volume[:, :, :, idx]
        else:  # dim == 2, Axial
            slice_tensor = volume[:, idx, :, :]
        
        return slice_tensor
    
    @staticmethod
    def multi_slice(volume: torch.Tensor, dim: int = 2, n_slices: int = 5) -> List[torch.Tensor]:
        """Extract multiple evenly spaced 2D slices from a 3D volume.
        
        Args:
            volume: 3D volume tensor of shape [C, D, H, W].
            dim: Dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            n_slices: Number of slices to extract.
            
        Returns:
            List of 2D slice tensors of shape [C, H, W].
        """
        # Ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # Get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # Calculate slice indices
        if n_slices > depth:
            n_slices = depth
        
        if n_slices == 1:
            # Just return the center slice
            indices = [depth // 2]
        else:
            # Calculate evenly spaced indices
            step = (depth - 1) / (n_slices - 1) if n_slices > 1 else 0
            indices = [int(round(i * step)) for i in range(n_slices)]
        
        # Extract slices
        slices = []
        for idx in indices:
            if dim == 0:  # Sagittal
                slice_tensor = volume[:, :, idx, :]
            elif dim == 1:  # Coronal
                slice_tensor = volume[:, :, :, idx]
            else:  # dim == 2, Axial
                slice_tensor = volume[:, idx, :, :]
            
            slices.append(slice_tensor)
        
        return slices