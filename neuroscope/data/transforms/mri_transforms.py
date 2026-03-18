"""3d mri transforms for neuroscope."""

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
    """3d mri transforms for data augmentation and preprocessing."""
    
    @staticmethod
    def get_training_transforms(
        normalize: bool = True,
        augment: bool = True,
        p_noise: float = 0.3,
        p_bias_field: float = 0.3,
        p_spatial: float = 0.3,
    ) -> Callable:
        """get transforms for training.
        
        args:
            normalize: whether to normalize the data.
            augment: whether to apply data augmentation.
            p_noise: probability of applying random noise.
            p_bias_field: probability of applying random bias field.
            p_spatial: probability of applying spatial transformations.
            
        returns:
            composition of transforms.
        """
        if tio is None:
            raise ImportError("TorchIO is required for MRI transforms. Install with 'pip install torchio'.")
        
        transforms_list = []
        
        # always convert to canonical orientation
        transforms_list.append(tio.ToCanonical())
        
        # normalization
        if normalize:
            transforms_list.append(tio.ZNormalization())
        
        # data augmentation
        if augment:
            # random noise
            if p_noise > 0:
                transforms_list.append(tio.RandomNoise(std=(0.01, 0.1), p=p_noise))
            
            # random bias field
            if p_bias_field > 0:
                transforms_list.append(tio.RandomBiasField(p=p_bias_field))
            
            # spatial transformations
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
        """get transforms for validation.
        
        args:
            normalize: whether to normalize the data.
            
        returns:
            composition of transforms.
        """
        if tio is None:
            raise ImportError("TorchIO is required for MRI transforms. Install with 'pip install torchio'.")
        
        transforms_list = []
        
        # always convert to canonical orientation
        transforms_list.append(tio.ToCanonical())
        
        # normalization
        if normalize:
            transforms_list.append(tio.ZNormalization())
        
        return tio.Compose(transforms_list)


class SliceExtractor:
    """extract 2d slices from 3d volumes."""
    
    @staticmethod
    def random_slice(volume: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """extract a random 2d slice from a 3d volume.
        
        args:
            volume: 3d volume tensor of shape [c, d, h, w].
            dim: dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            
        returns:
            2d slice tensor of shape [c, h, w].
        """
        # ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # choose random slice index
        idx = random.randint(0, depth - 1)
        
        # extract slice
        if dim == 0:  # sagittal
            slice_tensor = volume[:, :, idx, :]
        elif dim == 1:  # coronal
            slice_tensor = volume[:, :, :, idx]
        else:  # dim == 2, axial
            slice_tensor = volume[:, idx, :, :]
        
        return slice_tensor
    
    @staticmethod
    def center_slice(volume: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """extract the center 2d slice from a 3d volume.
        
        args:
            volume: 3d volume tensor of shape [c, d, h, w].
            dim: dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            
        returns:
            2d slice tensor of shape [c, h, w].
        """
        # ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # choose center slice index
        idx = depth // 2
        
        # extract slice
        if dim == 0:  # sagittal
            slice_tensor = volume[:, :, idx, :]
        elif dim == 1:  # coronal
            slice_tensor = volume[:, :, :, idx]
        else:  # dim == 2, axial
            slice_tensor = volume[:, idx, :, :]
        
        return slice_tensor
    
    @staticmethod
    def multi_slice(volume: torch.Tensor, dim: int = 2, n_slices: int = 5) -> List[torch.Tensor]:
        """extract multiple evenly spaced 2d slices from a 3d volume.
        
        args:
            volume: 3d volume tensor of shape [c, d, h, w].
            dim: dimension to extract slice from (0=sagittal, 1=coronal, 2=axial).
            n_slices: number of slices to extract.
            
        returns:
            list of 2d slice tensors of shape [c, h, w].
        """
        # ensure volume has 4 dimensions (batch, channels, height, width)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
        
        # get depth along selected dimension
        depth = volume.shape[dim + 1]  # +1 because of batch dimension
        
        # calculate slice indices
        if n_slices > depth:
            n_slices = depth
        
        if n_slices == 1:
            # just return the center slice
            indices = [depth // 2]
        else:
            # calculate evenly spaced indices
            step = (depth - 1) / (n_slices - 1) if n_slices > 1 else 0
            indices = [int(round(i * step)) for i in range(n_slices)]
        
        # extract slices
        slices = []
        for idx in indices:
            if dim == 0:  # sagittal
                slice_tensor = volume[:, :, idx, :]
            elif dim == 1:  # coronal
                slice_tensor = volume[:, :, :, idx]
            else:  # dim == 2, axial
                slice_tensor = volume[:, idx, :, :]
            
            slices.append(slice_tensor)
        
        return slices