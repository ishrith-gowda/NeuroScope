"""Data augmentation methods for 3D medical volumes.

This module provides various augmentation techniques for medical imaging data,
including geometric transformations and intensity modifications.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class DataAugmentation:
    """Data augmentation methods for 3D medical volumes."""
    
    @staticmethod
    def random_flip(
        volume: np.ndarray,
        axes: List[int] = None,
        p: float = 0.5,
    ) -> np.ndarray:
        """Randomly flip the volume along specified axes.
        
        Args:
            volume: Input volume as numpy array.
            axes: Axes along which to potentially flip. Default: all axes.
            p: Probability of flipping along each axis.
            
        Returns:
            Augmented volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Default to all axes if none specified
        if axes is None:
            axes = list(range(volume.ndim))
        
        # Apply random flips along each specified axis
        for axis in axes:
            if np.random.random() < p:
                result = np.flip(result, axis=axis)
        
        return result
    
    @staticmethod
    def random_rotation_3d(
        volume: np.ndarray,
        max_angle: float = 15.0,  # degrees
        axes: List[Tuple[int, int]] = None,
        mode: str = "constant",
        order: int = 1,
    ) -> np.ndarray:
        """Randomly rotate the volume in 3D.
        
        Args:
            volume: Input 3D volume as numpy array.
            max_angle: Maximum rotation angle in degrees.
            axes: Rotation planes. Default: all combinations.
            mode: Interpolation mode.
            order: Interpolation order.
            
        Returns:
            Rotated volume.
        """
        try:
            from scipy.ndimage import rotate
        except ImportError:
            logger.warning("scipy.ndimage not available, skipping rotation augmentation")
            return volume
        
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Default to all planes if none specified
        if axes is None:
            axes = [(0, 1), (1, 2), (0, 2)]
        
        # Apply random rotations in specified planes
        for axis_pair in axes:
            angle = np.random.uniform(-max_angle, max_angle)
            result = rotate(result, angle, axes=axis_pair, reshape=False, 
                           mode=mode, order=order)
        
        return result
    
    @staticmethod
    def random_crop(
        volume: np.ndarray,
        crop_size: Tuple[int, ...],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Randomly crop the volume.
        
        Args:
            volume: Input volume as numpy array.
            crop_size: Size of the crop.
            mask: Optional mask to guide cropping (prefer foreground regions).
            
        Returns:
            Cropped volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Check if crop is possible
        for i in range(len(crop_size)):
            if crop_size[i] > volume.shape[i]:
                raise ValueError(f"Crop size {crop_size[i]} is larger than volume size {volume.shape[i]} along axis {i}")
        
        # Calculate valid crop ranges
        valid_starts = [volume.shape[i] - crop_size[i] for i in range(len(crop_size))]
        
        # If mask is provided, prefer foreground regions
        if mask is not None and np.any(mask > 0):
            # Get foreground positions
            foreground_positions = np.where(mask > 0)
            
            # Select a random foreground voxel as the center
            idx = np.random.randint(0, len(foreground_positions[0]))
            center = [foreground_positions[i][idx] for i in range(len(crop_size))]
            
            # Calculate crop starts with the foreground voxel as center
            starts = [max(0, min(valid_starts[i], center[i] - crop_size[i] // 2)) for i in range(len(crop_size))]
        else:
            # Completely random crop
            starts = [np.random.randint(0, valid_start + 1) for valid_start in valid_starts]
        
        # Create slices for cropping
        slices = tuple(slice(starts[i], starts[i] + crop_size[i]) for i in range(len(crop_size)))
        
        # Apply crop
        return volume[slices]
    
    @staticmethod
    def random_intensity_shift(
        volume: np.ndarray,
        shift_range: Tuple[float, float],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply random intensity shift augmentation.
        
        Args:
            volume: Input volume as numpy array.
            shift_range: Range of intensity shifts as fraction of intensity range.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Augmented volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Determine intensity range
        if mask is not None:
            foreground = volume[mask > 0]
            if len(foreground) > 0:
                v_min = np.min(foreground)
                v_max = np.max(foreground)
            else:
                v_min = np.min(volume)
                v_max = np.max(volume)
        else:
            v_min = np.min(volume)
            v_max = np.max(volume)
        
        # Calculate intensity range and shift amount
        intensity_range = v_max - v_min
        shift_amount = np.random.uniform(shift_range[0], shift_range[1]) * intensity_range
        
        # Apply shift
        result = result + shift_amount
        
        return result
    
    @staticmethod
    def random_intensity_scale(
        volume: np.ndarray,
        scale_range: Tuple[float, float],
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply random intensity scaling augmentation.
        
        Args:
            volume: Input volume as numpy array.
            scale_range: Range of intensity scaling factors.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Augmented volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Calculate scaling factor
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        
        # Determine mean intensity for mean-preserving scaling
        if mask is not None:
            foreground = volume[mask > 0]
            if len(foreground) > 0:
                mean_val = np.mean(foreground)
            else:
                mean_val = np.mean(volume)
        else:
            mean_val = np.mean(volume)
        
        # Apply scaling while preserving mean
        result = mean_val + scale_factor * (result - mean_val)
        
        return result
    
    @staticmethod
    def random_noise(
        volume: np.ndarray,
        noise_type: str = "gaussian",
        params: Dict[str, Any] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Add random noise to the volume.
        
        Args:
            volume: Input volume as numpy array.
            noise_type: Type of noise ('gaussian', 'poisson', 'salt', 'pepper', 'salt_and_pepper').
            params: Parameters for the noise.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Noisy volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Default parameters
        if params is None:
            params = {}
        
        # Determine if noise should be applied to foreground only
        if mask is not None:
            noise_mask = mask > 0
        else:
            noise_mask = np.ones_like(volume, dtype=bool)
        
        # Apply noise based on type
        if noise_type == "gaussian":
            # Default standard deviation as percentage of intensity range
            std_dev = params.get("std_dev", 0.01)
            if "intensity_range" in params:
                intensity_range = params["intensity_range"]
            else:
                intensity_range = np.max(result[noise_mask]) - np.min(result[noise_mask])
            
            # Generate Gaussian noise
            noise = np.random.normal(
                loc=0, 
                scale=std_dev * intensity_range, 
                size=result.shape
            )
            result[noise_mask] += noise[noise_mask]
            
        elif noise_type == "poisson":
            # Scale image for Poisson noise
            scale_factor = params.get("scale_factor", 1.0)
            lambda_factor = params.get("lambda_factor", 10.0)
            
            # Generate scaled image for Poisson simulation
            scaled = np.max(np.abs(result[noise_mask])) * scale_factor
            
            if scaled > 0:
                # Generate Poisson noise
                noisy_scaled = np.random.poisson(
                    lam=result[noise_mask] * lambda_factor / scaled
                )
                result[noise_mask] = noisy_scaled * scaled / lambda_factor
            
        elif noise_type in ["salt", "pepper", "salt_and_pepper"]:
            # Amount of salt/pepper noise (fraction of voxels)
            amount = params.get("amount", 0.01)
            salt_val = params.get("salt_val", np.max(result))
            pepper_val = params.get("pepper_val", np.min(result))
            
            # Number of noise voxels
            num_noise_voxels = int(amount * np.sum(noise_mask))
            
            # Get indices of foreground voxels
            noise_indices = np.where(noise_mask)
            if len(noise_indices[0]) == 0:
                return result
            
            # Randomly select voxels to modify
            idx = np.random.choice(len(noise_indices[0]), num_noise_voxels, replace=False)
            noise_coords = tuple(noise_indices[i][idx] for i in range(len(noise_indices)))
            
            if noise_type == "salt":
                # Add salt noise (high intensity)
                result[noise_coords] = salt_val
                
            elif noise_type == "pepper":
                # Add pepper noise (low intensity)
                result[noise_coords] = pepper_val
                
            else:  # salt_and_pepper
                # Split between salt and pepper
                salt_idx = np.random.choice(len(idx), len(idx) // 2, replace=False)
                salt_coords = tuple(noise_coords[i][salt_idx] for i in range(len(noise_coords)))
                pepper_coords = tuple(
                    np.delete(noise_coords[i], salt_idx) for i in range(len(noise_coords))
                )
                
                # Apply salt and pepper
                if len(salt_coords[0]) > 0:
                    result[salt_coords] = salt_val
                if len(pepper_coords[0]) > 0:
                    result[pepper_coords] = pepper_val
        
        return result