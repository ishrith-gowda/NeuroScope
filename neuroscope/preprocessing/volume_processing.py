"""Medical image preprocessing utilities.

This module contains utilities for preprocessing 3D medical imaging data,
including normalization, standardization, and data augmentation.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class VolumeNormalization:
    """Normalization methods for 3D medical volumes."""
    
    @staticmethod
    def min_max_normalization(
        volume: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        target_range: Tuple[float, float] = (0, 1),
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalize volume to [0, 1] or custom range using min-max normalization.
        
        Args:
            volume: Input volume as numpy array.
            min_val: Optional minimum value for normalization.
            max_val: Optional maximum value for normalization.
            target_range: Target range for normalized values.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # Determine min and max values
        if mask is not None:
            # Use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            v_min = min_val if min_val is not None else np.min(foreground)
            v_max = max_val if max_val is not None else np.max(foreground)
        else:
            v_min = min_val if min_val is not None else np.min(volume)
            v_max = max_val if max_val is not None else np.max(volume)
        
        # Avoid division by zero
        if v_min == v_max:
            logger.warning(f"Min and max values are equal: {v_min}. Setting normalized volume to {target_range[0]}.")
            return np.ones_like(volume) * target_range[0]
        
        # Normalize to [0, 1]
        normalized = (normalized - v_min) / (v_max - v_min)
        
        # Scale to target range if not [0, 1]
        if target_range != (0, 1):
            target_min, target_max = target_range
            normalized = normalized * (target_max - target_min) + target_min
        
        return normalized
    
    @staticmethod
    def z_score_normalization(
        volume: np.ndarray,
        mean_val: Optional[float] = None,
        std_val: Optional[float] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalize volume using z-score normalization.
        
        Args:
            volume: Input volume as numpy array.
            mean_val: Optional mean value for normalization.
            std_val: Optional standard deviation for normalization.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # Determine mean and std values
        if mask is not None:
            # Use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            v_mean = mean_val if mean_val is not None else np.mean(foreground)
            v_std = std_val if std_val is not None else np.std(foreground)
        else:
            v_mean = mean_val if mean_val is not None else np.mean(volume)
            v_std = std_val if std_val is not None else np.std(volume)
        
        # Avoid division by zero
        if v_std == 0:
            logger.warning(f"Standard deviation is zero. Setting normalized volume to zeros.")
            return np.zeros_like(volume)
        
        # Apply z-score normalization
        normalized = (normalized - v_mean) / v_std
        
        return normalized
    
    @staticmethod
    def percentile_normalization(
        volume: np.ndarray,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        target_range: Tuple[float, float] = (0, 1),
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Normalize volume using percentile clipping.
        
        Args:
            volume: Input volume as numpy array.
            low_percentile: Lower percentile for clipping.
            high_percentile: Upper percentile for clipping.
            target_range: Target range for normalized values.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # Calculate percentiles
        if mask is not None:
            # Use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            low_val = np.percentile(foreground, low_percentile)
            high_val = np.percentile(foreground, high_percentile)
        else:
            low_val = np.percentile(volume, low_percentile)
            high_val = np.percentile(volume, high_percentile)
        
        # Clip to percentile range
        normalized = np.clip(normalized, low_val, high_val)
        
        # Apply min-max normalization to target range
        return VolumeNormalization.min_max_normalization(
            normalized, low_val, high_val, target_range
        )
    
    @staticmethod
    def histogram_equalization(
        volume: np.ndarray,
        num_bins: int = 256,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply histogram equalization to the volume.
        
        Args:
            volume: Input volume as numpy array.
            num_bins: Number of histogram bins.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Histogram-equalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        equalized = volume.copy()
        
        # Normalize to [0, 1] first for binning
        normalized = VolumeNormalization.min_max_normalization(volume, mask=mask)
        
        # Create histogram
        if mask is not None:
            # Use only foreground voxels for histogram
            foreground = normalized[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                hist, bins = np.histogram(normalized.flatten(), num_bins, range=(0, 1))
            else:
                hist, bins = np.histogram(foreground, num_bins, range=(0, 1))
        else:
            hist, bins = np.histogram(normalized.flatten(), num_bins, range=(0, 1))
        
        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        cdf = cdf / float(cdf[-1])  # Normalize CDF
        
        # Apply equalization using the CDF
        equalized = np.interp(normalized.flatten(), bins[:-1], cdf)
        equalized = equalized.reshape(volume.shape)
        
        return equalized
    
    @staticmethod
    def adaptive_histogram_equalization(
        volume: np.ndarray,
        block_size: Tuple[int, int, int] = (32, 32, 32),
        clip_limit: float = 0.01,
    ) -> np.ndarray:
        """Apply adaptive histogram equalization to the volume.
        
        This method applies a 3D version of CLAHE (Contrast Limited Adaptive
        Histogram Equalization) to the volume.
        
        Args:
            volume: Input volume as numpy array.
            block_size: Size of blocks for local histogram equalization.
            clip_limit: Clipping limit to prevent over-amplification of noise.
            
        Returns:
            CLAHE-equalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Normalize to [0, 1] first
        normalized = VolumeNormalization.min_max_normalization(volume)
        
        # Scale to uint8 for processing
        uint8_volume = (normalized * 255).astype(np.uint8)
        
        # Try to use skimage for 3D CLAHE if available
        try:
            from skimage import exposure
            
            # Process each axis separately (pseudo-3D CLAHE)
            clahe_xy = np.zeros_like(uint8_volume, dtype=np.float32)
            clahe_xz = np.zeros_like(uint8_volume, dtype=np.float32)
            clahe_yz = np.zeros_like(uint8_volume, dtype=np.float32)
            
            # Apply CLAHE along different planes
            for i in range(uint8_volume.shape[0]):  # XY planes
                clahe_xy[i] = exposure.equalize_adapthist(
                    uint8_volume[i], 
                    kernel_size=block_size[1:],
                    clip_limit=clip_limit
                )
            
            for i in range(uint8_volume.shape[1]):  # XZ planes
                clahe_xz[:, i, :] = exposure.equalize_adapthist(
                    uint8_volume[:, i, :], 
                    kernel_size=(block_size[0], block_size[2]),
                    clip_limit=clip_limit
                )
            
            for i in range(uint8_volume.shape[2]):  # YZ planes
                clahe_yz[:, :, i] = exposure.equalize_adapthist(
                    uint8_volume[:, :, i], 
                    kernel_size=block_size[:2],
                    clip_limit=clip_limit
                )
            
            # Average the results from different planes
            equalized = (clahe_xy + clahe_xz + clahe_yz) / 3.0
            
        except ImportError:
            logger.warning("skimage not available, falling back to global histogram equalization")
            equalized = VolumeNormalization.histogram_equalization(volume)
        
        return equalized
    
    @staticmethod
    def white_stripe_normalization(
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        stripe_width: float = 0.1,
        target_value: float = 1.0,
    ) -> np.ndarray:
        """Apply white stripe normalization.
        
        This method is specifically designed for brain MRI normalization.
        It identifies a stripe of white matter and normalizes based on it.
        
        Args:
            volume: Input volume as numpy array.
            mask: Optional mask for brain voxels.
            stripe_width: Width of the stripe as proportion of intensity range.
            target_value: Target value for the white stripe.
            
        Returns:
            Normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # Calculate histogram for brain region
        if mask is not None:
            brain_voxels = normalized[mask > 0]
            if len(brain_voxels) == 0:
                logger.warning("Empty brain mask, using whole volume")
                brain_voxels = normalized.flatten()
        else:
            brain_voxels = normalized.flatten()
        
        # Calculate the mode of the histogram (approximating white matter peak)
        hist, bin_edges = np.histogram(brain_voxels, bins=100)
        mode_index = np.argmax(hist)
        mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
        
        # Define stripe around the mode
        intensity_range = np.max(brain_voxels) - np.min(brain_voxels)
        stripe_half_width = stripe_width * intensity_range / 2
        stripe_min = mode_value - stripe_half_width
        stripe_max = mode_value + stripe_half_width
        
        # Create white matter mask
        wm_mask = (normalized >= stripe_min) & (normalized <= stripe_max)
        if mask is not None:
            wm_mask = wm_mask & (mask > 0)
        
        # Calculate mean of white stripe
        if np.sum(wm_mask) > 0:
            wm_mean = np.mean(normalized[wm_mask])
        else:
            logger.warning("No voxels in white stripe, using mode value")
            wm_mean = mode_value
        
        # Scale the volume so that white stripe mean is at target value
        scale_factor = target_value / wm_mean
        normalized = normalized * scale_factor
        
        return normalized


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


class VolumePreprocessor:
    """Pipeline for preprocessing 3D medical volumes."""
    
    def __init__(
        self,
        preprocessing_steps: List[Tuple[str, Dict[str, Any]]] = None,
    ):
        """Initialize VolumePreprocessor.
        
        Args:
            preprocessing_steps: List of preprocessing steps and their parameters.
                Each step is a tuple of (step_name, parameters).
        """
        self.preprocessing_steps = preprocessing_steps or []
    
    def add_step(self, step_name: str, parameters: Dict[str, Any] = None):
        """Add a preprocessing step.
        
        Args:
            step_name: Name of the preprocessing step.
            parameters: Parameters for the step.
        """
        if parameters is None:
            parameters = {}
        self.preprocessing_steps.append((step_name, parameters))
    
    def preprocess(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply preprocessing pipeline to a volume.
        
        Args:
            volume: Input volume.
            mask: Optional mask for foreground voxels.
            
        Returns:
            Preprocessed volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # Create a copy to avoid modifying the original
        result = volume.copy()
        
        # Apply each preprocessing step
        for step_name, params in self.preprocessing_steps:
            logger.debug(f"Applying preprocessing step: {step_name}")
            
            if step_name == "min_max_normalization":
                result = VolumeNormalization.min_max_normalization(result, mask=mask, **params)
                
            elif step_name == "z_score_normalization":
                result = VolumeNormalization.z_score_normalization(result, mask=mask, **params)
                
            elif step_name == "percentile_normalization":
                result = VolumeNormalization.percentile_normalization(result, mask=mask, **params)
                
            elif step_name == "histogram_equalization":
                result = VolumeNormalization.histogram_equalization(result, mask=mask, **params)
                
            elif step_name == "adaptive_histogram_equalization":
                result = VolumeNormalization.adaptive_histogram_equalization(result, **params)
                
            elif step_name == "white_stripe_normalization":
                result = VolumeNormalization.white_stripe_normalization(result, mask=mask, **params)
                
            elif step_name == "crop":
                # Handle crop parameters
                if "crop_size" in params:
                    crop_size = params["crop_size"]
                    if "method" in params and params["method"] == "random":
                        result = DataAugmentation.random_crop(result, crop_size, mask=mask)
                    else:
                        # Center crop
                        starts = [(result.shape[i] - crop_size[i]) // 2 for i in range(len(crop_size))]
                        slices = tuple(slice(starts[i], starts[i] + crop_size[i]) for i in range(len(crop_size)))
                        result = result[slices]
            
            elif step_name == "rescale":
                try:
                    from scipy.ndimage import zoom
                except ImportError:
                    logger.warning("scipy.ndimage not available, skipping rescaling")
                    continue
                
                # Handle rescale parameters
                if "scale_factor" in params:
                    scale_factor = params["scale_factor"]
                    order = params.get("order", 1)  # Default to linear interpolation
                    result = zoom(result, scale_factor, order=order)
                elif "target_shape" in params:
                    target_shape = params["target_shape"]
                    scale_factor = [target_shape[i] / result.shape[i] for i in range(len(target_shape))]
                    order = params.get("order", 1)  # Default to linear interpolation
                    result = zoom(result, scale_factor, order=order)
            
            else:
                logger.warning(f"Unknown preprocessing step: {step_name}")
        
        return result
    
    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        mask_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch process multiple volumes.
        
        Args:
            input_dir: Input directory with volumes.
            output_dir: Output directory for preprocessed volumes.
            file_pattern: Glob pattern for input files.
            mask_dir: Optional directory with masks.
            
        Returns:
            Dictionary with preprocessing metadata.
        """
        import glob
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # List input files
        input_files = sorted(glob.glob(str(input_dir / file_pattern)))
        
        if not input_files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return {}
        
        # Process each file
        results = {}
        for input_file in input_files:
            try:
                file_path = Path(input_file)
                file_name = file_path.name
                output_file = output_dir / file_name
                
                logger.info(f"Processing {file_name}")
                
                # Load volume
                volume = self._load_volume(input_file)
                
                # Load mask if available
                mask = None
                if mask_dir:
                    mask_file = Path(mask_dir) / file_name
                    if os.path.exists(mask_file):
                        mask = self._load_volume(mask_file)
                
                # Apply preprocessing
                processed_volume = self.preprocess(volume, mask)
                
                # Save preprocessed volume
                self._save_volume(processed_volume, output_file, reference_file=input_file)
                
                # Record metadata
                metadata = {
                    "input_shape": volume.shape,
                    "output_shape": processed_volume.shape,
                    "input_min": float(np.min(volume)),
                    "input_max": float(np.max(volume)),
                    "output_min": float(np.min(processed_volume)),
                    "output_max": float(np.max(processed_volume)),
                    "preprocessing_steps": self.preprocessing_steps,
                }
                results[file_name] = metadata
                
                logger.info(f"Saved preprocessed {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
        
        return results
    
    def _load_volume(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load a volume from file.
        
        Args:
            file_path: Path to volume file.
            
        Returns:
            Volume as numpy array.
        """
        if str(file_path).endswith(".nii") or str(file_path).endswith(".nii.gz"):
            try:
                import nibabel as nib
                nii_img = nib.load(str(file_path))
                return np.asarray(nii_img.dataobj)
            except ImportError:
                raise ImportError("nibabel is required for loading NIfTI files")
        elif str(file_path).endswith(".npy"):
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _save_volume(
        self,
        volume: np.ndarray,
        output_file: Union[str, Path],
        reference_file: Optional[Union[str, Path]] = None,
    ):
        """Save a volume to file.
        
        Args:
            volume: Volume as numpy array.
            output_file: Output file path.
            reference_file: Optional reference file for header information.
        """
        if str(output_file).endswith(".nii") or str(output_file).endswith(".nii.gz"):
            try:
                import nibabel as nib
                
                # Copy header information from reference file if available
                if reference_file:
                    # Load reference file
                    ref_img = nib.load(str(reference_file))
                    # Create new image with reference header and affine
                    affine = ref_img.affine if hasattr(ref_img, 'affine') else np.eye(4)
                    new_img = nib.Nifti1Image(volume, affine)
                    # Copy header if possible
                    if hasattr(ref_img, 'header') and hasattr(new_img, 'header'):
                        for field in ref_img.header:
                            if field != 'dim':  # Don't copy dimensions
                                new_img.header[field] = ref_img.header[field]
                else:
                    # Create new NIfTI image
                    new_img = nib.Nifti1Image(volume, np.eye(4))
                
                # Save to file
                nib.save(new_img, str(output_file))
                
            except ImportError:
                raise ImportError("nibabel is required for saving NIfTI files")
        elif str(output_file).endswith(".npy"):
            np.save(output_file, volume)
        else:
            raise ValueError(f"Unsupported file format: {output_file}")


# Available preprocessing functions
PREPROCESSING_FUNCTIONS = {
    "min_max_normalization": VolumeNormalization.min_max_normalization,
    "z_score_normalization": VolumeNormalization.z_score_normalization,
    "percentile_normalization": VolumeNormalization.percentile_normalization,
    "histogram_equalization": VolumeNormalization.histogram_equalization,
    "adaptive_histogram_equalization": VolumeNormalization.adaptive_histogram_equalization,
    "white_stripe_normalization": VolumeNormalization.white_stripe_normalization,
    "random_flip": DataAugmentation.random_flip,
    "random_rotation_3d": DataAugmentation.random_rotation_3d,
    "random_crop": DataAugmentation.random_crop,
    "random_intensity_shift": DataAugmentation.random_intensity_shift,
    "random_intensity_scale": DataAugmentation.random_intensity_scale,
    "random_noise": DataAugmentation.random_noise,
}