"""volume normalization methods for medical imaging data.

this module provides various normalization techniques specifically designed
for 3d medical imaging volumes, including brain mri preprocessing.
"""

import numpy as np
import torch
from typing import Optional, Tuple

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class VolumeNormalization:
    """normalization methods for 3d medical volumes."""
    
    @staticmethod
    def min_max_normalization(
        volume: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        target_range: Tuple[float, float] = (0, 1),
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """normalize volume to [0, 1] or custom range using min-max normalization.
        
        args:
            volume: input volume as numpy array.
            min_val: optional minimum value for normalization.
            max_val: optional maximum value for normalization.
            target_range: target range for normalized values.
            mask: optional mask for foreground voxels.
            
        returns:
            normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # determine min and max values
        if mask is not None:
            # use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            v_min = min_val if min_val is not None else np.min(foreground)
            v_max = max_val if max_val is not None else np.max(foreground)
        else:
            v_min = min_val if min_val is not None else np.min(volume)
            v_max = max_val if max_val is not None else np.max(volume)
        
        # avoid division by zero
        if v_min == v_max:
            logger.warning(f"Min and max values are equal: {v_min}. Setting normalized volume to {target_range[0]}.")
            return np.ones_like(volume) * target_range[0]
        
        # normalize to [0, 1]
        normalized = (normalized - v_min) / (v_max - v_min)
        
        # scale to target range if not [0, 1]
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
        """normalize volume using z-score normalization.
        
        args:
            volume: input volume as numpy array.
            mean_val: optional mean value for normalization.
            std_val: optional standard deviation for normalization.
            mask: optional mask for foreground voxels.
            
        returns:
            normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # determine mean and std values
        if mask is not None:
            # use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            v_mean = mean_val if mean_val is not None else np.mean(foreground)
            v_std = std_val if std_val is not None else np.std(foreground)
        else:
            v_mean = mean_val if mean_val is not None else np.mean(volume)
            v_std = std_val if std_val is not None else np.std(volume)
        
        # avoid division by zero
        if v_std == 0:
            logger.warning(f"Standard deviation is zero. Setting normalized volume to zeros.")
            return np.zeros_like(volume)
        
        # apply z-score normalization
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
        """normalize volume using percentile clipping.
        
        args:
            volume: input volume as numpy array.
            low_percentile: lower percentile for clipping.
            high_percentile: upper percentile for clipping.
            target_range: target range for normalized values.
            mask: optional mask for foreground voxels.
            
        returns:
            normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # calculate percentiles
        if mask is not None:
            # use only foreground voxels for normalization
            foreground = volume[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                foreground = volume
            low_val = np.percentile(foreground, low_percentile)
            high_val = np.percentile(foreground, high_percentile)
        else:
            low_val = np.percentile(volume, low_percentile)
            high_val = np.percentile(volume, high_percentile)
        
        # clip to percentile range
        normalized = np.clip(normalized, low_val, high_val)
        
        # apply min-max normalization to target range
        return VolumeNormalization.min_max_normalization(
            normalized, low_val, high_val, target_range
        )
    
    @staticmethod
    def histogram_equalization(
        volume: np.ndarray,
        num_bins: int = 256,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """apply histogram equalization to the volume.
        
        args:
            volume: input volume as numpy array.
            num_bins: number of histogram bins.
            mask: optional mask for foreground voxels.
            
        returns:
            histogram-equalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # create a copy to avoid modifying the original
        equalized = volume.copy()
        
        # normalize to [0, 1] first for binning
        normalized = VolumeNormalization.min_max_normalization(volume, mask=mask)
        
        # create histogram
        if mask is not None:
            # use only foreground voxels for histogram
            foreground = normalized[mask > 0]
            if len(foreground) == 0:
                logger.warning("Empty foreground mask, using whole volume")
                hist, bins = np.histogram(normalized.flatten(), num_bins, range=(0, 1))
            else:
                hist, bins = np.histogram(foreground, num_bins, range=(0, 1))
        else:
            hist, bins = np.histogram(normalized.flatten(), num_bins, range=(0, 1))
        
        # calculate cumulative distribution function (cdf)
        cdf = hist.cumsum()
        cdf = cdf / float(cdf[-1])  # normalize cdf
        
        # apply equalization using the cdf
        equalized = np.interp(normalized.flatten(), bins[:-1], cdf)
        equalized = equalized.reshape(volume.shape)
        
        return equalized
    
    @staticmethod
    def adaptive_histogram_equalization(
        volume: np.ndarray,
        block_size: Tuple[int, int, int] = (32, 32, 32),
        clip_limit: float = 0.01,
    ) -> np.ndarray:
        """apply adaptive histogram equalization to the volume.
        
        this method applies a 3d version of clahe (contrast limited adaptive
        histogram equalization) to the volume.
        
        args:
            volume: input volume as numpy array.
            block_size: size of blocks for local histogram equalization.
            clip_limit: clipping limit to prevent over-amplification of noise.
            
        returns:
            clahe-equalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # normalize to [0, 1] first
        normalized = VolumeNormalization.min_max_normalization(volume)
        
        # scale to uint8 for processing
        uint8_volume = (normalized * 255).astype(np.uint8)
        
        # try to use skimage for 3d clahe if available
        try:
            from skimage import exposure
            
            # process each axis separately (pseudo-3d clahe)
            clahe_xy = np.zeros_like(uint8_volume, dtype=np.float32)
            clahe_xz = np.zeros_like(uint8_volume, dtype=np.float32)
            clahe_yz = np.zeros_like(uint8_volume, dtype=np.float32)
            
            # apply clahe along different planes
            for i in range(uint8_volume.shape[0]):  # xy planes
                clahe_xy[i] = exposure.equalize_adapthist(
                    uint8_volume[i], 
                    kernel_size=block_size[1:],
                    clip_limit=clip_limit
                )
            
            for i in range(uint8_volume.shape[1]):  # xz planes
                clahe_xz[:, i, :] = exposure.equalize_adapthist(
                    uint8_volume[:, i, :], 
                    kernel_size=(block_size[0], block_size[2]),
                    clip_limit=clip_limit
                )
            
            for i in range(uint8_volume.shape[2]):  # yz planes
                clahe_yz[:, :, i] = exposure.equalize_adapthist(
                    uint8_volume[:, :, i], 
                    kernel_size=block_size[:2],
                    clip_limit=clip_limit
                )
            
            # average the results from different planes
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
        """apply white stripe normalization.
        
        this method is specifically designed for brain mri normalization.
        it identifies a stripe of white matter and normalizes based on it.
        
        args:
            volume: input volume as numpy array.
            mask: optional mask for brain voxels.
            stripe_width: width of the stripe as proportion of intensity range.
            target_value: target value for the white stripe.
            
        returns:
            normalized volume.
        """
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()
        
        # create a copy to avoid modifying the original
        normalized = volume.copy()
        
        # calculate histogram for brain region
        if mask is not None:
            brain_voxels = normalized[mask > 0]
            if len(brain_voxels) == 0:
                logger.warning("Empty brain mask, using whole volume")
                brain_voxels = normalized.flatten()
        else:
            brain_voxels = normalized.flatten()
        
        # calculate the mode of the histogram (approximating white matter peak)
        hist, bin_edges = np.histogram(brain_voxels, bins=100)
        mode_index = np.argmax(hist)
        mode_value = (bin_edges[mode_index] + bin_edges[mode_index + 1]) / 2
        
        # define stripe around the mode
        intensity_range = np.max(brain_voxels) - np.min(brain_voxels)
        stripe_half_width = stripe_width * intensity_range / 2
        stripe_min = mode_value - stripe_half_width
        stripe_max = mode_value + stripe_half_width
        
        # create white matter mask
        wm_mask = (normalized >= stripe_min) & (normalized <= stripe_max)
        if mask is not None:
            wm_mask = wm_mask & (mask > 0)
        
        # calculate mean of white stripe
        if np.sum(wm_mask) > 0:
            wm_mean = np.mean(normalized[wm_mask])
        else:
            logger.warning("No voxels in white stripe, using mode value")
            wm_mean = mode_value
        
        # scale the volume so that white stripe mean is at target value
        scale_factor = target_value / wm_mean
        normalized = normalized * scale_factor
        
        return normalized