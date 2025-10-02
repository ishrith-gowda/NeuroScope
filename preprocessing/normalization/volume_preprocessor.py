"""Volume preprocessing pipeline for medical imaging data.

This module provides a flexible pipeline for preprocessing 3D medical volumes
with various normalization and augmentation steps.
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from neuroscope.core.logging import get_logger
from .volume_normalization import VolumeNormalization
from .data_augmentation import DataAugmentation

logger = get_logger(__name__)


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