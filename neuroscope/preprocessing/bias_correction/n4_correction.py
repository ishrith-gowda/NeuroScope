"""n4 bias field correction for mri."""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class N4BiasFieldCorrection:
    """n4 bias field correction for mri volumes.
    
    this class provides methods for performing n4 bias field correction
    on mri volumes using simpleitk.
    """
    
    def __init__(
        self,
        shrink_factor: int = 4,
        iterations: List[int] = None,
        convergence_threshold: float = 0.001,
        spline_order: int = 3,
        spline_distance: float = 200.0,
        history_weight: float = 0.5,
        save_bias_field: bool = False,
    ):
        """initialize n4biasfieldcorrection.
        
        args:
            shrink_factor: shrink factor for downsampling.
            iterations: number of iterations at each resolution level.
            convergence_threshold: convergence threshold.
            spline_order: order of bspline used in the approximation.
            spline_distance: distance between b-spline control points.
            history_weight: weight for historical gradient values.
            save_bias_field: whether to save the estimated bias field.
        """
        self.shrink_factor = shrink_factor
        self.iterations = iterations or [50, 50, 30, 20]
        self.convergence_threshold = convergence_threshold
        self.spline_order = spline_order
        self.spline_distance = spline_distance
        self.history_weight = history_weight
        self.save_bias_field = save_bias_field
    
    def correct_volume(
        self,
        input_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """apply n4 bias field correction to a volume.
        
        args:
            input_image: input volume as numpy array.
            mask: optional mask to specify foreground voxels.
            
        returns:
            tuple of (corrected_image, bias_field).
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for N4BiasFieldCorrection")
        
        # convert numpy array to simpleitk image
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.numpy()
        
        sitk_image = sitk.GetImageFromArray(input_image)
        
        # create mask if not provided
        if mask is None:
            # create an otsu mask based on the input image
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetInsideValue(0)
            otsu_filter.SetOutsideValue(1)
            sitk_mask = otsu_filter.Execute(sitk_image)
        else:
            # use provided mask
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        # create n4 bias field correction filter
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        
        # set parameters
        n4_filter.SetMaximumNumberOfIterations(self.iterations)
        n4_filter.SetConvergenceThreshold(self.convergence_threshold)
        n4_filter.SetSplineOrder(self.spline_order)
        n4_filter.SetBiasFieldFullWidthAtHalfMaximum(self.spline_distance)
        n4_filter.SetWienerFilterNoise(0.01)
        n4_filter.SetBiasFieldFullWidthAtHalfMaximum(0.15)
        n4_filter.SetUseOptimalMetricValue(True)
        
        # apply n4 correction
        try:
            corrected_image = n4_filter.Execute(sitk_image, sitk_mask)
        except RuntimeError as e:
            logger.warning(f"N4 correction failed: {e}. Using original image.")
            return input_image, np.ones_like(input_image)
        
        # convert back to numpy array
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        
        # get bias field (log domain)
        if self.save_bias_field:
            bias_field = sitk.GetArrayFromImage(n4_filter.GetBiasField(sitk_image))
        else:
            # compute bias field by dividing the corrected image by the original image
            epsilon = 1e-6
            bias_field = np.divide(
                corrected_array + epsilon,
                input_image + epsilon
            )
        
        return corrected_array, bias_field
    
    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        mask_dir: Optional[Union[str, Path]] = None,
        save_bias: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """batch process multiple volumes.
        
        args:
            input_dir: input directory with mri volumes.
            output_dir: output directory for corrected volumes.
            file_pattern: glob pattern for input files.
            mask_dir: optional directory with masks.
            save_bias: whether to save bias fields.
            
        returns:
            dictionary with correction metrics.
        """
        import glob
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # list input files
        input_files = glob.glob(str(input_dir / file_pattern))
        
        if not input_files:
            logger.warning(f"No files found matching pattern: {file_pattern}")
            return {}
        
        # process each file
        results = {}
        for input_file in input_files:
            try:
                file_path = Path(input_file)
                file_name = file_path.name
                output_file = output_dir / file_name
                bias_file = output_dir / f"bias_{file_name}"
                
                # load volume
                volume = self._load_volume(input_file)
                
                # load mask if available
                mask = None
                if mask_dir:
                    mask_file = Path(mask_dir) / file_name
                    if os.path.exists(mask_file):
                        mask = self._load_volume(mask_file)
                
                # apply correction
                corrected_volume, bias_field = self.correct_volume(volume, mask)
                
                # save corrected volume
                self._save_volume(corrected_volume, output_file, reference_file=input_file)
                
                # save bias field if requested
                if save_bias:
                    self._save_volume(bias_field, bias_file, reference_file=input_file)
                
                # calculate metrics
                metrics = self._calculate_metrics(volume, corrected_volume, bias_field)
                results[file_name] = metrics
                
                logger.info(f"Processed {file_name}: CV = {metrics['coefficient_of_variation']:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
        
        return results
    
    def _load_volume(self, file_path: Union[str, Path]) -> np.ndarray:
        """load a volume from file.
        
        args:
            file_path: path to volume file.
            
        returns:
            volume as numpy array.
        """
        if str(file_path).endswith(".nii") or str(file_path).endswith(".nii.gz"):
            try:
                import nibabel as nib
                return nib.load(file_path).get_fdata()
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
        """save a volume to file.
        
        args:
            volume: volume as numpy array.
            output_file: output file path.
            reference_file: optional reference file for header information.
        """
        if str(output_file).endswith(".nii") or str(output_file).endswith(".nii.gz"):
            try:
                import nibabel as nib
                
                # copy header information from reference file if available
                if reference_file:
                    ref_nii = nib.load(reference_file)
                    new_nii = nib.Nifti1Image(volume, ref_nii.affine, ref_nii.header)
                else:
                    # create new nifti image
                    new_nii = nib.Nifti1Image(volume, np.eye(4))
                
                # save to file
                nib.save(new_nii, output_file)
                
            except ImportError:
                raise ImportError("nibabel is required for saving NIfTI files")
        elif str(output_file).endswith(".npy"):
            np.save(output_file, volume)
        else:
            raise ValueError(f"Unsupported file format: {output_file}")
    
    def _calculate_metrics(
        self,
        original_volume: np.ndarray,
        corrected_volume: np.ndarray,
        bias_field: np.ndarray,
    ) -> Dict[str, float]:
        """calculate correction metrics.
        
        args:
            original_volume: original volume.
            corrected_volume: corrected volume.
            bias_field: estimated bias field.
            
        returns:
            dictionary of metrics.
        """
        # create a foreground mask to exclude background voxels
        mean_volume = (original_volume + corrected_volume) / 2
        threshold = np.mean(mean_volume) * 0.1
        mask = mean_volume > threshold
        
        # apply mask to volumes
        original_masked = original_volume[mask]
        corrected_masked = corrected_volume[mask]
        bias_masked = bias_field[mask]
        
        # calculate metrics
        original_cv = np.std(original_masked) / np.mean(original_masked)
        corrected_cv = np.std(corrected_masked) / np.mean(corrected_masked)
        cv_improvement = (original_cv - corrected_cv) / original_cv * 100
        
        # bias field statistics
        bias_mean = np.mean(bias_masked)
        bias_std = np.std(bias_masked)
        bias_max = np.max(bias_masked)
        bias_min = np.min(bias_masked)
        
        return {
            "original_cv": float(original_cv),
            "corrected_cv": float(corrected_cv),
            "cv_improvement_percent": float(cv_improvement),
            "coefficient_of_variation": float(corrected_cv),
            "bias_mean": float(bias_mean),
            "bias_std": float(bias_std),
            "bias_max": float(bias_max),
            "bias_min": float(bias_min),
        }