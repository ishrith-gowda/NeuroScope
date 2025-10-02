"""Registration utilities for MRI volumes."""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class MRIRegistration:
    """Registration of MRI volumes.
    
    This class provides methods for performing registration between MRI volumes
    using SimpleITK, supporting rigid, affine, and deformable registration.
    """
    
    def __init__(
        self,
        registration_type: str = "rigid",
        metric: str = "mutual_information",
        optimizer: str = "gradient_descent",
        sampling_percentage: float = 0.1,
        learning_rate: float = 0.1,
        number_of_iterations: int = 100,
        shrink_factors: Optional[List[int]] = None,
        smoothing_sigmas: Optional[List[float]] = None,
        final_interpolator: str = "linear",
        verbose: bool = False,
    ):
        """Initialize MRIRegistration.
        
        Args:
            registration_type: Type of registration ('rigid', 'affine', or 'deformable').
            metric: Similarity metric ('mutual_information', 'mean_squares', etc.).
            optimizer: Optimizer for registration ('gradient_descent', 'lbfgs', etc.).
            sampling_percentage: Percentage of voxels to sample for metric evaluation.
            learning_rate: Learning rate for optimizer.
            number_of_iterations: Maximum number of iterations.
            shrink_factors: Shrink factors at each resolution level.
            smoothing_sigmas: Smoothing sigmas at each resolution level.
            final_interpolator: Interpolator for final resampling.
            verbose: Whether to print verbose output during registration.
        """
        self.registration_type = registration_type.lower()
        self.metric = metric.lower()
        self.optimizer = optimizer.lower()
        self.sampling_percentage = sampling_percentage
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.shrink_factors = shrink_factors or [8, 4, 2, 1]
        self.smoothing_sigmas = smoothing_sigmas or [3, 2, 1, 0]
        self.final_interpolator = final_interpolator.lower()
        self.verbose = verbose
        
        # Validate parameters
        if self.registration_type not in ["rigid", "affine", "deformable"]:
            raise ValueError(
                f"Invalid registration type: {self.registration_type}. "
                "Must be one of 'rigid', 'affine', or 'deformable'."
            )
    
    def register_volumes(
        self,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_mask: Optional[np.ndarray] = None,
        moving_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Any]:
        """Register moving image to fixed image.
        
        Args:
            fixed_image: Fixed (target) image as numpy array.
            moving_image: Moving (source) image as numpy array.
            fixed_mask: Optional mask for fixed image.
            moving_mask: Optional mask for moving image.
            
        Returns:
            Tuple of (registered_image, transform).
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for MRIRegistration")
        
        # Convert numpy arrays to SimpleITK images
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        
        # Convert masks if provided
        fixed_mask_sitk = None
        moving_mask_sitk = None
        
        if fixed_mask is not None:
            fixed_mask_sitk = sitk.GetImageFromArray(fixed_mask.astype(np.uint8))
        if moving_mask is not None:
            moving_mask_sitk = sitk.GetImageFromArray(moving_mask.astype(np.uint8))
        
        # Set up registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set similarity metric
        if self.metric == "mutual_information":
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif self.metric == "mean_squares":
            registration_method.SetMetricAsMeanSquares()
        elif self.metric == "correlation":
            registration_method.SetMetricAsCorrelation()
        else:
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            logger.warning(f"Unknown metric: {self.metric}. Using mutual information.")
        
        # Set optimizer
        if self.optimizer == "gradient_descent":
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.learning_rate,
                numberOfIterations=self.number_of_iterations,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
        elif self.optimizer == "lbfgs":
            registration_method.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-5,
                numberOfIterations=self.number_of_iterations,
                deltaConvergenceTolerance=1e-5,
            )
        else:
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=self.learning_rate,
                numberOfIterations=self.number_of_iterations,
            )
            logger.warning(f"Unknown optimizer: {self.optimizer}. Using gradient descent.")
        
        # Set sampling strategy
        registration_method.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
        registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
        
        # Set multi-resolution framework
        registration_method.SetShrinkFactorsPerLevel(self.shrink_factors)
        registration_method.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        
        # Set masks if provided
        if fixed_mask_sitk is not None:
            registration_method.SetMetricFixedMask(fixed_mask_sitk)
        if moving_mask_sitk is not None:
            registration_method.SetMetricMovingMask(moving_mask_sitk)
        
        # Initialize transform based on registration type
        initial_transform = None
        if self.registration_type == "rigid":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif self.registration_type == "affine":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif self.registration_type == "deformable":
            # First perform affine registration
            affine_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            registration_method.SetInitialTransform(affine_transform)
            affine_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            
            # Then perform deformable registration
            # Reset registration method for BSpline registration
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            registration_method.SetOptimizerAsLBFGS2(
                solutionAccuracy=1e-5,
                numberOfIterations=self.number_of_iterations,
            )
            registration_method.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
            registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
            registration_method.SetShrinkFactorsPerLevel(self.shrink_factors)
            registration_method.SetSmoothingSigmasPerLevel(self.smoothing_sigmas)
            registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            
            if fixed_mask_sitk is not None:
                registration_method.SetMetricFixedMask(fixed_mask_sitk)
            if moving_mask_sitk is not None:
                registration_method.SetMetricMovingMask(moving_mask_sitk)
            
            # Apply the affine transform to the moving image
            moving_sitk = sitk.Resample(
                moving_sitk,
                fixed_sitk,
                affine_transform,
                sitk.sitkLinear,
                0.0,
                moving_sitk.GetPixelID(),
            )
            
            # Set up the BSpline transform
            transform_domain_mesh_size = [8] * moving_sitk.GetDimension()
            initial_transform = sitk.BSplineTransformInitializer(
                fixed_sitk, transform_domain_mesh_size
            )
        
        # Set the initial transform
        if initial_transform is not None:
            registration_method.SetInitialTransform(initial_transform)
        
        # Set additional options
        if self.verbose:
            registration_method.AddCommand(sitk.sitkIterationEvent, self._registration_callback)
        
        # Perform registration
        transform = None
        try:
            transform = registration_method.Execute(fixed_sitk, moving_sitk)
            logger.info(f"Registration complete. Final metric value: {registration_method.GetMetricValue()}")
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return moving_image, None
        
        # Apply transform to the moving image
        interpolator = sitk.sitkLinear
        if self.final_interpolator == "nearest":
            interpolator = sitk.sitkNearestNeighbor
        elif self.final_interpolator == "bspline":
            interpolator = sitk.sitkBSpline
        
        registered_sitk = sitk.Resample(
            moving_sitk,
            fixed_sitk,
            transform,
            interpolator,
            0.0,
            moving_sitk.GetPixelID(),
        )
        
        # Convert back to numpy array
        registered_image = sitk.GetArrayFromImage(registered_sitk)
        
        return registered_image, transform
    
    def apply_transform(
        self,
        moving_image: np.ndarray,
        transform,
        reference_image: np.ndarray,
        interpolator: str = "linear",
    ) -> np.ndarray:
        """Apply a transformation to a moving image.
        
        Args:
            moving_image: Moving (source) image as numpy array.
            transform: Transform to apply.
            reference_image: Reference image for output dimensions.
            interpolator: Interpolation method.
            
        Returns:
            Transformed image.
        """
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK is required for MRIRegistration")
        
        # Convert numpy arrays to SimpleITK images
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        reference_sitk = sitk.GetImageFromArray(reference_image.astype(np.float32))
        
        # Set interpolator
        interp = sitk.sitkLinear
        if interpolator == "nearest":
            interp = sitk.sitkNearestNeighbor
        elif interpolator == "bspline":
            interp = sitk.sitkBSpline
        
        # Apply transform
        transformed_sitk = sitk.Resample(
            moving_sitk,
            reference_sitk,
            transform,
            interp,
            0.0,
            moving_sitk.GetPixelID(),
        )
        
        # Convert back to numpy array
        transformed_image = sitk.GetArrayFromImage(transformed_sitk)
        
        return transformed_image
    
    def _registration_callback(self, filter):
        """Callback for registration progress.
        
        Args:
            filter: SimpleITK registration filter.
        """
        if not self.verbose:
            return
        
        try:
            print(f"Iteration: {filter.GetElapsedIterations()}, Metric: {filter.GetMetricValue()}")
        except Exception:
            pass
    
    def batch_register(
        self,
        fixed_path: Union[str, Path],
        moving_path: Union[str, Path],
        output_path: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        fixed_mask_path: Optional[Union[str, Path]] = None,
        moving_mask_path: Optional[Union[str, Path]] = None,
        save_transforms: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Batch registration of multiple volumes.
        
        Args:
            fixed_path: Path to fixed (target) image or directory.
            moving_path: Path to moving (source) image or directory.
            output_path: Output directory for registered images.
            file_pattern: Glob pattern for input files when directories are provided.
            fixed_mask_path: Optional path to fixed image mask or directory.
            moving_mask_path: Optional path to moving image mask or directory.
            save_transforms: Whether to save transforms.
            
        Returns:
            Dictionary with registration metrics.
        """
        import glob
        
        fixed_path = Path(fixed_path)
        moving_path = Path(moving_path)
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Handle single file or directory
        if fixed_path.is_file() and moving_path.is_file():
            # Single file registration
            fixed_files = [fixed_path]
            moving_files = [moving_path]
            
            # Load masks if provided
            fixed_mask = None
            if fixed_mask_path and Path(fixed_mask_path).is_file():
                fixed_mask = self._load_volume(fixed_mask_path)
            
            moving_mask = None
            if moving_mask_path and Path(moving_mask_path).is_file():
                moving_mask = self._load_volume(moving_mask_path)
            
        elif fixed_path.is_dir() and moving_path.is_dir():
            # Directory-based registration
            fixed_files = sorted(glob.glob(str(fixed_path / file_pattern)))
            moving_files = sorted(glob.glob(str(moving_path / file_pattern)))
            
            if len(fixed_files) != len(moving_files):
                logger.warning(
                    f"Number of fixed files ({len(fixed_files)}) does not match "
                    f"number of moving files ({len(moving_files)}). Using available pairs."
                )
        else:
            raise ValueError("Both fixed_path and moving_path must be files or directories")
        
        # Process each file pair
        results = {}
        for i, (fixed_file, moving_file) in enumerate(zip(fixed_files, moving_files)):
            try:
                fixed_name = Path(fixed_file).name
                moving_name = Path(moving_file).name
                output_file = output_path / f"registered_{moving_name}"
                transform_file = output_path / f"transform_{moving_name}.txt"
                
                logger.info(f"Registering {moving_name} to {fixed_name}")
                
                # Load images
                fixed_image = self._load_volume(fixed_file)
                moving_image = self._load_volume(moving_file)
                
                # Load masks if directory-based
                fixed_mask = None
                moving_mask = None
                
                if fixed_mask_path and Path(fixed_mask_path).is_dir():
                    fixed_mask_file = Path(fixed_mask_path) / fixed_name
                    if fixed_mask_file.exists():
                        fixed_mask = self._load_volume(fixed_mask_file)
                
                if moving_mask_path and Path(moving_mask_path).is_dir():
                    moving_mask_file = Path(moving_mask_path) / moving_name
                    if moving_mask_file.exists():
                        moving_mask = self._load_volume(moving_mask_file)
                
                # Register volumes
                registered_image, transform = self.register_volumes(
                    fixed_image, moving_image, fixed_mask, moving_mask
                )
                
                # Save registered image
                self._save_volume(registered_image, output_file, reference_file=fixed_file)
                
                # Save transform if requested
                if save_transforms and transform is not None:
                    try:
                        import SimpleITK as sitk
                        sitk.WriteTransform(transform, str(transform_file))
                    except Exception as e:
                        logger.error(f"Failed to save transform: {e}")
                
                # Calculate metrics
                metrics = self._calculate_metrics(fixed_image, moving_image, registered_image)
                results[moving_name] = metrics
                
                logger.info(
                    f"Completed {moving_name}: "
                    f"Initial similarity: {metrics['initial_similarity']:.3f}, "
                    f"Final similarity: {metrics['final_similarity']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Error processing {moving_file}: {e}")
        
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
    
    def _calculate_metrics(
        self,
        fixed_image: np.ndarray,
        original_moving_image: np.ndarray,
        registered_image: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate registration metrics.
        
        Args:
            fixed_image: Fixed (target) image.
            original_moving_image: Original moving (source) image before registration.
            registered_image: Registered moving image.
            
        Returns:
            Dictionary of metrics.
        """
        # Create a foreground mask to exclude background voxels
        mean_image = (fixed_image + registered_image) / 2
        threshold = np.mean(mean_image) * 0.1
        mask = mean_image > threshold
        
        # Apply mask to images
        fixed_masked = fixed_image[mask]
        original_moving_masked = original_moving_image[mask] if original_moving_image.shape == fixed_image.shape else None
        registered_masked = registered_image[mask]
        
        # Calculate metrics
        # Mean squared error (lower is better)
        if original_moving_masked is not None:
            initial_mse = np.mean((fixed_masked - original_moving_masked) ** 2)
        else:
            initial_mse = float('nan')
        
        final_mse = np.mean((fixed_masked - registered_masked) ** 2)
        
        # Correlation coefficient (higher is better)
        if original_moving_masked is not None:
            initial_correlation = np.corrcoef(fixed_masked, original_moving_masked)[0, 1]
        else:
            initial_correlation = float('nan')
        
        final_correlation = np.corrcoef(fixed_masked, registered_masked)[0, 1]
        
        # Mutual information (higher is better) - simplified approximation
        # For proper MI calculation, use sklearn or other libraries
        initial_similarity = initial_correlation
        final_similarity = final_correlation
        
        return {
            "initial_mse": float(initial_mse),
            "final_mse": float(final_mse),
            "mse_improvement": float(initial_mse - final_mse) if not np.isnan(initial_mse) else float('nan'),
            "initial_correlation": float(initial_correlation),
            "final_correlation": float(final_correlation),
            "correlation_improvement": float(final_correlation - initial_correlation) if not np.isnan(initial_correlation) else float('nan'),
            "initial_similarity": float(initial_similarity),
            "final_similarity": float(final_similarity),
            "similarity_improvement": float(final_similarity - initial_similarity) if not np.isnan(initial_similarity) else float('nan'),
        }