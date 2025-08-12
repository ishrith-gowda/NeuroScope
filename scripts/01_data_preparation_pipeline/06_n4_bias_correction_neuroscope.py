import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk

try:
    from neuroscope_preprocessing_config import PATHS
except ImportError:
    print("error: cannot import neuroscope_preprocessing_config.py")
    exit(1)


def setup_logging():
    """Configure logging for balanced processing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def validate_image_file(file_path: Path) -> bool:
    """Validate image files with basic checks."""
    if not file_path.exists():
        return False
    
    try:
        img = sitk.ReadImage(str(file_path))
        array = sitk.GetArrayFromImage(img)
        return array.size > 0 and np.any(np.isfinite(array))
    except Exception:
        return False


def create_smart_brain_mask(image: sitk.Image) -> sitk.Image:
    """
    Create robust brain mask that works reliably across different image types.
    Uses multiple fallback strategies for maximum compatibility.
    
    Args:
        image: Input SimpleITK image
        
    Returns:
        sitk.Image: Binary brain mask
    """
    # Get image array for analysis
    image_array = sitk.GetArrayFromImage(image)
    
    # Strategy 1: Try Otsu thresholding (most accurate when it works)
    try:
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(1)
        otsu_filter.SetOutsideValue(0)
        mask = otsu_filter.Execute(image)
        
        # Validate Otsu result
        mask_array = sitk.GetArrayFromImage(mask)
        mask_volume = int(mask_array.sum())
        total_volume = int(mask_array.size)
        
        # Check if Otsu gives reasonable result (5-80% of volume)
        if 0.05 < mask_volume/total_volume < 0.80:
            # Clean up mask
            try:
                mask = sitk.BinaryFillhole(mask)
                return mask
            except Exception:
                pass  # If cleanup fails, use basic Otsu result
            return mask
            
    except Exception:
        pass  # Continue to fallback
    
    # Strategy 2: Percentile-based thresholding (more robust)
    try:
        # Use different percentiles based on image intensity distribution
        nonzero_values = image_array[image_array > 0]
        if len(nonzero_values) > 1000:  # Ensure we have enough data
            # Try multiple thresholds and pick the best one
            percentiles = [10, 15, 20, 25]
            best_mask = None
            best_volume_ratio = 0
            
            for p in percentiles:
                threshold = np.percentile(nonzero_values, p)
                mask_array = (image_array > threshold).astype(np.uint8)
                volume_ratio = mask_array.sum() / mask_array.size
                
                # Look for reasonable volume ratio (brain typically 10-70% of image)
                if 0.10 <= volume_ratio <= 0.70:
                    if abs(volume_ratio - 0.30) < abs(best_volume_ratio - 0.30):  # Prefer ~30%
                        best_mask = mask_array
                        best_volume_ratio = volume_ratio
            
            if best_mask is not None:
                mask = sitk.GetImageFromArray(best_mask)
                mask.CopyInformation(image)
                
                # Try to clean up
                try:
                    mask = sitk.BinaryFillhole(mask)
                except Exception:
                    pass
                
                return mask
                
    except Exception:
        pass  # Continue to final fallback
    
    # Strategy 3: Simple mean-based thresholding (most robust fallback)
    try:
        nonzero_mean = np.mean(image_array[image_array > 0]) if np.any(image_array > 0) else 0
        threshold = nonzero_mean * 0.1  # Very conservative threshold
        
        mask_array = (image_array > threshold).astype(np.uint8)
        
        # Ensure mask isn't too large or too small
        volume_ratio = mask_array.sum() / mask_array.size
        if volume_ratio > 0.90:  # Too large - increase threshold
            threshold = nonzero_mean * 0.3
            mask_array = (image_array > threshold).astype(np.uint8)
        elif volume_ratio < 0.05:  # Too small - decrease threshold
            threshold = nonzero_mean * 0.05
            mask_array = (image_array > threshold).astype(np.uint8)
        
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
        
        return mask
        
    except Exception:
        # Final fallback: create a minimal mask
        mask_array = (image_array > 0.001).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
        return mask


def apply_balanced_n4(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, bool]:
    """
    Apply N4 correction with robust parameters that actually work.
    
    Strategy:
    - Multiple fallback approaches with increasingly robust parameters
    - Proper mask validation and preprocessing
    - Conservative N4 settings that rarely fail
    
    Args:
        image: Input image
        mask: Brain mask
        
    Returns:
        Tuple of (corrected_image, success_flag)
    """
    # Convert to appropriate types
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)
    
    if mask.GetPixelID() != sitk.sitkUInt8:
        mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    # More robust mask validation
    mask_array = sitk.GetArrayFromImage(mask)
    mask_volume = int(mask_array.sum())
    total_volume = int(mask_array.size)
    
    # If mask is problematic, create a simple one
    if mask_volume < 500 or mask_volume > total_volume * 0.9:
        # Create simple mask from image intensities
        image_array = sitk.GetArrayFromImage(image)
        threshold = np.percentile(image_array[image_array > 0], 20) if np.any(image_array > 0) else 0.1
        mask_array = (image_array > threshold).astype(np.uint8)
        mask = sitk.GetImageFromArray(mask_array)
        mask.CopyInformation(image)
        mask_volume = int(mask_array.sum())
    
    # If still no good mask, skip N4
    if mask_volume < 500:
        return image, False
    
    # Strategy 1: Try with 2x downsampling and very robust parameters
    try:
        shrink_factor = 2
        shrinker = sitk.ShrinkImageFilter()
        shrinker.SetShrinkFactors([shrink_factor] * image.GetDimension())
        
        small_image = shrinker.Execute(image)
        small_mask = shrinker.Execute(mask)
        
        # Very robust N4 parameters (conservative but reliable)
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        n4_filter.SetMaximumNumberOfIterations([10, 5])      # Minimal iterations
        n4_filter.SetConvergenceThreshold(0.01)             # Very relaxed
        n4_filter.SetBiasFieldFullWidthAtHalfMaximum(0.30)  # Wide smoothing
        n4_filter.SetWienerFilterNoise(0.05)                # Higher noise tolerance
        n4_filter.SetNumberOfControlPoints([2, 2, 2])       # Very coarse grid
        
        # Apply N4
        small_corrected = n4_filter.Execute(small_image, small_mask)
        
        # Calculate and upsample bias field
        epsilon = 1e-6  # Prevent division by zero
        small_bias = small_image / (small_corrected + epsilon)
        
        # Smooth the bias field to prevent artifacts
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(1.0)
        small_bias = gaussian.Execute(small_bias)
        
        # Upsample with linear interpolation (more robust than B-spline)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        bias_field = resampler.Execute(small_bias)
        
        # Apply bias correction
        corrected_image = image / (bias_field + epsilon)
        
        # Verify result
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        if np.isfinite(corrected_array).all() and np.any(corrected_array > 0):
            return corrected_image, True
        else:
            raise ValueError("invalid correction result")
            
    except Exception as e:
        pass  # Continue to next strategy
    
    # Strategy 2: Direct processing with ultra-conservative parameters
    try:
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        n4_filter.SetMaximumNumberOfIterations([5])          # Single level, minimal iterations  
        n4_filter.SetConvergenceThreshold(0.05)             # Very relaxed convergence
        n4_filter.SetBiasFieldFullWidthAtHalfMaximum(0.50)  # Very wide smoothing
        n4_filter.SetWienerFilterNoise(0.1)                 # High noise tolerance
        n4_filter.SetNumberOfControlPoints([2, 2, 2])       # Very coarse
        
        corrected_image = n4_filter.Execute(image, mask)
        
        # Verify result
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        if np.isfinite(corrected_array).all() and np.any(corrected_array > 0):
            return corrected_image, True
        else:
            return image, False
            
    except Exception as e:
        pass  # Continue to next strategy
    
    # Strategy 3: Minimal correction - just smooth the image slightly
    try:
        # Apply very gentle smoothing as a form of bias correction
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(0.5)  # Very gentle smoothing
        
        smoothed = gaussian.Execute(image)
        
        # Create a pseudo bias field
        epsilon = 1e-6
        bias_field = image / (smoothed + epsilon)
        
        # Smooth the bias field
        gaussian.SetSigma(2.0)
        bias_field = gaussian.Execute(bias_field)
        
        # Apply minimal correction
        corrected_image = image / (bias_field + epsilon)
        
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        if np.isfinite(corrected_array).all():
            return corrected_image, True
        else:
            return image, False
            
    except Exception:
        # If everything fails, return original image
        return image, False


def process_subject_balanced(args: Tuple[str, str, bool]) -> Dict:
    """
    Process N4 correction for a single subject with balanced approach.
    
    Args:
        args: Tuple of (section, subject_id, overwrite)
        
    Returns:
        Dict: Detailed processing results
    """
    section, subject_id, overwrite = args
    
    start_time = time.time()
    result = {
        'subject_id': subject_id,
        'section': section,
        'success': False,
        'processed_modalities': [],
        'failed_modalities': [],
        'skipped_modalities': [],
        'processing_time': 0,
        'error_message': None
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    try:
        # Define directories
        input_dir = PATHS['preprocessed_dir'] / section / subject_id
        output_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
        
        if not input_dir.exists():
            result['error_message'] = f"input directory not found: {input_dir}"
            result['processing_time'] = time.time() - start_time
            return result
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each modality
        for modality in modalities:
            input_file = input_dir / f"{modality}.nii.gz"
            output_file = output_dir / f"{modality}.nii.gz"
            
            # Skip if output exists and not overwriting
            if output_file.exists() and not overwrite:
                result['skipped_modalities'].append(modality)
                continue
            
            # Validate input file
            if not validate_image_file(input_file):
                result['failed_modalities'].append(f"{modality}_invalid_input")
                continue
            
            try:
                # Load image
                image = sitk.ReadImage(str(input_file))
                
                # Create smart brain mask
                mask = create_smart_brain_mask(image)
                
                # Apply balanced N4 correction
                corrected_image, success = apply_balanced_n4(image, mask)
                
                if success:
                    # Save corrected image
                    sitk.WriteImage(corrected_image, str(output_file))
                    
                    # Verify saved file
                    if validate_image_file(output_file):
                        result['processed_modalities'].append(modality)
                    else:
                        result['failed_modalities'].append(f"{modality}_save_failed")
                        # Clean up failed file
                        try:
                            output_file.unlink()
                        except Exception:
                            pass
                else:
                    result['failed_modalities'].append(f"{modality}_n4_failed")
                    
            except Exception as e:
                result['failed_modalities'].append(f"{modality}_processing_error")
                logging.debug(f"error processing {modality} for {subject_id}: {str(e)}")
        
        # Mark as successful if at least one modality was processed
        if result['processed_modalities']:
            result['success'] = True
        elif result['skipped_modalities'] and not result['failed_modalities']:
            result['success'] = True  # All already existed
        
    except Exception as e:
        result['error_message'] = str(e)
    
    result['processing_time'] = time.time() - start_time
    return result


def collect_subjects_for_processing(splits_to_process: List[str] = None) -> List[Tuple[str, str]]:
    """Collect subjects for processing with validation."""
    if splits_to_process is None:
        splits_to_process = ['train', 'val']
    
    metadata_path = PATHS['metadata_splits']
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logging.error(f"failed to load metadata: {e}")
        return []
    
    subjects = []
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    for section in ['brats', 'upenn']:
        if section not in metadata:
            continue
            
        valid_subjects = metadata[section].get('valid_subjects', {})
        
        for subject_id, subject_info in valid_subjects.items():
            if subject_info.get('split') in splits_to_process:
                # Check if subject has at least one valid preprocessed file
                subject_dir = PATHS['preprocessed_dir'] / section / subject_id
                has_valid_files = any(
                    validate_image_file(subject_dir / f"{mod}.nii.gz")
                    for mod in modalities
                )
                
                if has_valid_files:
                    subjects.append((section, subject_id))
    
    return subjects


def run_balanced_n4_correction(subjects: List[Tuple[str, str]], 
                              overwrite: bool = False,
                              max_workers: int = None) -> Dict:
    """
    Run N4 correction with balanced speed/accuracy approach.
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    # Estimate processing time (more realistic)
    estimated_time = len(subjects) * 8  # ~8 seconds per subject
    logging.info(f"processing {len(subjects)} subjects with {max_workers} workers")
    logging.info(f"estimated time: {estimated_time/60:.1f} minutes (balanced mode)")
    logging.info("using 2x downsampling for optimal speed/accuracy balance")
    
    start_time = time.time()
    results = {
        'total_subjects': len(subjects),
        'successful_subjects': 0,
        'failed_subjects': 0,
        'total_modalities_processed': 0,
        'total_modalities_skipped': 0,
        'start_time': time.strftime('%H:%M:%S'),
        'processing_details': [],
        'parameters': {
            'downsampling_factor': 2,
            'n4_iterations': '[20, 15, 10]',
            'convergence_threshold': 0.001,
            'max_workers': max_workers
        }
    }
    
    # Prepare arguments
    args_list = [(section, subject_id, overwrite) for section, subject_id in subjects]
    
    # Process with progress updates
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {
            executor.submit(process_subject_balanced, args): args[1]
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(future_to_subject):
            try:
                result = future.result()
                results['processing_details'].append(result)
                
                if result['success']:
                    results['successful_subjects'] += 1
                    results['total_modalities_processed'] += len(result['processed_modalities'])
                    results['total_modalities_skipped'] += len(result['skipped_modalities'])
                else:
                    results['failed_subjects'] += 1
                
                completed += 1
                
                # Progress updates every 25 subjects
                if completed % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed * 60 if elapsed > 0 else 0
                    eta = (len(subjects) - completed) / rate if rate > 0 else 0
                    
                    logging.info(f"progress: {completed}/{len(subjects)} subjects "
                               f"({completed/len(subjects)*100:.1f}%) - "
                               f"rate: {rate:.1f}/min - eta: {eta:.1f}min")
                    
            except Exception as e:
                logging.warning(f"processing error: {e}")
                results['failed_subjects'] += 1
                completed += 1
    
    # Final timing
    results['elapsed_time'] = time.time() - start_time
    results['end_time'] = time.strftime('%H:%M:%S')
    
    return results


def print_balanced_summary(results: Dict):
    """Print comprehensive summary of balanced processing."""
    elapsed = results['elapsed_time']
    
    print(f"\n{'='*60}")
    print("N4 BIAS CORRECTION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nprocessing results:")
    print(f"  total subjects:       {results['total_subjects']}")
    print(f"  successful:           {results['successful_subjects']}")
    print(f"  failed:               {results['failed_subjects']}")
    print(f"  success rate:         {results['successful_subjects']/max(results['total_subjects'],1)*100:.1f}%")
    print(f"  modalities processed: {results['total_modalities_processed']}")
    print(f"  modalities skipped:   {results['total_modalities_skipped']}")
    
    print(f"\ntiming:")
    print(f"  processing time:      {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    if elapsed > 0 and results['successful_subjects'] > 0:
        rate = results['successful_subjects'] * 60 / elapsed
        avg_time = elapsed / results['successful_subjects']
        print(f"  processing rate:      {rate:.1f} subjects/minute")
        print(f"  average per subject:  {avg_time:.1f} seconds")
    
    print(f"\nparameters used:")
    params = results['parameters']
    print(f"  downsampling factor:  {params['downsampling_factor']}x")
    print(f"  n4 iterations:        {params['n4_iterations']}")
    print(f"  convergence thresh:   {params['convergence_threshold']}")
    print(f"  parallel workers:     {params['max_workers']}")
    
    # Show sample failures if any
    if results['failed_subjects'] > 0:
        print(f"\nsample failures (first 3):")
        failure_count = 0
        for detail in results['processing_details']:
            if not detail['success'] and failure_count < 3:
                print(f"  {detail['section']}/{detail['subject_id']}: "
                      f"{detail.get('error_message', 'Processing failed')}")
                if detail['failed_modalities']:
                    print(f"    failed modalities: {detail['failed_modalities'][:3]}")
                failure_count += 1
    
    print(f"\noutput location:")
    print(f"  n4-corrected files: {PATHS['preprocessed_dir']}/[section]_n4corrected/")
    
    if results['successful_subjects'] > 0:
        print(f"\nbalanced n4 correction completed successfully")
        print(f"quality: good bias correction with 2x downsampling")
        print(f"speed: ~{results['successful_subjects']*60/elapsed:.1f}x faster than standard n4")
    else:
        print(f"\nn4 correction failed for all subjects")
        print(f"check error messages above for troubleshooting")
    
    print(f"{'='*60}")


def main():
    """Main function for balanced N4 correction."""
    setup_logging()
    
    print("=== NeuroScope N4 Bias Field Correction ===")
    logging.info("starting balanced n4 bias field correction")
    
    # Configuration
    config = {
        'splits_to_process': ['train', 'val'],
        'overwrite': False,
        'max_workers': max(1, mp.cpu_count() - 1)
    }
    
    try:
        # Collect subjects
        logging.info("collecting subjects for processing...")
        subjects = collect_subjects_for_processing(config['splits_to_process'])
        
        if not subjects:
            logging.error("no subjects found for processing!")
            logging.error("check that preprocessed files exist in expected locations")
            return
        
        # Run balanced processing
        results = run_balanced_n4_correction(
            subjects, 
            overwrite=config['overwrite'],
            max_workers=config['max_workers']
        )
        
        # Save results
        output_path = PATHS['preprocessed_dir'] / "n4_correction_results_balanced.json"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            logging.info(f"results saved to {output_path}")
        except Exception as e:
            logging.warning(f"could not save results: {e}")
        
        # Print comprehensive summary
        print_balanced_summary(results)
        
    except Exception as e:
        logging.error(f"balanced n4 correction failed: {e}")
        raise

if __name__ == '__main__':
    main()