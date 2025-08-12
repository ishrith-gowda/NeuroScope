import os
import json
import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """Configure logging format and level for N4 bias correction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_metadata_with_validation(metadata_path: Path) -> Dict[str, Any]:
    """
    Load and validate the metadata with splits.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Dict: Validated metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If metadata structure is invalid
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info("loaded metadata from: %s", metadata_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json in metadata file: {e}")
    
    # Validate structure
    required_sections = ['brats', 'upenn']
    for section in required_sections:
        if section not in metadata:
            raise ValueError(f"missing section: {section}")
        if 'valid_subjects' not in metadata[section]:
            raise ValueError(f"missing 'valid_subjects' in section: {section}")
    
    total_subjects = sum(len(metadata[section]['valid_subjects']) for section in required_sections)
    if total_subjects == 0:
        raise ValueError("no valid subjects found in metadata")
    
    logging.info("metadata validation passed: %d total subjects", total_subjects)
    return metadata


def verify_preprocessed_file(file_path: Path) -> bool:
    """
    Verify that a preprocessed file exists and is readable.
    
    Args:
        file_path: Path to the preprocessed NIfTI file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path.exists():
        return False
    
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        # Basic sanity checks
        if arr.size == 0 or not np.isfinite(arr).all():
            return False
        
        return True
        
    except Exception:
        return False


def create_fast_brain_mask(image: sitk.Image, background_threshold: float = 0.01) -> sitk.Image:
    """
    Create a fast brain mask optimized for N4 correction speed.
    
    Args:
        image: Input SimpleITK image
        background_threshold: Threshold for background exclusion
        
    Returns:
        sitk.Image: Binary brain mask
    """
    # For normalized images, simple thresholding is very fast and effective
    arr = sitk.GetArrayFromImage(image)
    mask_array = (arr > background_threshold).astype(np.uint8)
    
    # Create mask image
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    
    # Minimal morphological cleanup (very fast)
    mask = sitk.BinaryFillhole(mask)
    
    return mask


def apply_fast_n4_correction(
    image: sitk.Image,
    mask: sitk.Image,
    shrink_factor: int = 4,
    max_iterations: List[int] = None,
    convergence_threshold: float = 0.0001
) -> sitk.Image:
    """
    Apply highly optimized N4 bias field correction.
    
    Args:
        image: Input SimpleITK image
        mask: Brain mask
        shrink_factor: Factor to shrink image for faster processing (default: 4)
        max_iterations: Maximum iterations per level (default: [20, 10])
        convergence_threshold: Convergence threshold (default: 0.0001)
        
    Returns:
        sitk.Image: Bias-corrected image
    """
    if max_iterations is None:
        max_iterations = [20, 10]  # Reduced from typical [50, 50, 50, 50]
    
    # Create N4 corrector with aggressive speed optimizations
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    # Key optimization: reduce iterations dramatically
    corrector.SetMaximumNumberOfIterations(max_iterations)
    
    # Optimization: increase convergence threshold (less precise but much faster)
    corrector.SetConvergenceThreshold(convergence_threshold)
    
    # Optimization: reduce number of fitting levels
    corrector.SetNumberOfFittingLevels(len(max_iterations))
    
    # Optimization: use wider B-spline grid (less precise but faster)
    corrector.SetNumberOfControlPoints(4)  # Reduced from default 8
    
    # Optimization: shrink image for processing if large
    if shrink_factor > 1:
        # Shrink image and mask
        shrinker = sitk.ShrinkImageFilter()
        shrinker.SetShrinkFactors([shrink_factor] * image.GetDimension())
        
        small_image = shrinker.Execute(image)
        small_mask = shrinker.Execute(mask)
        
        # Apply N4 to smaller image
        try:
            small_corrected = corrector.Execute(small_image, small_mask)
            
            # Resample bias field back to original size
            bias_field = small_image / (small_corrected + 1e-8)
            
            # Resample to original size
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetInterpolator(sitk.sitkBSpline)
            bias_field_full = resampler.Execute(bias_field)
            
            # Apply bias correction
            corrected_image = image / (bias_field_full + 1e-8)
            
        except Exception as e:
            logging.warning("shrunk N4 correction failed, trying full resolution: %s", e)
            # Fallback to full resolution
            corrected_image = corrector.Execute(image, mask)
    else:
        # Direct N4 correction without shrinking
        corrected_image = corrector.Execute(image, mask)
    
    return corrected_image


def process_subject_n4_correction(
    section: str,
    subject_id: str,
    modalities: List[str] = None,
    shrink_factor: int = 4,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Apply N4 bias correction to all modalities of a single subject.
    
    Args:
        section: Dataset section ('brats' or 'upenn')
        subject_id: Subject identifier
        modalities: List of modalities to process (default: ['t1', 't1gd', 't2', 'flair'])
        shrink_factor: Image shrinking factor for speed optimization
        overwrite: Whether to overwrite existing corrected files
        
    Returns:
        Dict with processing results
    """
    if modalities is None:
        modalities = ['t1', 't1gd', 't2', 'flair']
    
    start_time = time.time()
    results = {
        'subject_id': subject_id,
        'section': section,
        'success': False,
        'modalities_processed': [],
        'modalities_failed': [],
        'modalities_skipped': [],
        'processing_time': 0,
        'error_message': None,
        'debug_info': {}
    }
    
    try:
        # Define paths
        preprocessed_dir = PATHS['preprocessed_dir'] / section / subject_id
        corrected_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
        
        # Debug: Check if preprocessed directory exists
        if not preprocessed_dir.exists():
            results['error_message'] = f"preprocessed directory not found: {preprocessed_dir}"
            results['processing_time'] = time.time() - start_time
            return results
        
        corrected_dir.mkdir(parents=True, exist_ok=True)
        results['debug_info']['preprocessed_dir'] = str(preprocessed_dir)
        results['debug_info']['corrected_dir'] = str(corrected_dir)
        
        # List available files for debugging
        available_files = list(preprocessed_dir.glob("*.nii.gz")) if preprocessed_dir.exists() else []
        results['debug_info']['available_files'] = [f.name for f in available_files]
        
        # Process each modality
        for modality in modalities:
            input_path = preprocessed_dir / f"{modality}.nii.gz"
            output_path = corrected_dir / f"{modality}.nii.gz"
            
            # Skip if output exists and not overwriting
            if output_path.exists() and not overwrite:
                results['modalities_skipped'].append(f"{modality}_exists")
                continue
            
            # Verify input file exists and is valid
            if not verify_preprocessed_file(input_path):
                results['modalities_failed'].append(f"{modality}_input_invalid")
                results['debug_info'][f'{modality}_input_path'] = str(input_path)
                results['debug_info'][f'{modality}_input_exists'] = input_path.exists()
                continue
            
            try:
                mod_start = time.time()
                
                # Load image
                image = sitk.ReadImage(str(input_path))
                
                # Debug: Check image properties
                img_size = image.GetSize()
                img_spacing = image.GetSpacing()
                
                # Convert to float for processing
                if image.GetPixelID() != sitk.sitkFloat32:
                    image = sitk.Cast(image, sitk.sitkFloat32)
                
                # Create fast brain mask with validation
                mask = create_fast_brain_mask(image)
                
                # Validate mask
                mask_array = sitk.GetArrayFromImage(mask)
                mask_volume = int(mask_array.sum())
                total_volume = int(mask_array.size)
                
                if mask_volume < 1000:  # Too small brain mask
                    results['modalities_failed'].append(f"{modality}_mask_too_small")
                    results['debug_info'][f'{modality}_mask_volume'] = mask_volume
                    continue
                
                if mask_volume > total_volume * 0.8:  # Mask too large (likely invalid)
                    results['modalities_failed'].append(f"{modality}_mask_too_large")
                    results['debug_info'][f'{modality}_mask_volume'] = mask_volume
                    continue
                
                # Apply optimized N4 correction with better error handling
                try:
                    corrected_image = apply_fast_n4_correction(
                        image, mask, 
                        shrink_factor=shrink_factor,
                        max_iterations=[20, 10],
                        convergence_threshold=0.001
                    )
                except Exception as n4_error:
                    # Fallback: try with minimal shrinking
                    try:
                        corrected_image = apply_fast_n4_correction(
                            image, mask, 
                            shrink_factor=2,  # Less aggressive shrinking
                            max_iterations=[10],  # Single level
                            convergence_threshold=0.01
                        )
                    except Exception as fallback_error:
                        results['modalities_failed'].append(f"{modality}_n4_failed")
                        results['debug_info'][f'{modality}_n4_error'] = str(n4_error)
                        results['debug_info'][f'{modality}_fallback_error'] = str(fallback_error)
                        continue
                
                # Ensure corrected image has reasonable values
                corrected_array = sitk.GetArrayFromImage(corrected_image)
                if not np.isfinite(corrected_array).all():
                    results['modalities_failed'].append(f"{modality}_output_invalid_values")
                    continue
                
                # Save corrected image
                sitk.WriteImage(corrected_image, str(output_path))
                
                # Verify output
                if verify_preprocessed_file(output_path):
                    results['modalities_processed'].append(modality)
                    mod_time = time.time() - mod_start
                    results['debug_info'][f'{modality}_processing_time'] = mod_time
                else:
                    results['modalities_failed'].append(f"{modality}_output_verification_failed")
                
            except Exception as e:
                results['modalities_failed'].append(f"{modality}_processing_error")
                results['debug_info'][f'{modality}_error'] = str(e)
        
        # Mark as successful if at least one modality was processed
        if results['modalities_processed']:
            results['success'] = True
        elif not results['modalities_failed'] and results['modalities_skipped']:
            results['success'] = True  # All files already existed and were skipped
        
        results['processing_time'] = time.time() - start_time
        
    except Exception as e:
        results['error_message'] = str(e)
        results['processing_time'] = time.time() - start_time
    
    return results


def process_subject_wrapper(args: Tuple[str, str, List[str], int, bool]) -> Dict[str, Any]:
    """
    Wrapper function for multiprocessing.
    
    Args:
        args: Tuple of (section, subject_id, modalities, shrink_factor, overwrite)
        
    Returns:
        Dict with processing results
    """
    section, subject_id, modalities, shrink_factor, overwrite = args
    return process_subject_n4_correction(section, subject_id, modalities, shrink_factor, overwrite)


def collect_subjects_for_processing(
    metadata: Dict[str, Any],
    splits_to_process: List[str] = None
) -> List[Tuple[str, str]]:
    """
    Collect subjects that need N4 bias correction.
    
    Args:
        metadata: Complete metadata dictionary
        splits_to_process: List of splits to process (default: ['train', 'val'])
        
    Returns:
        List of (section, subject_id) tuples
    """
    if splits_to_process is None:
        splits_to_process = ['train', 'val']
    
    subjects_to_process = []
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    for section in ['brats', 'upenn']:
        for subject_id, subject_info in metadata[section]['valid_subjects'].items():
            # Check if subject is in target splits
            if subject_info.get('split') not in splits_to_process:
                continue
            
            # Check if subject has preprocessed files
            preprocessed_dir = PATHS['preprocessed_dir'] / section / subject_id
            has_preprocessed_files = any(
                verify_preprocessed_file(preprocessed_dir / f"{mod}.nii.gz") 
                for mod in modalities
            )
            
            if has_preprocessed_files:
                subjects_to_process.append((section, subject_id))
    
    return subjects_to_process


def run_parallel_n4_correction(
    subjects_to_process: List[Tuple[str, str]],
    modalities: List[str] = None,
    shrink_factor: int = 4,
    overwrite: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run N4 bias correction on multiple subjects in parallel.
    
    Args:
        subjects_to_process: List of (section, subject_id) tuples
        modalities: List of modalities to process
        shrink_factor: Image shrinking factor for speed
        overwrite: Whether to overwrite existing files
        max_workers: Maximum number of parallel workers (default: CPU count - 1)
        
    Returns:
        Dict with processing summary
    """
    if modalities is None:
        modalities = ['t1', 't1gd', 't2', 'flair']
    
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    start_time = time.time()
    logging.info("starting parallel n4 correction with %d workers", max_workers)
    
    # Prepare arguments for parallel processing
    args_list = [
        (section, subject_id, modalities, shrink_factor, overwrite)
        for section, subject_id in subjects_to_process
    ]
    
    results = {
        'successful_subjects': 0,
        'failed_subjects': 0,
        'total_subjects': len(subjects_to_process),
        'total_modalities_processed': 0,
        'processing_details': [],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'shrink_factor': shrink_factor,
            'max_workers': max_workers,
            'overwrite': overwrite,
            'modalities': modalities
        }
    }
    
    # Process subjects in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_subject = {
            executor.submit(process_subject_wrapper, args): args[1]  # subject_id
            for args in args_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_subject):
            subject_id = future_to_subject[future]
            try:
                result = future.result()
                results['processing_details'].append(result)
                
                if result['success']:
                    results['successful_subjects'] += 1
                    results['total_modalities_processed'] += len(result['modalities_processed'])
                else:
                    results['failed_subjects'] += 1
                
                completed += 1
                if completed % 50 == 0:  # Log progress every 50 subjects
                    logging.info("completed %d/%d subjects", completed, len(subjects_to_process))
                    
            except Exception as e:
                logging.error("parallel processing error for subject %s: %s", subject_id, e)
                results['failed_subjects'] += 1
    
    # Add timing info
    elapsed_time = time.time() - start_time
    results['elapsed_time_seconds'] = elapsed_time
    results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    results['average_time_per_subject'] = elapsed_time / max(len(subjects_to_process), 1)
    
    return results


def save_n4_correction_results(results: Dict[str, Any], output_path: Path) -> None:
    """
    Save N4 correction results to JSON file.
    
    Args:
        results: Processing results dictionary
        output_path: Path to save results
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        
        file_size = output_path.stat().st_size
        logging.info("n4 correction results saved to: %s (%.1f KB)", output_path, file_size / 1024)
        
    except Exception as e:
        logging.error("failed to save n4 correction results: %s", e)
        raise


def print_n4_correction_summary(results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of N4 bias correction results.
    
    Args:
        results: N4 correction results
    """
    print("\n" + "="*70)
    print("N4 BIAS FIELD CORRECTION SUMMARY")
    print("="*70)
    
    # Processing summary
    print(f"\nProcessing Summary:")
    print(f"  Total subjects:       {results['total_subjects']}")
    print(f"  Successful:           {results['successful_subjects']}")
    print(f"  Failed:               {results['failed_subjects']}")
    print(f"  Success rate:         {results['successful_subjects']/max(results['total_subjects'],1)*100:.1f}%")
    print(f"  Total modalities:     {results['total_modalities_processed']}")
    print(f"  Processing time:      {results['elapsed_time_seconds']:.1f} seconds")
    print(f"  Average per subject:  {results['average_time_per_subject']:.1f} seconds")
    
    # Performance info
    params = results['parameters']
    print(f"\nPerformance Parameters:")
    print(f"  Shrink factor:        {params['shrink_factor']}x")
    print(f"  Parallel workers:     {params['max_workers']}")
    print(f"  Max iterations:       [20, 10] (optimized)")
    print(f"  Convergence thresh:   0.001 (relaxed)")
    
    # Calculate throughput
    if results['elapsed_time_seconds'] > 0:
        subjects_per_hour = results['successful_subjects'] * 3600 / results['elapsed_time_seconds']
        modalities_per_hour = results['total_modalities_processed'] * 3600 / results['elapsed_time_seconds']
        print(f"\nThroughput:")
        print(f"  Subjects/hour:        {subjects_per_hour:.1f}")
        print(f"  Modalities/hour:      {modalities_per_hour:.1f}")
    
    # Detailed failure analysis
    if results['failed_subjects'] > 0:
        print(f"\nFailure Analysis:")
        failure_reasons = {}
        input_issues = 0
        mask_issues = 0
        n4_issues = 0
        output_issues = 0
        other_issues = 0
        
        for detail in results['processing_details']:
            if not detail['success']:
                for failed_mod in detail['modalities_failed']:
                    if 'input' in failed_mod:
                        input_issues += 1
                    elif 'mask' in failed_mod:
                        mask_issues += 1
                    elif 'n4' in failed_mod:
                        n4_issues += 1
                    elif 'output' in failed_mod:
                        output_issues += 1
                    else:
                        other_issues += 1
        
        if input_issues > 0:
            print(f"  Input file issues:    {input_issues}")
        if mask_issues > 0:
            print(f"  Brain mask issues:    {mask_issues}")
        if n4_issues > 0:
            print(f"  N4 algorithm issues:  {n4_issues}")
        if output_issues > 0:
            print(f"  Output file issues:   {output_issues}")
        if other_issues > 0:
            print(f"  Other issues:         {other_issues}")
    
    # Debug information for first few failures
    if results['failed_subjects'] > 0:
        print(f"\nDebugging Information (first 3 failures):")
        failure_count = 0
        for detail in results['processing_details']:
            if not detail['success'] and failure_count < 3:
                print(f"\n  Subject: {detail['section']}/{detail['subject_id']}")
                if detail.get('error_message'):
                    print(f"    Error: {detail['error_message']}")
                
                debug_info = detail.get('debug_info', {})
                if 'preprocessed_dir' in debug_info:
                    print(f"    Preprocessed dir: {debug_info['preprocessed_dir']}")
                if 'available_files' in debug_info:
                    print(f"    Available files: {debug_info['available_files']}")
                
                failed_mods = detail.get('modalities_failed', [])
                if failed_mods:
                    print(f"    Failed modalities: {failed_mods[:3]}")  # Show first 3
                
                failure_count += 1
    
    print(f"\nOutput Location:")
    print(f"  N4-corrected files saved in: {PATHS['preprocessed_dir']}")
    print(f"  Directory naming: [section]_n4corrected/[subject_id]/")
    
    # Next steps recommendations
    print(f"\nNext Steps:")
    if results['successful_subjects'] > 0:
        print(f"  ✅ {results['successful_subjects']} subjects processed successfully")
        print(f"  1. Re-run bias assessment (script 05) on N4-corrected data")
        print(f"  2. Compare bias metrics before/after N4 correction")
        print(f"  3. Use N4-corrected data for CycleGAN training")
    else:
        print(f"  ❌ No subjects processed successfully - troubleshooting needed")
        print(f"  1. Check if preprocessed files exist in expected locations")
        print(f"  2. Verify SimpleITK N4 installation: pip install SimpleITK")
        print(f"  3. Check debug information above for specific issues")
    
    print("="*70)


def diagnose_preprocessing_files(metadata: Dict[str, Any], max_subjects: int = 10) -> None:
    """
    Diagnose preprocessing file issues by checking a sample of subjects.
    
    Args:
        metadata: Complete metadata dictionary
        max_subjects: Maximum number of subjects to check for diagnosis
    """
    logging.info("=== PREPROCESSING FILES DIAGNOSIS ===")
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    checked_subjects = 0
    
    for section in ['brats', 'upenn']:
        logging.info("checking %s section...", section)
        
        for subject_id, subject_info in metadata[section]['valid_subjects'].items():
            if checked_subjects >= max_subjects:
                break
                
            # Check if in train/val splits
            if subject_info.get('split') not in ['train', 'val']:
                continue
            
            preprocessed_dir = PATHS['preprocessed_dir'] / section / subject_id
            
            logging.info("subject %s/%s:", section, subject_id)
            logging.info("  preprocessed dir: %s", preprocessed_dir)
            logging.info("  directory exists: %s", preprocessed_dir.exists())
            
            if preprocessed_dir.exists():
                all_files = list(preprocessed_dir.glob("*"))
                nifti_files = list(preprocessed_dir.glob("*.nii.gz"))
                
                logging.info("  all files: %s", [f.name for f in all_files])
                logging.info("  nifti files: %s", [f.name for f in nifti_files])
                
                for modality in modalities:
                    mod_path = preprocessed_dir / f"{modality}.nii.gz"
                    exists = mod_path.exists()
                    valid = verify_preprocessed_file(mod_path) if exists else False
                    
                    logging.info("  %s: exists=%s, valid=%s", modality, exists, valid)
                    
                    if exists and not valid:
                        # Additional diagnosis for invalid files
                        try:
                            img = sitk.ReadImage(str(mod_path))
                            arr = sitk.GetArrayFromImage(img)
                            logging.info("    size: %s, dtype: %s, finite: %s", 
                                       arr.shape, arr.dtype, np.isfinite(arr).all())
                        except Exception as e:
                            logging.info("    read error: %s", e)
            else:
                logging.info("  directory does not exist")
            
            checked_subjects += 1
            
            if checked_subjects >= max_subjects:
                break
    
    logging.info("diagnosis complete for %d subjects", checked_subjects)


def main() -> None:
    """
    Main function to run fast N4 bias field correction.
    """
    configure_logging()
    
    logging.info("=== FAST N4 BIAS FIELD CORRECTION ===")
    logging.info("using neuroscope_preprocessing_config.py for path management")
    
    # Define paths
    metadata_path = PATHS['metadata_splits']
    output_path = PATHS['preprocessed_dir'] / "n4_correction_results.json"
    
    logging.info("input metadata: %s", metadata_path)
    logging.info("output results: %s", output_path)
    
    # Configuration parameters (optimized for speed)
    config = {
        'splits_to_process': ['train', 'val'],  # Only process training data
        'shrink_factor': 4,  # 4x shrinking for 16x speed improvement
        'overwrite': False,  # Skip existing files
        'max_workers': max(1, mp.cpu_count() - 1)  # Use most available CPUs
    }
    
    logging.info("configuration: %s", config)
    
    try:
        # Step 1: Load and validate metadata
        logging.info("step 1: loading metadata...")
        metadata = load_metadata_with_validation(metadata_path)
        
        # Step 1.5: Diagnose preprocessing files
        logging.info("step 1.5: diagnosing preprocessing files...")
        diagnose_preprocessing_files(metadata, max_subjects=5)
        
        # Step 2: Collect subjects for processing
        logging.info("step 2: collecting subjects for n4 correction...")
        subjects_to_process = collect_subjects_for_processing(
            metadata, 
            config['splits_to_process']
        )
        
        if not subjects_to_process:
            logging.error("no subjects found for processing")
            logging.info("this suggests preprocessed files are not in expected locations")
            return
        
        logging.info("found %d subjects for n4 correction", len(subjects_to_process))
        
        # Step 3: Run parallel N4 correction
        logging.info("step 3: running parallel n4 bias correction...")
        logging.info("estimated time: %.1f minutes (based on 4-6 sec/subject)", 
                    len(subjects_to_process) * 5 / 60)
        
        results = run_parallel_n4_correction(
            subjects_to_process,
            shrink_factor=config['shrink_factor'],
            overwrite=config['overwrite'],
            max_workers=config['max_workers']
        )
        
        # Step 4: Save results
        logging.info("step 4: saving processing results...")
        save_n4_correction_results(results, output_path)
        
        # Step 5: Display summary
        print_n4_correction_summary(results)
        
        # Recommendation for next steps
        if results['successful_subjects'] > 0:
            logging.info("✅ N4 bias correction completed successfully!")
        else:
            logging.error("❌ N4 bias correction failed for all subjects")
            logging.info("Check the debugging information above to identify issues")
        
    except Exception as e:
        logging.error("n4 bias correction failed: %s", e)
        raise


if __name__ == '__main__':
    main()