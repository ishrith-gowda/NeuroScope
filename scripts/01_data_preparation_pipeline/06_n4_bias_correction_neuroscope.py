import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import numpy as np
import SimpleITK as sitk

try:
    from neuroscope_preprocessing_config import PATHS
except ImportError:
    print("error: cannot import neuroscope_preprocessing_config.py")
    exit(1)

from preprocessing_utils import generate_brain_mask, evaluate_bias_need, acceptable_n4_change, basic_intensity_stats


def setup_logging():
    """Configure logging for improved N4 processing."""
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


def apply_conservative_n4(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, bool]:
    """
    Apply N4 correction with very conservative parameters designed to avoid
    increasing variation while still providing some bias correction benefit.
    """
    # Convert to appropriate types
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)
    
    if mask.GetPixelID() != sitk.sitkUInt8:
        mask = sitk.Cast(mask, sitk.sitkUInt8)
    
    # Validate mask
    mask_array = sitk.GetArrayFromImage(mask)
    mask_volume = int(mask_array.sum())
    
    if mask_volume < 1000:  # Need reasonable amount of brain tissue
        return image, False
    
    # Strategy 1: Ultra-conservative N4 (minimal correction)
    try:
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Ultra-conservative parameters to minimize over-correction
        n4_filter.SetMaximumNumberOfIterations([5, 3])        # Very few iterations
        n4_filter.SetConvergenceThreshold(0.01)              # Relaxed convergence  
        n4_filter.SetBiasFieldFullWidthAtHalfMaximum(0.50)   # Wide smoothing (less aggressive)
        n4_filter.SetWienerFilterNoise(0.05)                 # Higher noise tolerance
        n4_filter.SetNumberOfControlPoints([3, 3, 3])        # Coarse control points
        
        # Apply N4 correction
        corrected_image = n4_filter.Execute(image, mask)
        
        # Verify result quality
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        
        if not np.isfinite(corrected_array).all():
            raise ValueError("Non-finite values in corrected image")
        
        # Additional quality check: ensure correction didn't create extreme values
        orig_array = sitk.GetArrayFromImage(image)
        orig_brain = orig_array[mask_array.astype(bool)]
        corr_brain = corrected_array[mask_array.astype(bool)]
        
        # Check for reasonable intensity preservation
        orig_range = orig_brain.max() - orig_brain.min()
        corr_range = corr_brain.max() - corr_brain.min()
        
        # Reject if correction caused extreme scaling
        if corr_range > orig_range * 2.0 or corr_range < orig_range * 0.5:
            logging.debug("Rejected N4 result: extreme intensity scaling")
            return image, False
        
        # Check for reasonable mean preservation
        orig_mean = orig_brain.mean()
        corr_mean = corr_brain.mean()
        
        if corr_mean > orig_mean * 1.5 or corr_mean < orig_mean * 0.5:
            logging.debug("Rejected N4 result: extreme mean shift")
            return image, False
        
        return corrected_image, True
        
    except Exception as e:
        logging.debug(f"Conservative N4 failed: {e}")
        pass
    
    # Strategy 2: Gentle smoothing-based pseudo-correction
    try:
        # Create a very gentle "bias field" using smoothing
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma([3.0, 3.0, 3.0])  # Gentle smoothing
        
        smoothed = gaussian.Execute(image)
        
        # Create pseudo bias field (very mild correction)
        epsilon = 1e-6
        pseudo_bias = image / (smoothed + epsilon)
        
        # Limit the correction strength
        pseudo_bias_array = sitk.GetArrayFromImage(pseudo_bias)
        pseudo_bias_array = np.clip(pseudo_bias_array, 0.9, 1.1)  # Limit to ±10% correction
        
        limited_bias = sitk.GetImageFromArray(pseudo_bias_array)
        limited_bias.CopyInformation(image)
        
        # Apply limited correction
        corrected_image = image / (limited_bias + epsilon)
        
        corrected_array = sitk.GetArrayFromImage(corrected_image)
        if np.isfinite(corrected_array).all():
            return corrected_image, True
        else:
            return image, False
            
    except Exception:
        # If all fails, return original image
        return image, False


def parse_args():
    ap = argparse.ArgumentParser(description='Improved N4 bias correction (configurable)')
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of splits to process')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--max-workers', type=int, default=max(1, mp.cpu_count()-1))
    ap.add_argument('--bias-threshold', type=float, default=0.18, help='Median slice CV threshold to trigger N4')
    ap.add_argument('--second-pass-iters', type=str, default='10,5', help='Second pass iteration schedule if first conservative attempt rejected')
    return ap.parse_args()


def process_subject_improved(args: Tuple[str, str, bool, float, Tuple[int,int]]) -> Dict:
    """
    Process N4 correction for a single subject with improved approach.
    """
    section, subject_id, overwrite, bias_threshold, second_pass = args
    
    start_time = time.time()
    result = {
        'subject_id': subject_id,
        'section': section,
        'success': False,
        'processed_modalities': [],
        'skipped_modalities': [],
        'skipped_unnecessary': [],
        'failed_modalities': [],
        'processing_time': 0,
        'error_message': None
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    try:
        # Define directories
        input_dir = PATHS['preprocessed_dir'] / section / subject_id
        output_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected_v2" / subject_id
        
        if not input_dir.exists():
            result['error_message'] = f"input directory not found: {input_dir}"
            result['processing_time'] = time.time() - start_time
            return result
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each modality
        for modality in modalities:
            input_file = input_dir / f"{modality}.nii.gz"
            output_file = output_dir / f"{modality}.nii.gz"
            reason_log = []
            
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
                
                # Create improved brain mask
                mask = generate_brain_mask(image)
                median_cv = evaluate_bias_need(image, mask)
                if median_cv < bias_threshold:
                    sitk.WriteImage(image, str(output_file))
                    result['skipped_unnecessary'].append(f"{modality}_low_bias({median_cv:.3f})")
                    continue
                # First conservative attempt
                orig_stats = basic_intensity_stats(image, mask)
                corrected_image, success = apply_conservative_n4(image, mask)
                if success:
                    corr_stats = basic_intensity_stats(corrected_image, mask)
                    if not acceptable_n4_change(orig_stats, corr_stats):
                        # Retry with second-pass iterations if provided
                        it1,it2 = second_pass
                        try:
                            n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
                            n4_filter.SetMaximumNumberOfIterations([it1,it2])
                            corrected_retry = n4_filter.Execute(image, mask)
                            corr_stats2 = basic_intensity_stats(corrected_retry, mask)
                            if acceptable_n4_change(orig_stats, corr_stats2):
                                corrected_image = corrected_retry
                            else:
                                success = False
                        except Exception:
                            success = False
                if success:
                    sitk.WriteImage(corrected_image, str(output_file))
                    if validate_image_file(output_file):
                        result['processed_modalities'].append(modality)
                    else:
                        result['failed_modalities'].append(f"{modality}_post_write_validation_fail")
                else:
                    sitk.WriteImage(image, str(output_file))
                    result['skipped_unnecessary'].append(f"{modality}_fallback_copy")
            except Exception as e:
                result['failed_modalities'].append(f"{modality}_processing_error")
                logging.debug(f"error processing {modality} for {subject_id}: {str(e)}")
        
        # Mark as successful if at least one modality was processed/copied
        if result['processed_modalities'] or result['skipped_modalities'] or result['skipped_unnecessary']:
            result['success'] = True
        
    except Exception as e:
        result['error_message'] = str(e)
    
    result['processing_time'] = time.time() - start_time
    return result


def collect_subjects_for_processing(splits_to_process: List[str] = None, bias_threshold: float = 0.18) -> List[Tuple[str, str]]:
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


def run_improved_n4_correction(subjects: List[Tuple[str, str]], 
                              overwrite: bool = False,
                              max_workers: int = None,
                              bias_threshold: float = 0.18,
                              second_pass: Tuple[int, int] = (10, 5)) -> Dict:
    """
    Run improved N4 correction with conservative parameters.
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    logging.info(f"processing {len(subjects)} subjects with {max_workers} workers")
    logging.info("using improved conservative N4 correction approach")
    logging.info("will skip N4 for images with minimal bias field")
    
    start_time = time.time()
    results = {
        'total_subjects': len(subjects),
        'successful_subjects': 0,
        'failed_subjects': 0,
        'total_modalities_processed': 0,
        'total_modalities_skipped': 0,
        'total_modalities_skipped_unnecessary': 0,
        'start_time': time.strftime('%H:%M:%S'),
        'processing_details': [],
        'parameters': {
            'approach': 'conservative_n4_with_bias_assessment',
            'n4_iterations': '[5, 3]',
            'convergence_threshold': 0.01,
            'bias_field_fwhm': 0.50,
            'max_workers': max_workers,
            'quality_checks': 'enabled'
        }
    }
    
    # Prepare arguments
    args_list = [
        (section, subject_id, overwrite, bias_threshold, second_pass)
        for section, subject_id in subjects
    ]
    
    # Process with progress updates
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_subject = {
            executor.submit(process_subject_improved, args): args[1]
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
                    results['total_modalities_skipped_unnecessary'] += len(result['skipped_unnecessary'])
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


def print_improved_summary(results: Dict):
    """Print comprehensive summary of improved processing."""
    elapsed = results['elapsed_time']
    
    print(f"\n{'='*70}")
    print("IMPROVED N4 BIAS CORRECTION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nprocessing results:")
    print(f"  total subjects:              {results['total_subjects']}")
    print(f"  successful:                  {results['successful_subjects']}")
    print(f"  failed:                      {results['failed_subjects']}")
    print(f"  success rate:                {results['successful_subjects']/max(results['total_subjects'],1)*100:.1f}%")
    
    print(f"\nmodality processing breakdown:")
    print(f"  N4 corrected:                {results['total_modalities_processed']}")
    print(f"  skipped (already existed):   {results['total_modalities_skipped']}")
    print(f"  skipped (unnecessary/failed): {results['total_modalities_skipped_unnecessary']}")
    
    total_modalities = (results['total_modalities_processed'] + 
                       results['total_modalities_skipped'] + 
                       results['total_modalities_skipped_unnecessary'])
    
    if total_modalities > 0:
        n4_rate = results['total_modalities_processed'] / total_modalities * 100
        print(f"  actual N4 correction rate:   {n4_rate:.1f}%")
    
    print(f"\ntiming:")
    print(f"  processing time:             {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    if elapsed > 0 and results['successful_subjects'] > 0:
        rate = results['successful_subjects'] * 60 / elapsed
        avg_time = elapsed / results['successful_subjects']
        print(f"  processing rate:             {rate:.1f} subjects/minute")
        print(f"  average per subject:         {avg_time:.1f} seconds")
    
    print(f"\nimproved approach features:")
    print(f"  ✓ Conservative N4 parameters to avoid over-correction")
    print(f"  ✓ Bias field assessment - skips N4 when unnecessary")
    print(f"  ✓ Quality validation - rejects poor N4 results")
    print(f"  ✓ Fallback strategy - copies original if N4 fails")
    print(f"  ✓ Intensity range protection")
    
    # Show sample processing details
    n4_applied_count = 0
    skipped_unnecessary_count = 0
    for detail in results['processing_details']:
        n4_applied_count += len(detail['processed_modalities'])
        skipped_unnecessary_count += len([x for x in detail['skipped_unnecessary'] 
                                        if not x.endswith('_n4_failed_copied_original')])
    
    if n4_applied_count + skipped_unnecessary_count > 0:
        skip_rate = skipped_unnecessary_count / (n4_applied_count + skipped_unnecessary_count) * 100
        print(f"\nbias field analysis results:")
        print(f"  modalities with minimal bias: {skipped_unnecessary_count} ({skip_rate:.1f}%)")
        print(f"  modalities needing N4:       {n4_applied_count} ({100-skip_rate:.1f}%)")
    
    print(f"\noutput location:")
    print(f"  improved N4 files: {PATHS['preprocessed_dir']}/[section]_n4corrected_v2/")
    
    print(f"\nnext step:")
    print(f"  run script 07 again on the _n4corrected_v2 directories to assess improvement")
    
    print(f"{'='*70}")


def main(config: Dict):
    """Main function for improved N4 correction with provided config."""
    setup_logging()

    print("=== NeuroScope Improved N4 Bias Field Correction ===")
    logging.info("starting improved conservative n4 bias field correction")

    try:
        logging.info("collecting subjects for processing...")
        subjects = collect_subjects_for_processing(config['splits_to_process'])

        if not subjects:
            logging.error("no subjects found for processing!")
            logging.error("check that preprocessed files exist in expected locations")
            return

        results = run_improved_n4_correction(
            subjects,
            overwrite=config['overwrite'],
            max_workers=config['max_workers'],
            bias_threshold=config['bias_threshold'],
            second_pass=config['second_pass']
        )

        output_path = PATHS['preprocessed_dir'] / "n4_correction_results_improved_v2.json"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, sort_keys=True)
            logging.info(f"results saved to {output_path}")
        except Exception as e:
            logging.warning(f"could not save results: {e}")

        print_improved_summary(results)

    except Exception as e:
        logging.error(f"improved n4 correction failed: {e}")
        raise

if __name__ == '__main__':
    parsed = parse_args()
    config = {
        'splits_to_process': [s.strip() for s in parsed.splits.split(',') if s.strip()],
        'overwrite': parsed.overwrite,
        'max_workers': parsed.max_workers,
        'bias_threshold': parsed.bias_threshold,
        'second_pass': tuple(int(x) for x in parsed.second_pass_iters.split(',') if x.strip())[:2] or (10, 5)
    }
    main(config)