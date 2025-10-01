import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from neuroscope_preprocessing_config import PATHS
from preprocessing_utils import write_json_with_schema
import argparse
from preprocessing_utils import generate_brain_mask

def configure_logging() -> None:
    """Configure logging format and level for N4 effectiveness assessment."""
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


def verify_image_file(file_path: Path) -> bool:
    """
    Verify that an image file exists and is readable.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not file_path.exists():
        logging.debug("file not found: %s", file_path)
        return False
    
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        # Basic sanity checks
        if arr.size == 0:
            logging.debug("empty image: %s", file_path)
            return False
        
        if not np.isfinite(arr).all():
            logging.debug("non-finite values in image: %s", file_path)
            return False
        
        return True
        
    except Exception as e:
        logging.debug("error reading image %s: %s", file_path, e)
        return False


def parse_args():
    ap = argparse.ArgumentParser(description='Assess N4 correction effectiveness (configurable)')
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of dataset splits to assess')
    ap.add_argument('--output', type=str, default=None, help='Override output JSON path')
    ap.add_argument('--summary-only', action='store_true', help='Skip saving detailed results, only summary')
    return ap.parse_args()


def compute_bias_field_residual_metrics(
    image: sitk.Image, 
    mask: sitk.Image,
    smoothing_sigma_mm: float = 10.0
) -> Dict[str, float]:
    """
    Compute comprehensive bias field residual metrics after N4 correction.
    
    Args:
        image: Input SimpleITK image (should be N4-corrected)
        mask: Brain mask
        smoothing_sigma_mm: Gaussian smoothing sigma in mm for bias estimation
        
    Returns:
        Dict containing bias residual metrics
    """
    # Get image arrays
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    brain_values = arr[mask_arr]
    
    if len(brain_values) < 1000:  # Not enough brain tissue
        return {
            'original_std': np.nan,
            'residual_std': np.nan,
            'residual_ratio': np.nan,
            'bias_field_strength': np.nan,
            'uniformity_coefficient': np.nan,
            'brain_volume_voxels': int(mask_arr.sum()),
            'error': 'insufficient_brain_tissue'
        }
    
    # Original statistics
    original_std = float(brain_values.std())
    original_mean = float(brain_values.mean())
    
    # Estimate remaining bias field using Gaussian smoothing
    spacing = image.GetSpacing()
    sigma_voxels = [smoothing_sigma_mm / sp for sp in spacing]
    
    try:
        # Create smoothed version to estimate low-frequency bias
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma_voxels)
        smoothed_image = gaussian_filter.Execute(image)
        smoothed_arr = sitk.GetArrayFromImage(smoothed_image).astype(np.float32)
        
        # Compute residual (high-frequency anatomical variation)
        residual = arr - smoothed_arr
        residual_values = residual[mask_arr]
        residual_std = float(residual_values.std())
        
        # Bias field strength estimation
        smoothed_brain_values = smoothed_arr[mask_arr]
        bias_field_variation = float(smoothed_brain_values.std())
        
        # Residual ratio (should be high after good N4 correction)
        residual_ratio = residual_std / (original_std + 1e-8)
        
        # Uniformity coefficient (bias field strength relative to signal)
        uniformity_coeff = bias_field_variation / (original_mean + 1e-8)
        
    except Exception as e:
        logging.debug("error in bias field estimation: %s", e)
        return {
            'original_std': original_std,
            'residual_std': np.nan,
            'residual_ratio': np.nan,
            'bias_field_strength': np.nan,
            'uniformity_coefficient': np.nan,
            'brain_volume_voxels': int(mask_arr.sum()),
            'error': f'smoothing_failed: {str(e)}'
        }
    
    # Additional quality metrics
    metrics = {
        'original_std': original_std,
        'residual_std': residual_std,
        'residual_ratio': residual_ratio,
        'bias_field_strength': bias_field_variation,
        'uniformity_coefficient': uniformity_coeff,
        'brain_volume_voxels': int(mask_arr.sum()),
        'original_mean': original_mean,
        'smoothed_mean': float(smoothed_brain_values.mean()),
        'residual_mean': float(residual_values.mean())
    }
    
    return metrics


def compare_before_after_n4(
    original_image: sitk.Image,
    corrected_image: sitk.Image,
    mask: sitk.Image
) -> Dict[str, float]:
    """
    Compare bias metrics before and after N4 correction.
    
    Args:
        original_image: Pre-N4 image
        corrected_image: Post-N4 image  
        mask: Brain mask
        
    Returns:
        Dict containing comparison metrics
    """
    # Get arrays
    orig_arr = sitk.GetArrayFromImage(original_image).astype(np.float32)
    corr_arr = sitk.GetArrayFromImage(corrected_image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    
    orig_brain = orig_arr[mask_arr]
    corr_brain = corr_arr[mask_arr]
    
    if len(orig_brain) < 1000:
        return {'error': 'insufficient_brain_tissue'}
    
    # Compute slice-wise coefficients of variation
    def compute_slice_cv(arr, mask_arr):
        slice_cvs = []
        for z in range(arr.shape[0]):
            slice_mask = mask_arr[z]
            if slice_mask.sum() > 10:
                slice_vals = arr[z][slice_mask]
                if slice_vals.mean() > 0:
                    cv = slice_vals.std() / slice_vals.mean()
                    slice_cvs.append(cv)
        return np.array(slice_cvs) if slice_cvs else np.array([np.nan])
    
    orig_slice_cvs = compute_slice_cv(orig_arr, mask_arr)
    corr_slice_cvs = compute_slice_cv(corr_arr, mask_arr)
    
    # Overall statistics
    metrics = {
        'original_cv': float(orig_brain.std() / (orig_brain.mean() + 1e-8)),
        'corrected_cv': float(corr_brain.std() / (corr_brain.mean() + 1e-8)),
        'original_slice_cv_median': float(np.nanmedian(orig_slice_cvs)),
        'corrected_slice_cv_median': float(np.nanmedian(corr_slice_cvs)),
        'cv_improvement_ratio': float(np.nanmedian(orig_slice_cvs) / (np.nanmedian(corr_slice_cvs) + 1e-8)),
        'intensity_preservation': float(np.corrcoef(orig_brain, corr_brain)[0, 1]) if len(orig_brain) > 1 else np.nan
    }
    
    return metrics


def assess_n4_effectiveness_for_subject(
    section: str,
    subject_id: str,
    modalities: List[str],
    splits_to_assess: List[str]
) -> Dict[str, Any]:
    """
    Assess N4 correction effectiveness for a single subject.
    
    Args:
        section: Dataset section ('brats' or 'upenn')
        subject_id: Subject identifier
        modalities: List of modality names ['t1', 't1gd', 't2', 'flair']
        splits_to_assess: List of splits to process
        
    Returns:
        Dict containing assessment results for all modalities
    """
    # Define paths
    original_dir = PATHS['preprocessed_dir'] / section / subject_id
    n4_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
    
    subject_results = {
        'subject_info': {
            'section': section,
            'subject_id': subject_id,
            'assessment_time': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'modality_results': {},
        'overall_effectiveness': {}
    }
    
    processed_modalities = 0
    total_improvements = []
    
    for modality in modalities:
        orig_file = original_dir / f"{modality}.nii.gz"
        n4_file = n4_dir / f"{modality}.nii.gz"
        
        # Check file availability
        orig_exists = verify_image_file(orig_file)
        n4_exists = verify_image_file(n4_file)
        
        modality_result = {
            'files_status': {
                'original_exists': orig_exists,
                'n4_corrected_exists': n4_exists
            }
        }
        
        if not orig_exists:
            modality_result['error'] = 'original_file_missing'
            subject_results['modality_results'][modality] = modality_result
            continue
        
        if not n4_exists:
            modality_result['error'] = 'n4_corrected_file_missing'
            subject_results['modality_results'][modality] = modality_result
            continue
        
        try:
            # Load images
            original_image = sitk.ReadImage(str(orig_file))
            n4_image = sitk.ReadImage(str(n4_file))
            
            # Generate brain mask from N4-corrected image (should be cleaner)
            mask = generate_brain_mask(n4_image)
            
            # Compute bias residual metrics for N4-corrected image
            n4_metrics = compute_bias_field_residual_metrics(n4_image, mask)
            
            # Compare before/after N4 correction
            comparison_metrics = compare_before_after_n4(original_image, n4_image, mask)
            
            modality_result.update({
                'n4_residual_metrics': n4_metrics,
                'before_after_comparison': comparison_metrics,
                'assessment_successful': True
            })
            
            # Track overall effectiveness
            if 'cv_improvement_ratio' in comparison_metrics and np.isfinite(comparison_metrics['cv_improvement_ratio']):
                total_improvements.append(comparison_metrics['cv_improvement_ratio'])
                processed_modalities += 1
            
        except Exception as e:
            logging.error("error assessing %s/%s %s: %s", section, subject_id, modality, e)
            modality_result['error'] = str(e)
        
        subject_results['modality_results'][modality] = modality_result
    
    # Compute overall effectiveness summary
    if processed_modalities > 0:
        subject_results['overall_effectiveness'] = {
            'processed_modalities': processed_modalities,
            'mean_cv_improvement': float(np.mean(total_improvements)),
            'median_cv_improvement': float(np.median(total_improvements)),
            'effectiveness_rating': 'good' if np.mean(total_improvements) > 1.2 else 'moderate' if np.mean(total_improvements) > 1.1 else 'poor'
        }
    else:
        subject_results['overall_effectiveness'] = {
            'processed_modalities': 0,
            'effectiveness_rating': 'unknown'
        }
    
    return subject_results


def run_comprehensive_n4_assessment(
    metadata: Dict[str, Any],
    splits_to_assess: List[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive N4 effectiveness assessment across datasets.
    
    Args:
        metadata: Complete metadata dictionary
        splits_to_assess: List of splits to assess
        
    Returns:
        Dict containing comprehensive assessment results
    """
    if splits_to_assess is None:
        splits_to_assess = ['train', 'val']
    
    start_time = time.time()
    logging.info("starting comprehensive n4 effectiveness assessment...")
    logging.info("assessing splits: %s", splits_to_assess)
    
    results = {
        'assessment_info': {
            'script_version': '07_assess_n4_correction_effectiveness_neuroscope.py v1.0',
            'assessment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'splits_assessed': splits_to_assess,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'processing_summary': {
            'total_subjects': 0,
            'successful_subjects': 0,
            'failed_subjects': 0,
            'subjects_with_missing_files': 0,
            'subjects_with_partial_assessment': 0
        },
        'detailed_results': {
            'brats': {},
            'upenn': {}
        },
        'dataset_summaries': {}
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    # Process each dataset
    for section in ['brats', 'upenn']:
        logging.info("processing %s dataset...", section)
        valid_subjects = metadata[section]['valid_subjects']
        
        section_processed = 0
        section_successful = 0
        
        for subject_id, subject_info in valid_subjects.items():
            # Check if subject is in the splits we want to assess
            subject_split = subject_info.get('split')
            if subject_split not in splits_to_assess:
                continue
            
            results['processing_summary']['total_subjects'] += 1
            section_processed += 1
            
            try:
                # Assess N4 effectiveness for this subject
                subject_results = assess_n4_effectiveness_for_subject(
                    section, subject_id, modalities, splits_to_assess
                )
                
                results['detailed_results'][section][subject_id] = subject_results
                
                # Check assessment success
                successful_modalities = sum(
                    1 for mod_result in subject_results['modality_results'].values()
                    if mod_result.get('assessment_successful', False)
                )
                
                if successful_modalities >= 3:  # At least 3/4 modalities successful
                    results['processing_summary']['successful_subjects'] += 1
                    section_successful += 1
                elif successful_modalities > 0:
                    results['processing_summary']['subjects_with_partial_assessment'] += 1
                else:
                    results['processing_summary']['failed_subjects'] += 1
                
                # Check for missing files
                has_missing_files = any(
                    not mod_result['files_status']['n4_corrected_exists']
                    for mod_result in subject_results['modality_results'].values()
                )
                if has_missing_files:
                    results['processing_summary']['subjects_with_missing_files'] += 1
                
                logging.info("completed %s/%s (%s) - %d/%d modalities assessed", 
                           section, subject_id, subject_split, successful_modalities, len(modalities))
                
            except Exception as e:
                logging.error("error processing %s/%s: %s", section, subject_id, e)
                results['detailed_results'][section][subject_id] = {
                    'subject_info': {'section': section, 'subject_id': subject_id},
                    'error': str(e)
                }
                results['processing_summary']['failed_subjects'] += 1
        
        logging.info("%s assessment complete: %d processed, %d successful", 
                    section, section_processed, section_successful)
    
    # Add timing info
    elapsed_time = time.time() - start_time
    results['processing_summary']['elapsed_time_seconds'] = elapsed_time
    results['processing_summary']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info("n4 effectiveness assessment completed in %.1f seconds", elapsed_time)
    return results


def generate_n4_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for N4 effectiveness assessment.
    
    Args:
        results: N4 assessment results
        
    Returns:
        Dict containing summary statistics
    """
    summary = {
        'overall_statistics': {},
        'dataset_statistics': {'brats': {}, 'upenn': {}},
        'modality_statistics': {'t1': {}, 't1gd': {}, 't2': {}, 'flair': {}},
        'effectiveness_thresholds': {
            'cv_improvement_ratio': {'excellent': 1.5, 'good': 1.2, 'moderate': 1.1, 'poor': 1.0},
            'residual_ratio': {'excellent': 0.8, 'good': 0.7, 'moderate': 0.6, 'poor': 0.5}
        }
    }
    
    # Collect metrics from all successful assessments
    all_improvements = []
    all_residual_ratios = []
    dataset_improvements = defaultdict(list)
    modality_improvements = defaultdict(list)
    
    for section in ['brats', 'upenn']:
        for subject_id, subject_results in results['detailed_results'][section].items():
            if 'error' in subject_results:
                continue
                
            for modality, mod_result in subject_results['modality_results'].items():
                if mod_result.get('assessment_successful', False):
                    # Extract improvement metrics
                    comparison = mod_result.get('before_after_comparison', {})
                    residual = mod_result.get('n4_residual_metrics', {})
                    
                    cv_improvement = comparison.get('cv_improvement_ratio')
                    residual_ratio = residual.get('residual_ratio')
                    
                    if cv_improvement is not None and np.isfinite(cv_improvement):
                        all_improvements.append(cv_improvement)
                        dataset_improvements[section].append(cv_improvement)
                        modality_improvements[modality].append(cv_improvement)
                    
                    if residual_ratio is not None and np.isfinite(residual_ratio):
                        all_residual_ratios.append(residual_ratio)
    
    # Compute overall statistics
    if all_improvements:
        summary['overall_statistics']['cv_improvement_ratio'] = {
            'mean': float(np.mean(all_improvements)),
            'median': float(np.median(all_improvements)),
            'std': float(np.std(all_improvements)),
            'min': float(np.min(all_improvements)),
            'max': float(np.max(all_improvements)),
            'count': len(all_improvements)
        }
    
    if all_residual_ratios:
        summary['overall_statistics']['residual_ratio'] = {
            'mean': float(np.mean(all_residual_ratios)),
            'median': float(np.median(all_residual_ratios)),
            'std': float(np.std(all_residual_ratios)),
            'min': float(np.min(all_residual_ratios)),
            'max': float(np.max(all_residual_ratios)),
            'count': len(all_residual_ratios)
        }
    
    # Dataset-specific statistics
    for section, improvements in dataset_improvements.items():
        if improvements:
            summary['dataset_statistics'][section]['cv_improvement_ratio'] = {
                'mean': float(np.mean(improvements)),
                'median': float(np.median(improvements)),
                'count': len(improvements)
            }
    
    # Modality-specific statistics
    for modality, improvements in modality_improvements.items():
        if improvements:
            summary['modality_statistics'][modality]['cv_improvement_ratio'] = {
                'mean': float(np.mean(improvements)),
                'median': float(np.median(improvements)),
                'count': len(improvements)
            }
    
    return summary


def save_n4_assessment_results(
    results: Dict[str, Any],
    summary: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save N4 effectiveness assessment results to JSON file.
    
    Args:
        results: Detailed assessment results
        summary: Summary statistics
        output_path: Path to save results
    """
    # Combine results and summary
    output_data = {
        'assessment_info': results.get('assessment_info', {}),
        'processing_summary': results.get('processing_summary', {}),
        'summary_statistics': summary,
        'detailed_results': results.get('detailed_results', {})
    }
    
    try:
        schema = {
            'assessment_info': dict,
            'processing_summary': dict,
            'summary_statistics': dict,
            'detailed_results': dict
        }
        write_json_with_schema(output_data, output_path, schema=schema)
        file_size = output_path.stat().st_size
        logging.info("N4 assessment results saved to: %s (%.1f KB)", output_path, file_size / 1024)
        summary_path = output_path.parent / (output_path.stem + '_summary.json')
        summary_light = {
            'script_version': output_data['assessment_info'].get('script_version'),
            'assessment_date': output_data['assessment_info'].get('assessment_date'),
            'subjects': output_data['processing_summary']['total_subjects']
        }
        write_json_with_schema(summary_light, summary_path)
        logging.info("summary file written: %s", summary_path)
    except Exception as e:
        logging.error("failed to save n4 assessment results: %s", e)
        raise


def print_n4_assessment_summary(summary: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of N4 effectiveness assessment."""
    proc_summary = results.get('processing_summary', {})
    overall_stats = summary.get('overall_statistics', {})
    dataset_stats = summary.get('dataset_statistics', {})
    print("\n" + "="*80)
    print("N4 BIAS FIELD CORRECTION EFFECTIVENESS ASSESSMENT")
    print("="*80)
    print(f"\nprocessing summary:")
    print(f"  total subjects:           {proc_summary.get('total_subjects', 0)}")
    print(f"  successful assessments:   {proc_summary.get('successful_subjects', 0)}")
    print(f"  partial assessments:      {proc_summary.get('subjects_with_partial_assessment', 0)}")
    print(f"  failed assessments:       {proc_summary.get('failed_subjects', 0)}")
    print(f"  subjects with missing files: {proc_summary.get('subjects_with_missing_files', 0)}")
    total_assessed = proc_summary.get('successful_subjects', 0) + proc_summary.get('subjects_with_partial_assessment', 0)
    if proc_summary.get('total_subjects', 0) > 0:
        success_rate = total_assessed / proc_summary['total_subjects'] * 100
        print(f"  assessment success rate:  {success_rate:.1f}%")
    print(f"  processing time:          {proc_summary.get('elapsed_time_seconds', 0):.1f} seconds")
    if 'cv_improvement_ratio' in overall_stats:
        cv_stats = overall_stats['cv_improvement_ratio']
        print(f"\nn4 correction effectiveness (cv improvement ratio):")
        print(f"  mean improvement:     {cv_stats['mean']:.3f}x")
        print(f"  median improvement:   {cv_stats['median']:.3f}x")
        print(f"  standard deviation:   {cv_stats['std']:.3f}")
        print(f"  range:               [{cv_stats['min']:.3f}, {cv_stats['max']:.3f}]")
        print(f"  assessments count:    {cv_stats['count']}")
    # Dataset and modality comparisons printed later (reuse existing logic below)
    
    # Modality comparison
    modality_stats = summary.get('modality_statistics', {})
    modality_names = {'t1': 'T1', 't1gd': 'T1-Gd', 't2': 'T2', 'flair': 'FLAIR'}
    print(f"\nmodality comparison (cv improvement):")
    for modality, display_name in modality_names.items():
        if modality in modality_stats and 'cv_improvement_ratio' in modality_stats[modality]:
            stats_dict = modality_stats[modality]['cv_improvement_ratio']
            print(f"  {display_name}:")
            print(f"    mean improvement:   {stats_dict['mean']:.3f}x")
            print(f"    median improvement: {stats_dict['median']:.3f}x")
            print(f"    count:              {stats_dict['count']}")
    
    # Recommendations
    print(f"\nrecommendations:")
    if overall_stats.get('cv_improvement_ratio', {}).get('median', 1.0) >= 1.2:
        print(f"n4 correction is working effectively")
        print(f"  coefficient of variation improved by {overall_stats['cv_improvement_ratio']['median']:.1f}x on average")
    elif overall_stats.get('cv_improvement_ratio', {}).get('median', 1.0) >= 1.1:
        print(f"~ n4 correction is working moderately well")
        print(f"  consider adjusting n4 parameters for better results")
    else:
        print(f"n4 correction effectiveness is limited")
        print(f"  consider alternative bias correction methods or parameter optimization")
    
    if proc_summary.get('subjects_with_missing_files', 0) > 0:
        print(f" {proc_summary['subjects_with_missing_files']} subjects missing N4-corrected files")
        print(f"  ensure script 06_n4_bias_correction_neuroscope.py completed successfully")
    
    print("="*80)


def check_n4_correction_status(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check which subjects have been N4-corrected and provide a summary.
    
    Args:
        metadata: Complete metadata dictionary
        
    Returns:
        Dict with N4 correction status summary
    """
    status = {
        'brats': {'train': {'total': 0, 'n4_corrected': 0}, 
                 'val': {'total': 0, 'n4_corrected': 0},
                 'test': {'total': 0, 'n4_corrected': 0}},
        'upenn': {'train': {'total': 0, 'n4_corrected': 0},
                 'val': {'total': 0, 'n4_corrected': 0}, 
                 'test': {'total': 0, 'n4_corrected': 0}}
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    for section in ['brats', 'upenn']:
        for subject_id, subject_info in metadata[section]['valid_subjects'].items():
            split = subject_info.get('split', 'unknown')
            if split in status[section]:
                status[section][split]['total'] += 1
                
                # Check if at least one modality is N4-corrected
                n4_corrected = False
                n4_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
                for modality in modalities:
                    n4_file = n4_dir / f"{modality}.nii.gz"
                    if verify_image_file(n4_file):
                        n4_corrected = True
                        break
                
                if n4_corrected:
                    status[section][split]['n4_corrected'] += 1
    
    return status



if __name__ == '__main__':
        def main():
            configure_logging()
            args = parse_args()
            logging.info("=== N4 BIAS FIELD CORRECTION EFFECTIVENESS ASSESSMENT ===")
            metadata_path = PATHS['metadata_splits']
            output_path = PATHS['preprocessed_dir'] / "n4_effectiveness_assessment.json"
            if args.output:
                output_path = Path(args.output)
            splits = [s.strip() for s in args.splits.split(',') if s.strip()]
            try:
                metadata = load_metadata_with_validation(metadata_path)
                n4_status = check_n4_correction_status(metadata)
                for section in ['brats', 'upenn']:
                    section_name = 'BraTS-TCGA-GBM' if section == 'brats' else 'UPenn-GBM'
                    logging.info("%s n4 correction status:", section_name)
                    for split in ['train', 'val', 'test']:
                        total = n4_status[section][split]['total']
                        n4_corrected = n4_status[section][split]['n4_corrected']
                        if total > 0:
                            logging.info("  %s: %d/%d (%.1f%%) n4-corrected", split, n4_corrected, total, n4_corrected/total*100)
                results = run_comprehensive_n4_assessment(metadata, splits_to_assess=splits)
                summary = generate_n4_summary_statistics(results)
                save_n4_assessment_results(results, summary, output_path)
                print_n4_assessment_summary(summary, results)
                logging.info("n4 effectiveness assessment completed successfully")
            except Exception as e:
                logging.error("n4 effectiveness assessment failed: %s", e)
                raise
        main()