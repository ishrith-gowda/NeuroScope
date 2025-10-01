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
import argparse

from neuroscope_preprocessing_config import PATHS
from preprocessing_utils import write_json_with_schema, generate_brain_mask


def configure_logging() -> None:
    """Configure logging format and level for bias assessment."""
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


def verify_preprocessed_file(file_path: str) -> bool:
    """
    Verify that a preprocessed file exists and is readable.
    
    Args:
        file_path: Path to the preprocessed NIfTI file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.isfile(file_path):
        logging.debug("file not found: %s", file_path)
        return False
    
    try:
        img = sitk.ReadImage(file_path)
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


def compute_slice_wise_statistics(image: sitk.Image, mask: sitk.Image) -> Dict[str, float]:
    """
    Compute comprehensive slice-wise intensity statistics.
    
    Args:
        image: Input SimpleITK image
        mask: Brain mask
        
    Returns:
        Dict containing various bias metrics
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    
    # Compute slice-wise statistics (axial slices)
    slice_means = []
    slice_stds = []
    slice_medians = []
    slice_volumes = []
    
    for z in range(arr.shape[0]):
        slice_mask = mask_arr[z]
        if slice_mask.sum() > 10:  # Minimum voxels for reliable statistics
            slice_vals = arr[z][slice_mask]
            slice_means.append(slice_vals.mean())
            slice_stds.append(slice_vals.std())
            slice_medians.append(np.median(slice_vals))
            slice_volumes.append(slice_mask.sum())
        else:
            # Skip slices with insufficient brain tissue
            slice_means.append(np.nan)
            slice_stds.append(np.nan)
            slice_medians.append(np.nan)
            slice_volumes.append(0)
    
    # Convert to numpy arrays and remove NaN values for analysis
    slice_means = np.array(slice_means)
    slice_stds = np.array(slice_stds)
    slice_medians = np.array(slice_medians)
    slice_volumes = np.array(slice_volumes)
    
    valid_indices = ~np.isnan(slice_means)
    if valid_indices.sum() < 3:
        # Not enough valid slices
        return {
            'slice_mean_variation': np.nan,
            'slice_std_variation': np.nan,
            'slice_median_variation': np.nan,
            'volume_weighted_variation': np.nan,
            'linear_trend_slope': np.nan,
            'linear_trend_r_squared': np.nan,
            'cv_coefficient': np.nan,
            'valid_slices': int(valid_indices.sum()),
            'total_slices': len(slice_means)
        }
    
    # Extract valid statistics
    valid_means = slice_means[valid_indices]
    valid_stds = slice_stds[valid_indices]
    valid_medians = slice_medians[valid_indices]
    valid_volumes = slice_volumes[valid_indices]
    valid_slice_indices = np.where(valid_indices)[0]
    
    # Global brain statistics for normalization
    global_brain_vals = arr[mask_arr]
    global_mean = global_brain_vals.mean()
    global_std = global_brain_vals.std()
    
    # Compute bias metrics
    metrics = {}
    
    # 1. Slice mean variation (normalized by global std)
    if global_std > 0:
        metrics['slice_mean_variation'] = valid_means.std() / global_std
    else:
        metrics['slice_mean_variation'] = np.nan
    
    # 2. Slice standard deviation variation
    metrics['slice_std_variation'] = valid_stds.std() / (valid_stds.mean() + 1e-8)
    
    # 3. Slice median variation
    if global_std > 0:
        metrics['slice_median_variation'] = valid_medians.std() / global_std
    else:
        metrics['slice_median_variation'] = np.nan
    
    # 4. Volume-weighted variation (accounts for slice thickness differences)
    if valid_volumes.sum() > 0:
        weighted_means = np.average(valid_means, weights=valid_volumes)
        weighted_var = np.average((valid_means - weighted_means)**2, weights=valid_volumes)
        metrics['volume_weighted_variation'] = np.sqrt(weighted_var) / global_std if global_std > 0 else np.nan
    else:
        metrics['volume_weighted_variation'] = np.nan
    
    # 5. Linear trend analysis (z-direction bias)
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_slice_indices, valid_means)
        metrics['linear_trend_slope'] = slope
        metrics['linear_trend_r_squared'] = r_value**2
        metrics['linear_trend_p_value'] = p_value
    except:
        metrics['linear_trend_slope'] = np.nan
        metrics['linear_trend_r_squared'] = np.nan
        metrics['linear_trend_p_value'] = np.nan
    
    # 6. Coefficient of variation for slice means
    if global_mean > 0:
        metrics['cv_coefficient'] = valid_means.std() / valid_means.mean()
    else:
        metrics['cv_coefficient'] = np.nan
    
    # 7. Additional statistics
    metrics['valid_slices'] = int(valid_indices.sum())
    metrics['total_slices'] = len(slice_means)
    metrics['brain_volume_voxels'] = int(mask_arr.sum())
    
    return metrics


def assess_subject_bias(
    section: str,
    subject_id: str,
    modality_files: Dict[str, str],
    standardized_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Assess intensity bias for all modalities of a single subject.
    
    Args:
        section: Dataset section ('brats' or 'upenn')
        subject_id: Subject identifier
        modality_files: Dictionary mapping modality suffixes to file paths
        standardized_names: List of standardized modality names
        
    Returns:
        Dict mapping standardized modality names to bias metrics
    """
    subject_results = {}
    
    # Map original suffixes to standardized names
    if section == 'brats':
        suffix_mapping = {
            '_t1.nii.gz': 't1',
            '_t1Gd.nii.gz': 't1gd', 
            '_t2.nii.gz': 't2',
            '_flair.nii.gz': 'flair'
        }
    else:  # upenn
        suffix_mapping = {
            '_T1.nii.gz': 't1',
            '_T1GD.nii.gz': 't1gd',
            '_T2.nii.gz': 't2', 
            '_FLAIR.nii.gz': 'flair'
        }
    
    for suffix, file_path in modality_files.items():
        if suffix not in suffix_mapping:
            logging.debug("unknown modality suffix: %s", suffix)
            continue
            
        modality_name = suffix_mapping[suffix]
        
        # Check if preprocessed file exists
        preprocessed_path = PATHS['preprocessed_dir'] / section / subject_id / f"{modality_name}.nii.gz"
        
        if not verify_preprocessed_file(str(preprocessed_path)):
            logging.warning("preprocessed file missing or invalid: %s", preprocessed_path)
            subject_results[modality_name] = {'error': 'file_missing_or_invalid'}
            continue
        
        try:
            # Load preprocessed image
            image = sitk.ReadImage(str(preprocessed_path))
            
            # Generate brain mask
            mask = generate_brain_mask(image)
            
            # Compute bias metrics
            bias_metrics = compute_slice_wise_statistics(image, mask)
            subject_results[modality_name] = bias_metrics
            
            logging.debug("%s/%s %s: slice_var=%.4f, trend_slope=%.6f", 
                         section, subject_id, modality_name,
                         bias_metrics.get('slice_mean_variation', np.nan),
                         bias_metrics.get('linear_trend_slope', np.nan))
            
        except Exception as e:
            logging.error("error processing %s/%s %s: %s", section, subject_id, modality_name, e)
            subject_results[modality_name] = {'error': str(e)}
    
    return subject_results


def analyze_dataset_bias(metadata: Dict[str, Any], splits_to_assess: List[str] = None) -> Dict[str, Any]:
    """
    Analyze intensity bias across subjects in both datasets.
    
    Args:
        metadata: Complete metadata dictionary
        splits_to_assess: List of splits to assess (default: ['train', 'val'])
        
    Returns:
        Dict containing comprehensive bias analysis results
    """
    if splits_to_assess is None:
        splits_to_assess = ['train', 'val']  # Default to splits that were preprocessed
    
    start_time = time.time()
    logging.info("starting comprehensive bias assessment...")
    logging.info("assessing splits: %s", splits_to_assess)
    
    results = {
        'brats': {},
        'upenn': {},
        'processing_info': {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'splits_assessed': splits_to_assess,
            'successful_subjects': 0,
            'failed_subjects': 0,
            'skipped_subjects': 0,
            'total_subjects': 0,
            'missing_files': []
        }
    }
    
    # Define modality mappings
    modality_configs = {
        'brats': {
            'suffixes': ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz'],
            'names': ['t1', 't1gd', 't2', 'flair']
        },
        'upenn': {
            'suffixes': ['_T1.nii.gz', '_T1GD.nii.gz', '_T2.nii.gz', '_FLAIR.nii.gz'],
            'names': ['t1', 't1gd', 't2', 'flair']
        }
    }
    
    # Process each dataset
    for section in ['brats', 'upenn']:
        logging.info("processing %s dataset...", section)
        valid_subjects = metadata[section]['valid_subjects']
        config = modality_configs[section]
        
        section_successful = 0
        section_failed = 0
        section_skipped = 0
        section_total = 0
        
        for subject_id, subject_info in valid_subjects.items():
            # Check if subject is in the splits we want to assess
            subject_split = subject_info.get('split')
            if subject_split not in splits_to_assess:
                section_skipped += 1
                continue
            
            section_total += 1
            try:
                # Extract modality file paths from subject info
                modality_files = {}
                for suffix in config['suffixes']:
                    if suffix in subject_info:
                        modality_files[suffix] = subject_info[suffix]
                
                if not modality_files:
                    logging.warning("no modality files found for %s/%s", section, subject_id)
                    section_failed += 1
                    continue
                
                # Check if any preprocessed files exist before attempting assessment
                preprocessed_exist = False
                for suffix in config['suffixes']:
                    modality_name = config['names'][config['suffixes'].index(suffix)]
                    preprocessed_path = PATHS['preprocessed_dir'] / section / subject_id / f"{modality_name}.nii.gz"
                    if verify_preprocessed_file(str(preprocessed_path)):
                        preprocessed_exist = True
                        break
                
                if not preprocessed_exist:
                    logging.debug("no preprocessed files found for %s/%s (split: %s) - skipping", 
                                section, subject_id, subject_split)
                    results['processing_info']['missing_files'].append(f"{section}/{subject_id}")
                    section_skipped += 1
                    continue
                
                # Assess bias for this subject
                subject_results = assess_subject_bias(section, subject_id, modality_files, config['names'])
                results[section][subject_id] = subject_results
                
                # Check if assessment was successful
                if any('error' not in metrics for metrics in subject_results.values()):
                    section_successful += 1
                    logging.info("completed %s/%s (%s)", section, subject_id, subject_split)
                else:
                    section_failed += 1
                    logging.warning("failed to assess %s/%s", section, subject_id)
                    
            except Exception as e:
                logging.error("error processing %s/%s: %s", section, subject_id, e)
                results[section][subject_id] = {'error': str(e)}
                section_failed += 1
        
        logging.info("%s assessment complete: %d successful, %d failed, %d skipped", 
                    section, section_successful, section_failed, section_skipped)
        
        # Update processing info
        results['processing_info']['successful_subjects'] += section_successful
        results['processing_info']['failed_subjects'] += section_failed
        results['processing_info']['skipped_subjects'] += section_skipped
        results['processing_info']['total_subjects'] += section_total
    
    # Add timing info
    elapsed_time = time.time() - start_time
    results['processing_info']['elapsed_time_seconds'] = elapsed_time
    results['processing_info']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info("bias assessment completed in %.1f seconds", elapsed_time)
    return results


def generate_bias_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for bias assessment results.
    
    Args:
        results: Bias assessment results
        
    Returns:
        Dict containing summary statistics
    """
    summary = {
        'dataset_statistics': {},
        'modality_statistics': {},
        'overall_statistics': {},
        'bias_thresholds': {
            'slice_mean_variation': {'low': 0.05, 'moderate': 0.15, 'high': 0.3},
            'linear_trend_slope': {'low': 0.001, 'moderate': 0.005, 'high': 0.01},
            'cv_coefficient': {'low': 0.1, 'moderate': 0.2, 'high': 0.4}
        }
    }
    
    # Collect all valid metrics
    all_metrics = defaultdict(list)
    dataset_metrics = defaultdict(lambda: defaultdict(list))
    modality_metrics = defaultdict(lambda: defaultdict(list))
    
    for section in ['brats', 'upenn']:
        for subject_id, subject_results in results[section].items():
            if isinstance(subject_results, dict) and 'error' not in subject_results:
                for modality, metrics in subject_results.items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                all_metrics[metric_name].append(value)
                                dataset_metrics[section][metric_name].append(value)
                                modality_metrics[modality][metric_name].append(value)
    
    # Compute overall statistics
    for metric_name, values in all_metrics.items():
        if values:
            summary['overall_statistics'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
    
    # Compute dataset-specific statistics
    for section, section_metrics in dataset_metrics.items():
        summary['dataset_statistics'][section] = {}
        for metric_name, values in section_metrics.items():
            if values:
                summary['dataset_statistics'][section][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
    
    # Compute modality-specific statistics  
    for modality, modality_metrics_dict in modality_metrics.items():
        summary['modality_statistics'][modality] = {}
        for metric_name, values in modality_metrics_dict.items():
            if values:
                summary['modality_statistics'][modality][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
    
    return summary


def save_bias_assessment_results(
    results: Dict[str, Any],
    summary: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save comprehensive bias assessment results to JSON file.
    
    Args:
        results: Detailed bias assessment results
        summary: Summary statistics
        output_path: Path to save results
    """
    # Combine results and summary
    output_data = {
        'assessment_info': {
            'script_version': '05_comprehensive_intensity_bias_assessment_neuroscope.py v2.0',
            'assessment_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_computed': [
                'slice_mean_variation',
                'slice_std_variation', 
                'slice_median_variation',
                'volume_weighted_variation',
                'linear_trend_slope',
                'linear_trend_r_squared',
                'cv_coefficient'
            ]
        },
        'processing_info': results.get('processing_info', {}),
        'detailed_results': {
            'brats': results['brats'],
            'upenn': results['upenn']
        },
        'summary_statistics': summary
    }
    
    try:
        schema = {
            'assessment_info': dict,
            'processing_info': dict,
            'detailed_results': dict,
            'summary_statistics': dict
        }
        write_json_with_schema(output_data, output_path, schema=schema)
        file_size = output_path.stat().st_size
        logging.info("bias assessment results saved to: %s (%.1f KB)", output_path, file_size / 1024)
        summary_path = output_path.parent / (output_path.stem + '_summary.json')
        light = {
            'script_version': output_data['assessment_info']['script_version'],
            'generated': output_data['assessment_info']['assessment_date'],
            'splits': output_data['processing_info'].get('splits_assessed'),
            'subjects': {
                'total': output_data['processing_info'].get('total_subjects'),
                'successful': output_data['processing_info'].get('successful_subjects'),
                'failed': output_data['processing_info'].get('failed_subjects')
            }
        }
        write_json_with_schema(light, summary_path)
        logging.info("summary file written: %s", summary_path)
    except Exception as e:
        logging.error("failed to save bias assessment results: %s", e)
        raise


def print_bias_assessment_summary(summary: Dict[str, Any], results: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of bias assessment results.
    
    Args:
        summary: Summary statistics
        results: Detailed results for processing info
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE INTENSITY BIAS ASSESSMENT SUMMARY")
    print("="*80)
    
    # Processing summary
    proc_info = results.get('processing_info', {})
    print(f"\nprocessing summary:")
    print(f"  splits assessed:     {proc_info.get('splits_assessed', [])}")
    print(f"  total subjects:      {proc_info.get('total_subjects', 0)}")
    print(f"  successful:          {proc_info.get('successful_subjects', 0)}")
    print(f"  failed:              {proc_info.get('failed_subjects', 0)}")
    print(f"  skipped (no files):  {proc_info.get('skipped_subjects', 0)}")
    print(f"  success rate:        {proc_info.get('successful_subjects', 0) / max(proc_info.get('total_subjects', 1), 1) * 100:.1f}%")
    print(f"  processing time:     {proc_info.get('elapsed_time_seconds', 0):.1f} seconds")
    
    # Show missing files summary if any
    missing_files = proc_info.get('missing_files', [])
    if missing_files:
        print(f"\nmissing preprocessed files ({len(missing_files)} subjects):")
        print("  note: subjects may not have been processed by script 01")
        if len(missing_files) <= 10:
            for missing in missing_files:
                print(f"    - {missing}")
        else:
            for missing in missing_files[:5]:
                print(f"    - {missing}")
            print(f"    ... and {len(missing_files) - 5} more")
    
    # Overall bias metrics
    overall_stats = summary.get('overall_statistics', {})
    if overall_stats:
        print(f"\noverall bias metrics (across all assessed subjects and modalities):")
        
        key_metrics = ['slice_mean_variation', 'linear_trend_slope', 'cv_coefficient']
        thresholds = summary.get('bias_thresholds', {})
        
        for metric in key_metrics:
            if metric in overall_stats:
                stats_dict = overall_stats[metric]
                thresh = thresholds.get(metric, {})
                
                print(f"\n  {metric.replace('_', ' ').title()}:")
                print(f"    mean ± std:      {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
                print(f"    median:          {stats_dict['median']:.4f}")
                print(f"    range:           [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
                
                # Classify severity based on median
                median_val = stats_dict['median']
                if median_val < thresh.get('low', 0):
                    severity = "low"
                elif median_val < thresh.get('moderate', 0):
                    severity = "moderate"
                else:
                    severity = "high"
                print(f"    severity:        {severity}")
    
    # Dataset comparison
    dataset_stats = summary.get('dataset_statistics', {})
    if len(dataset_stats) >= 2:
        print(f"\ndataset comparison (slice mean variation):")
        for section in ['brats', 'upenn']:
            section_name = 'BraTS-TCGA-GBM' if section == 'brats' else 'UPenn-GBM'
            if section in dataset_stats and 'slice_mean_variation' in dataset_stats[section]:
                stats_dict = dataset_stats[section]['slice_mean_variation']
                print(f"  {section_name}:")
                print(f"    mean:            {stats_dict['mean']:.4f}")
                print(f"    median:          {stats_dict['median']:.4f}")
                print(f"    count:           {stats_dict['count']}")
    
    # Recommendations
    print(f"\nrecommendations:")
    if overall_stats.get('slice_mean_variation', {}).get('median', 0) > 0.15:
        print(f"high slice variation detected, need to consider N4 bias correction")
    else:
        print(f"acceptable slice variation levels")
    
    if abs(overall_stats.get('linear_trend_slope', {}).get('median', 0)) > 0.005:
        print(f" significant linear trend detected, need to check acquisition protocols")
    else:
        print(f"no significant linear trends detected")
    
    # Preprocessing recommendations
    total_subjects_in_metadata = sum(len(results[section]) for section in ['brats', 'upenn'] 
                                   if section in results)
    if proc_info.get('skipped_subjects', 0) > 0:
        print(f"{proc_info.get('skipped_subjects', 0)} subjects skipped due to missing preprocessed files")
        print(f"consider running script 01 with additional splits if needed")
    
    print("="*80)


def check_preprocessing_status(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check which subjects have been preprocessed and provide a summary.
    
    Args:
        metadata: Complete metadata dictionary
        
    Returns:
        Dict with preprocessing status summary
    """
    status = {
        'brats': {'train': {'total': 0, 'preprocessed': 0}, 
                 'val': {'total': 0, 'preprocessed': 0},
                 'test': {'total': 0, 'preprocessed': 0}},
        'upenn': {'train': {'total': 0, 'preprocessed': 0},
                 'val': {'total': 0, 'preprocessed': 0}, 
                 'test': {'total': 0, 'preprocessed': 0}}
    }
    
    modality_configs = {
        'brats': ['t1', 't1gd', 't2', 'flair'],
        'upenn': ['t1', 't1gd', 't2', 'flair']
    }
    
    for section in ['brats', 'upenn']:
        for subject_id, subject_info in metadata[section]['valid_subjects'].items():
            split = subject_info.get('split', 'unknown')
            if split in status[section]:
                status[section][split]['total'] += 1
                
                # Check if at least one modality is preprocessed
                preprocessed = False
                for modality in modality_configs[section]:
                    preprocessed_path = PATHS['preprocessed_dir'] / section / subject_id / f"{modality}.nii.gz"
                    if verify_preprocessed_file(str(preprocessed_path)):
                        preprocessed = True
                        break
                
                if preprocessed:
                    status[section][split]['preprocessed'] += 1
    
    return status


def parse_args():
    ap = argparse.ArgumentParser(description='Comprehensive intensity bias assessment (configurable)')
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of dataset splits to assess')
    ap.add_argument('--output', type=str, default=None, help='Override output JSON path')
    ap.add_argument('--no-summary', action='store_true', help='Disable writing separate summary JSON')
    return ap.parse_args()


def main() -> None:
    """
    Main function to run comprehensive intensity bias assessment.
    """
    configure_logging()
    
    logging.info("=== COMPREHENSIVE INTENSITY BIAS ASSESSMENT ===")
    logging.info("using neuroscope_preprocessing_config.py for path management")
    
    # Define paths
    metadata_path = PATHS['metadata_splits']
    output_path = PATHS['slice_bias_assessment']
    
    logging.info("input metadata: %s", metadata_path)
    logging.info("output path: %s", output_path)
    
    try:
        # Step 1: Load and validate metadata
        logging.info("step 1: loading metadata with validation...")
        metadata = load_metadata_with_validation(metadata_path)
        
        # Step 1.5: Check preprocessing status
        logging.info("step 1.5: checking preprocessing status...")
        prep_status = check_preprocessing_status(metadata)
        
        # Log preprocessing summary
        for section in ['brats', 'upenn']:
            section_name = 'BraTS-TCGA-GBM' if section == 'brats' else 'UPenn-GBM'
            logging.info("%s preprocessing status:", section_name)
            for split in ['train', 'val', 'test']:
                total = prep_status[section][split]['total']
                preprocessed = prep_status[section][split]['preprocessed']
                if total > 0:
                    logging.info("  %s: %d/%d (%.1f%%) preprocessed", 
                               split, preprocessed, total, preprocessed/total*100)
        
        # Step 2: Run comprehensive bias assessment
        logging.info("step 2: analyzing intensity bias across datasets...")
        # Only assess train and val splits by default (since test split may not be preprocessed)
        splits_to_assess = ['train', 'val']
        args = parse_args()
        if args.splits:
            splits_to_assess = [s.strip() for s in args.splits.split(',')]
        
        results = analyze_dataset_bias(metadata, splits_to_assess=splits_to_assess)
        
        # Step 3: Generate summary statistics
        logging.info("step 3: generating summary statistics...")
        summary = generate_bias_summary_statistics(results)
        
        # Step 4: Save results
        logging.info("step 4: saving assessment results...")
        if args.output:
            output_path = Path(args.output)
        
        save_bias_assessment_results(results, summary, output_path)
        
        # Step 5: Display summary
        print_bias_assessment_summary(summary, results)
        
        logging.info("comprehensive bias assessment completed successfully")
        
    except Exception as e:
        logging.error("bias assessment failed: %s", e)
        raise


if __name__ == '__main__':
    main()