import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from neuroscope_preprocessing_config import PATHS
from preprocessing_utils import write_json_with_schema
import argparse
from preprocessing_utils import generate_brain_mask


def configure_logging() -> None:
    """Configure logging for N4 diagnostic analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def verify_image_file(file_path: Path) -> bool:
    """Verify that an image file exists and is readable."""
    if not file_path.exists():
        return False
    
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        return arr.size > 0 and np.isfinite(arr).all()
    except Exception:
        return False


def parse_args():
    ap = argparse.ArgumentParser(description='Diagnose N4 correction issues (configurable sample)')
    ap.add_argument('--splits', type=str, default='train,val', help='Splits to sample from')
    ap.add_argument('--max-per-section', type=int, default=15, help='Max subjects per section to diagnose')
    ap.add_argument('--output', type=str, default=None, help='Override output JSON path')
    return ap.parse_args()


def analyze_intensity_distribution(image: sitk.Image, mask: sitk.Image, label: str) -> Dict[str, float]:
    """Analyze intensity distribution within brain mask."""
    arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    brain_values = arr[mask_arr]
    
    if len(brain_values) < 100:
        return {'error': 'insufficient_brain_tissue'}
    
    return {
        'mean': float(brain_values.mean()),
        'std': float(brain_values.std()),
        'median': float(np.median(brain_values)),
        'cv': float(brain_values.std() / (brain_values.mean() + 1e-8)),
        'min': float(brain_values.min()),
        'max': float(brain_values.max()),
        'percentile_5': float(np.percentile(brain_values, 5)),
        'percentile_95': float(np.percentile(brain_values, 95)),
        'voxel_count': len(brain_values),
        'label': label
    }


def analyze_slice_uniformity(image: sitk.Image, mask: sitk.Image) -> Dict[str, float]:
    """Analyze slice-wise uniformity."""
    arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    
    slice_means = []
    slice_stds = []
    slice_cvs = []
    
    for z in range(arr.shape[0]):
        slice_mask = mask_arr[z]
        if slice_mask.sum() > 50:  # Minimum voxels for reliable stats
            slice_vals = arr[z][slice_mask]
            slice_mean = slice_vals.mean()
            slice_std = slice_vals.std()
            
            slice_means.append(slice_mean)
            slice_stds.append(slice_std)
            if slice_mean > 0:
                slice_cvs.append(slice_std / slice_mean)
    
    if not slice_means:
        return {'error': 'no_valid_slices'}
    
    slice_means = np.array(slice_means)
    slice_stds = np.array(slice_stds)
    slice_cvs = np.array(slice_cvs) if slice_cvs else np.array([np.nan])
    
    return {
        'slice_mean_variation': float(slice_means.std()),
        'slice_std_variation': float(slice_stds.std()),
        'slice_cv_median': float(np.nanmedian(slice_cvs)),
        'slice_cv_std': float(np.nanstd(slice_cvs)),
        'valid_slices': len(slice_means),
        'total_slices': arr.shape[0]
    }


def check_intensity_scaling_issues(original: sitk.Image, n4_corrected: sitk.Image, mask: sitk.Image) -> Dict[str, Any]:
    """Check for intensity scaling or range issues after N4 correction."""
    orig_arr = sitk.GetArrayFromImage(original)
    n4_arr = sitk.GetArrayFromImage(n4_corrected)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    
    orig_brain = orig_arr[mask_arr]
    n4_brain = n4_arr[mask_arr]
    
    if len(orig_brain) < 100:
        return {'error': 'insufficient_brain_tissue'}
    
    # Compute intensity range changes
    orig_range = orig_brain.max() - orig_brain.min()
    n4_range = n4_brain.max() - n4_brain.min()
    
    # Compute mean intensity changes
    orig_mean = orig_brain.mean()
    n4_mean = n4_brain.mean()
    
    # Check for problematic scaling
    range_ratio = n4_range / (orig_range + 1e-8)
    mean_ratio = n4_mean / (orig_mean + 1e-8)
    
    # Correlation between original and corrected
    correlation = np.corrcoef(orig_brain, n4_brain)[0, 1] if len(orig_brain) > 1 else np.nan
    
    # Check for negative values (shouldn't happen with MRI)
    negative_values = (n4_brain < 0).sum()
    
    # Check for extreme values
    orig_99th = np.percentile(orig_brain, 99)
    n4_99th = np.percentile(n4_brain, 99)
    
    return {
        'original_range': float(orig_range),
        'n4_range': float(n4_range),
        'range_ratio': float(range_ratio),
        'original_mean': float(orig_mean),
        'n4_mean': float(n4_mean),
        'mean_ratio': float(mean_ratio),
        'correlation': float(correlation),
        'negative_values_count': int(negative_values),
        'original_99th_percentile': float(orig_99th),
        'n4_99th_percentile': float(n4_99th),
        'extreme_scaling_detected': range_ratio > 2.0 or range_ratio < 0.5,
        'mean_shift_detected': mean_ratio > 1.5 or mean_ratio < 0.5,
        'poor_correlation_detected': correlation < 0.8
    }


def diagnose_single_subject(section: str, subject_id: str, modalities: List[str]) -> Dict[str, Any]:
    """Comprehensive diagnosis of N4 correction issues for one subject."""
    original_dir = PATHS['preprocessed_dir'] / section / subject_id
    n4_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
    
    subject_diagnosis = {
        'subject_info': {
            'section': section,
            'subject_id': subject_id
        },
        'modality_diagnoses': {},
        'overall_issues': []
    }
    
    for modality in modalities:
        orig_file = original_dir / f"{modality}.nii.gz"
        n4_file = n4_dir / f"{modality}.nii.gz"
        
        modality_diagnosis = {
            'files_exist': {
                'original': verify_image_file(orig_file),
                'n4_corrected': verify_image_file(n4_file)
            }
        }
        
        if not modality_diagnosis['files_exist']['original'] or not modality_diagnosis['files_exist']['n4_corrected']:
            modality_diagnosis['error'] = 'missing_files'
            subject_diagnosis['modality_diagnoses'][modality] = modality_diagnosis
            continue
        
        try:
            # Load images
            original = sitk.ReadImage(str(orig_file))
            n4_corrected = sitk.ReadImage(str(n4_file))
            
            # Generate mask from original (more reliable for comparison)
            mask = generate_brain_mask(original)
            
            # Analyze intensity distributions
            orig_stats = analyze_intensity_distribution(original, mask, 'original')
            n4_stats = analyze_intensity_distribution(n4_corrected, mask, 'n4_corrected')
            
            # Analyze slice uniformity
            orig_uniformity = analyze_slice_uniformity(original, mask)
            n4_uniformity = analyze_slice_uniformity(n4_corrected, mask)
            
            # Check for scaling/range issues
            scaling_analysis = check_intensity_scaling_issues(original, n4_corrected, mask)
            
            modality_diagnosis.update({
                'intensity_analysis': {
                    'original': orig_stats,
                    'n4_corrected': n4_stats
                },
                'uniformity_analysis': {
                    'original': orig_uniformity,
                    'n4_corrected': n4_uniformity
                },
                'scaling_analysis': scaling_analysis,
                'diagnosis_successful': True
            })
            
            # Detect specific issues
            issues = []
            
            # Issue 1: CV got worse instead of better
            if 'cv' in orig_stats and 'cv' in n4_stats:
                cv_ratio = n4_stats['cv'] / (orig_stats['cv'] + 1e-8)
                if cv_ratio > 1.1:
                    issues.append(f"CV increased by {cv_ratio:.2f}x (should decrease)")
            
            # Issue 2: Extreme intensity scaling
            if scaling_analysis.get('extreme_scaling_detected', False):
                issues.append(f"Extreme intensity scaling detected (ratio: {scaling_analysis.get('range_ratio', 'N/A'):.2f})")
            
            # Issue 3: Mean intensity shift
            if scaling_analysis.get('mean_shift_detected', False):
                issues.append(f"Large mean intensity shift (ratio: {scaling_analysis.get('mean_ratio', 'N/A'):.2f})")
            
            # Issue 4: Poor correlation
            if scaling_analysis.get('poor_correlation_detected', False):
                issues.append(f"Poor correlation between original and N4 ({scaling_analysis.get('correlation', 'N/A'):.3f})")
            
            # Issue 5: Negative values
            if scaling_analysis.get('negative_values_count', 0) > 0:
                issues.append(f"Negative values introduced: {scaling_analysis['negative_values_count']}")
            
            modality_diagnosis['detected_issues'] = issues
            
            if issues:
                subject_diagnosis['overall_issues'].extend([f"{modality}: {issue}" for issue in issues])
        
        except Exception as e:
            logging.error("Error diagnosing %s/%s %s: %s", section, subject_id, modality, e)
            modality_diagnosis['error'] = str(e)
        
        subject_diagnosis['modality_diagnoses'][modality] = modality_diagnosis
    
    return subject_diagnosis


def run_diagnostic_analysis(metadata: Dict[str, Any], 
                          max_subjects_per_section: int = 10,
                          splits: List[str] = None) -> Dict[str, Any]:
    """Run diagnostic analysis on a sample of subjects."""
    logging.info("Starting N4 diagnostic analysis...")
    
    results = {
        'diagnostic_info': {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'max_subjects_per_section': max_subjects_per_section
        },
        'subjects_analyzed': {
            'brats': [],
            'upenn': []
        },
        'common_issues': defaultdict(int),
        'detailed_diagnoses': {
            'brats': {},
            'upenn': {}
        }
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    # Analyze sample of subjects from each dataset
    for section in ['brats', 'upenn']:
        logging.info(f"Analyzing {section} subjects...")
        valid_subjects = metadata[section]['valid_subjects']
        
        # Get subjects with both original and N4-corrected files
        available_subjects = []
        for subject_id, subject_info in valid_subjects.items():
            if subject_info.get('split') in (splits or ['train','val']):  # Focus on chosen splits
                # Check if subject has N4-corrected files
                n4_dir = PATHS['preprocessed_dir'] / f"{section}_n4corrected" / subject_id
                if n4_dir.exists():
                    available_subjects.append(subject_id)
        
        # Sample subjects for analysis
        import random
        random.seed(42)  # For reproducible results
        sample_subjects = random.sample(available_subjects, 
                                      min(max_subjects_per_section, len(available_subjects)))
        
        results['subjects_analyzed'][section] = sample_subjects
        
        for subject_id in sample_subjects:
            logging.info(f"Diagnosing {section}/{subject_id}...")
            diagnosis = diagnose_single_subject(section, subject_id, modalities)
            results['detailed_diagnoses'][section][subject_id] = diagnosis
            
            # Collect common issues
            for issue in diagnosis.get('overall_issues', []):
                results['common_issues'][issue] += 1
    
    return results


def generate_diagnostic_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of diagnostic findings."""
    summary = {
        'analysis_overview': {
            'subjects_analyzed': sum(len(subjects) for subjects in results['subjects_analyzed'].values()),
            'sections_analyzed': len(results['subjects_analyzed'])
        },
        'issue_frequency': dict(results['common_issues']),
        'top_issues': [],
        'recommendations': []
    }
    
    # Sort issues by frequency
    sorted_issues = sorted(results['common_issues'].items(), key=lambda x: x[1], reverse=True)
    summary['top_issues'] = sorted_issues[:10]  # Top 10 most common issues
    
    # Generate recommendations based on findings
    recommendations = []
    
    for issue, count in sorted_issues:
        if 'CV increased' in issue:
            recommendations.append("N4 correction is increasing rather than decreasing intensity variation - check N4 parameters")
        elif 'Extreme intensity scaling' in issue:
            recommendations.append("N4 correction is causing extreme intensity scaling - consider more conservative parameters")
        elif 'mean intensity shift' in issue:
            recommendations.append("Large mean intensity shifts detected - verify intensity normalization steps")
        elif 'Poor correlation' in issue:
            recommendations.append("Poor correlation suggests N4 correction is distorting image content")
        elif 'Negative values' in issue:
            recommendations.append("Negative values introduced - check N4 implementation and input image preprocessing")
    
    # Add general recommendations
    recommendations.extend([
        "Consider using more conservative N4 parameters (fewer iterations, larger smoothing)",
        "Verify that input images are properly intensity normalized before N4 correction", 
        "Check if N4 correction is necessary - images may already be sufficiently uniform",
        "Consider alternative bias correction methods if N4 consistently performs poorly"
    ])
    
    summary['recommendations'] = list(set(recommendations))  # Remove duplicates
    
    return summary


def save_diagnostic_results(results: Dict[str, Any], summary: Dict[str, Any], output_path: Path) -> None:
    """Save diagnostic results to JSON file."""
    output_data = {
        'diagnostic_info': results.get('diagnostic_info', {}),
        'analysis_summary': summary,
        'subjects_analyzed': results.get('subjects_analyzed', {}),
        'detailed_diagnoses': results.get('detailed_diagnoses', {})
    }
    
    try:
        schema = {
            'diagnostic_info': dict,
            'analysis_summary': dict,
            'subjects_analyzed': dict,
            'detailed_diagnoses': dict
        }
        write_json_with_schema(output_data, output_path, schema=schema)
        file_size = output_path.stat().st_size
        logging.info("Diagnostic results saved to: %s (%.1f KB)", output_path, file_size / 1024)
        summary_path = output_path.parent / (output_path.stem + '_summary.json')
        light = {
            'analysis_date': output_data['diagnostic_info'].get('analysis_date'),
            'subjects_analyzed': summary['analysis_overview']['subjects_analyzed']
        }
        write_json_with_schema(light, summary_path)
        logging.info("Summary file written: %s", summary_path)
    except Exception as e:
        logging.error("Failed to save diagnostic results: %s", e)
        raise


def print_diagnostic_summary(summary: Dict[str, Any]) -> None:
    """Print diagnostic summary."""
    print("\n" + "="*80)
    print("N4 CORRECTION DIAGNOSTIC ANALYSIS")
    print("="*80)
    
    print(f"\nAnalysis Overview:")
    print(f"  Subjects analyzed: {summary['analysis_overview']['subjects_analyzed']}")
    print(f"  Sections analyzed: {summary['analysis_overview']['sections_analyzed']}")
    
    print(f"\nTop Issues Detected:")
    for i, (issue, count) in enumerate(summary['top_issues'][:5], 1):
        print(f"  {i}. {issue} (occurred {count} times)")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(summary['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    print("="*80)


def main():
    configure_logging()
    args = parse_args()
    logging.info("=== N4 CORRECTION DIAGNOSTIC ANALYSIS ===")
    try:
        metadata_path = PATHS['metadata_splits']
        metadata = load_metadata(metadata_path)
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]
        results = run_diagnostic_analysis(metadata, max_subjects_per_section=args.max_per_section, splits=splits)
        summary = generate_diagnostic_summary(results)
        output_path = PATHS['preprocessed_dir'] / "n4_diagnostic_analysis.json"
        if args.output:
            output_path = Path(args.output)
        save_diagnostic_results(results, summary, output_path)
        print_diagnostic_summary(summary)
        logging.info("N4 diagnostic analysis completed successfully")
    except Exception as e:
        logging.error("N4 diagnostic analysis failed: %s", e)
        raise


if __name__ == '__main__':
    main()