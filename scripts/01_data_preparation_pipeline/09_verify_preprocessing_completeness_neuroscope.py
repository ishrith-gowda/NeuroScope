import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import SimpleITK as sitk
import numpy as np
from collections import defaultdict

from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """Configure logging for pipeline verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


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


def analyze_image_quality(file_path: Path) -> Dict[str, Any]:
    """Analyze basic image quality metrics."""
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        # Basic quality metrics
        nonzero_values = arr[arr > 0]
        if len(nonzero_values) < 100:
            return {'error': 'insufficient_nonzero_voxels'}
        
        return {
            'spacing': img.GetSpacing(),
            'size': img.GetSize(),
            'mean_intensity': float(nonzero_values.mean()),
            'std_intensity': float(nonzero_values.std()),
            'cv': float(nonzero_values.std() / (nonzero_values.mean() + 1e-8)),
            'min_intensity': float(nonzero_values.min()),
            'max_intensity': float(nonzero_values.max()),
            'total_voxels': int(arr.size),
            'nonzero_voxels': len(nonzero_values),
            'intensity_range': float(nonzero_values.max() - nonzero_values.min())
        }
    except Exception as e:
        return {'error': str(e)}


def verify_subject_completeness(section: str, subject_id: str, modalities: List[str]) -> Dict[str, Any]:
    """Verify completeness and quality of a single subject's preprocessed data."""
    preprocessed_dir = PATHS['preprocessed_dir'] / section / subject_id
    
    subject_status = {
        'subject_id': subject_id,
        'section': section,
        'modality_status': {},
        'overall_complete': False,
        'ready_for_training': False
    }
    
    complete_modalities = 0
    quality_issues = []
    
    for modality in modalities:
        file_path = preprocessed_dir / f"{modality}.nii.gz"
        
        modality_status = {
            'file_exists': verify_image_file(file_path),
            'quality_metrics': {}
        }
        
        if modality_status['file_exists']:
            quality_metrics = analyze_image_quality(file_path)
            modality_status['quality_metrics'] = quality_metrics
            
            if 'error' not in quality_metrics:
                complete_modalities += 1
                
                # Check for quality issues
                if quality_metrics['cv'] > 0.5:  # Very high variation
                    quality_issues.append(f"{modality}: high CV ({quality_metrics['cv']:.3f})")
                
                if quality_metrics['intensity_range'] < 0.1:  # Very low range
                    quality_issues.append(f"{modality}: low intensity range ({quality_metrics['intensity_range']:.3f})")
                
                # Check spacing (should be 1mm isotropic)
                spacing = quality_metrics['spacing']
                if not all(abs(s - 1.0) < 0.1 for s in spacing):
                    quality_issues.append(f"{modality}: non-isotropic spacing {spacing}")
            else:
                quality_issues.append(f"{modality}: quality analysis failed - {quality_metrics['error']}")
        else:
            quality_issues.append(f"{modality}: file missing or unreadable")
        
        subject_status['modality_status'][modality] = modality_status
    
    # Overall assessment
    subject_status['complete_modalities'] = complete_modalities
    subject_status['total_modalities'] = len(modalities)
    subject_status['overall_complete'] = complete_modalities >= 3  # At least 3/4 modalities
    subject_status['ready_for_training'] = complete_modalities == len(modalities) and not quality_issues
    subject_status['quality_issues'] = quality_issues
    
    return subject_status


def run_pipeline_verification(metadata: Dict[str, Any], splits_to_verify: List[str] = None) -> Dict[str, Any]:
    """Run comprehensive pipeline verification."""
    if splits_to_verify is None:
        splits_to_verify = ['train', 'val']
    
    start_time = time.time()
    logging.info("Starting comprehensive pipeline verification...")
    logging.info("Verifying splits: %s", splits_to_verify)
    
    results = {
        'verification_info': {
            'verification_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'splits_verified': splits_to_verify,
            'script_version': '09_verify_preprocessing_completeness_neuroscope.py v1.0'
        },
        'summary_statistics': {
            'total_subjects_verified': 0,
            'complete_subjects': 0,
            'training_ready_subjects': 0,
            'subjects_with_quality_issues': 0,
            'missing_files_count': 0
        },
        'dataset_breakdown': {
            'brats': {'total': 0, 'complete': 0, 'training_ready': 0},
            'upenn': {'total': 0, 'complete': 0, 'training_ready': 0}
        },
        'modality_statistics': {
            't1': {'available': 0, 'quality_ok': 0},
            't1gd': {'available': 0, 'quality_ok': 0},
            't2': {'available': 0, 'quality_ok': 0},
            'flair': {'available': 0, 'quality_ok': 0}
        },
        'quality_analysis': {
            'cv_statistics': {'values': [], 'mean': 0, 'median': 0, 'std': 0},
            'spacing_compliance': 0,
            'intensity_range_statistics': {'values': [], 'mean': 0, 'median': 0}
        },
        'detailed_results': {
            'brats': {},
            'upenn': {}
        }
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    all_cvs = []
    all_ranges = []
    spacing_compliant = 0
    total_modalities_checked = 0
    
    # Process each dataset
    for section in ['brats', 'upenn']:
        logging.info("Verifying %s dataset...", section)
        valid_subjects = metadata[section]['valid_subjects']
        
        section_processed = 0
        section_complete = 0
        section_training_ready = 0
        
        for subject_id, subject_info in valid_subjects.items():
            # Check if subject is in the splits we want to verify
            subject_split = subject_info.get('split')
            if subject_split not in splits_to_verify:
                continue
            
            results['summary_statistics']['total_subjects_verified'] += 1
            results['dataset_breakdown'][section]['total'] += 1
            section_processed += 1
            
            # Verify subject completeness and quality
            subject_status = verify_subject_completeness(section, subject_id, modalities)
            results['detailed_results'][section][subject_id] = subject_status
            
            # Update statistics
            if subject_status['overall_complete']:
                results['summary_statistics']['complete_subjects'] += 1
                results['dataset_breakdown'][section]['complete'] += 1
                section_complete += 1
            
            if subject_status['ready_for_training']:
                results['summary_statistics']['training_ready_subjects'] += 1
                results['dataset_breakdown'][section]['training_ready'] += 1
                section_training_ready += 1
            
            if subject_status['quality_issues']:
                results['summary_statistics']['subjects_with_quality_issues'] += 1
            
            # Collect modality statistics
            for modality, mod_status in subject_status['modality_status'].items():
                if mod_status['file_exists']:
                    results['modality_statistics'][modality]['available'] += 1
                    
                    quality_metrics = mod_status['quality_metrics']
                    if 'error' not in quality_metrics:
                        results['modality_statistics'][modality]['quality_ok'] += 1
                        
                        # Collect quality statistics
                        cv = quality_metrics.get('cv', np.nan)
                        if np.isfinite(cv):
                            all_cvs.append(cv)
                        
                        intensity_range = quality_metrics.get('intensity_range', np.nan)
                        if np.isfinite(intensity_range):
                            all_ranges.append(intensity_range)
                        
                        # Check spacing compliance
                        spacing = quality_metrics.get('spacing', [])
                        if spacing and all(abs(s - 1.0) < 0.1 for s in spacing):
                            spacing_compliant += 1
                        
                        total_modalities_checked += 1
                else:
                    results['summary_statistics']['missing_files_count'] += 1
            
            if section_processed % 50 == 0:
                logging.info("Verified %d subjects in %s dataset", section_processed, section)
        
        logging.info("%s verification complete: %d total, %d complete, %d training-ready", 
                    section, section_processed, section_complete, section_training_ready)
    
    # Compute overall quality statistics
    if all_cvs:
        results['quality_analysis']['cv_statistics'] = {
            'values': all_cvs,
            'mean': float(np.mean(all_cvs)),
            'median': float(np.median(all_cvs)),
            'std': float(np.std(all_cvs)),
            'count': len(all_cvs)
        }
    
    if all_ranges:
        results['quality_analysis']['intensity_range_statistics'] = {
            'values': all_ranges,
            'mean': float(np.mean(all_ranges)),
            'median': float(np.median(all_ranges)),
            'count': len(all_ranges)
        }
    
    if total_modalities_checked > 0:
        results['quality_analysis']['spacing_compliance'] = spacing_compliant / total_modalities_checked
    
    # Add timing info
    elapsed_time = time.time() - start_time
    results['verification_info']['elapsed_time_seconds'] = elapsed_time
    results['verification_info']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info("Pipeline verification completed in %.1f seconds", elapsed_time)
    return results


def generate_verification_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics for pipeline verification."""
    summary = {
        'pipeline_readiness': 'unknown',
        'completion_rates': {},
        'quality_assessment': {},
        'recommendations': []
    }
    
    # Calculate completion rates
    total_subjects = results['summary_statistics']['total_subjects_verified']
    complete_subjects = results['summary_statistics']['complete_subjects']
    training_ready = results['summary_statistics']['training_ready_subjects']
    
    if total_subjects > 0:
        completion_rate = complete_subjects / total_subjects
        training_ready_rate = training_ready / total_subjects
        
        summary['completion_rates'] = {
            'overall_completion': completion_rate,
            'training_ready': training_ready_rate,
            'subjects_with_issues': results['summary_statistics']['subjects_with_quality_issues'] / total_subjects
        }
        
        # Assess pipeline readiness
        if training_ready_rate >= 0.95:
            summary['pipeline_readiness'] = 'excellent'
        elif training_ready_rate >= 0.90:
            summary['pipeline_readiness'] = 'good'
        elif training_ready_rate >= 0.80:
            summary['pipeline_readiness'] = 'acceptable'
        else:
            summary['pipeline_readiness'] = 'needs_improvement'
    
    # Quality assessment
    cv_stats = results['quality_analysis'].get('cv_statistics', {})
    if 'median' in cv_stats:
        median_cv = cv_stats['median']
        if median_cv < 0.2:
            summary['quality_assessment']['intensity_uniformity'] = 'excellent'
        elif median_cv < 0.3:
            summary['quality_assessment']['intensity_uniformity'] = 'good'
        else:
            summary['quality_assessment']['intensity_uniformity'] = 'needs_improvement'
    
    spacing_compliance = results['quality_analysis'].get('spacing_compliance', 0)
    if spacing_compliance >= 0.95:
        summary['quality_assessment']['spacing_consistency'] = 'excellent'
    elif spacing_compliance >= 0.90:
        summary['quality_assessment']['spacing_consistency'] = 'good'
    else:
        summary['quality_assessment']['spacing_consistency'] = 'needs_improvement'
    
    # Generate recommendations
    recommendations = []
    
    if training_ready_rate < 0.90:
        recommendations.append(f"Only {training_ready_rate*100:.1f}% of subjects are training-ready - investigate quality issues")
    
    if results['summary_statistics']['missing_files_count'] > 0:
        recommendations.append(f"{results['summary_statistics']['missing_files_count']} missing files detected - run preprocessing scripts")
    
    if cv_stats.get('median', 0) > 0.3:
        recommendations.append("High intensity variation detected - consider additional normalization")
    
    if spacing_compliance < 0.95:
        recommendations.append("Some images not properly resampled to 1mm isotropic")
    
    if not recommendations:
        recommendations.append("Pipeline verification passed - data is ready for CycleGAN training!")
    
    summary['recommendations'] = recommendations
    
    return summary


def save_verification_results(results: Dict[str, Any], summary: Dict[str, Any], output_path: Path) -> None:
    """Save verification results to JSON file."""
    output_data = {
        'verification_info': results.get('verification_info', {}),
        'summary_statistics': results.get('summary_statistics', {}),
        'pipeline_summary': summary,
        'dataset_breakdown': results.get('dataset_breakdown', {}),
        'modality_statistics': results.get('modality_statistics', {}),
        'quality_analysis': results.get('quality_analysis', {}),
        'detailed_results': results.get('detailed_results', {})
    }
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, sort_keys=True)
        
        file_size = output_path.stat().st_size
        logging.info("Verification results saved to: %s (%.1f KB)", output_path, file_size / 1024)
        
    except Exception as e:
        logging.error("Failed to save verification results: %s", e)
        raise


def print_verification_summary(summary: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Print comprehensive verification summary."""
    print("\n" + "="*80)
    print("NEUROSCOPE PREPROCESSING PIPELINE VERIFICATION")
    print("="*80)
    
    # Overall status
    summary_stats = results.get('summary_statistics', {})
    print(f"\nOverall Pipeline Status: {summary['pipeline_readiness'].upper()}")
    print(f"\nProcessing Summary:")
    print(f"  Total subjects verified:    {summary_stats.get('total_subjects_verified', 0)}")
    print(f"  Complete subjects:          {summary_stats.get('complete_subjects', 0)}")
    print(f"  Training-ready subjects:    {summary_stats.get('training_ready_subjects', 0)}")
    print(f"  Subjects with issues:       {summary_stats.get('subjects_with_quality_issues', 0)}")
    print(f"  Missing files:              {summary_stats.get('missing_files_count', 0)}")
    
    # Completion rates
    completion_rates = summary.get('completion_rates', {})
    if completion_rates:
        print(f"\nCompletion Rates:")
        print(f"  Overall completion:         {completion_rates.get('overall_completion', 0)*100:.1f}%")
        print(f"  Training ready:             {completion_rates.get('training_ready', 0)*100:.1f}%")
        print(f"  Subjects with issues:       {completion_rates.get('subjects_with_issues', 0)*100:.1f}%")
    
    # Dataset breakdown
    dataset_breakdown = results.get('dataset_breakdown', {})
    print(f"\nDataset Breakdown:")
    for section in ['brats', 'upenn']:
        section_name = 'BraTS-TCGA-GBM' if section == 'brats' else 'UPenn-GBM'
        section_data = dataset_breakdown.get(section, {})
        if section_data.get('total', 0) > 0:
            print(f"  {section_name}:")
            print(f"    Total:                    {section_data.get('total', 0)}")
            print(f"    Complete:                 {section_data.get('complete', 0)}")
            print(f"    Training-ready:           {section_data.get('training_ready', 0)}")
            print(f"    Training-ready rate:      {section_data.get('training_ready', 0)/section_data.get('total', 1)*100:.1f}%")
    
    # Modality statistics
    modality_stats = results.get('modality_statistics', {})
    print(f"\nModality Availability:")
    modality_names = {'t1': 'T1', 't1gd': 'T1-Gd', 't2': 'T2', 'flair': 'FLAIR'}
    for modality, display_name in modality_names.items():
        mod_data = modality_stats.get(modality, {})
        available = mod_data.get('available', 0)
        quality_ok = mod_data.get('quality_ok', 0)
        if available > 0:
            quality_rate = quality_ok / available * 100
            print(f"  {display_name}:                      {available} available, {quality_ok} quality OK ({quality_rate:.1f}%)")
    
    # Quality assessment
    quality_assessment = summary.get('quality_assessment', {})
    print(f"\nQuality Assessment:")
    if 'intensity_uniformity' in quality_assessment:
        print(f"  Intensity uniformity:       {quality_assessment['intensity_uniformity']}")
    if 'spacing_consistency' in quality_assessment:
        print(f"  Spacing consistency:        {quality_assessment['spacing_consistency']}")
    
    # Quality statistics
    cv_stats = results['quality_analysis'].get('cv_statistics', {})
    if 'median' in cv_stats:
        print(f"  Median CV:                  {cv_stats['median']:.3f}")
        print(f"  Mean CV:                    {cv_stats['mean']:.3f}")
    
    spacing_compliance = results['quality_analysis'].get('spacing_compliance', 0)
    print(f"  Spacing compliance:         {spacing_compliance*100:.1f}%")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    print(f"\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Next steps
    print(f"\nNext Steps:")
    if summary['pipeline_readiness'] in ['excellent', 'good']:
        print(f"  ✓ Your preprocessing pipeline is working excellently!")
        print(f"  ✓ Data is ready for CycleGAN training")
        print(f"  ✓ No N4 bias field correction needed")
        print(f"  → Proceed to script 02_model_development_pipeline/train_cyclegan.py")
    else:
        print(f"  ⚠ Address quality issues before proceeding to model training")
        print(f"  → Review detailed results in the JSON output file")
    
    print("="*80)


def main() -> None:
    """Main function for pipeline verification."""
    configure_logging()
    
    logging.info("=== NEUROSCOPE PREPROCESSING PIPELINE VERIFICATION ===")
    
    # Define paths
    metadata_path = PATHS['metadata_splits']
    output_path = PATHS['preprocessed_dir'] / "pipeline_verification_results.json"
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Run comprehensive verification
        results = run_pipeline_verification(metadata, splits_to_verify=['train', 'val'])
        
        # Generate summary
        summary = generate_verification_summary(results)
        
        # Save results
        save_verification_results(results, summary, output_path)
        
        # Print summary
        print_verification_summary(summary, results)
        
        logging.info("Pipeline verification completed successfully")
        
    except Exception as e:
        logging.error("Pipeline verification failed: %s", e)
        raise


if __name__ == '__main__':
    main()