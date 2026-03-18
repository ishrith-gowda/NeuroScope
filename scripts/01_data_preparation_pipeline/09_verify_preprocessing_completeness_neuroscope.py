import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import SimpleITK as sitk
import numpy as np
from collections import defaultdict

from neuroscope_preprocessing_config import PATHS
import argparse
from preprocessing_utils import write_json_with_schema


def configure_logging() -> None:
    """configure logging for pipeline verification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def verify_image_file(file_path: Path) -> bool:
    """verify that an image file exists and is readable."""
    if not file_path.exists():
        return False
    
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        return arr.size > 0 and np.isfinite(arr).all()
    except Exception:
        return False


def sanitize_for_json(obj: Any) -> Any:
    """
    recursively sanitize objects to be json serializable.
    converts numpy types and other non-serializable types.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(item) for item in obj]  # convert tuple to list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # convert numpy array to list
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)  # ensure it's python bool
    elif isinstance(obj, np.str_):
        return str(obj)
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        # for any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None


def analyze_image_quality(file_path: Path) -> Dict[str, Any]:
    """analyze basic image quality metrics."""
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        # basic quality metrics
        nonzero_values = arr[arr > 0]
        if len(nonzero_values) < 100:
            return {'error': 'insufficient_nonzero_voxels'}
        
        # ensure all values are json serializable
        return {
            'spacing': [float(s) for s in img.GetSpacing()],  # convert to regular float
            'size': [int(s) for s in img.GetSize()],  # convert to regular int
            'mean_intensity': float(nonzero_values.mean()),
            'std_intensity': float(nonzero_values.std()),
            'cv': float(nonzero_values.std() / (nonzero_values.mean() + 1e-8)),
            'min_intensity': float(nonzero_values.min()),
            'max_intensity': float(nonzero_values.max()),
            'total_voxels': int(arr.size),
            'nonzero_voxels': int(len(nonzero_values)),
            'intensity_range': float(nonzero_values.max() - nonzero_values.min())
        }
    except Exception as e:
        return {'error': str(e)}


def verify_subject_completeness(section: str, subject_id: str, modalities: List[str]) -> Dict[str, Any]:
    """verify completeness and quality of a single subject's preprocessed data."""
    preprocessed_dir = PATHS['preprocessed_dir'] / section / subject_id
    
    subject_status = {
        'subject_id': str(subject_id),
        'section': str(section),
        'modality_status': {},
        'overall_complete': False,
        'ready_for_training': False
    }
    
    complete_modalities = 0
    quality_issues = []
    
    for modality in modalities:
        file_path = preprocessed_dir / f"{modality}.nii.gz"
        
        modality_status = {
            'file_exists': bool(verify_image_file(file_path)),  # ensure python bool
            'quality_metrics': {}
        }
        
        if modality_status['file_exists']:
            quality_metrics = analyze_image_quality(file_path)
            modality_status['quality_metrics'] = quality_metrics
            
            if 'error' not in quality_metrics:
                complete_modalities += 1
                
                # check for quality issues
                if quality_metrics['cv'] > 0.5:  # very high variation
                    quality_issues.append(f"{modality}: high CV ({quality_metrics['cv']:.3f})")
                
                if quality_metrics['intensity_range'] < 0.1:  # very low range
                    quality_issues.append(f"{modality}: low intensity range ({quality_metrics['intensity_range']:.3f})")
                
                # check spacing (should be 1mm isotropic)
                spacing = quality_metrics['spacing']
                if not all(abs(s - 1.0) < 0.1 for s in spacing):
                    quality_issues.append(f"{modality}: non-isotropic spacing {spacing}")
            else:
                quality_issues.append(f"{modality}: quality analysis failed - {quality_metrics['error']}")
        else:
            quality_issues.append(f"{modality}: file missing or unreadable")
        
        subject_status['modality_status'][modality] = modality_status
    
    # overall assessment - ensure all values are json serializable
    subject_status['complete_modalities'] = int(complete_modalities)
    subject_status['total_modalities'] = int(len(modalities))
    subject_status['overall_complete'] = bool(complete_modalities >= 3)  # at least 3/4 modalities
    subject_status['ready_for_training'] = bool(complete_modalities == len(modalities) and not quality_issues)
    subject_status['quality_issues'] = quality_issues
    
    return subject_status


def calculate_dataset_statistics(detailed_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """calculate comprehensive statistics from detailed results."""
    stats = {
        'total_subjects': 0,
        'complete_subjects': 0,
        'training_ready_subjects': 0,
        'subjects_with_issues': 0,
        'missing_files_total': 0,
        'modality_breakdown': {
            't1': {'available': 0, 'quality_ok': 0, 'missing': 0},
            't1gd': {'available': 0, 'quality_ok': 0, 'missing': 0},
            't2': {'available': 0, 'quality_ok': 0, 'missing': 0},
            'flair': {'available': 0, 'quality_ok': 0, 'missing': 0}
        },
        'quality_metrics': {
            'cv_values': [],
            'intensity_ranges': [],
            'spacing_compliant': 0,
            'total_modalities_checked': 0
        }
    }
    
    for section, subjects in detailed_results.items():
        for subject_id, subject_data in subjects.items():
            stats['total_subjects'] += 1
            
            if subject_data.get('overall_complete', False):
                stats['complete_subjects'] += 1
            
            if subject_data.get('ready_for_training', False):
                stats['training_ready_subjects'] += 1
            
            if subject_data.get('quality_issues', []):
                stats['subjects_with_issues'] += 1
            
            # analyze modalities
            for modality, mod_status in subject_data.get('modality_status', {}).items():
                if modality not in stats['modality_breakdown']:
                    continue
                
                if mod_status.get('file_exists', False):
                    stats['modality_breakdown'][modality]['available'] += 1
                    
                    quality_metrics = mod_status.get('quality_metrics', {})
                    if 'error' not in quality_metrics:
                        stats['modality_breakdown'][modality]['quality_ok'] += 1
                        
                        # collect quality statistics
                        cv = quality_metrics.get('cv')
                        if cv is not None and np.isfinite(cv):
                            stats['quality_metrics']['cv_values'].append(float(cv))
                        
                        intensity_range = quality_metrics.get('intensity_range')
                        if intensity_range is not None and np.isfinite(intensity_range):
                            stats['quality_metrics']['intensity_ranges'].append(float(intensity_range))
                        
                        # check spacing compliance
                        spacing = quality_metrics.get('spacing', [])
                        if spacing and all(abs(s - 1.0) < 0.1 for s in spacing):
                            stats['quality_metrics']['spacing_compliant'] += 1
                        
                        stats['quality_metrics']['total_modalities_checked'] += 1
                else:
                    stats['modality_breakdown'][modality]['missing'] += 1
                    stats['missing_files_total'] += 1
    
    return stats


def run_pipeline_verification(metadata: Dict[str, Any], splits_to_verify: List[str] = None) -> Dict[str, Any]:
    """run comprehensive pipeline verification."""
    if splits_to_verify is None:
        splits_to_verify = ['train', 'val']
    
    start_time = time.time()
    logging.info("Starting comprehensive pipeline verification...")
    logging.info("Verifying splits: %s", splits_to_verify)
    
    results = {
        'verification_info': {
            'verification_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'splits_verified': splits_to_verify,
            'script_version': '09_verify_preprocessing_completeness_neuroscope.py v2.0 (fixed)',
            'neuroscope_version': 'v1.0'
        },
        'detailed_results': {
            'brats': {},
            'upenn': {}
        }
    }
    
    modalities = ['t1', 't1gd', 't2', 'flair']
    
    # process each dataset
    for section in ['brats', 'upenn']:
        logging.info("Verifying %s dataset...", section)
        
        if section not in metadata:
            logging.warning("Section %s not found in metadata", section)
            continue
            
        valid_subjects = metadata[section].get('valid_subjects', {})
        if not valid_subjects:
            logging.warning("No valid subjects found for section %s", section)
            continue
        
        section_processed = 0
        
        for subject_id, subject_info in valid_subjects.items():
            # check if subject is in the splits we want to verify
            subject_split = subject_info.get('split')
            if subject_split not in splits_to_verify:
                continue
            
            section_processed += 1
            
            # verify subject completeness and quality
            subject_status = verify_subject_completeness(section, subject_id, modalities)
            results['detailed_results'][section][subject_id] = subject_status
            
            if section_processed % 50 == 0:
                logging.info("Verified %d subjects in %s dataset", section_processed, section)
        
        section_complete = sum(1 for s in results['detailed_results'][section].values() if s.get('overall_complete', False))
        section_training_ready = sum(1 for s in results['detailed_results'][section].values() if s.get('ready_for_training', False))
        
        logging.info("%s verification complete: %d total, %d complete, %d training-ready", 
                    section, section_processed, section_complete, section_training_ready)
    
    # calculate comprehensive statistics
    stats = calculate_dataset_statistics(results['detailed_results'])
    
    # add summary statistics to results
    results['summary_statistics'] = {
        'total_subjects_verified': int(stats['total_subjects']),
        'complete_subjects': int(stats['complete_subjects']),
        'training_ready_subjects': int(stats['training_ready_subjects']),
        'subjects_with_quality_issues': int(stats['subjects_with_issues']),
        'missing_files_count': int(stats['missing_files_total'])
    }
    
    # add dataset breakdown
    results['dataset_breakdown'] = {
        section: {
            'total': len(results['detailed_results'][section]),
            'complete': sum(1 for s in results['detailed_results'][section].values() if s.get('overall_complete', False)),
            'training_ready': sum(1 for s in results['detailed_results'][section].values() if s.get('ready_for_training', False))
        }
        for section in ['brats', 'upenn']
    }
    
    # add modality statistics
    results['modality_statistics'] = stats['modality_breakdown']
    
    # add quality analysis with proper json serialization
    cv_values = stats['quality_metrics']['cv_values']
    intensity_ranges = stats['quality_metrics']['intensity_ranges']
    
    quality_analysis = {
        'spacing_compliance': float(stats['quality_metrics']['spacing_compliant'] / max(stats['quality_metrics']['total_modalities_checked'], 1))
    }
    
    if cv_values:
        quality_analysis['cv_statistics'] = {
            'count': len(cv_values),
            'mean': float(np.mean(cv_values)),
            'median': float(np.median(cv_values)),
            'std': float(np.std(cv_values)),
            'min': float(np.min(cv_values)),
            'max': float(np.max(cv_values))
        }
    
    if intensity_ranges:
        quality_analysis['intensity_range_statistics'] = {
            'count': len(intensity_ranges),
            'mean': float(np.mean(intensity_ranges)),
            'median': float(np.median(intensity_ranges)),
            'min': float(np.min(intensity_ranges)),
            'max': float(np.max(intensity_ranges))
        }
    
    results['quality_analysis'] = quality_analysis
    
    # add timing info
    elapsed_time = time.time() - start_time
    results['verification_info']['elapsed_time_seconds'] = float(elapsed_time)
    results['verification_info']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info("Pipeline verification completed in %.1f seconds", elapsed_time)
    return results


def generate_verification_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """generate summary statistics for pipeline verification."""
    summary = {
        'pipeline_readiness': 'unknown',
        'completion_rates': {},
        'quality_assessment': {},
        'recommendations': []
    }
    
    # calculate completion rates
    total_subjects = results['summary_statistics']['total_subjects_verified']
    complete_subjects = results['summary_statistics']['complete_subjects']
    training_ready = results['summary_statistics']['training_ready_subjects']
    
    if total_subjects > 0:
        completion_rate = complete_subjects / total_subjects
        training_ready_rate = training_ready / total_subjects
        
        summary['completion_rates'] = {
            'overall_completion': float(completion_rate),
            'training_ready': float(training_ready_rate),
            'subjects_with_issues': float(results['summary_statistics']['subjects_with_quality_issues'] / total_subjects)
        }
        
        # assess pipeline readiness
        if training_ready_rate >= 0.95:
            summary['pipeline_readiness'] = 'excellent'
        elif training_ready_rate >= 0.90:
            summary['pipeline_readiness'] = 'good'
        elif training_ready_rate >= 0.80:
            summary['pipeline_readiness'] = 'acceptable'
        else:
            summary['pipeline_readiness'] = 'needs_improvement'
    
    # quality assessment
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
    
    # generate recommendations
    recommendations = []
    
    if total_subjects > 0:
        training_ready_rate = training_ready / total_subjects
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
    """save verification results to json file with proper serialization."""
    output_data = {
        'verification_info': results.get('verification_info', {}),
        'summary_statistics': results.get('summary_statistics', {}),
        'pipeline_summary': summary,
        'dataset_breakdown': results.get('dataset_breakdown', {}),
        'modality_statistics': results.get('modality_statistics', {}),
        'quality_analysis': results.get('quality_analysis', {}),
        'detailed_results': results.get('detailed_results', {})
    }
    
    # sanitize all data for json serialization
    output_data = sanitize_for_json(output_data)
    
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
    """print comprehensive verification summary."""
    print("\n" + "="*80)
    print("neuroscope preprocessing pipeline verification")
    print("="*80)
    
    # overall status
    summary_stats = results.get('summary_statistics', {})
    pipeline_status = summary['pipeline_readiness'].upper()
    status_emoji = {
        'EXCELLENT': '🟢',
        'GOOD': '🟡', 
        'ACCEPTABLE': '🟠',
        'NEEDS_IMPROVEMENT': '🔴'
    }.get(pipeline_status, '⚪')
    
    print(f"\noverall pipeline status: {status_emoji} {pipeline_status}")
    
    print(f"\nprocessing summary:")
    print(f"  total subjects verified:    {summary_stats.get('total_subjects_verified', 0)}")
    print(f"  complete subjects:          {summary_stats.get('complete_subjects', 0)}")
    print(f"  training-ready subjects:    {summary_stats.get('training_ready_subjects', 0)}")
    print(f"  subjects with issues:       {summary_stats.get('subjects_with_quality_issues', 0)}")
    print(f"  missing files:              {summary_stats.get('missing_files_count', 0)}")
    
    # completion rates
    completion_rates = summary.get('completion_rates', {})
    if completion_rates:
        print(f"\ncompletion rates:")
        print(f"  overall completion:         {completion_rates.get('overall_completion', 0)*100:.1f}%")
        print(f"  training ready:             {completion_rates.get('training_ready', 0)*100:.1f}%")
        print(f"  subjects with issues:       {completion_rates.get('subjects_with_issues', 0)*100:.1f}%")
    
    # dataset breakdown
    dataset_breakdown = results.get('dataset_breakdown', {})
    print(f"\ndataset breakdown:")
    for section in ['brats', 'upenn']:
        section_name = 'BraTS-TCGA-GBM' if section == 'brats' else 'UPenn-GBM'
        section_data = dataset_breakdown.get(section, {})
        if section_data.get('total', 0) > 0:
            total = section_data.get('total', 0)
            complete = section_data.get('complete', 0)
            training_ready = section_data.get('training_ready', 0)
            
            print(f"  {section_name}:")
            print(f"    total:                    {total}")
            print(f"    complete:                 {complete} ({complete/max(total,1)*100:.1f}%)")
            print(f"    training-ready:           {training_ready} ({training_ready/max(total,1)*100:.1f}%)")
    
    # modality statistics
    modality_stats = results.get('modality_statistics', {})
    print(f"\nmodality availability:")
    modality_names = {'t1': 'T1', 't1gd': 'T1-Gd', 't2': 'T2', 'flair': 'FLAIR'}
    for modality, display_name in modality_names.items():
        mod_data = modality_stats.get(modality, {})
        available = mod_data.get('available', 0)
        quality_ok = mod_data.get('quality_ok', 0)
        missing = mod_data.get('missing', 0)
        
        total_expected = available + missing
        if total_expected > 0:
            availability_rate = available / total_expected * 100
            quality_rate = quality_ok / max(available, 1) * 100
            print(f"  {display_name:6}: {available:3}/{total_expected:3} available ({availability_rate:5.1f}%), "
                  f"{quality_ok:3} quality OK ({quality_rate:5.1f}%)")
    
    # quality assessment
    quality_assessment = summary.get('quality_assessment', {})
    print(f"\nquality assessment:")
    if 'intensity_uniformity' in quality_assessment:
        uniformity = quality_assessment['intensity_uniformity']
        uniformity_emoji = {'excellent': '🟢', 'good': '🟡', 'needs_improvement': '🔴'}.get(uniformity, '⚪')
        print(f"  intensity uniformity:       {uniformity_emoji} {uniformity}")
    
    if 'spacing_consistency' in quality_assessment:
        spacing = quality_assessment['spacing_consistency']
        spacing_emoji = {'excellent': '🟢', 'good': '🟡', 'needs_improvement': '🔴'}.get(spacing, '⚪')
        print(f"  spacing consistency:        {spacing_emoji} {spacing}")
    
    # quality statistics
    cv_stats = results['quality_analysis'].get('cv_statistics', {})
    if 'median' in cv_stats:
        print(f"  median cv:                  {cv_stats['median']:.3f}")
        print(f"  mean cv:                    {cv_stats['mean']:.3f} (±{cv_stats['std']:.3f})")
    
    spacing_compliance = results['quality_analysis'].get('spacing_compliance', 0)
    print(f"  spacing compliance:         {spacing_compliance*100:.1f}%")
    
    # recommendations
    recommendations = summary.get('recommendations', [])
    print(f"\nrecommendations:")
    for i, rec in enumerate(recommendations, 1):
        rec_symbol = "+" if "passed" in rec.lower() else "-"
        print(f"  {rec_symbol} {rec}")

    # next steps
    print(f"\nnext steps:")
    if summary['pipeline_readiness'] in ['excellent', 'good']:
        print(f"  + preprocessing pipeline is working excellently")
        print(f"  + data is ready for cyclegan training")
        print(f"  + proceed to: scripts/02_model_development_pipeline/train_cyclegan.py")
        print(f"  + review detailed json report for specifics")
    else:
        print(f"  - address quality issues before proceeding to model training")
        print(f"  - review detailed results in the json output file")
        print(f"  - run preprocessing scripts again if needed")
    
    # timing info
    elapsed_time = results['verification_info'].get('elapsed_time_seconds', 0)
    print(f"\nprocessing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    
    print("="*80)


def parse_args():
    ap = argparse.ArgumentParser(description='Verify preprocessing completeness & quality')
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of splits to verify')
    ap.add_argument('--output', type=str, default=None, help='Override output JSON path')
    ap.add_argument('--summary-only', action='store_true', help='Write only summary file (skip details)')
    return ap.parse_args()

def main():
    configure_logging()
    args = parse_args()
    logging.info("=== NEUROSCOPE PREPROCESSING PIPELINE VERIFICATION ===")
    metadata_path = PATHS['metadata_splits']
    output_path = PATHS['preprocessed_dir'] / "neuroscope_pipeline_verification_results.json"
    if args.output:
        output_path = Path(args.output)
    try:
        if not metadata_path.exists():
            logging.error("Metadata file not found: %s", metadata_path)
            return
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for section in ['brats', 'upenn']:
            if section not in metadata or 'valid_subjects' not in metadata[section]:
                logging.error("Invalid metadata structure for section %s", section)
                return
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]
        results = run_pipeline_verification(metadata, splits_to_verify=splits)
        summary = generate_verification_summary(results)
        # use schema writer after generating results
        save_verification_results(results, summary, output_path)
        summary_path = output_path.parent / (output_path.stem + '_summary.json')
        light = {
            'verification_date': results['verification_info']['verification_date'],
            'splits': results['verification_info']['splits_verified'],
            'total_subjects': results['summary_statistics']['total_subjects_verified'],
            'training_ready_subjects': results['summary_statistics']['training_ready_subjects'],
            'pipeline_readiness': summary['pipeline_readiness']
        }
        write_json_with_schema(light, summary_path)
        logging.info("Summary file written: %s", summary_path)
        print_verification_summary(summary, results)
        logging.info("Pipeline verification completed successfully")
    except Exception as e:
        logging.error("Pipeline verification failed: %s", e)
        raise


if __name__ == '__main__':
    main()