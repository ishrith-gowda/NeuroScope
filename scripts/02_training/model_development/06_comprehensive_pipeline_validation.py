#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation Script

This script validates the entire preprocessing -> CycleGAN pipeline to ensure:
1. Preprocessing produces [0,1] normalized tensors
2. Dataset loader correctly maps [0,1] -> [-1,1] 
3. Domain mapping is correct (A=brats, B=upenn)
4. Data consistency between preprocessing and training
5. All required files and directories exist
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from neuroscope_dataset_loader import get_cycle_domain_loaders

# Import PATHS from preprocessing config
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / '01_data_preparation_pipeline'
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))
import neuroscope_preprocessing_config as npc  # type: ignore
PATHS = npc.PATHS


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_preprocessing_outputs() -> Dict[str, any]:
    """
    Validate that preprocessing outputs are correctly normalized to [0,1].
    
    Returns:
        Dict with validation results
    """
    logging.info("=== VALIDATING PREPROCESSING OUTPUTS ===")
    
    results = {
        'validated_subjects': 0,
        'failed_subjects': [],
        'tensor_ranges': [],
        'issues': []
    }
    
    preprocessed_dir = PATHS['preprocessed_dir']
    metadata_path = PATHS['metadata_splits']
    
    if not preprocessed_dir.exists():
        results['issues'].append(f"Preprocessed directory not found: {preprocessed_dir}")
        return results
    
    if not metadata_path.exists():
        results['issues'].append(f"Metadata file not found: {metadata_path}")
        return results
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    # Check a sample of subjects from each domain
    sample_size = 5  # Check first 5 subjects from each domain
    
    for section in ['brats', 'upenn']:
        subjects = list(meta.get(section, {}).get('valid_subjects', {}).keys())[:sample_size]
        
        for subject_id in subjects:
            subject_dir = preprocessed_dir / section / subject_id
            if not subject_dir.exists():
                results['failed_subjects'].append(f"{section}/{subject_id} (directory not found)")
                continue
            
            # Check all modalities
            modalities = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
            subject_valid = True
            
            for modality in modalities:
                modality_path = subject_dir / modality
                if not modality_path.exists():
                    results['failed_subjects'].append(f"{section}/{subject_id} (missing {modality})")
                    subject_valid = False
                    continue
                
                try:
                    # Load and check tensor values
                    img = sitk.ReadImage(str(modality_path))
                    arr = sitk.GetArrayFromImage(img).astype(np.float32)
                    
                    # Check for non-finite values
                    if not np.isfinite(arr).all():
                        results['issues'].append(f"{section}/{subject_id}/{modality} contains non-finite values")
                        subject_valid = False
                        continue
                    
                    # Check value range
                    min_val, max_val = arr.min(), arr.max()
                    results['tensor_ranges'].append({
                        'subject': f"{section}/{subject_id}",
                        'modality': modality,
                        'min': float(min_val),
                        'max': float(max_val),
                        'range_valid': 0.0 <= min_val <= max_val <= 1.0
                    })
                    
                    if not (0.0 <= min_val <= max_val <= 1.0):
                        results['issues'].append(
                            f"{section}/{subject_id}/{modality} values outside [0,1]: [{min_val:.3f}, {max_val:.3f}]"
                        )
                        subject_valid = False
                    
                except Exception as e:
                    results['failed_subjects'].append(f"{section}/{subject_id}/{modality} (error: {str(e)})")
                    subject_valid = False
            
            if subject_valid:
                results['validated_subjects'] += 1
    
    return results


def validate_dataset_loader() -> Dict[str, any]:
    """
    Validate that dataset loader correctly handles tensor normalization.
    
    Returns:
        Dict with validation results
    """
    logging.info("=== VALIDATING DATASET LOADER ===")
    
    results = {
        'loaders_created': [],
        'loaders_failed': [],
        'tensor_ranges': [],
        'domain_mapping': {},
        'issues': []
    }
    
    try:
        # Create loaders
        loaders = get_cycle_domain_loaders(
            preprocessed_dir=str(PATHS['preprocessed_dir']),
            metadata_json=str(PATHS['metadata_splits']),
            batch_size=2,  # Small batch for testing
            num_workers=0,
            slices_per_subject=1,
            seed=42
        )
        
        # Check domain mapping
        expected_mapping = {'brats': 'A', 'upenn': 'B'}
        for section, domain in expected_mapping.items():
            train_key = f'train_{domain}'
            val_key = f'val_{domain}'
            
            if train_key in loaders:
                results['domain_mapping'][section] = domain
                results['loaders_created'].append(train_key)
            else:
                results['loaders_failed'].append(train_key)
                results['issues'].append(f"Missing loader for {section} -> {domain}")
            
            if val_key in loaders:
                results['loaders_created'].append(val_key)
            else:
                results['loaders_failed'].append(val_key)
        
        # Test tensor ranges
        for loader_name, loader in loaders.items():
            try:
                batch = next(iter(loader))
                min_val, max_val = batch.min().item(), batch.max().item()
                
                results['tensor_ranges'].append({
                    'loader': loader_name,
                    'batch_shape': list(batch.shape),
                    'min': min_val,
                    'max': max_val,
                    'range_valid': -1.0 <= min_val <= max_val <= 1.0
                })
                
                if not (-1.0 <= min_val <= max_val <= 1.0):
                    results['issues'].append(
                        f"{loader_name} tensor values outside [-1,1]: [{min_val:.3f}, {max_val:.3f}]"
                    )
                
            except Exception as e:
                results['issues'].append(f"Error testing {loader_name}: {str(e)}")
    
    except Exception as e:
        results['issues'].append(f"Failed to create loaders: {str(e)}")
    
    return results


def validate_data_consistency() -> Dict[str, any]:
    """
    Validate consistency between preprocessing outputs and dataset loader inputs.
    
    Returns:
        Dict with validation results
    """
    logging.info("=== VALIDATING DATA CONSISTENCY ===")
    
    results = {
        'consistency_checks': [],
        'issues': []
    }
    
    try:
        # Load metadata
        with open(PATHS['metadata_splits'], 'r') as f:
            meta = json.load(f)
        
        # Get a sample subject from each domain
        sample_subjects = {}
        for section in ['brats', 'upenn']:
            subjects = list(meta.get(section, {}).get('valid_subjects', {}).keys())
            if subjects:
                sample_subjects[section] = subjects[0]
        
        # Check preprocessing outputs directly
        preprocessed_dir = PATHS['preprocessed_dir']
        
        for section, subject_id in sample_subjects.items():
            subject_dir = preprocessed_dir / section / subject_id
            
            # Load one modality directly from preprocessing
            modality_path = subject_dir / 't1.nii.gz'
            if modality_path.exists():
                img = sitk.ReadImage(str(modality_path))
                arr = sitk.GetArrayFromImage(img).astype(np.float32)
                
                # Check preprocessing range
                preproc_min, preproc_max = arr.min(), arr.max()
                
                # Now check what dataset loader produces
                loaders = get_cycle_domain_loaders(
                    preprocessed_dir=str(preprocessed_dir),
                    metadata_json=str(PATHS['metadata_splits']),
                    batch_size=1,
                    num_workers=0,
                    slices_per_subject=1,
                    seed=42
                )
                
                domain = 'A' if section == 'brats' else 'B'
                loader_key = f'train_{domain}'
                
                if loader_key in loaders:
                    # Get a batch from the loader
                    batch = next(iter(loaders[loader_key]))
                    loader_min, loader_max = batch.min().item(), batch.max().item()
                    
                    results['consistency_checks'].append({
                        'subject': f"{section}/{subject_id}",
                        'preprocessing_range': [float(preproc_min), float(preproc_max)],
                        'loader_range': [loader_min, loader_max],
                        'preprocessing_valid': 0.0 <= preproc_min <= preproc_max <= 1.0,
                        'loader_valid': -1.0 <= loader_min <= loader_max <= 1.0,
                        'transformation_correct': abs(loader_min - (preproc_min * 2.0 - 1.0)) < 0.01
                    })
                    
                    # Validate transformation
                    expected_min = preproc_min * 2.0 - 1.0
                    expected_max = preproc_max * 2.0 - 1.0
                    
                    if abs(loader_min - expected_min) > 0.01 or abs(loader_max - expected_max) > 0.01:
                        results['issues'].append(
                            f"Incorrect transformation for {section}/{subject_id}: "
                            f"expected [{expected_min:.3f}, {expected_max:.3f}], "
                            f"got [{loader_min:.3f}, {loader_max:.3f}]"
                        )
    
    except Exception as e:
        results['issues'].append(f"Data consistency check failed: {str(e)}")
    
    return results


def validate_file_structure() -> Dict[str, any]:
    """
    Validate that all required files and directories exist.
    
    Returns:
        Dict with validation results
    """
    logging.info("=== VALIDATING FILE STRUCTURE ===")
    
    results = {
        'required_files': {},
        'required_dirs': {},
        'issues': []
    }
    
    # Required files
    required_files = {
        'metadata_splits': PATHS['metadata_splits'],
        'preprocessing_config': PATHS['metadata_base'],
        'mni_template': PATHS['mni_template']
    }
    
    for name, path in required_files.items():
        exists = path.exists()
        results['required_files'][name] = {
            'path': str(path),
            'exists': exists
        }
        if not exists:
            results['issues'].append(f"Required file missing: {name} at {path}")
    
    # Required directories
    required_dirs = {
        'preprocessed_dir': PATHS['preprocessed_dir'],
        'checkpoints_dir': PATHS['checkpoints_dir'],
        'samples_dir': PATHS['samples_dir'],
        'figures_dir': PATHS['figures_dir']
    }
    
    for name, path in required_dirs.items():
        exists = path.exists()
        results['required_dirs'][name] = {
            'path': str(path),
            'exists': exists
        }
        if not exists:
            results['issues'].append(f"Required directory missing: {name} at {path}")
    
    return results


def generate_validation_report(results: Dict[str, any]) -> None:
    """Generate a comprehensive validation report."""
    logging.info("=== GENERATING VALIDATION REPORT ===")
    
    report = {
        'timestamp': str(Path().cwd()),
        'validation_results': results,
        'summary': {
            'total_issues': sum(len(r.get('issues', [])) for r in results.values()),
            'preprocessing_valid': len(results.get('preprocessing', {}).get('issues', [])) == 0,
            'dataloader_valid': len(results.get('dataloader', {}).get('issues', [])) == 0,
            'consistency_valid': len(results.get('consistency', {}).get('issues', [])) == 0,
            'file_structure_valid': len(results.get('file_structure', {}).get('issues', [])) == 0
        }
    }
    
    # Save report
    report_path = PATHS['scripts_dir'] / '02_model_development_pipeline' / 'pipeline_validation_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Validation report saved to: {report_path}")
    
    # Print summary
    summary = report['summary']
    logging.info("=== VALIDATION SUMMARY ===")
    logging.info(f"Total issues found: {summary['total_issues']}")
    logging.info(f"Preprocessing valid: {summary['preprocessing_valid']}")
    logging.info(f"DataLoader valid: {summary['dataloader_valid']}")
    logging.info(f"Data consistency valid: {summary['consistency_valid']}")
    logging.info(f"File structure valid: {summary['file_structure_valid']}")
    
    if summary['total_issues'] == 0:
        logging.info("✅ ALL VALIDATIONS PASSED - Pipeline is ready for training!")
    else:
        logging.warning("⚠️  ISSUES FOUND - Please review and fix before training")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive pipeline validation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip preprocessing validation')
    parser.add_argument('--skip-dataloader', action='store_true', help='Skip dataloader validation')
    parser.add_argument('--skip-consistency', action='store_true', help='Skip consistency validation')
    parser.add_argument('--skip-files', action='store_true', help='Skip file structure validation')
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info("Starting comprehensive pipeline validation...")
    
    results = {}
    
    # Run validations
    if not args.skip_preprocessing:
        results['preprocessing'] = validate_preprocessing_outputs()
    
    if not args.skip_dataloader:
        results['dataloader'] = validate_dataset_loader()
    
    if not args.skip_consistency:
        results['consistency'] = validate_data_consistency()
    
    if not args.skip_files:
        results['file_structure'] = validate_file_structure()
    
    # Generate report
    generate_validation_report(results)


if __name__ == '__main__':
    main()