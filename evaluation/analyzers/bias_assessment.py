"""Comprehensive bias assessment for medical imaging data.

This module provides tools for assessing intensity bias in medical imaging datasets,
including slice-wise statistics, subject-level bias analysis, and dataset-wide
bias characterization.
"""

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

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


def verify_preprocessed_file(file_path: str) -> bool:
    """Verify that a preprocessed file exists and is readable.
    
    Args:
        file_path: Path to the preprocessed file
        
    Returns:
        bool: True if file exists and is readable, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"file not found: {file_path}")
            return False
        
        # Try to read the file
        image = sitk.ReadImage(file_path)
        if image is None:
            logger.warning(f"could not read image: {file_path}")
            return False
        
        # Check if image has valid dimensions
        size = image.GetSize()
        if any(dim <= 0 for dim in size):
            logger.warning(f"invalid image dimensions: {size} for {file_path}")
            return False
        
        logger.debug(f"verified file: {file_path}")
        return True
        
    except Exception as e:
        logger.warning(f"error verifying file {file_path}: {e}")
        return False


def compute_slice_wise_statistics(image: sitk.Image, mask: sitk.Image) -> Dict[str, float]:
    """Compute slice-wise statistics for bias assessment.
    
    Args:
        image: Input image as SimpleITK Image
        mask: Brain mask as SimpleITK Image
        
    Returns:
        Dict containing slice-wise statistics
    """
    try:
        # Convert to numpy arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Ensure mask is binary
        mask_array = (mask_array > 0).astype(np.float32)
        
        # Initialize statistics
        stats_dict = {
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'min_intensity': 0.0,
            'max_intensity': 0.0,
            'median_intensity': 0.0,
            'q25_intensity': 0.0,
            'q75_intensity': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'foreground_voxels': 0,
            'total_voxels': 0,
            'foreground_ratio': 0.0
        }
        
        # Get foreground voxels
        foreground_voxels = image_array[mask_array > 0]
        
        if len(foreground_voxels) == 0:
            logger.warning("no foreground voxels found in mask")
            return stats_dict
        
        # Compute basic statistics
        stats_dict['mean_intensity'] = float(np.mean(foreground_voxels))
        stats_dict['std_intensity'] = float(np.std(foreground_voxels))
        stats_dict['min_intensity'] = float(np.min(foreground_voxels))
        stats_dict['max_intensity'] = float(np.max(foreground_voxels))
        stats_dict['median_intensity'] = float(np.median(foreground_voxels))
        stats_dict['q25_intensity'] = float(np.percentile(foreground_voxels, 25))
        stats_dict['q75_intensity'] = float(np.percentile(foreground_voxels, 75))
        
        # Compute higher-order moments
        if len(foreground_voxels) > 1:
            stats_dict['skewness'] = float(stats.skew(foreground_voxels))
            stats_dict['kurtosis'] = float(stats.kurtosis(foreground_voxels))
        
        # Compute voxel counts
        stats_dict['foreground_voxels'] = int(np.sum(mask_array))
        stats_dict['total_voxels'] = int(mask_array.size)
        stats_dict['foreground_ratio'] = float(stats_dict['foreground_voxels'] / stats_dict['total_voxels'])
        
        return stats_dict
        
    except Exception as e:
        logger.error(f"error computing slice-wise statistics: {e}")
        return {
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'min_intensity': 0.0,
            'max_intensity': 0.0,
            'median_intensity': 0.0,
            'q25_intensity': 0.0,
            'q75_intensity': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'foreground_voxels': 0,
            'total_voxels': 0,
            'foreground_ratio': 0.0
        }


def assess_subject_bias(
    subject_id: str,
    modality_files: Dict[str, str],
    mask_file: str,
    splits_to_assess: List[str] = None
) -> Dict[str, Any]:
    """Assess bias for a single subject across modalities.
    
    Args:
        subject_id: Subject identifier
        modality_files: Dictionary mapping modality names to file paths
        mask_file: Path to brain mask file
        splits_to_assess: List of splits to assess (train, val, test)
        
    Returns:
        Dict containing subject-level bias assessment
    """
    if splits_to_assess is None:
        splits_to_assess = ['train', 'val', 'test']
    
    subject_results = {
        'subject_id': subject_id,
        'modalities': {},
        'overall_bias_score': 0.0,
        'assessment_timestamp': time.time(),
        'status': 'success'
    }
    
    try:
        # Verify mask file
        if not verify_preprocessed_file(mask_file):
            subject_results['status'] = 'failed'
            subject_results['error'] = 'mask file not found or invalid'
            return subject_results
        
        # Load mask
        mask_image = sitk.ReadImage(mask_file)
        
        # Assess each modality
        modality_scores = []
        for modality, file_path in modality_files.items():
            if not verify_preprocessed_file(file_path):
                logger.warning(f"skipping modality {modality} for subject {subject_id}: file not found")
                continue
            
            try:
                # Load image
                image = sitk.ReadImage(file_path)
                
                # Compute statistics
                stats_dict = compute_slice_wise_statistics(image, mask_image)
                
                # Store modality results
                subject_results['modalities'][modality] = {
                    'file_path': file_path,
                    'statistics': stats_dict,
                    'bias_indicators': {
                        'high_skewness': abs(stats_dict['skewness']) > 2.0,
                        'high_kurtosis': abs(stats_dict['kurtosis']) > 3.0,
                        'low_foreground_ratio': stats_dict['foreground_ratio'] < 0.1,
                        'extreme_intensity_range': (stats_dict['max_intensity'] - stats_dict['min_intensity']) > 1000
                    }
                }
                
                # Calculate modality bias score (0-1, higher = more biased)
                bias_score = 0.0
                if abs(stats_dict['skewness']) > 2.0:
                    bias_score += 0.3
                if abs(stats_dict['kurtosis']) > 3.0:
                    bias_score += 0.3
                if stats_dict['foreground_ratio'] < 0.1:
                    bias_score += 0.2
                if (stats_dict['max_intensity'] - stats_dict['min_intensity']) > 1000:
                    bias_score += 0.2
                
                modality_scores.append(bias_score)
                
            except Exception as e:
                logger.error(f"error assessing modality {modality} for subject {subject_id}: {e}")
                subject_results['modalities'][modality] = {
                    'file_path': file_path,
                    'error': str(e),
                    'bias_score': 1.0  # Maximum bias score for failed assessments
                }
                modality_scores.append(1.0)
        
        # Calculate overall bias score
        if modality_scores:
            subject_results['overall_bias_score'] = float(np.mean(modality_scores))
        else:
            subject_results['status'] = 'failed'
            subject_results['error'] = 'no valid modalities found'
        
    except Exception as e:
        logger.error(f"error assessing subject {subject_id}: {e}")
        subject_results['status'] = 'failed'
        subject_results['error'] = str(e)
    
    return subject_results


def analyze_dataset_bias(metadata: Dict[str, Any], splits_to_assess: List[str] = None) -> Dict[str, Any]:
    """Analyze bias across the entire dataset.
    
    Args:
        metadata: Dataset metadata with subject information
        splits_to_assess: List of splits to assess
        
    Returns:
        Dict containing dataset-wide bias analysis
    """
    if splits_to_assess is None:
        splits_to_assess = ['train', 'val', 'test']
    
    dataset_results = {
        'assessment_timestamp': time.time(),
        'splits_assessed': splits_to_assess,
        'subjects': {},
        'dataset_statistics': {},
        'bias_summary': {}
    }
    
    try:
        # Process each subject
        for subject_id, subject_data in metadata.get('subjects', {}).items():
            # Check if subject is in any of the splits to assess
            subject_splits = subject_data.get('splits', [])
            if not any(split in splits_to_assess for split in subject_splits):
                continue
            
            logger.info(f"assessing bias for subject: {subject_id}")
            
            # Get modality files
            modality_files = {}
            for modality in ['T1', 'T1ce', 'T2', 'FLAIR']:
                file_path = subject_data.get(f'{modality.lower()}_file')
                if file_path:
                    modality_files[modality] = file_path
            
            # Get mask file
            mask_file = subject_data.get('mask_file')
            if not mask_file:
                logger.warning(f"no mask file found for subject {subject_id}")
                continue
            
            # Assess subject bias
            subject_bias = assess_subject_bias(
                subject_id, modality_files, mask_file, splits_to_assess
            )
            
            dataset_results['subjects'][subject_id] = subject_bias
        
        # Compute dataset-wide statistics
        dataset_results['dataset_statistics'] = generate_bias_summary_statistics(dataset_results)
        
    except Exception as e:
        logger.error(f"error analyzing dataset bias: {e}")
        dataset_results['error'] = str(e)
    
    return dataset_results


def generate_bias_summary_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics from bias assessment results.
    
    Args:
        results: Bias assessment results
        
    Returns:
        Dict containing summary statistics
    """
    summary = {
        'total_subjects': 0,
        'successful_assessments': 0,
        'failed_assessments': 0,
        'modality_statistics': {},
        'bias_score_distribution': {},
        'common_bias_patterns': []
    }
    
    try:
        subjects = results.get('subjects', {})
        summary['total_subjects'] = len(subjects)
        
        # Collect bias scores and modality statistics
        bias_scores = []
        modality_stats = defaultdict(list)
        
        for subject_id, subject_data in subjects.items():
            if subject_data.get('status') == 'success':
                summary['successful_assessments'] += 1
                bias_scores.append(subject_data.get('overall_bias_score', 0.0))
                
                # Collect modality statistics
                for modality, modality_data in subject_data.get('modalities', {}).items():
                    if 'statistics' in modality_data:
                        stats_dict = modality_data['statistics']
                        modality_stats[modality].append({
                            'mean_intensity': stats_dict['mean_intensity'],
                            'std_intensity': stats_dict['std_intensity'],
                            'skewness': stats_dict['skewness'],
                            'kurtosis': stats_dict['kurtosis'],
                            'foreground_ratio': stats_dict['foreground_ratio']
                        })
            else:
                summary['failed_assessments'] += 1
        
        # Compute bias score distribution
        if bias_scores:
            summary['bias_score_distribution'] = {
                'mean': float(np.mean(bias_scores)),
                'std': float(np.std(bias_scores)),
                'min': float(np.min(bias_scores)),
                'max': float(np.max(bias_scores)),
                'median': float(np.median(bias_scores)),
                'q25': float(np.percentile(bias_scores, 25)),
                'q75': float(np.percentile(bias_scores, 75))
            }
        
        # Compute modality statistics
        for modality, stats_list in modality_stats.items():
            if stats_list:
                summary['modality_statistics'][modality] = {
                    'mean_intensity': {
                        'mean': float(np.mean([s['mean_intensity'] for s in stats_list])),
                        'std': float(np.std([s['mean_intensity'] for s in stats_list]))
                    },
                    'std_intensity': {
                        'mean': float(np.mean([s['std_intensity'] for s in stats_list])),
                        'std': float(np.std([s['std_intensity'] for s in stats_list]))
                    },
                    'skewness': {
                        'mean': float(np.mean([s['skewness'] for s in stats_list])),
                        'std': float(np.std([s['skewness'] for s in stats_list]))
                    },
                    'kurtosis': {
                        'mean': float(np.mean([s['kurtosis'] for s in stats_list])),
                        'std': float(np.std([s['kurtosis'] for s in stats_list]))
                    },
                    'foreground_ratio': {
                        'mean': float(np.mean([s['foreground_ratio'] for s in stats_list])),
                        'std': float(np.std([s['foreground_ratio'] for s in stats_list]))
                    }
                }
        
        # Identify common bias patterns
        high_bias_subjects = [sid for sid, data in subjects.items() 
                             if data.get('overall_bias_score', 0) > 0.5]
        if high_bias_subjects:
            summary['common_bias_patterns'].append(f"High bias detected in {len(high_bias_subjects)} subjects")
        
    except Exception as e:
        logger.error(f"error generating summary statistics: {e}")
        summary['error'] = str(e)
    
    return summary