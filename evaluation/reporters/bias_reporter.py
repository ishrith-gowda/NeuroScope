"""Bias assessment reporting and visualization.

This module provides tools for generating reports and visualizations
from bias assessment results.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


def save_bias_assessment_results(
    results: Dict[str, Any],
    output_path: Path,
    include_summary: bool = True
) -> None:
    """Save bias assessment results to JSON file.
    
    Args:
        results: Bias assessment results
        output_path: Output file path
        include_summary: Whether to include summary statistics
    """
    try:
        # Prepare data for saving
        save_data = {
            'assessment_timestamp': results.get('assessment_timestamp', time.time()),
            'splits_assessed': results.get('splits_assessed', []),
            'subjects': results.get('subjects', {}),
            'dataset_statistics': results.get('dataset_statistics', {})
        }
        
        if include_summary:
            save_data['summary'] = generate_bias_summary_statistics(results)
        
        # Write to file with proper formatting
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        logger.info(f"saved bias assessment results to: {output_path}")
        
    except Exception as e:
        logger.error(f"error saving bias assessment results: {e}")
        raise


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
        modality_stats = {}
        
        for subject_id, subject_data in subjects.items():
            if subject_data.get('status') == 'success':
                summary['successful_assessments'] += 1
                bias_scores.append(subject_data.get('overall_bias_score', 0.0))
                
                # Collect modality statistics
                for modality, modality_data in subject_data.get('modalities', {}).items():
                    if 'statistics' in modality_data:
                        stats_dict = modality_data['statistics']
                        if modality not in modality_stats:
                            modality_stats[modality] = []
                        modality_stats[modality].append(stats_dict)
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


def print_bias_assessment_summary(summary: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Print a formatted summary of bias assessment results.
    
    Args:
        summary: Summary statistics
        results: Full bias assessment results
    """
    print("\n" + "="*80)
    print("BIAS ASSESSMENT SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"Total subjects assessed: {summary.get('total_subjects', 0)}")
    print(f"Successful assessments: {summary.get('successful_assessments', 0)}")
    print(f"Failed assessments: {summary.get('failed_assessments', 0)}")
    
    # Bias score distribution
    bias_dist = summary.get('bias_score_distribution', {})
    if bias_dist:
        print(f"\nBias Score Distribution:")
        print(f"  Mean: {bias_dist.get('mean', 0):.3f}")
        print(f"  Std:  {bias_dist.get('std', 0):.3f}")
        print(f"  Min:  {bias_dist.get('min', 0):.3f}")
        print(f"  Max:  {bias_dist.get('max', 0):.3f}")
        print(f"  Median: {bias_dist.get('median', 0):.3f}")
    
    # Modality statistics
    modality_stats = summary.get('modality_statistics', {})
    if modality_stats:
        print(f"\nModality Statistics:")
        for modality, stats in modality_stats.items():
            print(f"  {modality}:")
            print(f"    Mean Intensity: {stats.get('mean_intensity', {}).get('mean', 0):.2f} ± {stats.get('mean_intensity', {}).get('std', 0):.2f}")
            print(f"    Skewness: {stats.get('skewness', {}).get('mean', 0):.3f} ± {stats.get('skewness', {}).get('std', 0):.3f}")
            print(f"    Kurtosis: {stats.get('kurtosis', {}).get('mean', 0):.3f} ± {stats.get('kurtosis', {}).get('std', 0):.3f}")
    
    # Common bias patterns
    patterns = summary.get('common_bias_patterns', [])
    if patterns:
        print(f"\nCommon Bias Patterns:")
        for pattern in patterns:
            print(f"  - {pattern}")
    
    print("="*80)


def create_bias_visualization(
    results: Dict[str, Any],
    output_dir: Path,
    modality: str = None
) -> None:
    """Create visualizations for bias assessment results.
    
    Args:
        results: Bias assessment results
        output_dir: Output directory for visualizations
        modality: Specific modality to visualize (None for all)
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bias score histogram
        bias_scores = []
        for subject_data in results.get('subjects', {}).values():
            if subject_data.get('status') == 'success':
                bias_scores.append(subject_data.get('overall_bias_score', 0.0))
        
        if bias_scores:
            plt.figure(figsize=(10, 6))
            plt.hist(bias_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Bias Score')
            plt.ylabel('Number of Subjects')
            plt.title('Distribution of Bias Scores Across Subjects')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'bias_score_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create modality-specific visualizations
        modalities_to_plot = [modality] if modality else ['T1', 'T1ce', 'T2', 'FLAIR']
        
        for mod in modalities_to_plot:
            mod_data = []
            for subject_data in results.get('subjects', {}).values():
                if (subject_data.get('status') == 'success' and 
                    mod in subject_data.get('modalities', {}) and
                    'statistics' in subject_data['modalities'][mod]):
                    mod_data.append(subject_data['modalities'][mod]['statistics'])
            
            if mod_data:
                # Create intensity distribution plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{mod} Modality Statistics', fontsize=16)
                
                # Mean intensity
                mean_intensities = [d['mean_intensity'] for d in mod_data]
                axes[0, 0].hist(mean_intensities, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('Mean Intensity')
                axes[0, 0].set_ylabel('Number of Subjects')
                axes[0, 0].set_title('Mean Intensity Distribution')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Skewness
                skewness_values = [d['skewness'] for d in mod_data]
                axes[0, 1].hist(skewness_values, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Skewness')
                axes[0, 1].set_ylabel('Number of Subjects')
                axes[0, 1].set_title('Skewness Distribution')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Kurtosis
                kurtosis_values = [d['kurtosis'] for d in mod_data]
                axes[1, 0].hist(kurtosis_values, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('Kurtosis')
                axes[1, 0].set_ylabel('Number of Subjects')
                axes[1, 0].set_title('Kurtosis Distribution')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Foreground ratio
                fg_ratios = [d['foreground_ratio'] for d in mod_data]
                axes[1, 1].hist(fg_ratios, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Foreground Ratio')
                axes[1, 1].set_ylabel('Number of Subjects')
                axes[1, 1].set_title('Foreground Ratio Distribution')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{mod.lower()}_statistics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"created bias visualizations in: {output_dir}")
        
    except Exception as e:
        logger.error(f"error creating bias visualizations: {e}")
        raise