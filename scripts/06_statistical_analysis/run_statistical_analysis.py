#!/usr/bin/env python3
"""
runner script for comprehensive statistical analysis.

orchestrates all statistical validation tasks:
1. load downstream evaluation results
2. run combat baseline comparison
3. compute comprehensive statistics
4. generate method comparison figures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from combat_comparison import (
    ComBatConfig, ComBatHarmonizer, evaluate_combat_harmonization
)
from comprehensive_statistics import (
    BootstrapCI, StatisticalResult
)


def extract_features_from_slices(data_dir: Path, domain: str) -> np.ndarray:
    """
    extract simple features from preprocessed mri slices.

    uses intensity statistics as proxy features for combat comparison.
    """
    import glob

    # find all npy files
    pattern = str(data_dir / '*.npy')
    files = sorted(glob.glob(pattern))

    if not files:
        raise ValueError(f'no npy files found in {data_dir}')

    print(f'[features] extracting features from {len(files)} slices in {domain}...')

    features_list = []
    for f in files[:500]:  # limit for speed
        try:
            data = np.load(f)
            # compute intensity features per modality
            if data.ndim == 3:
                # shape: (modalities, h, w) or (h, w, modalities)
                if data.shape[0] <= 4:
                    modalities = data.shape[0]
                    feat = []
                    for m in range(modalities):
                        mod_data = data[m].flatten()
                        mod_data = mod_data[mod_data > 0]  # non-zero only
                        if len(mod_data) > 100:
                            feat.extend([
                                np.mean(mod_data),
                                np.std(mod_data),
                                np.percentile(mod_data, 25),
                                np.percentile(mod_data, 50),
                                np.percentile(mod_data, 75),
                                np.percentile(mod_data, 95),
                            ])
                        else:
                            feat.extend([0] * 6)
                    features_list.append(feat)
                else:
                    # (h, w, modalities)
                    modalities = data.shape[-1]
                    feat = []
                    for m in range(min(modalities, 4)):
                        mod_data = data[:, :, m].flatten()
                        mod_data = mod_data[mod_data > 0]
                        if len(mod_data) > 100:
                            feat.extend([
                                np.mean(mod_data),
                                np.std(mod_data),
                                np.percentile(mod_data, 25),
                                np.percentile(mod_data, 50),
                                np.percentile(mod_data, 75),
                                np.percentile(mod_data, 95),
                            ])
                        else:
                            feat.extend([0] * 6)
                    features_list.append(feat)
            elif data.ndim == 2:
                mod_data = data.flatten()
                mod_data = mod_data[mod_data > 0]
                if len(mod_data) > 100:
                    feat = [
                        np.mean(mod_data),
                        np.std(mod_data),
                        np.percentile(mod_data, 25),
                        np.percentile(mod_data, 50),
                        np.percentile(mod_data, 75),
                        np.percentile(mod_data, 95),
                    ]
                else:
                    feat = [0] * 6
                features_list.append(feat)
        except Exception as e:
            print(f'[features] error loading {f}: {e}')
            continue

    features = np.array(features_list)
    print(f'[features] extracted shape: {features.shape}')
    return features


def run_combat_comparison(
    domain_a_dir: Path,
    domain_b_dir: Path,
    output_dir: Path
) -> dict:
    """
    run combat harmonization baseline comparison.
    """
    print('=' * 60)
    print('[combat] running combat baseline comparison...')
    print('=' * 60)

    # extract features
    features_a = extract_features_from_slices(domain_a_dir, 'domain_a')
    features_b = extract_features_from_slices(domain_b_dir, 'domain_b')

    # ensure same number of features
    n_features = min(features_a.shape[1], features_b.shape[1])
    features_a = features_a[:, :n_features]
    features_b = features_b[:, :n_features]

    # save raw features
    np.save(output_dir / 'raw_features_a.npy', features_a)
    np.save(output_dir / 'raw_features_b.npy', features_b)

    # run combat evaluation
    results = evaluate_combat_harmonization(features_a, features_b, output_dir)

    return results


def run_comprehensive_statistics(
    evaluation_results: dict,
    combat_results: dict,
    output_dir: Path
) -> dict:
    """
    run comprehensive statistical analysis.
    """
    print('=' * 60)
    print('[stats] running comprehensive statistical analysis...')
    print('=' * 60)

    # prepare metrics for analysis
    raw_metrics = evaluation_results.get('results', {}).get('domain_classification', {}).get('raw', {})
    harmonized_metrics = evaluation_results.get('results', {}).get('domain_classification', {}).get('harmonized', {})

    # domain classification metrics
    domain_class_results = {
        'raw_accuracy': raw_metrics.get('accuracy', 0),
        'harmonized_accuracy': harmonized_metrics.get('accuracy', 0),
        'raw_auc': raw_metrics.get('auc', 0),
        'harmonized_auc': harmonized_metrics.get('auc', 0),
    }

    # feature statistics
    raw_feature_stats = raw_metrics.get('feature_statistics', {})
    harm_feature_stats = harmonized_metrics.get('feature_statistics', {})

    feature_results = {
        'raw_mmd': raw_feature_stats.get('mmd', 0),
        'harmonized_mmd': harm_feature_stats.get('mmd', 0),
        'raw_cosine': raw_feature_stats.get('cosine_similarity', 0),
        'harmonized_cosine': harm_feature_stats.get('cosine_similarity', 0),
        'mmd_reduction': raw_feature_stats.get('mmd', 0) - harm_feature_stats.get('mmd', 0),
        'mmd_reduction_percent': 100 * (raw_feature_stats.get('mmd', 0) - harm_feature_stats.get('mmd', 0)) / (raw_feature_stats.get('mmd', 0) + 1e-10),
    }

    # combat comparison
    combat_comparison = {
        'sa_cyclegan_mmd': harm_feature_stats.get('mmd', 0),
        'combat_mmd': combat_results.get('combat', {}).get('mmd', 0),
        'raw_mmd': raw_feature_stats.get('mmd', 0),
        'sa_cyclegan_cosine': harm_feature_stats.get('cosine_similarity', 0),
        'combat_cosine': combat_results.get('combat', {}).get('cosine_similarity', 0),
    }

    comprehensive_results = {
        'domain_classification': domain_class_results,
        'feature_statistics': feature_results,
        'combat_comparison': combat_comparison,
        'summary': {
            'sa_cyclegan_mmd_reduction_percent': feature_results['mmd_reduction_percent'],
            'combat_mmd_reduction_percent': combat_results.get('improvement', {}).get('mmd_reduction_percent', 0),
            'sa_cyclegan_outperforms_combat': harm_feature_stats.get('mmd', 0) < combat_results.get('combat', {}).get('mmd', 0),
        }
    }

    return comprehensive_results


def run_combat_on_features(
    features_a: np.ndarray,
    features_b: np.ndarray,
    output_dir: Path
) -> dict:
    """
    run combat harmonization on pre-extracted features.
    """
    print('=' * 60)
    print('[combat] running combat baseline comparison on extracted features...')
    print('=' * 60)

    print(f'[combat] features a shape: {features_a.shape}')
    print(f'[combat] features b shape: {features_b.shape}')

    # ensure same number of features
    n_features = min(features_a.shape[1], features_b.shape[1])
    features_a = features_a[:, :n_features]
    features_b = features_b[:, :n_features]

    # save raw features
    np.save(output_dir / 'raw_features_a.npy', features_a)
    np.save(output_dir / 'raw_features_b.npy', features_b)

    # run combat evaluation
    results = evaluate_combat_harmonization(features_a, features_b, output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='run comprehensive statistical analysis for harmonization validation'
    )
    parser.add_argument('--features-a', type=str, required=True,
                       help='path to domain a extracted features (npy file)')
    parser.add_argument('--features-b', type=str, required=True,
                       help='path to domain b extracted features (npy file)')
    parser.add_argument('--evaluation-results', type=str, required=True,
                       help='path to downstream evaluation results json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for statistical analysis')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 60)
    print(f'[main] statistical analysis started at {timestamp}')
    print('=' * 60)

    # load existing evaluation results
    print('[main] loading evaluation results...')
    with open(args.evaluation_results, 'r') as f:
        evaluation_results = json.load(f)

    # load pre-extracted features
    print('[main] loading extracted features...')
    features_a = np.load(args.features_a)
    features_b = np.load(args.features_b)

    # run combat comparison
    combat_results = run_combat_on_features(
        features_a,
        features_b,
        output_dir
    )

    # save combat results
    with open(output_dir / 'combat_results.json', 'w') as f:
        json.dump(combat_results, f, indent=2)

    # run comprehensive statistics
    comprehensive_results = run_comprehensive_statistics(
        evaluation_results, combat_results, output_dir
    )

    # save comprehensive results
    final_results = {
        'timestamp': timestamp,
        'evaluation_summary': evaluation_results.get('results', {}),
        'combat_comparison': combat_results,
        'comprehensive_analysis': comprehensive_results,
    }

    with open(output_dir / 'statistical_analysis_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # print summary
    print('=' * 60)
    print('[main] statistical analysis summary:')
    print('=' * 60)
    print(f"  sa-cyclegan mmd: {comprehensive_results['combat_comparison']['sa_cyclegan_mmd']:.6f}")
    print(f"  combat mmd: {comprehensive_results['combat_comparison']['combat_mmd']:.6f}")
    print(f"  raw mmd: {comprehensive_results['combat_comparison']['raw_mmd']:.6f}")
    print(f"  sa-cyclegan mmd reduction: {comprehensive_results['summary']['sa_cyclegan_mmd_reduction_percent']:.1f}%")
    print(f"  combat mmd reduction: {comprehensive_results['summary']['combat_mmd_reduction_percent']:.1f}%")
    print(f"  sa-cyclegan outperforms combat: {comprehensive_results['summary']['sa_cyclegan_outperforms_combat']}")
    print('=' * 60)
    print(f'[main] results saved to {output_dir}')


if __name__ == '__main__':
    main()
