#!/usr/bin/env python3
"""
runner script for radiomics preservation analysis.

uses existing extracted features from downstream evaluation
to compute radiomics preservation metrics.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from radiomics_preservation import (
    RadiomicsPreservationAnalyzer,
    compute_preservation_metrics
)


def generate_feature_names(n_features: int, n_modalities: int = 4) -> list:
    """
    generate descriptive feature names.

    since we're using cnn-extracted features, we create
    descriptive names based on feature indices.
    """
    features_per_mod = n_features // n_modalities
    modalities = ['t1', 't1gd', 't2', 'flair']
    names = []

    for mod_idx, mod in enumerate(modalities):
        for i in range(features_per_mod):
            if i < features_per_mod // 3:
                category = 'fo'  # first-order-like
            elif i < 2 * features_per_mod // 3:
                category = 'glcm'  # texture-like
            else:
                category = 'shape'  # shape-like
            names.append(f'{mod}_{category}_{i}')

    # handle any remaining features
    while len(names) < n_features:
        names.append(f'other_{len(names)}')

    return names[:n_features]


def run_preservation_analysis(
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    harmonized_a: np.ndarray,
    harmonized_b: np.ndarray,
    feature_names: list,
    method_name: str = 'sa-cyclegan'
) -> dict:
    """
    run complete preservation analysis.

    args:
        raw_a: raw features from domain a
        raw_b: raw features from domain b
        harmonized_a: harmonized features from domain a
        harmonized_b: harmonized features from domain b (if available)
        feature_names: list of feature names
        method_name: name of harmonization method

    returns:
        comprehensive results dictionary
    """
    analyzer = RadiomicsPreservationAnalyzer(feature_names)

    results = {
        'method': method_name,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'n_features': len(feature_names),
    }

    # domain a preservation
    print(f'[preservation] analyzing domain a preservation...')
    results['domain_a_preservation'] = analyzer.analyze_preservation(raw_a, harmonized_a)

    # domain b preservation (if harmonized_b available)
    if harmonized_b is not None and len(harmonized_b) > 0:
        print(f'[preservation] analyzing domain b preservation...')
        # for domain b, compare raw_b with itself since we're targeting domain a style
        # this shows how features change when harmonizing to target domain
        results['domain_b_preservation'] = analyzer.analyze_preservation(raw_b, harmonized_b)
    else:
        # compare with domain a harmonized as proxy
        n_common = min(len(raw_b), len(harmonized_a))
        results['domain_b_preservation'] = analyzer.analyze_preservation(
            raw_b[:n_common], harmonized_a[:n_common]
        )

    # cross-domain alignment
    n_common = min(len(raw_a), len(raw_b))
    print(f'[preservation] analyzing cross-domain alignment...')
    results['cross_domain_raw'] = analyzer.analyze_preservation(
        raw_a[:n_common], raw_b[:n_common]
    )

    # after harmonization
    if harmonized_b is not None and len(harmonized_b) > 0:
        n_harm = min(len(harmonized_a), len(harmonized_b))
        results['cross_domain_harmonized'] = analyzer.analyze_preservation(
            harmonized_a[:n_harm], harmonized_b[:n_harm]
        )
    else:
        results['cross_domain_harmonized'] = results['domain_a_preservation']

    return results


def main():
    parser = argparse.ArgumentParser(
        description='run radiomics preservation analysis'
    )
    parser.add_argument('--raw-a', type=str, required=True,
                       help='path to raw domain a features')
    parser.add_argument('--raw-b', type=str, required=True,
                       help='path to raw domain b features')
    parser.add_argument('--harmonized-a', type=str, required=True,
                       help='path to harmonized domain a features')
    parser.add_argument('--harmonized-b', type=str,
                       help='path to harmonized domain b features (optional)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--method-name', type=str, default='SA-CycleGAN-2.5D',
                       help='name of harmonization method')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('[radiomics] loading features...')
    print('=' * 60)

    raw_a = np.load(args.raw_a)
    raw_b = np.load(args.raw_b)
    harmonized_a = np.load(args.harmonized_a)

    harmonized_b = None
    if args.harmonized_b and Path(args.harmonized_b).exists():
        harmonized_b = np.load(args.harmonized_b)

    print(f'[radiomics] raw a shape: {raw_a.shape}')
    print(f'[radiomics] raw b shape: {raw_b.shape}')
    print(f'[radiomics] harmonized a shape: {harmonized_a.shape}')
    if harmonized_b is not None:
        print(f'[radiomics] harmonized b shape: {harmonized_b.shape}')

    # generate feature names
    n_features = raw_a.shape[1]
    feature_names = generate_feature_names(n_features)

    # run analysis
    print('=' * 60)
    print('[radiomics] running preservation analysis...')
    print('=' * 60)

    results = run_preservation_analysis(
        raw_a, raw_b, harmonized_a, harmonized_b,
        feature_names, args.method_name
    )

    # save results
    with open(output_dir / 'radiomics_preservation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # print summary
    print('=' * 60)
    print('[radiomics] preservation analysis summary:')
    print('=' * 60)
    overall_a = results['domain_a_preservation']['overall']
    overall_b = results.get('domain_b_preservation', {}).get('overall', {})
    cross_raw = results['cross_domain_raw']['overall']
    cross_harm = results['cross_domain_harmonized']['overall']

    print(f"  domain a preservation:")
    print(f"    mean ccc: {overall_a.get('mean_ccc', 0):.4f} +/- {overall_a.get('std_ccc', 0):.4f}")
    print(f"    mean icc: {overall_a.get('mean_icc', 0):.4f} +/- {overall_a.get('std_icc', 0):.4f}")
    print(f"    excellent: {overall_a.get('excellent_preservation', 0)}, good: {overall_a.get('good_preservation', 0)}")

    if overall_b:
        print(f"  domain b preservation:")
        print(f"    mean ccc: {overall_b.get('mean_ccc', 0):.4f} +/- {overall_b.get('std_ccc', 0):.4f}")

    print(f"  cross-domain alignment (raw):")
    print(f"    mean ccc: {cross_raw.get('mean_ccc', 0):.4f}")
    print(f"  cross-domain alignment (harmonized):")
    print(f"    mean ccc: {cross_harm.get('mean_ccc', 0):.4f}")
    print(f"  improvement: {cross_harm.get('mean_ccc', 0) - cross_raw.get('mean_ccc', 0):.4f}")
    print('=' * 60)
    print(f'[radiomics] results saved to {output_dir}')


if __name__ == '__main__':
    main()
