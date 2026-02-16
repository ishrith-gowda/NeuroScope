#!/usr/bin/env python3
"""
regenerate fig_harmonization_summary.pdf with improved publication-quality styling.

uses the enhanced plot_harmonization_effect_summary function that matches
the quality of fig_domain_classification.pdf and fig_feature_distribution.pdf
"""

import json
import sys
from pathlib import Path

# add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / '05_downstream_evaluation'))

from generate_downstream_figures import plot_harmonization_effect_summary


def main():
    """regenerate the harmonization summary figure."""
    
    # get root directory
    root_dir = Path(__file__).parent.parent.parent
    
    # output directory
    output_dir = root_dir / 'figures' / 'downstream'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load domain classification results
    domain_file = root_dir / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'domain_classification_results.json'
    
    if not domain_file.exists():
        print(f'[error] domain classification results not found at {domain_file}')
        return 1
    
    with open(domain_file) as f:
        domain_results = json.load(f)
    
    # load feature distribution results
    feature_file = root_dir / 'experiments' / 'downstream_evaluation' / 'feature_distribution' / 'feature_distribution_results.json'
    
    if not feature_file.exists():
        print(f'[error] feature distribution results not found at {feature_file}')
        return 1
    
    with open(feature_file) as f:
        feature_results = json.load(f)
    
    print(f'[regenerate] domain results: {domain_results.keys()}')
    print(f'[regenerate] feature results: {feature_results.keys()}')
    
    # regenerate figure with improved styling
    output_path = output_dir / 'fig_harmonization_summary.pdf'
    
    print(f'[regenerate] generating improved harmonization summary figure...')
    plot_harmonization_effect_summary(domain_results, feature_results, output_path)
    
    print(f'[regenerate] ✓ saved improved figure to {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
