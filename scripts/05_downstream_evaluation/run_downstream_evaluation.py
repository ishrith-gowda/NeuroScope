#!/usr/bin/env python3
"""
main runner script for downstream task evaluation.

orchestrates the complete downstream evaluation pipeline:
1. domain classification evaluation
2. feature distribution analysis
3. figure generation

usage:
    python run_downstream_evaluation.py \
        --domain-a-dir /path/to/brats \
        --domain-b-dir /path/to/upenn \
        --harmonization-model /path/to/model.pth \
        --output-dir ./downstream_results
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, description: str) -> bool:
    """run command and return success status."""
    print(f'[run] {description}')
    print(f'[run] command: {" ".join(cmd)}')
    print('=' * 60)

    try:
        result = subprocess.run(cmd, check=True)
        print(f'[run] {description} completed successfully')
        return True
    except subprocess.CalledProcessError as e:
        print(f'[error] {description} failed with code {e.returncode}')
        return False
    except Exception as e:
        print(f'[error] {description} failed: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='run complete downstream evaluation pipeline'
    )
    parser.add_argument('--domain-a-dir', type=str, required=True,
                       help='path to domain a (brats) data')
    parser.add_argument('--domain-b-dir', type=str, required=True,
                       help='path to domain b (upenn) data')
    parser.add_argument('--harmonization-model', type=str, default=None,
                       help='path to trained harmonization model')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--n-epochs', type=int, default=50,
                       help='domain classifier training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='device')
    parser.add_argument('--skip-classification', action='store_true',
                       help='skip domain classification')
    parser.add_argument('--skip-features', action='store_true',
                       help='skip feature distribution analysis')
    parser.add_argument('--skip-figures', action='store_true',
                       help='skip figure generation')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'[downstream] starting evaluation at {timestamp}')
    print(f'[downstream] output directory: {output_dir}')
    print('=' * 60)

    # save configuration
    config = {
        'domain_a_dir': args.domain_a_dir,
        'domain_b_dir': args.domain_b_dir,
        'harmonization_model': args.harmonization_model,
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'device': args.device,
        'timestamp': timestamp,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # step 1: domain classification
    if not args.skip_classification:
        domain_output = output_dir / 'domain_classification'

        cmd = [
            sys.executable, str(script_dir / 'domain_classifier.py'),
            '--domain-a-dir', args.domain_a_dir,
            '--domain-b-dir', args.domain_b_dir,
            '--output-dir', str(domain_output),
            '--n-epochs', str(args.n_epochs),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
        ]

        if args.harmonization_model:
            cmd.extend(['--harmonization-model', args.harmonization_model])

        if not run_command(cmd, 'domain classification evaluation'):
            print('[warning] domain classification failed, continuing...')
    else:
        print('[downstream] skipping domain classification')
        domain_output = output_dir / 'domain_classification'

    # step 2: feature distribution analysis
    if not args.skip_features:
        feature_output = output_dir / 'feature_distribution'

        cmd = [
            sys.executable, str(script_dir / 'feature_distribution_analysis.py'),
            '--domain-a-dir', args.domain_a_dir,
            '--domain-b-dir', args.domain_b_dir,
            '--output-dir', str(feature_output),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
        ]

        if args.harmonization_model:
            cmd.extend(['--harmonization-model', args.harmonization_model])

        if not run_command(cmd, 'feature distribution analysis'):
            print('[warning] feature distribution analysis failed, continuing...')
    else:
        print('[downstream] skipping feature distribution analysis')
        feature_output = output_dir / 'feature_distribution'

    # step 3: generate figures
    if not args.skip_figures:
        figures_output = output_dir / 'figures'
        figures_output.mkdir(parents=True, exist_ok=True)

        domain_results = domain_output / 'domain_classification_results.json'
        feature_results = feature_output / 'feature_distribution_results.json'
        training_history = domain_output / 'training_history.json'

        if domain_results.exists() and feature_results.exists():
            cmd = [
                sys.executable, str(script_dir / 'generate_downstream_figures.py'),
                '--domain-results', str(domain_results),
                '--feature-results', str(feature_results),
                '--tsne-dir', str(feature_output),
                '--output-dir', str(figures_output),
            ]

            if training_history.exists():
                cmd.extend(['--training-history', str(training_history)])

            if not run_command(cmd, 'figure generation'):
                print('[warning] figure generation failed')
        else:
            print('[warning] results files not found, skipping figure generation')
    else:
        print('[downstream] skipping figure generation')

    # generate summary report
    print('=' * 60)
    print('[downstream] generating summary report...')

    summary = {
        'timestamp': timestamp,
        'configuration': config,
        'results': {}
    }

    # load domain results
    domain_results_file = domain_output / 'domain_classification_results.json'
    if domain_results_file.exists():
        with open(domain_results_file) as f:
            summary['results']['domain_classification'] = json.load(f)

    # load feature results
    feature_results_file = feature_output / 'feature_distribution_results.json'
    if feature_results_file.exists():
        with open(feature_results_file) as f:
            summary['results']['feature_distribution'] = json.load(f)

    # save summary
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # print summary
    print('=' * 60)
    print('[downstream] evaluation complete')
    print('=' * 60)

    if 'domain_classification' in summary['results']:
        dc = summary['results']['domain_classification']
        print('[domain classification]')
        print(f"  raw accuracy: {dc['raw']['accuracy']:.4f}")
        print(f"  raw auc: {dc['raw']['auc']:.4f}")
        if 'harmonized' in dc:
            print(f"  harmonized accuracy: {dc['harmonized']['accuracy']:.4f}")
            print(f"  harmonized auc: {dc['harmonized']['auc']:.4f}")
            if 'improvement' in dc:
                print(f"  accuracy reduction: {dc['improvement']['accuracy_reduction']:.4f}")

    if 'feature_distribution' in summary['results']:
        fd = summary['results']['feature_distribution']
        print('[feature distribution]')
        print(f"  raw fid: {fd['raw']['fid']:.4f}")
        print(f"  raw mmd: {fd['raw']['mmd_rbf']:.4f}")
        if 'harmonized' in fd:
            print(f"  harmonized fid: {fd['harmonized']['fid']:.4f}")
            print(f"  harmonized mmd: {fd['harmonized']['mmd_rbf']:.4f}")
            if 'improvement' in fd:
                print(f"  fid reduction: {fd['improvement']['fid_reduction_percent']:.1f}%")

    print('=' * 60)
    print(f'[downstream] results saved to {output_dir}')


if __name__ == '__main__':
    main()
