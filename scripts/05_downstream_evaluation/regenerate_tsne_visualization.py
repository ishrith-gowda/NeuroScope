#!/usr/bin/env python3
"""
regenerate fig_tsne_visualization.pdf with improved publication-quality styling.

converts to full latex rendering with professional typography and colors.
"""

import numpy as np
import sys
from pathlib import Path

# add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_downstream_figures import plot_tsne_visualization


def main():
    """regenerate the t-sne visualization figure."""
    
    # get root directory
    root_dir = Path(__file__).parent.parent.parent
    
    # output directory
    output_dir = root_dir / 'figures' / 'downstream'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load t-sne embeddings
    tsne_raw_file = root_dir / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'raw_tsne_embedding.npy'
    tsne_labels_file = root_dir / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'raw_tsne_labels.npy'
    tsne_harm_file = root_dir / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'harmonized_tsne_embedding.npy'
    
    if not tsne_raw_file.exists():
        print(f'[error] raw t-sne embeddings not found at {tsne_raw_file}')
        return 1
    
    if not tsne_labels_file.exists():
        print(f'[error] t-sne labels not found at {tsne_labels_file}')
        return 1
    
    tsne_raw = np.load(tsne_raw_file)
    labels = np.load(tsne_labels_file)
    
    tsne_harm = None
    if tsne_harm_file.exists():
        tsne_harm = np.load(tsne_harm_file)
        print(f'[regenerate] loaded harmonized t-sne: {tsne_harm.shape}')
    
    print(f'[regenerate] raw t-sne shape: {tsne_raw.shape}')
    print(f'[regenerate] labels shape: {labels.shape}')
    
    # regenerate figure with improved styling
    output_path = output_dir / 'fig_tsne_visualization.pdf'
    
    print(f'[regenerate] generating improved t-sne visualization figure...')
    plot_tsne_visualization(tsne_raw, labels, tsne_harm, output_path)
    
    print(f'[regenerate] ✓ saved improved figure to {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
