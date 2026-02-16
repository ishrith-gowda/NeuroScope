#!/usr/bin/env python3
"""
regenerate fig_training_curves.pdf with improved publication-quality styling.

converts to full latex rendering with professional typography and colors.
"""

import json
import sys
from pathlib import Path

# add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_downstream_figures import plot_training_curves


def main():
    """regenerate the training curves figure."""
    
    # get root directory
    root_dir = Path(__file__).parent.parent.parent
    
    # output directory
    output_dir = root_dir / 'figures' / 'downstream'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load training history
    history_file = root_dir / 'experiments' / 'downstream_evaluation' / 'domain_classification' / 'training_history.json'
    
    if not history_file.exists():
        print(f'[error] training history not found at {history_file}')
        return 1
    
    with open(history_file) as f:
        history = json.load(f)
    
    print(f'[regenerate] training history keys: {history.keys()}')
    
    # regenerate figure with improved styling
    output_path = output_dir / 'fig_training_curves.pdf'
    
    print(f'[regenerate] generating improved training curves figure...')
    plot_training_curves(history, output_path)
    
    print(f'[regenerate] ✓ saved improved figure to {output_path}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
