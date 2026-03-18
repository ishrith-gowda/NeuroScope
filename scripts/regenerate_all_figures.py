#!/usr/bin/env python3
"""
comprehensive figure regeneration script.

regenerates all publication-quality figures across the entire project.
updated with enhanced latex rendering, custom color palettes, and professional styling.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """run a figure generation script and report results."""
    print(f"\n{'='*70}")
    print(f"[regenerating] {description}...")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            ['python', script_path],
            cwd=str(Path(script_path).parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            # print last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:
                if line.strip():
                    print(f"  {line}")
            return True
        else:
            print(f"✗ {description} failed")
            print(f"error: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {description} timed out")
        return False
    except Exception as e:
        print(f"✗ {description} error: {e}")
        return False


def main():
    """regenerate all figures."""
    root = Path(__file__).parent.parent
    
    scripts = [
        # downstream evaluation (already upgraded)
        (str(root / 'scripts/05_downstream_evaluation/regenerate_harmonization_figure.py'),
         '[Downstream] Harmonization Summary'),
        (str(root / 'scripts/05_downstream_evaluation/regenerate_training_curves.py'),
         '[Downstream] Training Curves'),
        (str(root / 'scripts/05_downstream_evaluation/regenerate_tsne_visualization.py'),
         '[Downstream] t-SNE Visualization'),
         
        # training figures (upgraded)
        (str(root / 'scripts/04_figures/generate_training_figures.py'),
         '[Training] Loss Curves & Metrics'),
        
        # dataset figures
        (str(root / 'scripts/04_figures/generate_dataset_figures.py'),
         '[Dataset] Statistics & Preprocessing'),
         
        # architecture figures
        (str(root / 'scripts/04_figures/generate_architecture_figures.py'),
         '[Architecture] Diagrams & Comparisons'),
         
        # statistical figures
        (str(root / 'scripts/04_figures/generate_statistical_figures.py'),
         '[Statistical] Analysis & Comparisons'),
    ]
    
    results = {}
    
    print("\n" + "="*70)
    print("comprehensive figure regeneration")
    print(f"project: neuroscope - sa-cyclegan for mri harmonization")
    print("="*70)
    
    for script_path, description in scripts:
        if Path(script_path).exists():
            results[description] = run_script(script_path, description)
        else:
            print(f"\n✗ script not found: {script_path}")
            results[description] = False
    
    # summary
    print("\n" + "="*70)
    print("regeneration summary")
    print("="*70)
    
    completed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for description, success in results.items():
        status = "✓ COMPLETE" if success else "✗ FAILED"
        print(f"{status:12} {description}")
    
    print(f"\ntotal: {completed}/{total} figure generation scripts completed")
    
    if completed == total:
        print("\n✓ all figures regenerated successfully!")
        return 0
    else:
        print(f"\n⚠ {total - completed} figure generation script(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
