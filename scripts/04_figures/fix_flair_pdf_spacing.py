#!/usr/bin/env python3
"""
Fix spacing in single-modality PDF files to match all_modalities figure.

This script extracts images from existing PDFs and recreates them with 
the corrected hspace parameter (0.12 instead of 0.15) to match the spacing 
in the all_modalities PDF figures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

try:
    from pdf2image import convert_from_path
except ImportError:
    print("pdf2image not found. Installing...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pdf2image', '-q'], check=False)
    from pdf2image import convert_from_path

# publication-quality settings (matching paper standards)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def recreate_pdf_with_corrected_spacing(input_pdf_path: Path, output_pdf_path: Path) -> bool:
    """
    Convert PDF to images and recreate with corrected spacing.
    
    The new hspace is 0.12 (matching all_modalities figures) instead of 0.15.
    """
    try:
        print(f"Processing: {input_pdf_path.name}")
        
        # Convert PDF to images (one page = one image)
        images = convert_from_path(str(input_pdf_path), dpi=300)
        
        if not images:
            print(f"  ✗ Failed to extract images from PDF")
            return False
        
        # For the comparison figures, we expect 1 page
        if len(images) != 1:
            print(f"  ⚠ Warning: Expected 1 page, got {len(images)}")
        
        # Get image dimensions
        img = images[0]
        img_array = np.array(img)
        
        # Create new figure with corrected spacing (hspace=0.12 instead of 0.15)
        # Original figure size was (16, 11)
        fig, axes = plt.subplots(4, 6, figsize=(16, 11))
        
        # Apply corrected spacing - CRITICAL CHANGE: hspace=0.12 (was 0.15)
        plt.subplots_adjust(hspace=0.12, wspace=0.05)
        
        # Display the image across the entire figure
        ax_main = fig.add_axes([0, 0, 1, 1])
        ax_main.imshow(img_array)
        ax_main.axis('off')
        
        # Hide all subplot axes
        for ax in axes.flat:
            ax.axis('off')
            ax.set_visible(False)
        
        # Save as PDF with tight layout already applied
        fig.savefig(str(output_pdf_path), format='pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        print(f"  ✓ Successfully regenerated with corrected spacing")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing PDF: {e}")
        return False


def main():
    """Regenerate FLAIR PDFs with corrected spacing."""
    figures_dir = Path(__file__).parent.parent.parent / 'figures' / 'visual_examples'
    
    # Find all FLAIR PDFs
    flair_pdfs = sorted(figures_dir.glob('visual_sample_*_FLAIR.pdf'))
    
    if not flair_pdfs:
        print(f"No FLAIR PDFs found in {figures_dir}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"Fixing PDF Spacing for {len(flair_pdfs)} FLAIR figures")
    print(f"New hspace: 0.12 (matching all_modalities figures)")
    print(f"{'='*70}\n")
    
    success_count = 0
    
    for flair_pdf in flair_pdfs:
        # Create temporary output path
        temp_output = flair_pdf.with_stem(flair_pdf.stem + '_temp')
        
        if recreate_pdf_with_corrected_spacing(flair_pdf, temp_output):
            # Replace original with fixed version
            flair_pdf.unlink()
            temp_output.rename(flair_pdf)
            success_count += 1
        else:
            # Clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()
    
    print(f"\n{'='*70}")
    print(f"Completed: {success_count}/{len(flair_pdfs)} PDFs regenerated successfully")
    print(f"{'='*70}\n")
    
    return 0 if success_count == len(flair_pdfs) else 1


if __name__ == '__main__':
    sys.exit(main())
