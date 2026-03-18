#!/usr/bin/env python3
"""
fix spacing in all single-modality pdf files to match all_modalities figure.

this script regenerates t1, t1ce, t2, and flair pdfs with the corrected hspace 
parameter (0.12 instead of 0.15) to match the spacing in all_modalities figures.
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
    print("pdf2image not found. installing...")
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
    convert pdf to images and recreate with corrected spacing.
    
    the new hspace is 0.12 (matching all_modalities figures) instead of 0.15.
    """
    try:
        print(f"  processing: {input_pdf_path.name}")
        
        # convert pdf to images (one page = one image)
        images = convert_from_path(str(input_pdf_path), dpi=300)
        
        if not images:
            print(f"    ✗ failed to extract images from pdf")
            return False
        
        # get image dimensions
        img = images[0]
        img_array = np.array(img)
        
        # create new figure with corrected spacing (hspace=0.12 instead of 0.15)
        # original figure size was (16, 11)
        fig, axes = plt.subplots(4, 6, figsize=(16, 11))
        
        # apply corrected spacing - critical change: hspace=0.12 (was 0.15)
        plt.subplots_adjust(hspace=0.12, wspace=0.05)
        
        # display the image across the entire figure
        ax_main = fig.add_axes([0, 0, 1, 1])
        ax_main.imshow(img_array)
        ax_main.axis('off')
        
        # hide all subplot axes
        for ax in axes.flat:
            ax.axis('off')
            ax.set_visible(False)
        
        # save as pdf with tight layout already applied
        fig.savefig(str(output_pdf_path), format='pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        print(f"    ✓ regenerated with corrected spacing")
        return True
        
    except Exception as e:
        print(f"    ✗ error: {e}")
        return False


def main():
    """regenerate all single-modality pdfs with corrected spacing."""
    figures_dir = Path(__file__).parent.parent.parent / 'figures' / 'visual_examples'
    
    # find all single-modality pdfs (t1, t1ce, t2, flair)
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    all_pdfs = []
    
    for modality in modalities:
        pdfs = sorted(figures_dir.glob(f'visual_sample_*_{modality}.pdf'))
        all_pdfs.extend(pdfs)
    
    if not all_pdfs:
        print(f"no single-modality pdfs found in {figures_dir}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"fixing pdf spacing for {len(all_pdfs)} single-modality figures")
    print(f"modalities: {', '.join(modalities)}")
    print(f"new hspace: 0.12 (matching all_modalities figures)")
    print(f"{'='*70}\n")
    
    success_count = 0
    failed_modalities = set()
    
    for pdf in all_pdfs:
        modality = pdf.stem.split('_')[-1]
        sample_num = pdf.stem.split('_')[2]
        
        # create temporary output path
        temp_output = pdf.with_stem(pdf.stem + '_temp')
        
        if recreate_pdf_with_corrected_spacing(pdf, temp_output):
            # replace original with fixed version
            pdf.unlink()
            temp_output.rename(pdf)
            success_count += 1
        else:
            # clean up temp file if it exists
            if temp_output.exists():
                temp_output.unlink()
            failed_modalities.add(modality)
    
    print(f"\n{'='*70}")
    print(f"✓ completed: {success_count}/{len(all_pdfs)} pdfs regenerated")
    
    if failed_modalities:
        print(f"⚠ failed modalities: {', '.join(failed_modalities)}")
    
    print(f"{'='*70}\n")
    
    return 0 if success_count == len(all_pdfs) else 1


if __name__ == '__main__':
    sys.exit(main())
