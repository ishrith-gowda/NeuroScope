#!/usr/bin/env python3
"""
Fix individual modality figures to match all_modalities layout style.

Extracts brain images from existing individual modality PDFs (2x6 grid)
and recreates them with tight spacing matching all_modalities figures.
For FLAIR sample 00 (broken placeholder), extracts from all_modalities PDF.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path
from itertools import groupby
import warnings
warnings.filterwarnings('ignore')

import fitz  # PyMuPDF
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# publication-quality settings (matching all_modalities gold standard)
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
    'axes.grid': False,
})


def render_pdf_to_array(pdf_path: Path, dpi: int = 300) -> np.ndarray:
    """Render a PDF page to a numpy array."""
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    doc.close()
    return img


def find_white_bands(values: np.ndarray, threshold: float = 240, min_width: int = 10):
    """Find contiguous bands of high values (white regions)."""
    white = values > threshold
    bands = []
    for k, g in groupby(enumerate(white), key=lambda x: x[1]):
        if k:
            indices = [x[0] for x in g]
            if len(indices) >= min_width:
                bands.append((indices[0], indices[-1]))
    return bands


def detect_grid(img: np.ndarray, expected_rows: int, expected_cols: int):
    """
    Detect subplot grid positions in a rendered PDF image.
    Returns list of (row_start, row_end) and (col_start, col_end) tuples.
    """
    gray = np.mean(img[:, :, :3], axis=2)
    h, w = gray.shape

    # Find horizontal white bands (row separators)
    row_brightness = np.mean(gray, axis=1)
    h_bands = find_white_bands(row_brightness, threshold=240, min_width=5)

    # Find vertical white bands (column separators)
    col_brightness = np.mean(gray, axis=0)
    v_bands = find_white_bands(col_brightness, threshold=240, min_width=5)

    # Add image edges as implicit bands if not already captured
    if not h_bands or h_bands[0][0] > 5:
        h_bands.insert(0, (0, 0))
    if not h_bands or h_bands[-1][1] < h - 5:
        h_bands.append((h - 1, h - 1))
    if not v_bands or v_bands[0][0] > 5:
        v_bands.insert(0, (0, 0))
    if not v_bands or v_bands[-1][1] < w - 5:
        v_bands.append((w - 1, w - 1))

    # Extract row regions (between horizontal bands)
    row_regions = []
    for i in range(len(h_bands) - 1):
        start = h_bands[i][1] + 1
        end = h_bands[i + 1][0] - 1
        if end - start > 50:
            row_regions.append((start, end))

    # Extract column regions (between vertical bands)
    col_regions = []
    for i in range(len(v_bands) - 1):
        start = v_bands[i][1] + 1
        end = v_bands[i + 1][0] - 1
        if end - start > 50:
            col_regions.append((start, end))

    return row_regions, col_regions


def extract_subplot_images(img: np.ndarray, row_regions, col_regions):
    """Extract individual subplot images from a grid."""
    subplots = {}
    for r_idx, (r_start, r_end) in enumerate(row_regions):
        for c_idx, (c_start, c_end) in enumerate(col_regions):
            subplots[(r_idx, c_idx)] = img[r_start:r_end, c_start:c_end]
    return subplots


def create_single_modality_figure(
    images_row1: list,
    images_row2: list,
    output_path: Path,
    modality: str,
    sample_idx: int,
    dataset_idx: int = None
):
    """
    Create a single modality figure using the same plt.subplots approach
    as the all_modalities figures for identical tight spacing.
    """
    n_rows = 2 if images_row2 is not None else 1

    # Use same approach as all_modalities: plt.subplots with subplots_adjust
    fig, axes = plt.subplots(n_rows, 6, figsize=(16, 6 if n_rows == 2 else 3.5))
    plt.subplots_adjust(hspace=0.12, wspace=0.05)

    # Ensure axes is 2D even for single row
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    col_headers_a = [
        r'Input A', r'$\rightarrow$B (Base)', r'$\rightarrow$B (Attn)',
        r'Rec (Base)', r'Rec (Attn)', r'$|$Diff$|$ (Attn)'
    ]

    # Row 1: A→B→A
    for c_idx, img in enumerate(images_row1):
        axes[0, c_idx].imshow(img)
        axes[0, c_idx].set_title(col_headers_a[c_idx], fontsize=12)
        axes[0, c_idx].axis('off')

    # Row label on first column using set_ylabel (stays tight to image)
    axes[0, 0].set_ylabel(r'A$\rightarrow$B$\rightarrow$A',
                           fontsize=12, fontweight='bold', rotation=90,
                           labelpad=2)
    # Re-enable just the y-axis label (axis('off') hides it)
    axes[0, 0].yaxis.set_visible(True)
    axes[0, 0].yaxis.label.set_visible(True)

    # Row 2: B→A→B
    if images_row2 is not None:
        for c_idx, img in enumerate(images_row2):
            axes[1, c_idx].imshow(img)
            axes[1, c_idx].axis('off')

        axes[1, 0].set_ylabel(r'B$\rightarrow$A$\rightarrow$B',
                               fontsize=12, fontweight='bold', rotation=90,
                               labelpad=2)
        axes[1, 0].yaxis.set_visible(True)
        axes[1, 0].yaxis.label.set_visible(True)

    # Title
    sample_label = f'Sample {dataset_idx}' if dataset_idx is not None else f'Sample {sample_idx}'
    fig.suptitle(
        f'Visual Comparison: Baseline vs SA-CycleGAN ({modality} Modality, {sample_label})',
        fontsize=16, fontweight='bold', y=0.98 if n_rows == 2 else 1.02
    )

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    fig.savefig(str(output_path), format='pdf')
    plt.close(fig)


def is_placeholder_graph(pdf_path: Path) -> bool:
    """Check if a PDF is a placeholder graph rather than brain images."""
    img = render_pdf_to_array(pdf_path, dpi=72)
    gray = np.mean(img[:, :, :3], axis=2)

    # Brain image PDFs have significant dark pixel areas (black backgrounds)
    dark_pixel_ratio = np.mean(gray < 50)
    # Also check for very bright axis lines typical of matplotlib graphs
    very_bright_ratio = np.mean(gray > 250)

    # Placeholder graphs: few dark pixels, lots of white background
    # Brain images: >10% dark pixels from black backgrounds
    return dark_pixel_ratio < 0.05


def process_individual_modality_pdf(
    pdf_path: Path,
    output_path: Path,
    modality: str,
    sample_idx: int
):
    """
    Process an existing individual modality PDF and recreate it
    with tight spacing matching all_modalities style.
    """
    print(f"  Processing {pdf_path.name}...")

    img = render_pdf_to_array(pdf_path)
    row_regions, col_regions = detect_grid(img, expected_rows=2, expected_cols=6)

    print(f"    Detected grid: {len(row_regions)} rows x {len(col_regions)} cols")

    if len(row_regions) < 2 or len(col_regions) < 6:
        print(f"  ERROR: Could not detect 2x6 grid (got {len(row_regions)}x{len(col_regions)})")
        return False

    row_regions = row_regions[:2]
    col_regions = col_regions[:6]

    subplots = extract_subplot_images(img, row_regions, col_regions)

    images_row1 = [subplots[(0, c)] for c in range(6)]
    images_row2 = [subplots[(1, c)] for c in range(6)]

    create_single_modality_figure(
        images_row1, images_row2, output_path, modality, sample_idx
    )
    return True


def process_from_original_png(
    pdf_name: str,
    output_path: Path,
    modality: str,
    sample_idx: int,
    figures_dir: Path
):
    """
    Extract brain images from the original PNG committed in git history.

    The original generate_visual_examples.py created PNGs with a 4-row x 3-col
    layout (each col spans 2 gridspec columns):
        Row 0: Input A, Fake B baseline, Fake B attention   (A->B->A top)
        Row 1: Rec A baseline, Rec A attention, Diff        (A->B->A bottom)
        Row 2: Input B, Fake A baseline, Fake A attention   (B->A->B top)
        Row 3: Rec B baseline, Rec B attention, Diff        (B->A->B bottom)

    These are remapped to 2x6 for the publication layout.
    """
    import subprocess
    import tempfile

    png_name = pdf_name.replace('.pdf', '.png')

    # Try to extract original PNG from git history
    try:
        result = subprocess.run(
            ['git', 'log', '--all', '--format=%H', '-1', '--', f'figures/visual_examples/{png_name}'],
            capture_output=True, text=True,
            cwd=str(figures_dir.parent.parent)
        )
        commit_hash = result.stdout.strip()
        if not commit_hash:
            print(f"    No git history found for {png_name}")
            return False

        # Extract the PNG to a temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        result = subprocess.run(
            ['git', 'show', f'{commit_hash}:figures/visual_examples/{png_name}'],
            capture_output=True,
            cwd=str(figures_dir.parent.parent)
        )
        if result.returncode != 0:
            print(f"    Failed to extract {png_name} from git")
            return False

        with open(tmp_path, 'wb') as f:
            f.write(result.stdout)

    except Exception as e:
        print(f"    Git extraction failed: {e}")
        return False

    try:
        from PIL import Image
        img = np.array(Image.open(tmp_path).convert('RGB'))
        os.unlink(tmp_path)
    except Exception as e:
        print(f"    Failed to load PNG: {e}")
        return False

    print(f"    Extracted {png_name} from git ({img.shape[1]}x{img.shape[0]})")

    # Detect content regions using dark pixel ratio (brain images on black backgrounds)
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    def find_content_regions(dark_ratio_profile, min_width=100, threshold=0.1):
        in_content = False
        regions = []
        start = 0
        for i in range(len(dark_ratio_profile)):
            if dark_ratio_profile[i] > threshold and not in_content:
                start = i
                in_content = True
            elif dark_ratio_profile[i] <= threshold and in_content:
                if i - start > min_width:
                    regions.append((start, i - 1))
                in_content = False
        if in_content and len(dark_ratio_profile) - start > min_width:
            regions.append((start, len(dark_ratio_profile) - 1))
        return regions

    col_dark_ratio = np.mean(gray < 100, axis=0)
    content_cols = find_content_regions(col_dark_ratio)

    row_dark_ratio = np.mean(gray < 100, axis=1)
    content_rows = find_content_regions(row_dark_ratio)

    if len(content_rows) != 4 or len(content_cols) != 3:
        print(f"    ERROR: Expected 4x3 grid, got {len(content_rows)}x{len(content_cols)}")
        return False

    # Extract 4x3 subplot images
    subplots = {}
    for r_idx, (r_start, r_end) in enumerate(content_rows):
        for c_idx, (c_start, c_end) in enumerate(content_cols):
            subplots[(r_idx, c_idx)] = img[r_start:r_end+1, c_start:c_end+1]

    # Remap 4x3 to 2x6:
    # Row 0 (A->B->A): (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # Row 1 (B->A->B): (2,0), (2,1), (2,2), (3,0), (3,1), (3,2)
    images_row1 = [subplots[(0,c)] for c in range(3)] + [subplots[(1,c)] for c in range(3)]
    images_row2 = [subplots[(2,c)] for c in range(3)] + [subplots[(3,c)] for c in range(3)]

    create_single_modality_figure(
        images_row1, images_row2, output_path, modality, sample_idx
    )
    return True


def main():
    figures_dir = Path(__file__).parent.parent.parent / 'figures' / 'visual_examples'

    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    num_samples = 5  # samples 00-04

    print(f"\n{'='*70}")
    print("Fixing Individual Modality Figures")
    print(f"Tight spacing matching all_modalities layout")
    print(f"{'='*70}\n")

    success_count = 0
    total_count = 0

    for sample_idx in range(num_samples):
        print(f"\n--- Sample {sample_idx:02d} ---")
        for modality in modalities:
            total_count += 1
            pdf_name = f'visual_sample_{sample_idx:02d}_{modality}.pdf'
            pdf_path = figures_dir / pdf_name
            output_path = pdf_path  # Overwrite in place

            if not pdf_path.exists():
                print(f"  SKIP: {pdf_name} not found")
                continue

            # Special case: FLAIR sample 00 is a known broken placeholder
            if modality == 'FLAIR' and sample_idx == 0:
                print(f"  {pdf_name}: known broken placeholder, extracting from git PNG...")
                if process_from_original_png(pdf_name, output_path, modality, sample_idx, figures_dir):
                    success_count += 1
                    print(f"  OK: {pdf_name} (from original PNG in git)")
                else:
                    print(f"  FAIL: {pdf_name}")
                continue

            # Check if this is a placeholder graph (safety check for other broken files)
            if is_placeholder_graph(pdf_path):
                print(f"  {pdf_name} is a placeholder graph, extracting from git PNG...")
                if process_from_original_png(pdf_name, output_path, modality, sample_idx, figures_dir):
                    success_count += 1
                    print(f"  OK: {pdf_name} (from original PNG in git)")
                else:
                    print(f"  FAIL: {pdf_name}")
            else:
                if process_individual_modality_pdf(pdf_path, output_path, modality, sample_idx):
                    success_count += 1
                    print(f"  OK: {pdf_name}")
                else:
                    print(f"  FAIL: {pdf_name}")

    print(f"\n{'='*70}")
    print(f"Completed: {success_count}/{total_count} figures fixed")
    print(f"{'='*70}\n")

    return 0 if success_count == total_count else 1


if __name__ == '__main__':
    sys.exit(main())
