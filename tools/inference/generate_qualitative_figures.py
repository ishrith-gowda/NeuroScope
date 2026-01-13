"""
generate publication-grade qualitative comparison figures

creates visual comparison grids from inference results and attention maps.
follows top-tier medical imaging publication standards (miccai, ieee tmi).

usage:
    python generate_qualitative_figures.py --inference_dir results/inference
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.figures.latex_figure_config import COLORS, save_figure


def load_inference_results(inference_dir: Path, category: str):
    """load inference results for a category"""
    inference_file = inference_dir / f'inference_{category}.npz'

    if not inference_file.exists():
        print(f"warning: {inference_file} not found")
        return None

    data = np.load(inference_file)
    return data


def load_attention_results(inference_dir: Path, category: str):
    """load attention maps for a category"""
    attention_file = inference_dir / f'attention_{category}.npz'

    if not attention_file.exists():
        print(f"warning: {attention_file} not found")
        return None

    data = np.load(attention_file)
    return data


def extract_center_slice_modalities(volume: np.ndarray) -> np.ndarray:
    """
    extract 4 modalities from center slice

    input: [12, H, W] (3 slices × 4 modalities)
    output: [4, H, W] (center slice, 4 modalities)
    """
    # 2.5d format: [slice0_mod0-3, slice1_mod0-3, slice2_mod0-3]
    # center slice is indices 4-7
    center_modalities = volume[4:8, :, :]
    return center_modalities


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """normalize image to [0, 1] for display"""
    image = image.copy()
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    return image


def generate_comparison_grid(
    inputs_a: np.ndarray,
    generated_b: np.ndarray,
    reconstructed_a: np.ndarray,
    ssim_scores: list,
    output_path: Path,
    n_samples: int = 5,
    modality_idx: int = 0
):
    """
    figure: input | generated | reconstructed comparison grid

    rows: different samples
    cols: input (a) | generated (b) | reconstructed (a)

    args:
        inputs_a: [N, 12, H, W] input volumes
        generated_b: [N, 4, H, W] generated translations
        reconstructed_a: [N, 4, H, W] cycle reconstructions
        ssim_scores: list of ssim values for each sample
        output_path: where to save figure
        n_samples: number of samples to show
        modality_idx: which modality to visualize (0=t1, 1=t1ce, 2=t2, 3=flair)
    """
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    modality_name = modality_names[modality_idx]

    fig, axes = plt.subplots(n_samples, 3, figsize=(7, n_samples * 2))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_samples):
        if row >= len(inputs_a):
            break

        # extract center slice from input
        input_slice = extract_center_slice_modalities(inputs_a[row])
        input_img = normalize_for_display(input_slice[modality_idx])

        # generated and reconstructed
        generated_img = normalize_for_display(generated_b[row, modality_idx])
        reconstructed_img = normalize_for_display(reconstructed_a[row, modality_idx])

        # plot
        axes[row, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(r'\textbf{(a)} Input ($A$)')

        axes[row, 1].imshow(generated_img, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title(r'\textbf{(b)} Generated ($\hat{B}$)')

        axes[row, 2].imshow(reconstructed_img, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title(r'\textbf{(c)} Reconstructed ($\hat{A}$)')

        # add ssim score as ylabel
        if ssim_scores and row < len(ssim_scores):
            axes[row, 0].text(-0.1, 0.5, f'SSIM: {ssim_scores[row]:.3f}',
                            transform=axes[row, 0].transAxes,
                            rotation=90, va='center', ha='right', fontsize=9)

    plt.suptitle(f'$A \\rightarrow B \\rightarrow A$ Cycle Comparison ({modality_name})',
                fontsize=12, y=0.995)
    plt.tight_layout()

    save_figure(fig, output_path.stem, output_dir=output_path.parent)
    plt.close()

    print(f"saved comparison grid: {output_path}")


def generate_multimodality_grid(
    inputs_a: np.ndarray,
    generated_b: np.ndarray,
    case_idx: int,
    output_path: Path
):
    """
    figure: show all 4 modalities for a single case

    layout:
             Input    Generated
    T1        img       img
    T1ce      img       img
    T2        img       img
    FLAIR     img       img
    """
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']

    fig, axes = plt.subplots(4, 2, figsize=(4, 7))

    # extract center slice from input
    input_slice = extract_center_slice_modalities(inputs_a[case_idx])

    for mod_idx, mod_name in enumerate(modality_names):
        # input
        input_img = normalize_for_display(input_slice[mod_idx])
        axes[mod_idx, 0].imshow(input_img, cmap='gray', vmin=0, vmax=1)
        axes[mod_idx, 0].axis('off')
        axes[mod_idx, 0].set_ylabel(mod_name, fontsize=10)

        # generated
        generated_img = normalize_for_display(generated_b[case_idx, mod_idx])
        axes[mod_idx, 1].imshow(generated_img, cmap='gray', vmin=0, vmax=1)
        axes[mod_idx, 1].axis('off')

    # column titles
    axes[0, 0].set_title(r'\textbf{Input ($A$)}', fontsize=11)
    axes[0, 1].set_title(r'\textbf{Generated ($\hat{B}$)}', fontsize=11)

    plt.tight_layout()

    save_figure(fig, output_path.stem, output_dir=output_path.parent)
    plt.close()

    print(f"saved multimodality grid: {output_path}")


def generate_attention_overlay(
    input_img: np.ndarray,
    attention_map: np.ndarray,
    output_path: Path,
    title: str = 'Attention Heatmap'
):
    """
    figure: attention heatmap overlay on input image

    shows where model focuses during translation
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.5))

    # normalize input
    input_norm = normalize_for_display(input_img)

    # normalize attention (remove batch and channel dims if present)
    if attention_map.ndim == 4:  # [B, C, H, W]
        attention_map = attention_map[0, 0]
    elif attention_map.ndim == 3:  # [B, H, W]
        attention_map = attention_map[0]

    # resize attention to match input if needed
    if attention_map.shape != input_norm.shape:
        from scipy.ndimage import zoom
        zoom_factors = (input_norm.shape[0] / attention_map.shape[0],
                       input_norm.shape[1] / attention_map.shape[1])
        attention_map = zoom(attention_map, zoom_factors, order=1)

    attention_norm = normalize_for_display(attention_map)

    # original image
    ax1.imshow(input_norm, cmap='gray')
    ax1.axis('off')
    ax1.set_title(r'\textbf{(a)} Original')

    # attention heatmap
    im2 = ax2.imshow(attention_norm, cmap='hot')
    ax2.axis('off')
    ax2.set_title(r'\textbf{(b)} Attention')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # overlay
    ax3.imshow(input_norm, cmap='gray')
    ax3.imshow(attention_norm, cmap='hot', alpha=0.5)
    ax3.axis('off')
    ax3.set_title(r'\textbf{(c)} Overlay')

    plt.suptitle(title, fontsize=11, y=0.98)
    plt.tight_layout()

    save_figure(fig, output_path.stem, output_dir=output_path.parent)
    plt.close()

    print(f"saved attention overlay: {output_path}")


def generate_cycle_consistency_demo(
    inputs_a: np.ndarray,
    generated_b: np.ndarray,
    reconstructed_a: np.ndarray,
    inputs_b: np.ndarray,
    generated_a: np.ndarray,
    reconstructed_b: np.ndarray,
    case_idx: int,
    modality_idx: int,
    output_path: Path
):
    """
    figure: bidirectional cycle consistency demonstration

    top row: a → b → a
    bottom row: b → a → b
    """
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    modality_name = modality_names[modality_idx]

    fig, axes = plt.subplots(2, 4, figsize=(7, 3.5))

    # extract images
    input_a = normalize_for_display(
        extract_center_slice_modalities(inputs_a[case_idx])[modality_idx]
    )
    gen_b = normalize_for_display(generated_b[case_idx, modality_idx])
    rec_a = normalize_for_display(reconstructed_a[case_idx, modality_idx])

    input_b = normalize_for_display(
        extract_center_slice_modalities(inputs_b[case_idx])[modality_idx]
    )
    gen_a = normalize_for_display(generated_a[case_idx, modality_idx])
    rec_b = normalize_for_display(reconstructed_b[case_idx, modality_idx])

    # compute difference maps
    diff_a = np.abs(input_a - rec_a)
    diff_b = np.abs(input_b - rec_b)

    # top row: a → b → a
    axes[0, 0].imshow(input_a, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title(r'$A$', fontsize=10)

    axes[0, 1].imshow(gen_b, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title(r'$G_{A2B}(A) = \hat{B}$', fontsize=10)

    axes[0, 2].imshow(rec_a, cmap='gray')
    axes[0, 2].axis('off')
    axes[0, 2].set_title(r'$G_{B2A}(\hat{B}) = \hat{A}$', fontsize=10)

    im03 = axes[0, 3].imshow(diff_a, cmap='hot')
    axes[0, 3].axis('off')
    axes[0, 3].set_title(r'$|A - \hat{A}|$', fontsize=10)
    plt.colorbar(im03, ax=axes[0, 3], fraction=0.046)

    # bottom row: b → a → b
    axes[1, 0].imshow(input_b, cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title(r'$B$', fontsize=10)

    axes[1, 1].imshow(gen_a, cmap='gray')
    axes[1, 1].axis('off')
    axes[1, 1].set_title(r'$G_{B2A}(B) = \hat{A}$', fontsize=10)

    axes[1, 2].imshow(rec_b, cmap='gray')
    axes[1, 2].axis('off')
    axes[1, 2].set_title(r'$G_{A2B}(\hat{A}) = \hat{B}$', fontsize=10)

    im13 = axes[1, 3].imshow(diff_b, cmap='hot')
    axes[1, 3].axis('off')
    axes[1, 3].set_title(r'$|B - \hat{B}|$', fontsize=10)
    plt.colorbar(im13, ax=axes[1, 3], fraction=0.046)

    # row labels
    axes[0, 0].text(-0.2, 0.5, r'$A \rightarrow B \rightarrow A$',
                   transform=axes[0, 0].transAxes,
                   rotation=90, va='center', ha='right', fontsize=10)
    axes[1, 0].text(-0.2, 0.5, r'$B \rightarrow A \rightarrow B$',
                   transform=axes[1, 0].transAxes,
                   rotation=90, va='center', ha='right', fontsize=10)

    plt.suptitle(f'Cycle Consistency Demonstration ({modality_name})',
                fontsize=12, y=0.98)
    plt.tight_layout()

    save_figure(fig, output_path.stem, output_dir=output_path.parent)
    plt.close()

    print(f"saved cycle consistency demo: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='generate qualitative figures')
    parser.add_argument('--inference_dir', type=str,
                       default='results/inference',
                       help='directory containing inference results')
    parser.add_argument('--output_dir', type=str,
                       default='figures/qualitative',
                       help='output directory for figures')
    parser.add_argument('--modality', type=int, default=1,
                       help='modality index to visualize (0=t1, 1=t1ce, 2=t2, 3=flair)')

    args = parser.parse_args()

    # setup paths
    project_root = Path(__file__).parent.parent.parent
    inference_dir = project_root / args.inference_dir
    output_dir = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"generating qualitative figures from {inference_dir}")
    print(f"output directory: {output_dir}")
    print(f"visualizing modality: {['T1', 'T1ce', 'T2', 'FLAIR'][args.modality]}")

    # process best cases
    print(f"\n{'='*60}")
    print("generating figures for best cases")
    print(f"{'='*60}")

    best_data = load_inference_results(inference_dir, 'best')
    if best_data is not None:
        # comparison grid
        generate_comparison_grid(
            best_data['inputs_a'],
            best_data['generated_b'],
            best_data['reconstructed_a'],
            best_data['ssim_a2b'].tolist(),
            output_dir / 'fig_best_comparison',
            n_samples=5,
            modality_idx=args.modality
        )

        # multimodality grid for best case
        generate_multimodality_grid(
            best_data['inputs_a'],
            best_data['generated_b'],
            case_idx=0,
            output_path=output_dir / 'fig_best_multimodality'
        )

        # cycle consistency demo
        generate_cycle_consistency_demo(
            best_data['inputs_a'],
            best_data['generated_b'],
            best_data['reconstructed_a'],
            best_data['inputs_b'],
            best_data['generated_a'],
            best_data['reconstructed_b'],
            case_idx=0,
            modality_idx=args.modality,
            output_path=output_dir / 'fig_best_cycle_consistency'
        )

    # process worst cases
    print(f"\n{'='*60}")
    print("generating figures for worst cases")
    print(f"{'='*60}")

    worst_data = load_inference_results(inference_dir, 'worst')
    if worst_data is not None:
        generate_comparison_grid(
            worst_data['inputs_a'],
            worst_data['generated_b'],
            worst_data['reconstructed_a'],
            worst_data['ssim_a2b'].tolist(),
            output_dir / 'fig_worst_comparison',
            n_samples=5,
            modality_idx=args.modality
        )

        generate_multimodality_grid(
            worst_data['inputs_a'],
            worst_data['generated_b'],
            case_idx=0,
            output_path=output_dir / 'fig_worst_multimodality'
        )

    # process median cases
    print(f"\n{'='*60}")
    print("generating figures for median cases")
    print(f"{'='*60}")

    median_data = load_inference_results(inference_dir, 'median')
    if median_data is not None:
        generate_comparison_grid(
            median_data['inputs_a'],
            median_data['generated_b'],
            median_data['reconstructed_a'],
            median_data['ssim_a2b'].tolist(),
            output_dir / 'fig_median_comparison',
            n_samples=3,
            modality_idx=args.modality
        )

    # process attention maps if available
    print(f"\n{'='*60}")
    print("generating attention visualizations")
    print(f"{'='*60}")

    attention_data = load_attention_results(inference_dir, 'best')
    if attention_data is not None and best_data is not None:
        # find spatial attention maps
        spatial_attention_keys = [k for k in attention_data.keys()
                                 if 'spatial' in k.lower() and 'case0' in k]

        if spatial_attention_keys:
            # use first spatial attention map
            attn_key = spatial_attention_keys[0]
            attention_map = attention_data[attn_key]

            input_img = extract_center_slice_modalities(
                best_data['inputs_a'][0]
            )[args.modality]

            generate_attention_overlay(
                input_img,
                attention_map,
                output_dir / 'fig_best_attention_overlay',
                title='Spatial Attention (Best Case)'
            )
        else:
            print("no spatial attention maps found in results")

    print(f"\n{'='*60}")
    print("qualitative figure generation complete")
    print(f"{'='*60}")
    print(f"figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
