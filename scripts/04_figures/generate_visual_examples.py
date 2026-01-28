#!/usr/bin/env python3
"""
generate visual examples for sa-cyclegan mri harmonization paper.

this script creates side-by-side comparison images showing:
- input images from both domains
- translated images
- reconstructed (cycle-consistent) images
- difference maps highlighting changes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
import argparse
from tqdm import tqdm
import random

from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig
from neuroscope.data.datasets.dataset_25d import UnpairedMRIDataset25D

# publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})


def load_model(checkpoint_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    """
    load trained model from checkpoint.

    args:
        checkpoint_path: path to checkpoint file
        model_type: 'baseline' or 'attention'
        device: torch device

    returns:
        loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # create config - baseline disables attention, attention model enables it
    config = SACycleGAN25DConfig(
        n_input_slices=3,
        n_modalities=4,
        ngf=64,
        ndf=64,
        n_residual_blocks=9,
    )

    # note: the model architecture is the same for both but attention layers
    # are controlled by attention_layers config. for baseline, the attention
    # was disabled during training. both models use same SACycleGAN25D class
    # but the baseline was trained with attention weights zeroed or removed.
    model = SACycleGAN25D(config)

    # load state dict with dataparallel handling
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('.module.', '.')
        if new_key.startswith('module.'):
            new_key = new_key[7:]
        new_state_dict[new_key] = v

    # use strict=False to allow loading models with different attention configs
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def generate_translations(
    model: torch.nn.Module,
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    device: torch.device
) -> dict:
    """
    generate all translation outputs from the model.

    the model takes 12-channel input (3 slices x 4 modalities) and
    outputs 4-channel (center slice, 4 modalities). for cycle consistency,
    we repeat the 4-channel output to 12 channels.

    returns dict with:
        - fake_B: a translated to b
        - fake_A: b translated to a
        - rec_A: reconstructed a (a->b->a)
        - rec_B: reconstructed b (b->a->b)
    """
    with torch.no_grad():
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # forward translations (input: 12 channels, output: 4 channels)
        fake_B = model.G_A2B(real_A)
        fake_A = model.G_B2A(real_B)

        # create 3-slice input for cycle consistency (repeat 4-ch to 12-ch)
        # this matches the training procedure
        fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
        fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))

        # cycle translations
        rec_A = model.G_B2A(fake_B_3slice)
        rec_B = model.G_A2B(fake_A_3slice)

    return {
        'real_A': real_A.cpu(),
        'real_B': real_B.cpu(),
        'fake_B': fake_B.cpu(),
        'fake_A': fake_A.cpu(),
        'rec_A': rec_A.cpu(),
        'rec_B': rec_B.cpu(),
    }


def tensor_to_image(tensor: torch.Tensor, modality_idx: int = 0) -> np.ndarray:
    """
    convert tensor to displayable image.

    args:
        tensor: [b, c, h, w] tensor
        modality_idx: which modality channel to extract (0-3 for T1, T1CE, T2, FLAIR)

    returns:
        normalized numpy array for display
    """
    # extract single modality from output (4 channels)
    # for input with 12 channels (3 slices x 4 modalities), extract center slice
    if tensor.shape[1] == 12:
        # center slice, selected modality
        img = tensor[0, 4 + modality_idx].numpy()
    else:
        # output has 4 channels (4 modalities)
        img = tensor[0, modality_idx].numpy()

    # normalize to 0-1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def create_translation_figure(
    baseline_outputs: dict,
    attention_outputs: dict,
    output_path: Path,
    sample_idx: int,
    modality: str = 'T1'
) -> None:
    """
    create comprehensive translation comparison figure.

    shows input, translations, and reconstructions for both models.
    """
    modality_idx = {'T1': 0, 'T1CE': 1, 'T2': 2, 'FLAIR': 3}[modality]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.15, wspace=0.05)

    # row labels
    row_labels = ['Input', 'Translated', 'Reconstructed', 'Difference']

    # a->b direction (top half)
    fig.text(0.02, 0.77, 'A→B→A\n(BraTS)', fontsize=11, fontweight='bold',
             rotation=90, va='center', ha='center')

    # b->a direction (bottom half)
    fig.text(0.02, 0.27, 'B→A→B\n(UPenn)', fontsize=11, fontweight='bold',
             rotation=90, va='center', ha='center')

    images_to_show = []

    # a->b->a direction
    # row 0: input A
    real_A = tensor_to_image(baseline_outputs['real_A'], modality_idx)

    # row 1: fake B (baseline vs attention)
    fake_B_base = tensor_to_image(baseline_outputs['fake_B'], modality_idx)
    fake_B_attn = tensor_to_image(attention_outputs['fake_B'], modality_idx)

    # row 2: reconstructed A
    rec_A_base = tensor_to_image(baseline_outputs['rec_A'], modality_idx)
    rec_A_attn = tensor_to_image(attention_outputs['rec_A'], modality_idx)

    # row 3: difference maps
    diff_A_base = np.abs(real_A - rec_A_base)
    diff_A_attn = np.abs(real_A - rec_A_attn)

    # b->a->b direction
    real_B = tensor_to_image(baseline_outputs['real_B'], modality_idx)
    fake_A_base = tensor_to_image(baseline_outputs['fake_A'], modality_idx)
    fake_A_attn = tensor_to_image(attention_outputs['fake_A'], modality_idx)
    rec_B_base = tensor_to_image(baseline_outputs['rec_B'], modality_idx)
    rec_B_attn = tensor_to_image(attention_outputs['rec_B'], modality_idx)
    diff_B_base = np.abs(real_B - rec_B_base)
    diff_B_attn = np.abs(real_B - rec_B_attn)

    # create subplots
    # a->b->a section (rows 0-1)
    # row 0: input A, fake B baseline, fake B attention
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(real_A, cmap='gray')
    ax.set_title(f'Input A ({modality})', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2:4])
    ax.imshow(fake_B_base, cmap='gray')
    ax.set_title('→B (Baseline)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 4:6])
    ax.imshow(fake_B_attn, cmap='gray')
    ax.set_title('→B (SA-CycleGAN)', fontsize=9)
    ax.axis('off')

    # row 1: reconstructed A
    ax = fig.add_subplot(gs[1, 0:2])
    ax.imshow(rec_A_base, cmap='gray')
    ax.set_title('Rec A (Baseline)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2:4])
    ax.imshow(rec_A_attn, cmap='gray')
    ax.set_title('Rec A (SA-CycleGAN)', fontsize=9)
    ax.axis('off')

    # difference maps
    ax = fig.add_subplot(gs[1, 4:6])
    diff_combined = np.stack([diff_A_base, diff_A_attn, np.zeros_like(diff_A_base)], axis=-1)
    diff_combined = diff_combined / (diff_combined.max() + 1e-8) * 2  # enhance visibility
    diff_combined = np.clip(diff_combined, 0, 1)
    ax.imshow(diff_combined)
    ax.set_title('Diff (R=Base, G=Attn)', fontsize=9)
    ax.axis('off')

    # b->a->b section (rows 2-3)
    ax = fig.add_subplot(gs[2, 0:2])
    ax.imshow(real_B, cmap='gray')
    ax.set_title(f'Input B ({modality})', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 2:4])
    ax.imshow(fake_A_base, cmap='gray')
    ax.set_title('→A (Baseline)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 4:6])
    ax.imshow(fake_A_attn, cmap='gray')
    ax.set_title('→A (SA-CycleGAN)', fontsize=9)
    ax.axis('off')

    # row 3: reconstructed B
    ax = fig.add_subplot(gs[3, 0:2])
    ax.imshow(rec_B_base, cmap='gray')
    ax.set_title('Rec B (Baseline)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 2:4])
    ax.imshow(rec_B_attn, cmap='gray')
    ax.set_title('Rec B (SA-CycleGAN)', fontsize=9)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 4:6])
    diff_combined = np.stack([diff_B_base, diff_B_attn, np.zeros_like(diff_B_base)], axis=-1)
    diff_combined = diff_combined / (diff_combined.max() + 1e-8) * 2
    diff_combined = np.clip(diff_combined, 0, 1)
    ax.imshow(diff_combined)
    ax.set_title('Diff (R=Base, G=Attn)', fontsize=9)
    ax.axis('off')

    # main title
    fig.suptitle(f'Visual Comparison: Baseline vs SA-CycleGAN ({modality} Modality, Sample {sample_idx})',
                fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    for fmt in ['pdf', 'png']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'), format=fmt)

    plt.close(fig)


def create_multimodality_figure(
    baseline_outputs: dict,
    attention_outputs: dict,
    output_path: Path,
    sample_idx: int
) -> None:
    """
    create figure showing all 4 modalities for a single sample.
    """
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']

    fig, axes = plt.subplots(4, 6, figsize=(15, 10))

    for i, mod in enumerate(modalities):
        mod_idx = i

        # extract images
        real_A = tensor_to_image(baseline_outputs['real_A'], mod_idx)
        fake_B_base = tensor_to_image(baseline_outputs['fake_B'], mod_idx)
        fake_B_attn = tensor_to_image(attention_outputs['fake_B'], mod_idx)
        rec_A_base = tensor_to_image(baseline_outputs['rec_A'], mod_idx)
        rec_A_attn = tensor_to_image(attention_outputs['rec_A'], mod_idx)
        diff_base = np.abs(real_A - rec_A_base)
        diff_attn = np.abs(real_A - rec_A_attn)

        # plot row
        axes[i, 0].imshow(real_A, cmap='gray')
        axes[i, 0].set_ylabel(mod, fontsize=10, fontweight='bold')
        if i == 0:
            axes[i, 0].set_title('Input A', fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(fake_B_base, cmap='gray')
        if i == 0:
            axes[i, 1].set_title('→B (Base)', fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(fake_B_attn, cmap='gray')
        if i == 0:
            axes[i, 2].set_title('→B (Attn)', fontsize=9)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(rec_A_base, cmap='gray')
        if i == 0:
            axes[i, 3].set_title('Rec (Base)', fontsize=9)
        axes[i, 3].axis('off')

        axes[i, 4].imshow(rec_A_attn, cmap='gray')
        if i == 0:
            axes[i, 4].set_title('Rec (Attn)', fontsize=9)
        axes[i, 4].axis('off')

        axes[i, 5].imshow(diff_attn, cmap='hot', vmin=0, vmax=0.3)
        if i == 0:
            axes[i, 5].set_title('|Diff| (Attn)', fontsize=9)
        axes[i, 5].axis('off')

    fig.suptitle(f'Multi-Modality Translation: A→B→A Direction (Sample {sample_idx})',
                fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in ['pdf', 'png']:
        fig.savefig(output_path.with_suffix(f'.{fmt}'), format=fmt)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='generate visual examples for sa-cyclegan paper'
    )
    parser.add_argument('--baseline-checkpoint', type=str, required=True,
                       help='path to baseline model checkpoint')
    parser.add_argument('--attention-checkpoint', type=str, required=True,
                       help='path to attention model checkpoint')
    parser.add_argument('--brats-dir', type=str, required=True,
                       help='path to brats preprocessed data')
    parser.add_argument('--upenn-dir', type=str, required=True,
                       help='path to upenn preprocessed data')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for figures')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--num-samples', type=int, default=5,
                       help='number of samples to generate')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'[visual] device: {device}')

    # load models
    print('[visual] loading baseline model...')
    baseline_model = load_model(Path(args.baseline_checkpoint), 'baseline', device)

    print('[visual] loading attention model...')
    attention_model = load_model(Path(args.attention_checkpoint), 'attention', device)

    # load dataset
    print('[visual] loading dataset...')
    dataset = UnpairedMRIDataset25D(
        domain_a_dir=args.brats_dir,
        domain_b_dir=args.upenn_dir,
        cache_in_memory=False
    )

    # select random samples
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f'[visual] generating {len(indices)} visual examples...')

    for i, idx in enumerate(tqdm(indices, desc='generating visuals')):
        sample = dataset[idx]
        real_A = sample['A'].unsqueeze(0)
        real_B = sample['B'].unsqueeze(0)

        # generate translations
        baseline_outputs = generate_translations(baseline_model, real_A, real_B, device)
        attention_outputs = generate_translations(attention_model, real_A, real_B, device)

        # create figures for each modality
        for modality in ['T1', 'T1CE', 'T2', 'FLAIR']:
            output_path = output_dir / f'visual_sample_{i:02d}_{modality}'
            create_translation_figure(
                baseline_outputs, attention_outputs, output_path, idx, modality
            )

        # create multi-modality figure
        output_path = output_dir / f'visual_sample_{i:02d}_all_modalities'
        create_multimodality_figure(
            baseline_outputs, attention_outputs, output_path, idx
        )

    print(f'[visual] saved {len(indices) * 5} figures to {output_dir}')


if __name__ == '__main__':
    main()
