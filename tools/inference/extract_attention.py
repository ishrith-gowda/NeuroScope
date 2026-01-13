"""
extract attention maps from trained model during inference

uses pytorch forward hooks to capture attention weights from:
- cbam channel attention (per-channel weights)
- cbam spatial attention (spatial heatmaps)
- self-attention (multi-head attention weights)

usage:
    python extract_attention.py --cases case_ids.json --checkpoint path/to/checkpoint.pth
"""

import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neuroscope.models.cyclegan_25d import CycleGAN25D
from neuroscope.data.brats_dataset import BraTSDataset
from torch.utils.data import DataLoader, Subset


class AttentionExtractor:
    """
    extracts attention weights using forward hooks

    captures:
    - cbam channel attention weights
    - cbam spatial attention maps
    - self-attention weights (if present)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self):
        """register forward hooks on attention modules"""
        # find all cbam and self-attention modules
        for name, module in self.model.named_modules():
            # cbam channel attention
            if 'channel_attention' in name.lower() or 'cbam' in name.lower():
                hook = module.register_forward_hook(
                    self._get_channel_attention_hook(name)
                )
                self.hooks.append(hook)
                print(f"registered hook on {name}")

            # cbam spatial attention
            if 'spatial_attention' in name.lower() or 'spatial' in name.lower():
                hook = module.register_forward_hook(
                    self._get_spatial_attention_hook(name)
                )
                self.hooks.append(hook)
                print(f"registered hook on {name}")

            # self-attention
            if 'self_attention' in name.lower() or 'selfattention' in name.lower():
                hook = module.register_forward_hook(
                    self._get_self_attention_hook(name)
                )
                self.hooks.append(hook)
                print(f"registered hook on {name}")

    def _get_channel_attention_hook(self, name: str):
        """create hook for channel attention"""
        def hook(module, input, output):
            # channel attention typically outputs [B, C, 1, 1]
            if isinstance(output, torch.Tensor):
                self.attention_maps[f'{name}_output'] = output.detach().cpu()
        return hook

    def _get_spatial_attention_hook(self, name: str):
        """create hook for spatial attention"""
        def hook(module, input, output):
            # spatial attention typically outputs [B, 1, H, W]
            if isinstance(output, torch.Tensor):
                self.attention_maps[f'{name}_output'] = output.detach().cpu()
        return hook

    def _get_self_attention_hook(self, name: str):
        """create hook for self-attention"""
        def hook(module, input, output):
            # self-attention output can vary by implementation
            # typically [B, C, H, W] or attention weights [B, heads, HW, HW]
            if isinstance(output, tuple):
                # some implementations return (output, attention_weights)
                self.attention_maps[f'{name}_output'] = output[0].detach().cpu()
                if len(output) > 1:
                    self.attention_maps[f'{name}_weights'] = output[1].detach().cpu()
            elif isinstance(output, torch.Tensor):
                self.attention_maps[f'{name}_output'] = output.detach().cpu()
        return hook

    def clear(self):
        """clear stored attention maps"""
        self.attention_maps = {}

    def remove_hooks(self):
        """remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_summary(self) -> Dict:
        """get summary of captured attention maps"""
        summary = {}
        for name, tensor in self.attention_maps.items():
            summary[name] = {
                'shape': tuple(tensor.shape),
                'mean': float(tensor.mean()),
                'std': float(tensor.std()),
                'min': float(tensor.min()),
                'max': float(tensor.max())
            }
        return summary


def load_model(checkpoint_path: Path, device: str = 'cuda') -> CycleGAN25D:
    """load trained cyclegan model from checkpoint"""
    print(f"loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        from neuroscope.config.config import load_config
        config = load_config('neuroscope/config/experiments/sa_cyclegan_25d.yaml')

    # initialize model
    model = CycleGAN25D(config)
    model.to(device)

    # load weights
    if 'generator_A2B_state_dict' in checkpoint:
        model.G_A2B.load_state_dict(checkpoint['generator_A2B_state_dict'])
        model.G_B2A.load_state_dict(checkpoint['generator_B2A_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("checkpoint format not recognized")

    model.eval()
    print("model loaded and set to eval mode")

    return model, config


def load_test_dataset(config, case_indices: List[int]) -> DataLoader:
    """load test dataset with case filtering"""
    print("loading test dataset...")

    test_dataset = BraTSDataset(
        config=config,
        split='test',
        augment=False
    )

    print(f"test dataset size: {len(test_dataset)}")
    print(f"filtering to {len(case_indices)} selected cases")

    # create subset
    test_dataset = Subset(test_dataset, case_indices)

    # create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader


def extract_attention_for_cases(
    model: CycleGAN25D,
    dataloader: DataLoader,
    device: str,
    case_info: Dict
) -> Dict:
    """
    extract attention maps for selected cases

    returns dict with attention maps for each case
    """
    # create attention extractors for both generators
    extractor_a2b = AttentionExtractor(model.G_A2B)
    extractor_b2a = AttentionExtractor(model.G_B2A)

    print("\nregistering hooks on generator a2b...")
    extractor_a2b.register_hooks()

    print("\nregistering hooks on generator b2a...")
    extractor_b2a.register_hooks()

    if not extractor_a2b.hooks and not extractor_b2a.hooks:
        print("\nwarning: no attention modules found in model!")
        print("this may be a baseline model without attention mechanisms.")
        return None

    results = {
        'attention_a2b': [],
        'attention_b2a': [],
        'case_indices': case_info['indices']
    }

    print(f"\nextracting attention maps from {len(dataloader)} cases...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # extract data
            if isinstance(batch, dict):
                real_a = batch['domain_a'].to(device)
                real_b = batch['domain_b'].to(device)
            else:
                real_a, real_b = batch
                real_a = real_a.to(device)
                real_b = real_b.to(device)

            # forward pass a2b (captures attention via hooks)
            extractor_a2b.clear()
            _ = model.G_A2B(real_a)
            attention_a2b = {k: v.numpy() for k, v in extractor_a2b.attention_maps.items()}

            # forward pass b2a
            extractor_b2a.clear()
            _ = model.G_B2A(real_b)
            attention_b2a = {k: v.numpy() for k, v in extractor_b2a.attention_maps.items()}

            results['attention_a2b'].append(attention_a2b)
            results['attention_b2a'].append(attention_b2a)

            if (batch_idx + 1) % 5 == 0:
                print(f"  processed {batch_idx + 1}/{len(dataloader)} cases")

    # remove hooks
    extractor_a2b.remove_hooks()
    extractor_b2a.remove_hooks()

    print(f"\nattention extraction complete")
    print(f"  captured {len(results['attention_a2b'])} cases")

    # print summary of first case
    if results['attention_a2b']:
        print("\nattention map summary (first case, a2b):")
        first_case = results['attention_a2b'][0]
        for name, tensor in first_case.items():
            print(f"  {name}: shape={tensor.shape}, "
                  f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    return results


def save_attention_results(results: Dict, output_dir: Path, category: str):
    """save attention extraction results"""
    if results is None:
        print("no attention maps to save")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'attention_{category}.npz'

    # flatten nested structure for numpy savez
    save_dict = {
        'case_indices': results['case_indices']
    }

    # save each case's attention maps
    for case_idx, (attn_a2b, attn_b2a) in enumerate(
        zip(results['attention_a2b'], results['attention_b2a'])
    ):
        for name, tensor in attn_a2b.items():
            key = f'case{case_idx}_a2b_{name}'
            save_dict[key] = tensor

        for name, tensor in attn_b2a.items():
            key = f'case{case_idx}_b2a_{name}'
            save_dict[key] = tensor

    np.savez_compressed(output_file, **save_dict)

    print(f"saved {category} attention maps to {output_file}")
    print(f"  file size: {output_file.stat().st_size / 1024 / 1024:.2f} mb")


def main():
    parser = argparse.ArgumentParser(description='extract attention maps from model')
    parser.add_argument('--cases', type=str,
                       default='tools/inference/case_ids.json',
                       help='path to case selection json')
    parser.add_argument('--checkpoint', type=str,
                       default='experiments/sa_cyclegan_25d_rtx6000_resume_20260108_002543/checkpoints/checkpoint_best.pth',
                       help='path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='results/inference',
                       help='output directory for attention maps')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to run extraction on')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['best', 'worst', 'median'],
                       help='which case categories to process (subset for attention)')

    args = parser.parse_args()

    # setup paths
    project_root = Path(__file__).parent.parent.parent
    cases_path = project_root / args.cases
    checkpoint_path = project_root / args.checkpoint
    output_dir = project_root / args.output_dir

    # check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("warning: cuda requested but not available, using cpu")
        args.device = 'cpu'

    print(f"using device: {args.device}")

    # load case selections
    print(f"\nloading case selections from {cases_path}")
    with open(cases_path, 'r') as f:
        cases = json.load(f)

    # load model
    model, config = load_model(checkpoint_path, device=args.device)

    # process each category
    for category in args.categories:
        if category not in cases:
            print(f"\nwarning: category '{category}' not found, skipping")
            continue

        case_info = cases[category]
        n_cases = len(case_info['indices'])

        print(f"\n{'='*60}")
        print(f"extracting attention for {category} cases")
        print(f"{'='*60}")
        print(f"description: {case_info['description']}")
        print(f"number of cases: {n_cases}")

        # load test dataset
        test_loader = load_test_dataset(config, case_info['indices'])

        # extract attention
        results = extract_attention_for_cases(
            model, test_loader, args.device, case_info
        )

        # save results
        save_attention_results(results, output_dir, category)

    print(f"\n{'='*60}")
    print("attention extraction complete")
    print(f"{'='*60}")
    print(f"results saved to: {output_dir}")
    print("\nnext step:")
    print("  generate figures: python tools/inference/generate_qualitative_figures.py")


if __name__ == '__main__':
    main()
