"""
run inference on selected test cases for qualitative visualization

loads trained sa-cyclegan-2.5d model and generates translations
for best/worst/median/interesting cases. saves numpy arrays for
figure generation.

usage:
    python run_inference.py --cases case_ids.json --checkpoint path/to/checkpoint.pth
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
from torch.utils.data import DataLoader


def load_model(checkpoint_path: Path, device: str = 'cuda') -> CycleGAN25D:
    """load trained cyclegan model from checkpoint"""
    print(f"loading model from {checkpoint_path}")

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # fallback to default config
        print("warning: config not found in checkpoint, using default")
        from neuroscope.config.config import load_config
        config = load_config('neuroscope/config/experiments/sa_cyclegan_25d.yaml')

    # initialize model
    model = CycleGAN25D(config)
    model.to(device)

    # load weights
    if 'generator_A2B_state_dict' in checkpoint:
        model.G_A2B.load_state_dict(checkpoint['generator_A2B_state_dict'])
        model.G_B2A.load_state_dict(checkpoint['generator_B2A_state_dict'])
        print("loaded generator weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("loaded full model weights")
    else:
        raise ValueError("checkpoint format not recognized")

    model.eval()
    print("model loaded and set to eval mode")

    return model, config


def load_test_dataset(config, case_indices: List[int] = None) -> DataLoader:
    """load test dataset with optional case filtering"""
    print("loading test dataset...")

    # create test dataset
    test_dataset = BraTSDataset(
        config=config,
        split='test',
        augment=False
    )

    print(f"test dataset size: {len(test_dataset)}")

    # if case indices provided, filter dataset
    if case_indices is not None:
        print(f"filtering to {len(case_indices)} selected cases")
        # create subset using torch.utils.data.Subset
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, case_indices)

    # create dataloader (batch size 1 for qualitative inspection)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return test_loader


def run_inference_on_cases(
    model: CycleGAN25D,
    dataloader: DataLoader,
    device: str,
    case_info: Dict
) -> Dict[str, np.ndarray]:
    """
    run inference on selected cases

    returns dict with keys:
    - inputs_a: input from domain a
    - inputs_b: input from domain b
    - generated_b: a→b translations
    - generated_a: b→a translations
    - reconstructed_a: a→b→a cycle
    - reconstructed_b: b→a→b cycle
    """
    results = {
        'inputs_a': [],
        'inputs_b': [],
        'generated_b': [],
        'generated_a': [],
        'reconstructed_a': [],
        'reconstructed_b': [],
        'case_indices': case_info['indices'],
        'ssim_a2b': case_info.get('ssim_a2b', []),
        'psnr_a2b': case_info.get('psnr_a2b', [])
    }

    print(f"\nrunning inference on {len(dataloader)} cases...")

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

            # forward pass: a → b
            fake_b = model.G_A2B(real_a)

            # forward pass: b → a
            fake_a = model.G_B2A(real_b)

            # cycle reconstruction: a → b → a
            rec_a = model.G_B2A(fake_b)

            # cycle reconstruction: b → a → b
            rec_b = model.G_A2B(fake_a)

            # move to cpu and convert to numpy
            # note: for 2.5d, input is [B, 12, H, W], output is [B, 4, H, W]
            # we'll save the center slice (4 modalities) for visualization
            results['inputs_a'].append(real_a[0].cpu().numpy())  # [12, H, W]
            results['inputs_b'].append(real_b[0].cpu().numpy())  # [12, H, W]
            results['generated_b'].append(fake_b[0].cpu().numpy())  # [4, H, W]
            results['generated_a'].append(fake_a[0].cpu().numpy())  # [4, H, W]
            results['reconstructed_a'].append(rec_a[0].cpu().numpy())  # [4, H, W]
            results['reconstructed_b'].append(rec_b[0].cpu().numpy())  # [4, H, W]

            if (batch_idx + 1) % 5 == 0:
                print(f"  processed {batch_idx + 1}/{len(dataloader)} cases")

    # convert lists to numpy arrays
    for key in ['inputs_a', 'inputs_b', 'generated_b', 'generated_a',
                'reconstructed_a', 'reconstructed_b']:
        results[key] = np.stack(results[key], axis=0)

    print(f"\ninference complete. shapes:")
    print(f"  inputs_a: {results['inputs_a'].shape}")
    print(f"  generated_b: {results['generated_b'].shape}")
    print(f"  reconstructed_a: {results['reconstructed_a'].shape}")

    return results


def save_inference_results(results: Dict, output_dir: Path, category: str):
    """save inference results as numpy arrays"""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'inference_{category}.npz'

    np.savez_compressed(
        output_file,
        inputs_a=results['inputs_a'],
        inputs_b=results['inputs_b'],
        generated_b=results['generated_b'],
        generated_a=results['generated_a'],
        reconstructed_a=results['reconstructed_a'],
        reconstructed_b=results['reconstructed_b'],
        case_indices=results['case_indices'],
        ssim_a2b=results['ssim_a2b'],
        psnr_a2b=results['psnr_a2b']
    )

    print(f"saved {category} results to {output_file}")
    print(f"  file size: {output_file.stat().st_size / 1024 / 1024:.2f} mb")


def main():
    parser = argparse.ArgumentParser(description='run inference on selected cases')
    parser.add_argument('--cases', type=str,
                       default='tools/inference/case_ids.json',
                       help='path to case selection json')
    parser.add_argument('--checkpoint', type=str,
                       default='experiments/sa_cyclegan_25d_rtx6000_resume_20260108_002543/checkpoints/checkpoint_best.pth',
                       help='path to model checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='results/inference',
                       help='output directory for inference results')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device to run inference on')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['best', 'worst', 'median', 'interesting', 'random'],
                       help='which case categories to process')

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

    print(f"metadata:")
    print(f"  test samples: {cases['metadata']['n_samples']}")
    print(f"  checkpoint: {cases['metadata']['checkpoint']}")
    print(f"  checkpoint epoch: {cases['metadata']['checkpoint_epoch']}")

    # load model
    model, config = load_model(checkpoint_path, device=args.device)

    # process each category
    for category in args.categories:
        if category not in cases:
            print(f"\nwarning: category '{category}' not found in case selections, skipping")
            continue

        case_info = cases[category]
        n_cases = len(case_info['indices'])

        print(f"\n{'='*60}")
        print(f"processing {category} cases")
        print(f"{'='*60}")
        print(f"description: {case_info['description']}")
        print(f"number of cases: {n_cases}")

        # load test dataset filtered to these cases
        test_loader = load_test_dataset(config, case_info['indices'])

        # run inference
        results = run_inference_on_cases(
            model, test_loader, args.device, case_info
        )

        # save results
        save_inference_results(results, output_dir, category)

    print(f"\n{'='*60}")
    print("inference complete for all categories")
    print(f"{'='*60}")
    print(f"results saved to: {output_dir}")
    print("\nnext steps:")
    print("1. extract attention: python tools/inference/extract_attention.py")
    print("2. generate figures: python tools/inference/generate_qualitative_figures.py")


if __name__ == '__main__':
    main()
