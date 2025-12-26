import argparse
import logging
import os
from pathlib import Path
import sys
import json

import torch
import torchvision.utils as vutils
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from neuroscope_dataset_loader import get_cycle_domain_loaders
from train_cyclegan import ResNetGenerator


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def compute_metrics(real: torch.Tensor, fake: torch.Tensor):
    real_np = real.detach().cpu().numpy()
    fake_np = fake.detach().cpu().numpy()
    # scale from [-1,1] to [0,1]
    real_np = (real_np + 1.0) / 2.0
    fake_np = (fake_np + 1.0) / 2.0
    # per-channel average
    ssim_vals, psnr_vals = [], []
    for c in range(real_np.shape[0]):
        ssim_vals.append(ssim(real_np[c], fake_np[c], data_range=1.0))
        psnr_vals.append(psnr(real_np[c], fake_np[c], data_range=1.0))
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


def evaluate(generator_ckpt: str, data_root: str, meta_json: str, output_dir: str, split: str = 'val', max_batches: int = 20, batch_size: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    G = ResNetGenerator().to(device)
    state = torch.load(generator_ckpt, map_location=device)
    try:
        G.load_state_dict(state)
    except Exception:
        # maybe a full checkpoint dict
        G.load_state_dict(state.get('G_A2B_state', state))
    G.eval()

    loaders = get_cycle_domain_loaders(data_root, meta_json, batch_size=batch_size, num_workers=0, slices_per_subject=4)
    key = f'{split}_A'
    if key not in loaders:
        raise RuntimeError(f'missing loader {key}; ensure split exists')
    dl = loaders[key]

    ssim_scores, psnr_scores = [], []
    with torch.no_grad():
        for bidx, real in enumerate(dl, 1):
            real = real.to(device)
            fake = G(real)
            for i in range(min(real.size(0), 4)):
                s, p = compute_metrics(real[i], fake[i])
                ssim_scores.append(s)
                psnr_scores.append(p)
            if bidx == 1:
                grid = vutils.make_grid(torch.cat([(real+1)/2.0, (fake+1)/2.0], dim=0), nrow=real.size(0), normalize=True)
                vutils.save_image(grid, os.path.join(output_dir, f'samples_{split}.png'))
            if bidx >= max_batches:
                break

    report = {
        'split': split,
        'n': len(ssim_scores),
        'ssim_mean': float(np.mean(ssim_scores)) if ssim_scores else None,
        'psnr_mean': float(np.mean(psnr_scores)) if psnr_scores else None,
    }
    with open(os.path.join(output_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    logging.info('wrote evaluation report to %s', os.path.join(output_dir, 'evaluation_report.json'))


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate CycleGAN generator with SSIM/PSNR and samples')
    ap.add_argument('--generator_ckpt', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--meta_json', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--split', type=str, default='val')
    ap.add_argument('--max_batches', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=4)
    return ap.parse_args()


def main():
    setup_logging()
    args = parse_args()
    evaluate(args.generator_ckpt, args.data_root, args.meta_json, args.output_dir, args.split, args.max_batches, args.batch_size)


if __name__ == '__main__':
    main()
