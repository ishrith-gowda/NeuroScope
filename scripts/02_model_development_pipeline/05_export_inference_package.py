import argparse
import logging
import os
import torch
from pathlib import Path

from train_cyclegan import ResNetGenerator


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def export_full_model(ckpt_path: str, export_dir: str, name: str = 'G_A2B'):
    os.makedirs(export_dir, exist_ok=True)
    device = torch.device('cpu')
    G = ResNetGenerator()
    state = torch.load(ckpt_path, map_location=device)
    try:
        G.load_state_dict(state)
    except Exception:
        G.load_state_dict(state.get('G_A2B_state', state))
    obj = {'architecture': G, 'state_dict': G.state_dict()}
    out_path = os.path.join(export_dir, f'full_{name}.pt')
    torch.save(obj, out_path)
    logging.info('exported full model to %s', out_path)


def write_inference_readme(export_dir: str):
    txt = """
Minimal inference example (PyTorch):

import torch
bundle = torch.load('full_G_A2B.pt', map_location='cpu')
G = bundle['architecture']
G.load_state_dict(bundle['state_dict'])
G.eval()
# x must be a tensor [N,4,H,W] scaled to [-1,1]
with torch.no_grad():
    y = G(x)
    y01 = (y + 1) / 2.0
    """.strip()
    with open(os.path.join(export_dir, 'INFERENCE.md'), 'w') as f:
        f.write(txt)


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description='Export full-model checkpoint for inference')
    ap.add_argument('--ckpt', type=str, required=True, help='Path to generator weights (.pth or full ckpt)')
    ap.add_argument('--export_dir', type=str, required=True)
    ap.add_argument('--name', type=str, default='G_A2B')
    return ap.parse_args()


def main():
    setup_logging()
    args = parse_args()
    export_full_model(args.ckpt, args.export_dir, args.name)
    write_inference_readme(args.export_dir)


if __name__ == '__main__':
    main()
