# visualize_cyclegan_training.py

import os
import json
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchinfo import summary
from torch.utils.data import DataLoader
from neuroscope_dataset_loader import get_cycle_domain_loaders
from train_cyclegan import ResNetGenerator


# -----------------------------
# Utility and Logging Setup
# -----------------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_model_weights(generator, weight_path):
    generator.load_state_dict(torch.load(weight_path, map_location='cpu'))
    generator.eval()
    return generator


def save_figure(fig, output_dir, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# Visualization Functions
# -----------------------------


def plot_loss_curves(log_file_path, output_dir):
    with open(log_file_path, 'r') as f:
        logs = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(logs['loss_G'], label='Generator')
    ax.plot(logs['loss_D_A'], label='Discriminator A')
    ax.plot(logs['loss_D_B'], label='Discriminator B')
    ax.set_title('Loss Curves')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    save_figure(fig, output_dir, 'loss_curves')


def model_summary_visual(model_class, name, output_dir):
    dummy_input = torch.randn(1, 4, 240, 240)
    summary_str = str(summary(model_class(), input_data=dummy_input))
    with open(os.path.join(output_dir, f"{name}_summary.txt"), 'w') as f:
        f.write(summary_str)


def generate_sample_grid(real, fake, rec, output_dir, prefix):
    stacked = torch.cat([real, fake, rec], dim=0)
    grid = vutils.make_grid(stacked, nrow=real.size(0), normalize=True)
    vutils.save_image(grid, os.path.join(output_dir, f"{prefix}_translation_grid.png"))


def compute_ssim_psnr(real, fake):
    real_np = real.cpu().numpy().squeeze()
    fake_np = fake.cpu().numpy().squeeze()
    return ssim(real_np, fake_np, data_range=1), psnr(real_np, fake_np, data_range=1)


def evaluate_metrics(loader, G, device, output_dir):
    ssim_scores, psnr_scores = [], []
    to_pil = T.ToPILImage()
    to_tensor = T.ToTensor()

    for idx, real in enumerate(loader):
        if idx >= 20:
            break
        real = real.to(device)
        fake = G(real).detach()

        for i in range(min(4, real.size(0))):
            s, p = compute_ssim_psnr(real[i], fake[i])
            ssim_scores.append(s)
            psnr_scores.append(p)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(ssim_scores, bins=20, color='skyblue')
    ax[0].set_title('SSIM Distribution')
    ax[1].hist(psnr_scores, bins=20, color='salmon')
    ax[1].set_title('PSNR Distribution')
    save_figure(fig, output_dir, 'metric_distributions')


# -----------------------------
# Main
# -----------------------------

if __name__ == '__main__':
    setup_logging()

    # -----------------------------
    # Config
    # -----------------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT = os.path.expanduser('~/Downloads/neuroscope/checkpoints/G_A2B_100.pth')
    DATA_ROOT = '/Downloads/neuroscope/data/preprocessed'
    META = os.path.expanduser('~/Downloads/neuroscope/scripts/neuroscope_dataset_metadata_splits.json')
    OUTPUT_DIR = os.path.expanduser('~/Downloads/neuroscope/figures')
    LOG_PATH = os.path.expanduser('~/Downloads/neuroscope/scripts/training_loss_log.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------
    # Load Model
    # -----------------------------
    logging.info("Loading trained generator...")
    G = load_model_weights(ResNetGenerator().to(DEVICE), CHECKPOINT)

    # -----------------------------
    # Dataset
    # -----------------------------
    logging.info("Loading dataset...")
    loaders = get_cycle_domain_loaders(DATA_ROOT, META, batch_size=4, num_workers=0)
    val_loader = loaders['val_A']

    # -----------------------------
    # Figures
    # -----------------------------
    logging.info("Generating architecture summary...")
    model_summary_visual(ResNetGenerator, 'G_A2B_architecture', OUTPUT_DIR)

    logging.info("Generating metric evaluations...")
    evaluate_metrics(val_loader, G, DEVICE, OUTPUT_DIR)

    logging.info("Generating loss plots...")
    plot_loss_curves(LOG_PATH, OUTPUT_DIR)

    logging.info("Saving qualitative outputs...")
    sample_batch = next(iter(val_loader)).to(DEVICE)
    with torch.no_grad():
        fake = G(sample_batch)
        rec = G(fake)
    generate_sample_grid(sample_batch, fake, rec, OUTPUT_DIR, 'val_grid')

    logging.info("All visualizations saved successfully to %s", OUTPUT_DIR)
