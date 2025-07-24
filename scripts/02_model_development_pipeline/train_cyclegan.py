import os
import sys
import argparse
import logging
import itertools
import json
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from neuroscope_dataset_loader import get_dataloaders
import seaborn as sns
import matplotlib as mpl
import logging

# ─── Seaborn + Times New Roman setup ──────────────────────────────
sns.set_theme(style="whitegrid")
mpl.rcParams['font.family']      = 'serif'
mpl.rcParams['font.serif']       = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'    # if you use math in labels

# ─── Silence Matplotlib findfont DEBUG spam ───────────────────────
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)



def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
# at the top of train.py, right after configure_logging():
logging.getLogger('torch').setLevel(logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.INFO)  # you’ve already silenced findfont


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, n_residual=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_feat, out_feat = 64, 128
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat * 2
        for _ in range(n_residual):
            model += [ResidualBlock(in_feat)]
        out_feat = in_feat // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            in_feat, out_feat = out_feat, out_feat // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_ch, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        def d_layer(in_f, out_f, stride=2, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *d_layer(in_ch, 64, normalize=False),
            *d_layer(64, 128),
            *d_layer(128, 256),
            *d_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


def sample_images(step, G_A2B, G_B2A, loaders, output_dir, tb_writer=None):
    real_A = next(iter(loaders['train'])).to(device)
    fake_B = G_A2B(real_A)
    real_B = next(iter(loaders['train'])).to(device)
    fake_A = G_B2A(real_B)

    imgs = torch.cat((real_A, fake_B, real_B, fake_A), 0)
    grid = make_grid(imgs, nrow=4, normalize=True)
    save_image(grid, os.path.join(output_dir, f"sample_{step}.png"))
    if tb_writer:
        tb_writer.add_image('samples', grid, step)


def plot_loss_graph(loss_history, save_path, tb_writer=None):
    plt.figure(figsize=(10, 6))
    for key, values in loss_history.items():
        plt.plot(values, label=key)
    plt.title("Training Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(save_path, 'loss_curves.png')
    plt.savefig(out_png)
    plt.close()
    logging.info(f"Saved loss curves to {out_png}")
    if tb_writer:
        for key, values in loss_history.items():
            tb_writer.add_scalars('losses', {key: values[-1]}, len(values))


def train(args):
    configure_logging()
    # Ultra‑verbose PyTorch logs
    logging.getLogger('torch').setLevel(logging.DEBUG)

    logging.info("Starting CycleGAN training")
    tb_writer = SummaryWriter(log_dir=os.path.expanduser('~/Downloads/neuroscope/runs'))
    logging.debug("TensorBoard writer initialized")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Data
    loaders = get_dataloaders(args.data_root, args.meta_json,
                              args.batch_size, args.num_workers)
    train_loader = loaders['train']
    logging.info(f"Loaded train loader with {len(train_loader)} batches")

    # Debug raw inputs
    raw_A = next(iter(train_loader)).to(device)
    debug_path = os.path.join(args.sample_dir, "debug_raw_A.png")
    save_image(raw_A, debug_path, nrow=4, normalize=False)
    logging.info(f"Saved raw input batch to {debug_path} "
                 f"(min={raw_A.min():.4f}, max={raw_A.max():.4f})")

    # Model init
    G_A2B = ResNetGenerator().to(device)
    G_B2A = ResNetGenerator().to(device)
    D_A   = PatchDiscriminator().to(device)
    D_B   = PatchDiscriminator().to(device)
    for net in (G_A2B, G_B2A, D_A, D_B):
        net.apply(weights_init_normal)

    # Quick model summary
    from torchinfo import summary
    summary_str = summary(G_A2B, input_size=(1,4,256,256), device=device, verbose=0).__str__()
    with open(os.path.join(args.sample_dir, "model_summary.txt"), "w") as f:
        f.write(summary_str)
    logging.info("Saved model summary to model_summary.txt")

    # Losses & optimizers
    L_GAN   = nn.MSELoss()
    L_cycle = nn.L1Loss()
    L_id    = nn.L1Loss()
    opt_G   = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                         lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # LR schedulers with safe denominator
    decay_span = args.n_epochs - args.decay_epoch
    if decay_span <= 0:
        logging.warning(f"decay_epoch ({args.decay_epoch}) >= n_epochs ({args.n_epochs}); "
                        "forcing decay over final epoch only.")
    denom = max(1, decay_span)
        # ─── Safe LR scheduler setup ───────────────────────────────
    # ensure decay_epoch < n_epochs
    if args.decay_epoch >= args.n_epochs:
        logging.warning(
            f"decay_epoch ({args.decay_epoch}) >= n_epochs ({args.n_epochs}); "
            "clamping decay_epoch to n_epochs - 1"
        )
        args.decay_epoch = args.n_epochs - 1

    decay_span = args.n_epochs - args.decay_epoch  # guaranteed ≥ 1 now
    lambda_lr  = lambda e: 1 - max(0, e - args.decay_epoch) / decay_span

    sched_G   = optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda=lambda_lr)
    sched_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=lambda_lr)
    sched_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=lambda_lr)


    loss_history = {"G": [], "D_A": [], "D_B": [], "Cycle": [], "Id": []}
    total_start = datetime.now()

    # Training loop
    for epoch in range(1, args.n_epochs + 1):
        epoch_start = datetime.now()
        logging.info(f"Epoch {epoch}/{args.n_epochs} started")

        for i, real_A in enumerate(train_loader, 1):
            real_B = next(iter(train_loader))
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Labels
            pred_shape = D_A(real_A).shape
            valid      = torch.ones(pred_shape,  device=device)
            fake_label = torch.zeros(pred_shape, device=device)

            # Generator step
            opt_G.zero_grad()
            loss_id_A    = L_id(G_B2A(real_A), real_A) * args.lambda_identity
            loss_id_B    = L_id(G_A2B(real_B), real_B) * args.lambda_identity
            fake_B       = G_A2B(real_A)
            loss_GAN_A2B = L_GAN(D_B(fake_B), valid)
            fake_A       = G_B2A(real_B)
            loss_GAN_B2A = L_GAN(D_A(fake_A), valid)
            recov_A      = G_B2A(fake_B)
            recov_B      = G_A2B(fake_A)
            loss_cycle   = (L_cycle(recov_A, real_A) + L_cycle(recov_B, real_B)) * args.lambda_cycle

            loss_G = loss_id_A + loss_id_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle
            loss_G.backward()
            opt_G.step()

            # Log fake_B range once
            if i == 1:
                logging.info(f"Epoch {epoch} fake_B range: {fake_B.min():.4f}–{fake_B.max():.4f}")

            # Discriminator A
            opt_D_A.zero_grad()
            loss_D_A = (L_GAN(D_A(real_A), valid) +
                        L_GAN(D_A(fake_A.detach()), fake_label)) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # Discriminator B
            opt_D_B.zero_grad()
            loss_D_B = (L_GAN(D_B(real_B), valid) +
                        L_GAN(D_B(fake_B.detach()), fake_label)) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            # Record & TB
            loss_history["G"].append(loss_G.item())
            loss_history["D_A"].append(loss_D_A.item())
            loss_history["D_B"].append(loss_D_B.item())
            loss_history["Cycle"].append(loss_cycle.item())
            loss_history["Id"].append((loss_id_A.item() + loss_id_B.item()) / 2)

            step = (epoch - 1) * len(train_loader) + i
            tb_writer.add_scalar('Loss/G', loss_G.item(), step)
            tb_writer.add_scalar('Loss/D_A', loss_D_A.item(), step)
            tb_writer.add_scalar('Loss/D_B', loss_D_B.item(), step)
            tb_writer.add_scalar('Loss/Cycle', loss_cycle.item(), step)
            tb_writer.add_scalar('Loss/Identity',
                                 (loss_id_A.item() + loss_id_B.item()) / 2, step)

            if step % 500 == 0:
                for name, param in G_A2B.named_parameters():
                    tb_writer.add_histogram(f'Weights/G_A2B/{name}', param, step)

            if i % args.log_interval == 0:
                logging.debug(f"Batch {i}/{len(train_loader)} | "
                              f"Loss_G={loss_G:.4f} Loss_D_A={loss_D_A:.4f} "
                              f"Loss_D_B={loss_D_B:.4f} Cycle={loss_cycle:.4f}")

            if step % args.sample_interval == 0:
                sample_images(step, G_A2B, G_B2A, loaders, args.sample_dir, tb_writer)

        # End of epoch
        sched_G.step(); sched_D_A.step(); sched_D_B.step()
        epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f} min")
        tb_writer.add_scalar('Time/Epoch', epoch_time, epoch)

        if epoch % args.checkpoint_interval == 0:
            # Save state_dicts
            torch.save(G_A2B.state_dict(),
                       os.path.join(args.checkpoint_dir, f"G_A2B_{epoch}.pth"))
            torch.save(G_B2A.state_dict(),
                       os.path.join(args.checkpoint_dir, f"G_B2A_{epoch}.pth"))
            torch.save(D_A.state_dict(),
                       os.path.join(args.checkpoint_dir, f"D_A_{epoch}.pth"))
            torch.save(D_B.state_dict(),
                       os.path.join(args.checkpoint_dir, f"D_B_{epoch}.pth"))
            logging.info(f"Saved state_dicts at epoch {epoch}")

            # Save full checkpoint
            ckpt = {
                'epoch':         epoch,
                'G_A2B_state':   G_A2B.state_dict(),
                'G_B2A_state':   G_B2A.state_dict(),
                'D_A_state':     D_A.state_dict(),
                'D_B_state':     D_B.state_dict(),
                'opt_G_state':   opt_G.state_dict(),
                'opt_D_A_state': opt_D_A.state_dict(),
                'opt_D_B_state': opt_D_B.state_dict(),
            }
            torch.save(ckpt,
                       os.path.join(args.checkpoint_dir, f"full_models_epoch_{epoch}.pt"))
            logging.info(f"Saved full models at epoch {epoch}")

    # Wrap‑up
    loss_log_path = os.path.join(args.sample_dir, 'training_loss_log.json')
    with open(loss_log_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    logging.info(f"Saved training loss log to {loss_log_path}")

    plot_loss_graph(loss_history, args.sample_dir, tb_writer)
    logging.info("Training complete.")

    # Final full checkpoint
    final_ckpt = {
        'epoch':          args.n_epochs,
        'G_A2B_state':    G_A2B.state_dict(),
        'G_B2A_state':    G_B2A.state_dict(),
        'D_A_state':      D_A.state_dict(),
        'D_B_state':      D_B.state_dict(),
        'opt_G_state':    opt_G.state_dict(),
        'opt_D_A_state':  opt_D_A.state_dict(),
        'opt_D_B_state':  opt_D_B.state_dict(),
    }
    torch.save(final_ckpt,
               os.path.join(args.checkpoint_dir, "full_models_final.pt"))
    logging.info("✔ Saved final full-model checkpoint to full_models_final.pt")

    total_time = (datetime.now() - total_start).total_seconds() / 60
    logging.info(f"Total training time: {total_time:.2f} minutes")
    tb_writer.add_scalar('Time/Total', total_time, 0)
    tb_writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/Volumes/USB DRIVE/neuroscope/data/preprocessed')
    parser.add_argument('--meta_json', type=str,
                        default=os.path.expanduser('~/Downloads/neuroscope/scripts/neuroscope_dataset_metadata_splits.json'))
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--sample_interval', type=int, default=200)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str,
                        default=os.path.expanduser('~/Downloads/neuroscope/checkpoints'))
    parser.add_argument('--sample_dir', type=str,
                        default=os.path.expanduser('~/Downloads/neuroscope/samples'))
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    configure_logging()
    logging.info(f"Using device: {device}")
    train(args)
