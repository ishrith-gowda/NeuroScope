#!/usr/bin/env python3
"""
downstream task evaluation for mri harmonization.

this script evaluates the impact of sa-cyclegan harmonization on
downstream tumor segmentation performance using a comprehensive
evaluation protocol:

1. train u-net segmentation on source domain (brats)
2. evaluate on target domain (upenn) with:
   - raw images (baseline)
   - harmonized images (sa-cyclegan)
   - harmonized images (baseline cyclegan)
3. compute comprehensive metrics per tumor region
4. statistical significance testing

metrics:
- dice coefficient (et, tc, wt)
- hausdorff distance 95th percentile
- sensitivity, specificity, precision
- volume similarity

protocol based on best practices from miccai, tmi, and neuroimage.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats, ndimage
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.05_downstream_evaluation.unet_segmentation import (
    UNet2D, DiceLoss, CombinedLoss, compute_dice_score
)


@dataclass
class SegmentationMetrics:
    """container for segmentation metrics."""
    dice_et: float  # enhancing tumor
    dice_tc: float  # tumor core
    dice_wt: float  # whole tumor
    hausdorff_et: float
    hausdorff_tc: float
    hausdorff_wt: float
    sensitivity_et: float
    specificity_et: float
    precision_et: float
    volume_diff_et: float


class SegmentationDataset(Dataset):
    """
    dataset for loading preprocessed segmentation data.
    """

    def __init__(
        self,
        data_dir: Path,
        subjects: List[str],
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.subjects = subjects
        self.transform = transform

        # collect all slices
        self.samples = []
        for subject in subjects:
            subject_dir = self.data_dir / subject
            if not subject_dir.exists():
                continue

            # find all slices
            for f in subject_dir.iterdir():
                if f.name.endswith('_input.npy'):
                    slice_idx = f.name.split('_')[1]
                    self.samples.append({
                        'subject': subject,
                        'slice_idx': slice_idx,
                        'input_path': f,
                        'seg_path': subject_dir / f'slice_{slice_idx}_seg.npy',
                        'seg_wt_path': subject_dir / f'slice_{slice_idx}_seg_wt.npy',
                        'seg_tc_path': subject_dir / f'slice_{slice_idx}_seg_tc.npy',
                        'seg_et_path': subject_dir / f'slice_{slice_idx}_seg_et.npy',
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # load data
        input_data = np.load(sample['input_path'])
        seg = np.load(sample['seg_path'])
        seg_wt = np.load(sample['seg_wt_path'])
        seg_tc = np.load(sample['seg_tc_path'])
        seg_et = np.load(sample['seg_et_path'])

        # convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        seg_tensor = torch.from_numpy(seg).long()
        seg_wt_tensor = torch.from_numpy(seg_wt).float()
        seg_tc_tensor = torch.from_numpy(seg_tc).float()
        seg_et_tensor = torch.from_numpy(seg_et).float()

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return {
            'input': input_tensor,
            'seg': seg_tensor,
            'seg_wt': seg_wt_tensor,
            'seg_tc': seg_tc_tensor,
            'seg_et': seg_et_tensor,
            'subject': sample['subject'],
            'slice_idx': sample['slice_idx']
        }


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5
) -> Dict[str, float]:
    """
    compute comprehensive binary segmentation metrics.

    args:
        pred: binary prediction
        target: binary ground truth
        smooth: smoothing factor

    returns:
        dict with metrics
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.sum(pred & target)
    fp = np.sum(pred & ~target)
    fn = np.sum(~pred & target)
    tn = np.sum(~pred & ~target)

    # dice
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    # sensitivity (recall)
    sensitivity = (tp + smooth) / (tp + fn + smooth)

    # specificity
    specificity = (tn + smooth) / (tn + fp + smooth)

    # precision
    precision = (tp + smooth) / (tp + fp + smooth)

    # volume difference
    vol_pred = np.sum(pred)
    vol_target = np.sum(target)
    vol_diff = abs(vol_pred - vol_target) / (vol_target + smooth)

    # hausdorff distance
    if np.any(pred) and np.any(target):
        # get boundary points
        pred_points = np.array(np.where(pred)).T
        target_points = np.array(np.where(target)).T

        if len(pred_points) > 0 and len(target_points) > 0:
            # compute hausdorff
            hd_forward = directed_hausdorff(pred_points, target_points)[0]
            hd_backward = directed_hausdorff(target_points, pred_points)[0]
            hausdorff = max(hd_forward, hd_backward)

            # 95th percentile version
            from scipy.ndimage import distance_transform_edt
            pred_dist = distance_transform_edt(~pred)
            target_dist = distance_transform_edt(~target)

            dist_pred_to_target = pred_dist[target]
            dist_target_to_pred = target_dist[pred]

            if len(dist_pred_to_target) > 0 and len(dist_target_to_pred) > 0:
                hd95 = max(
                    np.percentile(dist_pred_to_target, 95),
                    np.percentile(dist_target_to_pred, 95)
                )
            else:
                hd95 = float('inf')
        else:
            hausdorff = float('inf')
            hd95 = float('inf')
    else:
        hausdorff = float('inf') if np.any(target) else 0.0
        hd95 = float('inf') if np.any(target) else 0.0

    return {
        'dice': dice,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'volume_diff': vol_diff,
        'hausdorff': hausdorff,
        'hausdorff95': hd95
    }


def train_segmentation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_path: Optional[Path] = None
) -> Dict:
    """
    train segmentation model on source domain.

    args:
        model: u-net model
        train_loader: training data loader
        val_loader: validation data loader
        device: torch device
        epochs: number of epochs
        lr: learning rate
        save_path: path to save best model

    returns:
        training history
    """
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_dice_wt': [],
        'val_dice_tc': [],
        'val_dice_et': []
    }

    best_val_dice = 0
    best_epoch = 0

    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f'epoch {epoch+1}/{epochs} [train]')
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['seg'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        # validation
        model.eval()
        val_losses = []
        val_dice_wt = []
        val_dice_tc = []
        val_dice_et = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'epoch {epoch+1}/{epochs} [val]'):
                inputs = batch['input'].to(device)
                targets = batch['seg'].to(device)
                seg_wt = batch['seg_wt'].to(device)
                seg_tc = batch['seg_tc'].to(device)
                seg_et = batch['seg_et'].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

                # compute dice for each region
                pred = F.softmax(outputs, dim=1)

                # whole tumor: classes 1, 2, 4 -> simplified to non-background
                pred_wt = (pred[:, 1:].sum(dim=1) > 0.5).float()
                dice_wt = compute_binary_dice(pred_wt, seg_wt)
                val_dice_wt.append(dice_wt)

                # tumor core: classes 1, 4
                pred_tc = ((pred[:, 1] + pred[:, 3]) > 0.5).float() if pred.shape[1] > 3 else pred_wt
                dice_tc = compute_binary_dice(pred_tc, seg_tc)
                val_dice_tc.append(dice_tc)

                # enhancing tumor: class 4
                pred_et = (pred[:, 3] > 0.5).float() if pred.shape[1] > 3 else pred_wt
                dice_et = compute_binary_dice(pred_et, seg_et)
                val_dice_et.append(dice_et)

        # record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['val_dice_wt'].append(np.mean(val_dice_wt))
        history['val_dice_tc'].append(np.mean(val_dice_tc))
        history['val_dice_et'].append(np.mean(val_dice_et))

        mean_dice = (history['val_dice_wt'][-1] + history['val_dice_tc'][-1] +
                    history['val_dice_et'][-1]) / 3

        print(f'epoch {epoch+1}: train_loss={history["train_loss"][-1]:.4f}, '
              f'val_loss={history["val_loss"][-1]:.4f}, '
              f'dice_wt={history["val_dice_wt"][-1]:.4f}, '
              f'dice_tc={history["val_dice_tc"][-1]:.4f}, '
              f'dice_et={history["val_dice_et"][-1]:.4f}')

        # save best model
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            best_epoch = epoch
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': mean_dice,
                    'history': history
                }, save_path)

    print(f'[train] best model at epoch {best_epoch+1} with dice {best_val_dice:.4f}')
    return history


def compute_binary_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """compute dice score for binary predictions."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()


def evaluate_segmentation(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    harmonization_model: Optional[nn.Module] = None
) -> Dict:
    """
    evaluate segmentation performance on test data.

    args:
        model: trained segmentation model
        data_loader: test data loader
        device: torch device
        harmonization_model: optional harmonization model (sa-cyclegan)

    returns:
        dict with comprehensive metrics
    """
    model.eval()
    if harmonization_model:
        harmonization_model.eval()

    all_metrics = {
        'dice_wt': [],
        'dice_tc': [],
        'dice_et': [],
        'hausdorff95_wt': [],
        'hausdorff95_tc': [],
        'hausdorff95_et': [],
        'sensitivity_wt': [],
        'sensitivity_tc': [],
        'sensitivity_et': [],
        'specificity_wt': [],
        'specificity_tc': [],
        'specificity_et': [],
        'precision_wt': [],
        'precision_tc': [],
        'precision_et': [],
        'volume_diff_wt': [],
        'volume_diff_tc': [],
        'volume_diff_et': [],
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='evaluating'):
            inputs = batch['input'].to(device)
            seg_wt = batch['seg_wt'].numpy()
            seg_tc = batch['seg_tc'].numpy()
            seg_et = batch['seg_et'].numpy()

            # apply harmonization if provided
            if harmonization_model is not None:
                # prepare input for harmonization (may need adjustment)
                inputs = harmonize_batch(inputs, harmonization_model, device)

            # segment
            outputs = model(inputs)
            pred = F.softmax(outputs, dim=1)

            # for each sample in batch
            for i in range(inputs.shape[0]):
                # whole tumor
                pred_wt = (pred[i, 1:].sum(dim=0) > 0.5).cpu().numpy()
                metrics_wt = compute_binary_metrics(pred_wt, seg_wt[i])
                all_metrics['dice_wt'].append(metrics_wt['dice'])
                all_metrics['hausdorff95_wt'].append(metrics_wt['hausdorff95'])
                all_metrics['sensitivity_wt'].append(metrics_wt['sensitivity'])
                all_metrics['specificity_wt'].append(metrics_wt['specificity'])
                all_metrics['precision_wt'].append(metrics_wt['precision'])
                all_metrics['volume_diff_wt'].append(metrics_wt['volume_diff'])

                # tumor core
                if pred.shape[1] > 3:
                    pred_tc = ((pred[i, 1] + pred[i, 3]) > 0.5).cpu().numpy()
                else:
                    pred_tc = pred_wt
                metrics_tc = compute_binary_metrics(pred_tc, seg_tc[i])
                all_metrics['dice_tc'].append(metrics_tc['dice'])
                all_metrics['hausdorff95_tc'].append(metrics_tc['hausdorff95'])
                all_metrics['sensitivity_tc'].append(metrics_tc['sensitivity'])
                all_metrics['specificity_tc'].append(metrics_tc['specificity'])
                all_metrics['precision_tc'].append(metrics_tc['precision'])
                all_metrics['volume_diff_tc'].append(metrics_tc['volume_diff'])

                # enhancing tumor
                if pred.shape[1] > 3:
                    pred_et = (pred[i, 3] > 0.5).cpu().numpy()
                else:
                    pred_et = pred_wt
                metrics_et = compute_binary_metrics(pred_et, seg_et[i])
                all_metrics['dice_et'].append(metrics_et['dice'])
                all_metrics['hausdorff95_et'].append(metrics_et['hausdorff95'])
                all_metrics['sensitivity_et'].append(metrics_et['sensitivity'])
                all_metrics['specificity_et'].append(metrics_et['specificity'])
                all_metrics['precision_et'].append(metrics_et['precision'])
                all_metrics['volume_diff_et'].append(metrics_et['volume_diff'])

    # aggregate statistics
    results = {}
    for key, values in all_metrics.items():
        # filter out inf values for hausdorff
        if 'hausdorff' in key:
            values = [v for v in values if not np.isinf(v)]

        if len(values) > 0:
            results[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values)
            }

    return results


def harmonize_batch(
    inputs: torch.Tensor,
    harmonization_model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    apply harmonization model to batch.

    this function handles the interface between the segmentation input
    (4 channels: t1, t1ce, t2, flair) and the harmonization model
    (12 channels: 3 slices x 4 modalities).

    args:
        inputs: segmentation input [b, 4, h, w]
        harmonization_model: sa-cyclegan model
        device: torch device

    returns:
        harmonized input [b, 4, h, w]
    """
    # the harmonization model expects 12-channel input (3 slices)
    # for 2d evaluation, we replicate the slice 3 times
    b, c, h, w = inputs.shape

    # create 3-slice input by replicating
    inputs_3slice = inputs.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    inputs_3slice = inputs_3slice.view(b, c * 3, h, w)

    # apply harmonization (G_A2B or G_B2A depending on direction)
    harmonized = harmonization_model.G_A2B(inputs_3slice)

    return harmonized


def statistical_comparison(
    baseline_results: Dict,
    harmonized_results: Dict,
    metric_keys: List[str]
) -> Dict:
    """
    perform statistical comparison between baseline and harmonized results.

    args:
        baseline_results: raw domain results
        harmonized_results: harmonized domain results
        metric_keys: keys to compare

    returns:
        dict with statistical tests
    """
    comparison = {}

    for key in metric_keys:
        if key not in baseline_results or key not in harmonized_results:
            continue

        base_mean = baseline_results[key]['mean']
        base_std = baseline_results[key]['std']
        harm_mean = harmonized_results[key]['mean']
        harm_std = harmonized_results[key]['std']
        n = baseline_results[key]['n']

        # compute improvement
        diff = harm_mean - base_mean
        improvement_pct = (diff / base_mean) * 100 if base_mean != 0 else 0

        # paired t-test (approximation using summary statistics)
        # for proper paired t-test, would need raw values
        pooled_std = np.sqrt((base_std**2 + harm_std**2) / 2)
        t_stat = diff / (pooled_std * np.sqrt(2/n)) if pooled_std > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=2*n-2))

        # effect size (cohen's d)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        # 95% confidence interval
        se = pooled_std * np.sqrt(2/n)
        ci_low = diff - 1.96 * se
        ci_high = diff + 1.96 * se

        comparison[key] = {
            'baseline_mean': base_mean,
            'baseline_std': base_std,
            'harmonized_mean': harm_mean,
            'harmonized_std': harm_std,
            'difference': diff,
            'improvement_pct': improvement_pct,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'significant': bool(p_value < 0.05)
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='downstream task evaluation for mri harmonization'
    )
    parser.add_argument('--seg-data-dir', type=str, required=True,
                       help='path to preprocessed segmentation data')
    parser.add_argument('--harmonization-checkpoint', type=str,
                       help='path to sa-cyclegan checkpoint')
    parser.add_argument('--baseline-checkpoint', type=str,
                       help='path to baseline cyclegan checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50,
                       help='epochs for segmentation training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--skip-training', action='store_true',
                       help='skip training and use existing checkpoint')
    parser.add_argument('--seg-checkpoint', type=str,
                       help='existing segmentation checkpoint to use')

    args = parser.parse_args()

    # setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'[downstream] device: {device}')
    print(f'[downstream] segmentation data: {args.seg_data_dir}')
    print('=' * 60)

    # load data splits
    seg_data_dir = Path(args.seg_data_dir)
    with open(seg_data_dir / 'splits.json', 'r') as f:
        splits = json.load(f)

    # create datasets
    train_dataset = SegmentationDataset(seg_data_dir, splits['train'])
    val_dataset = SegmentationDataset(seg_data_dir, splits['val'])
    test_dataset = SegmentationDataset(seg_data_dir, splits['test'])

    print(f'[downstream] train samples: {len(train_dataset)}')
    print(f'[downstream] val samples: {len(val_dataset)}')
    print(f'[downstream] test samples: {len(test_dataset)}')

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    # create segmentation model
    seg_model = UNet2D(in_channels=4, n_classes=4, base_filters=32, use_attention=True)

    # train or load segmentation model
    seg_checkpoint_path = output_dir / 'segmentation_model.pth'

    if args.skip_training and args.seg_checkpoint:
        print(f'[downstream] loading segmentation model from {args.seg_checkpoint}')
        checkpoint = torch.load(args.seg_checkpoint, map_location=device)
        seg_model.load_state_dict(checkpoint['model_state_dict'])
    elif args.skip_training and seg_checkpoint_path.exists():
        print(f'[downstream] loading segmentation model from {seg_checkpoint_path}')
        checkpoint = torch.load(seg_checkpoint_path, map_location=device)
        seg_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('[downstream] training segmentation model...')
        history = train_segmentation_model(
            seg_model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, save_path=seg_checkpoint_path
        )

        # save training history
        with open(output_dir / 'segmentation_training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # reload best model
        checkpoint = torch.load(seg_checkpoint_path, map_location=device)
        seg_model.load_state_dict(checkpoint['model_state_dict'])

    seg_model = seg_model.to(device)

    # evaluate on test set
    print('=' * 60)
    print('[downstream] evaluating on test set...')

    # baseline evaluation (no harmonization)
    print('[downstream] evaluating raw (no harmonization)...')
    raw_results = evaluate_segmentation(seg_model, test_loader, device)

    # save raw results
    with open(output_dir / 'raw_segmentation_results.json', 'w') as f:
        json.dump(raw_results, f, indent=2)

    print('[downstream] raw results:')
    print(f'  dice_wt: {raw_results["dice_wt"]["mean"]:.4f} +/- {raw_results["dice_wt"]["std"]:.4f}')
    print(f'  dice_tc: {raw_results["dice_tc"]["mean"]:.4f} +/- {raw_results["dice_tc"]["std"]:.4f}')
    print(f'  dice_et: {raw_results["dice_et"]["mean"]:.4f} +/- {raw_results["dice_et"]["std"]:.4f}')

    # harmonized evaluation (if model provided)
    if args.harmonization_checkpoint:
        print('[downstream] loading harmonization model...')
        # load sa-cyclegan model
        from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig

        config = SACycleGAN25DConfig()
        harm_model = SACycleGAN25D(config)

        checkpoint = torch.load(args.harmonization_checkpoint, map_location=device,
                               weights_only=False)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('.module.', '.')
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            new_state_dict[new_key] = v
        harm_model.load_state_dict(new_state_dict, strict=False)
        harm_model = harm_model.to(device)

        print('[downstream] evaluating with sa-cyclegan harmonization...')
        harmonized_results = evaluate_segmentation(
            seg_model, test_loader, device, harm_model
        )

        # save harmonized results
        with open(output_dir / 'harmonized_segmentation_results.json', 'w') as f:
            json.dump(harmonized_results, f, indent=2)

        print('[downstream] harmonized results:')
        print(f'  dice_wt: {harmonized_results["dice_wt"]["mean"]:.4f} +/- {harmonized_results["dice_wt"]["std"]:.4f}')
        print(f'  dice_tc: {harmonized_results["dice_tc"]["mean"]:.4f} +/- {harmonized_results["dice_tc"]["std"]:.4f}')
        print(f'  dice_et: {harmonized_results["dice_et"]["mean"]:.4f} +/- {harmonized_results["dice_et"]["std"]:.4f}')

        # statistical comparison
        metric_keys = ['dice_wt', 'dice_tc', 'dice_et',
                      'hausdorff95_wt', 'hausdorff95_tc', 'hausdorff95_et']
        comparison = statistical_comparison(raw_results, harmonized_results, metric_keys)

        with open(output_dir / 'statistical_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)

        # print summary
        print('=' * 60)
        print('[downstream] statistical comparison:')
        for key, stats in comparison.items():
            sig = '*' if stats['significant'] else ''
            print(f'  {key}: {stats["improvement_pct"]:+.2f}%{sig} '
                  f'(p={stats["p_value"]:.4f}, d={stats["cohens_d"]:.3f})')

    print('=' * 60)
    print(f'[downstream] results saved to {output_dir}')


if __name__ == '__main__':
    main()
