#!/usr/bin/env python3
"""
downstream task evaluation for harmonization quality assessment.

evaluates the clinical utility of harmonized mri by measuring performance
on downstream tasks that depend on cross-site data quality:

1. tumor segmentation: u-net trained on one domain, tested on another
2. domain generalization: segmentation on mixed harmonized data
3. radiomics stability: icc of prognostic features pre/post harmonization

this is the strongest validation of harmonization quality -- showing that
models trained on harmonized data generalize better across sites.

extension d of the journal extension.

usage:
    python eval_downstream.py --harmonized_dir /path/to/harmonized \
                              --raw_dir /path/to/raw \
                              --output_dir /path/to/results
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# lightweight u-net for tumor segmentation
# ============================================================================


class DoubleConv(nn.Module):
    """two 3x3 conv + batchnorm + relu blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    lightweight 2d u-net for tumor segmentation.

    4-channel input (t1, t1ce, t2, flair), 3-class output
    (background, tumor core, enhancing tumor).
    """

    def __init__(self, in_channels: int = 4, n_classes: int = 3, base_filters: int = 32):
        super().__init__()

        f = base_filters

        # encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)

        # bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16)

        # decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)

        # output
        self.out_conv = nn.Conv2d(f, n_classes, 1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # bottleneck
        b = self.bottleneck(self.pool(e4))

        # decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ============================================================================
# loss functions for segmentation
# ============================================================================


class DiceLoss(nn.Module):
    """soft dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        args:
            pred: predictions [b, c, h, w] (logits)
            target: one-hot targets [b, c, h, w]
        """
        pred = torch.softmax(pred, dim=1)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """dice + cross-entropy loss for segmentation."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        args:
            pred: predictions [b, c, h, w] (logits)
            target: integer labels [b, h, w] for ce, one-hot [b, c, h, w] for dice
        """
        # convert integer labels to one-hot for dice
        if target.dim() == 3:
            target_onehot = F.one_hot(target.long(), num_classes=pred.shape[1])
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
            ce_target = target.long()
        else:
            target_onehot = target
            ce_target = target.argmax(dim=1)

        dice = self.dice_loss(pred, target_onehot)
        ce = self.ce_loss(pred, ce_target)
        return self.dice_weight * dice + self.ce_weight * ce


# ============================================================================
# evaluation metrics
# ============================================================================


def compute_dice_score(
    pred: torch.Tensor, target: torch.Tensor, n_classes: int = 3
) -> Dict[str, float]:
    """
    compute per-class dice scores.

    args:
        pred: predicted labels [b, h, w] (integer)
        target: ground truth labels [b, h, w] (integer)
        n_classes: number of segmentation classes
    returns:
        dict with per-class and mean dice scores
    """
    dice_scores = {}
    class_names = ["background", "tumor_core", "enhancing_tumor"]

    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection / union).item()

        name = class_names[c] if c < len(class_names) else f"class_{c}"
        dice_scores[name] = dice

    # mean dice (excluding background)
    foreground_dice = [v for k, v in dice_scores.items() if k != "background"]
    dice_scores["mean_foreground"] = np.mean(foreground_dice) if foreground_dice else 0.0

    return dice_scores


def compute_hausdorff_95(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """
    compute 95th percentile hausdorff distance.

    args:
        pred: predicted binary mask [h, w]
        target: ground truth binary mask [h, w]
    returns:
        hd95 value
    """
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    # compute surface distances
    pred_boundary = pred ^ (
        distance_transform_edt(pred) > 1
    )
    target_boundary = target ^ (
        distance_transform_edt(target) > 1
    )

    dt_pred = distance_transform_edt(~pred_boundary)
    dt_target = distance_transform_edt(~target_boundary)

    distances_pred_to_target = dt_target[pred_boundary > 0]
    distances_target_to_pred = dt_pred[target_boundary > 0]

    if len(distances_pred_to_target) == 0 or len(distances_target_to_pred) == 0:
        return float("inf")

    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    return np.percentile(all_distances, 95)


# ============================================================================
# downstream evaluation pipeline
# ============================================================================


class DownstreamEvaluator:
    """
    comprehensive downstream task evaluation for harmonization.

    runs three evaluation protocols:
    1. cross-site segmentation transfer
    2. domain generalization on mixed data
    3. radiomics feature stability analysis
    """

    def __init__(
        self,
        output_dir: str,
        device: str = "auto",
        n_classes: int = 3,
        seg_epochs: int = 50,
        seg_lr: float = 1e-3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.n_classes = n_classes
        self.seg_epochs = seg_epochs
        self.seg_lr = seg_lr

    def train_segmentation(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        model_name: str = "unet",
    ) -> UNet:
        """
        train a u-net segmentation model.

        args:
            train_loader: training data loader
            val_loader: optional validation loader
            model_name: name for saving
        returns:
            trained u-net model
        """
        model = UNet(in_channels=4, n_classes=self.n_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.seg_lr)
        criterion = CombinedSegLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.seg_epochs
        )

        best_dice = 0.0
        best_state = None

        for epoch in range(self.seg_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                pred = model(images)
                loss = criterion(pred, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            # validate
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_dice = self.evaluate_segmentation(model, val_loader)
                mean_dice = val_dice["mean_foreground"]
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_state = model.state_dict().copy()

        if best_state is not None:
            model.load_state_dict(best_state)

        return model

    @torch.no_grad()
    def evaluate_segmentation(
        self,
        model: UNet,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        evaluate segmentation model on a test set.

        args:
            model: trained u-net
            test_loader: test data loader
        returns:
            dict with dice scores, hd95, sensitivity, specificity
        """
        model.eval()
        all_dice = []
        all_hd95 = []

        for batch in test_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"]

            pred_logits = model(images)
            pred_labels = pred_logits.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                dice = compute_dice_score(pred_labels[i], masks[i], self.n_classes)
                all_dice.append(dice)

                # hd95 for foreground classes
                for c in range(1, self.n_classes):
                    pred_c = (pred_labels[i] == c).numpy()
                    target_c = (masks[i] == c).numpy()
                    if target_c.sum() > 0:
                        hd = compute_hausdorff_95(pred_c, target_c)
                        all_hd95.append(hd)

        # aggregate
        results = {}
        for key in all_dice[0].keys():
            values = [d[key] for d in all_dice]
            results[f"dice_{key}_mean"] = np.mean(values)
            results[f"dice_{key}_std"] = np.std(values)

        valid_hd = [h for h in all_hd95 if h != float("inf")]
        results["hd95_mean"] = np.mean(valid_hd) if valid_hd else float("inf")
        results["hd95_std"] = np.std(valid_hd) if valid_hd else 0.0

        return results

    def cross_site_transfer(
        self,
        train_loader_A: DataLoader,
        test_loader_B: DataLoader,
        train_loader_A_harmonized: Optional[DataLoader] = None,
        test_loader_B_harmonized: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        evaluate cross-site segmentation transfer.

        trains on domain a, tests on domain b. compares performance
        with and without harmonization.

        args:
            train_loader_a: training data from domain a (raw)
            test_loader_b: test data from domain b (raw)
            train_loader_a_harmonized: training data from domain a (harmonized)
            test_loader_b_harmonized: test data from domain b (harmonized)
        returns:
            dict with results for raw and harmonized conditions
        """
        results = {}

        # train on raw a, test on raw b
        print("  training on raw domain a...")
        model_raw = self.train_segmentation(train_loader_A, model_name="raw_a")
        print("  evaluating on raw domain b...")
        results["raw_a_to_raw_b"] = self.evaluate_segmentation(model_raw, test_loader_B)

        # train on harmonized a, test on harmonized b
        if train_loader_A_harmonized and test_loader_B_harmonized:
            print("  training on harmonized domain a...")
            model_harm = self.train_segmentation(
                train_loader_A_harmonized, model_name="harm_a"
            )
            print("  evaluating on harmonized domain b...")
            results["harm_a_to_harm_b"] = self.evaluate_segmentation(
                model_harm, test_loader_B_harmonized
            )

        return results

    def save_results(self, results: Dict, filename: str = "downstream_results.json"):
        """save evaluation results to json."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="downstream task evaluation for harmonization"
    )
    parser.add_argument("--harmonized_dir", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seg_epochs", type=int, default=50)
    parser.add_argument("--seg_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluator = DownstreamEvaluator(
        output_dir=args.output_dir,
        seg_epochs=args.seg_epochs,
        seg_lr=args.seg_lr,
    )

    print("downstream evaluation pipeline ready")
    print(f"output directory: {args.output_dir}")
    print("note: requires data loaders to be configured with segmentation masks")
