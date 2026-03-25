#!/usr/bin/env python3
"""
downstream task evaluation for harmonization quality assessment.

evaluates the clinical utility of harmonized mri by measuring performance
on downstream segmentation tasks that depend on cross-site data quality:

1. train u-net on site a raw data, test on site b raw data (baseline)
2. train u-net on site a harmonized data, test on site b harmonized data
3. compare dice scores to show harmonization improves cross-site generalization

supports two modes:
  (a) pre-generated harmonized directories (default, better for reproducibility)
  (b) on-the-fly harmonization from a trained sa-cyclegan-2.5d checkpoint

extension d of the journal extension.

usage:
    python eval_downstream.py --brats_dir preprocessed/brats \
                              --upenn_dir preprocessed/upenn \
                              --harmonized_brats_dir results/harmonized/brats \
                              --harmonized_upenn_dir results/harmonized/upenn \
                              --output_dir results/downstream
    python eval_downstream.py --config ../configs/downstream.yaml
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

try:
    import nibabel as nib
except ImportError:
    nib = None

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# performance setup
# ============================================================================


def setup_torch_performance():
    """configure torch for maximum gpu throughput."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)


# ============================================================================
# brats segmentation label mapping
# ============================================================================

# brats convention: 0=background, 1=ncr/net, 2=ed, 4=et
# we remap to contiguous: 0=background, 1=ncr/net, 2=ed, 3=et
# derived regions: wt (1+2+4), tc (1+4), et (4)

BRATS_LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
CLASS_NAMES = ["background", "ncr_net", "ed", "et"]
REGION_NAMES = ["wt", "tc", "et"]


def remap_brats_labels(seg: np.ndarray) -> np.ndarray:
    """remap brats labels to contiguous 0-3 classes."""
    out = np.zeros_like(seg, dtype=np.int64)
    for src, dst in BRATS_LABEL_MAP.items():
        out[seg == src] = dst
    return out


def compute_region_masks(labels: np.ndarray) -> Dict[str, np.ndarray]:
    """
    compute derived brats tumor region masks from contiguous labels.
    input uses remapped labels: 0=bg, 1=ncr/net, 2=ed, 3=et
    """
    return {
        "wt": ((labels == 1) | (labels == 2) | (labels == 3)).astype(np.float32),
        "tc": ((labels == 1) | (labels == 3)).astype(np.float32),
        "et": (labels == 3).astype(np.float32),
    }


# ============================================================================
# 2d slice dataset from nifti volumes
# ============================================================================


class BraTSSliceDataset(Dataset):
    """
    2d axial slice dataset for tumor segmentation from nifti volumes.

    loads 4-channel (t1, t1gd, flair, t2) slices with segmentation labels.
    filters out slices with no tumor content for training efficiency.
    """

    MODALITIES = ["t1", "t1gd", "flair", "t2"]

    def __init__(
        self,
        data_dir: str,
        subject_ids: List[str],
        image_size: Tuple[int, int] = (128, 128),
        slice_range: Tuple[int, int] = (30, 125),
        require_tumor: bool = True,
        min_tumor_pixels: int = 10,
    ):
        """
        args:
            data_dir: path to preprocessed site directory (e.g. preprocessed/brats)
            subject_ids: list of subject ids to include
            image_size: target spatial resolution for resizing
            slice_range: axial slice range to sample from
            require_tumor: if true, skip slices with no tumor voxels
            min_tumor_pixels: minimum foreground pixels to keep a slice
        """
        assert nib is not None, "nibabel is required for nifti loading"

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.require_tumor = require_tumor
        self.min_tumor_pixels = min_tumor_pixels

        # build sample index: (subject_dir, slice_idx)
        self.samples: List[Tuple[Path, int]] = []
        self._volume_cache: Dict[str, np.ndarray] = {}
        self._seg_cache: Dict[str, np.ndarray] = {}

        for subj_id in sorted(subject_ids):
            subj_dir = self.data_dir / subj_id
            if not subj_dir.is_dir():
                continue
            if not self._has_required_files(subj_dir):
                continue

            # load seg to determine valid slices
            seg_path = subj_dir / "seg.nii.gz"
            seg = nib.load(str(seg_path)).get_fdata().astype(np.int64)
            seg = remap_brats_labels(seg)

            # seg shape is [h, w, d] -- axial slices along last dim
            n_slices = seg.shape[2]
            start = max(slice_range[0], 0)
            end = min(slice_range[1], n_slices)

            for s in range(start, end):
                axial_slice = seg[:, :, s]
                if require_tumor and axial_slice.sum() < min_tumor_pixels:
                    continue
                self.samples.append((subj_dir, s))

        print(f"  dataset: {len(self.samples)} slices from "
              f"{len(subject_ids)} subjects in {data_dir}")

    def _has_required_files(self, subj_dir: Path) -> bool:
        """check if subject has all modalities and segmentation."""
        for mod in self.MODALITIES:
            if not (subj_dir / f"{mod}.nii.gz").exists():
                return False
        if not (subj_dir / "seg.nii.gz").exists():
            return False
        return True

    def _load_volume(self, subj_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """load 4-modality volume and seg. returns ([4, h, w, d], [h, w, d])."""
        cache_key = str(subj_dir)
        if cache_key in self._volume_cache:
            return self._volume_cache[cache_key], self._seg_cache[cache_key]

        vols = []
        for mod in self.MODALITIES:
            path = subj_dir / f"{mod}.nii.gz"
            vol = nib.load(str(path)).get_fdata().astype(np.float32)
            vols.append(vol)
        volume = np.stack(vols, axis=0)  # [4, h, w, d]

        seg = nib.load(str(subj_dir / "seg.nii.gz")).get_fdata().astype(np.int64)
        seg = remap_brats_labels(seg)

        self._volume_cache[cache_key] = volume
        self._seg_cache[cache_key] = seg
        return volume, seg

    def _normalize_and_resize(self, slc: np.ndarray) -> torch.Tensor:
        """normalize to [0, 1] and resize. input: [c, h, w]."""
        tensor = torch.from_numpy(slc).float()
        for c in range(tensor.shape[0]):
            vmax = tensor[c].max()
            if vmax > 0:
                tensor[c] = tensor[c] / vmax
        if self.image_size and tensor.shape[-2:] != tuple(self.image_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=self.image_size,
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return tensor

    def _resize_mask(self, mask: np.ndarray) -> torch.Tensor:
        """resize integer mask with nearest interpolation. input: [h, w]."""
        tensor = torch.from_numpy(mask).long().unsqueeze(0).unsqueeze(0).float()
        if self.image_size and mask.shape != tuple(self.image_size):
            tensor = F.interpolate(
                tensor, size=self.image_size, mode="nearest",
            )
        return tensor.squeeze(0).squeeze(0).long()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj_dir, slice_idx = self.samples[idx]
        volume, seg = self._load_volume(subj_dir)

        # extract 2d axial slice: [4, h, w]
        img_slice = volume[:, :, :, slice_idx]  # [4, h, w]
        seg_slice = seg[:, :, slice_idx]  # [h, w]

        img_tensor = self._normalize_and_resize(img_slice)
        seg_tensor = self._resize_mask(seg_slice)

        return {
            "image": img_tensor,          # [4, h, w]
            "mask": seg_tensor,            # [h, w] integer labels 0-3
            "subject": subj_dir.name,
            "slice_idx": slice_idx,
        }


def get_subject_ids(data_dir: str) -> List[str]:
    """get all subject ids that have seg.nii.gz in a data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    ids = []
    for subj_dir in sorted(data_path.iterdir()):
        if subj_dir.is_dir() and (subj_dir / "seg.nii.gz").exists():
            ids.append(subj_dir.name)
    return ids


def split_subjects(
    subject_ids: List[str],
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """split subject ids into train and test sets."""
    rng = np.random.RandomState(seed)
    ids = list(subject_ids)
    rng.shuffle(ids)
    n_train = int(len(ids) * train_frac)
    return ids[:n_train], ids[n_train:]


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

    4-channel input (t1, t1gd, flair, t2), 4-class output
    (background, ncr/net, ed, et).
    """

    def __init__(self, in_channels: int = 4, n_classes: int = 4, base_filters: int = 32):
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
    pred: torch.Tensor, target: torch.Tensor, n_classes: int = 4
) -> Dict[str, float]:
    """
    compute per-class dice scores.

    args:
        pred: predicted labels [h, w] or [b, h, w] (integer)
        target: ground truth labels [h, w] or [b, h, w] (integer)
        n_classes: number of segmentation classes
    returns:
        dict with per-class and mean dice scores
    """
    dice_scores = {}

    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection / union).item()

        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        dice_scores[name] = dice

    # mean dice (excluding background)
    foreground_dice = [v for k, v in dice_scores.items() if k != "background"]
    dice_scores["mean_foreground"] = float(np.mean(foreground_dice)) if foreground_dice else 0.0

    return dice_scores


def compute_region_dice(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """
    compute dice for brats-derived tumor regions (wt, tc, et).

    args:
        pred: predicted labels [h, w] (integer, remapped 0-3)
        target: ground truth labels [h, w] (integer, remapped 0-3)
    returns:
        dict with region dice scores
    """
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target

    pred_regions = compute_region_masks(pred_np)
    target_regions = compute_region_masks(target_np)

    region_dice = {}
    for name in REGION_NAMES:
        p = pred_regions[name]
        t = target_regions[name]
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            region_dice[name] = 1.0 if intersection == 0 else 0.0
        else:
            region_dice[name] = float(2.0 * intersection / union)

    region_dice["mean_region"] = float(np.mean([region_dice[r] for r in REGION_NAMES]))
    return region_dice


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
    return float(np.percentile(all_distances, 95))


# ============================================================================
# on-the-fly harmonization (option b)
# ============================================================================


def generate_harmonized_data(
    checkpoint_path: str,
    source_dir: str,
    output_dir: str,
    direction: str = "a_to_b",
    device: str = "cuda",
) -> str:
    """
    generate harmonized data using a trained sa-cyclegan-2.5d checkpoint.

    loads the generator from a checkpoint and runs inference on all subjects
    in source_dir, saving harmonized volumes to output_dir.

    args:
        checkpoint_path: path to trained model checkpoint
        source_dir: directory with raw preprocessed data
        output_dir: directory to write harmonized volumes
        direction: 'a_to_b' or 'b_to_a' for generator selection
        device: torch device string
    returns:
        path to the harmonized output directory
    """
    from neuroscope.models.sa_cyclegan_25d import SACycleGAN25DConfig, GeneratorSA25D

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # determine config from checkpoint
    if "config" in checkpoint:
        config = SACycleGAN25DConfig(**checkpoint["config"])
    else:
        config = SACycleGAN25DConfig()

    # load generator
    gen_key = "generator_AB" if direction == "a_to_b" else "generator_BA"
    generator = GeneratorSA25D(config).to(device)
    generator.load_state_dict(checkpoint[gen_key])
    generator.eval()

    source_path = Path(source_dir)
    modalities = ["t1", "t1gd", "flair", "t2"]

    subject_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    print(f"harmonizing {len(subject_dirs)} subjects from {source_dir}")

    for subj_dir in tqdm(subject_dirs, desc="harmonizing"):
        out_subj = output_path / subj_dir.name
        out_subj.mkdir(exist_ok=True)

        # check if already done
        if all((out_subj / f"{m}.nii.gz").exists() for m in modalities):
            continue

        # load volumes
        vols = []
        affine = None
        for mod in modalities:
            nii = nib.load(str(subj_dir / f"{mod}.nii.gz"))
            if affine is None:
                affine = nii.affine
            vols.append(nii.get_fdata().astype(np.float32))

        volume = np.stack(vols, axis=0)  # [4, h, w, d]
        n_slices = volume.shape[3]
        harmonized = np.zeros_like(volume)

        with torch.no_grad():
            for s in range(1, n_slices - 1):
                # build 2.5d input: 3 adjacent slices x 4 modalities = 12 channels
                triplet = volume[:, :, :, s - 1:s + 2]  # [4, h, w, 3]
                triplet = np.transpose(triplet, (0, 3, 1, 2))  # [4, 3, h, w]
                input_25d = triplet.reshape(-1, triplet.shape[2], triplet.shape[3])  # [12, h, w]

                # normalize
                tensor = torch.from_numpy(input_25d).float().unsqueeze(0).to(device)
                vmax = tensor.max()
                if vmax > 0:
                    tensor = tensor / vmax * 2 - 1

                # inference
                output = generator(tensor)  # [1, 4, h, w]
                output = (output.squeeze(0).cpu().numpy() + 1) / 2 * vmax.cpu().item()
                harmonized[:, :, :, s] = output

            # copy edge slices
            harmonized[:, :, :, 0] = harmonized[:, :, :, 1]
            harmonized[:, :, :, -1] = harmonized[:, :, :, -2]

        # save harmonized volumes
        for i, mod in enumerate(modalities):
            out_nii = nib.Nifti1Image(harmonized[i], affine)
            nib.save(out_nii, str(out_subj / f"{mod}.nii.gz"))

        # copy segmentation as-is
        seg_src = subj_dir / "seg.nii.gz"
        if seg_src.exists():
            import shutil
            shutil.copy2(str(seg_src), str(out_subj / "seg.nii.gz"))

    print(f"harmonized data saved to {output_path}")
    return str(output_path)


# ============================================================================
# downstream evaluation pipeline
# ============================================================================


class DownstreamEvaluator:
    """
    downstream task evaluation for harmonization.

    runs cross-site segmentation transfer evaluation:
    - train u-net on site a, test on site b (raw baseline)
    - train u-net on site a, test on site b (harmonized)
    - compare dice scores to quantify harmonization benefit
    """

    def __init__(
        self,
        output_dir: str,
        device: str = "auto",
        n_classes: int = 4,
        seg_epochs: int = 50,
        seg_lr: float = 1e-3,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 128,
        use_amp: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.n_classes = n_classes
        self.seg_epochs = seg_epochs
        self.seg_lr = seg_lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = (image_size, image_size)
        self.use_amp = use_amp and self.device.type == "cuda"

    def _make_loader(
        self,
        data_dir: str,
        subject_ids: List[str],
        shuffle: bool = True,
        require_tumor: bool = True,
    ) -> DataLoader:
        """create a dataloader from a data directory and subject list."""
        dataset = BraTSSliceDataset(
            data_dir=data_dir,
            subject_ids=subject_ids,
            image_size=self.image_size,
            require_tumor=require_tumor,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    def train_segmentation(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        model_name: str = "unet",
        resume_path: Optional[str] = None,
    ) -> Tuple[UNet, Dict]:
        """
        train a u-net segmentation model.

        args:
            train_loader: training data loader
            val_loader: optional validation loader
            model_name: name for saving checkpoints
            resume_path: path to checkpoint for resume
        returns:
            (trained u-net model, training history dict)
        """
        model = UNet(in_channels=4, n_classes=self.n_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.seg_lr)
        criterion = CombinedSegLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.seg_epochs
        )
        scaler = GradScaler("cuda", enabled=self.use_amp)

        best_dice = 0.0
        best_state = None
        start_epoch = 0
        history = {"train_loss": [], "val_dice": []}

        # resume from checkpoint
        if resume_path and Path(resume_path).exists():
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt.get("best_dice", 0.0)
            history = ckpt.get("history", history)
            print(f"  resumed {model_name} from epoch {start_epoch}")

        for epoch in range(start_epoch, self.seg_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=self.use_amp):
                    pred = model(images)
                    loss = criterion(pred, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            # validate every 5 epochs
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_results = self.evaluate_segmentation(model, val_loader)
                mean_dice = val_results["dice_mean_foreground_mean"]
                history["val_dice"].append(mean_dice)

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if (epoch + 1) % 10 == 0:
                    print(f"    epoch {epoch + 1}/{self.seg_epochs} - "
                          f"loss: {avg_loss:.4f} - val dice: {mean_dice:.4f}")
            elif (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch + 1}/{self.seg_epochs} - loss: {avg_loss:.4f}")

            # save checkpoint
            if (epoch + 1) % 25 == 0:
                ckpt_path = self.output_dir / f"{model_name}_epoch{epoch + 1}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_dice": best_dice,
                    "history": history,
                }, ckpt_path)

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history

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
            dict with per-class dice, region dice, hd95
        """
        model.eval()
        all_dice = []
        all_region_dice = []
        all_hd95 = []
        per_subject: Dict[str, List[Dict]] = {}

        for batch in test_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"]
            subjects = batch["subject"]

            with autocast("cuda", enabled=self.use_amp):
                pred_logits = model(images)
            pred_labels = pred_logits.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                # per-class dice
                dice = compute_dice_score(pred_labels[i], masks[i], self.n_classes)
                all_dice.append(dice)

                # region dice (wt, tc, et)
                rdice = compute_region_dice(pred_labels[i], masks[i])
                all_region_dice.append(rdice)

                # per-subject tracking
                subj = subjects[i]
                if subj not in per_subject:
                    per_subject[subj] = []
                per_subject[subj].append({**dice, **rdice})

                # hd95 for foreground classes
                for c in range(1, self.n_classes):
                    pred_c = (pred_labels[i] == c).numpy()
                    target_c = (masks[i] == c).numpy()
                    if target_c.sum() > 0:
                        hd = compute_hausdorff_95(pred_c, target_c)
                        all_hd95.append(hd)

        # aggregate slice-level metrics
        results = {}
        if all_dice:
            for key in all_dice[0].keys():
                values = [d[key] for d in all_dice]
                results[f"dice_{key}_mean"] = float(np.mean(values))
                results[f"dice_{key}_std"] = float(np.std(values))

        if all_region_dice:
            for key in all_region_dice[0].keys():
                values = [d[key] for d in all_region_dice]
                results[f"region_{key}_mean"] = float(np.mean(values))
                results[f"region_{key}_std"] = float(np.std(values))

        valid_hd = [h for h in all_hd95 if h != float("inf")]
        results["hd95_mean"] = float(np.mean(valid_hd)) if valid_hd else float("inf")
        results["hd95_std"] = float(np.std(valid_hd)) if valid_hd else 0.0

        # aggregate per-subject metrics
        subject_summaries = {}
        for subj, slice_results in per_subject.items():
            subj_summary = {}
            for key in slice_results[0].keys():
                values = [s[key] for s in slice_results]
                subj_summary[key] = float(np.mean(values))
            subject_summaries[subj] = subj_summary
        results["per_subject"] = subject_summaries

        return results

    def cross_site_transfer(
        self,
        site_a_dir: str,
        site_b_dir: str,
        harm_a_dir: Optional[str] = None,
        harm_b_dir: Optional[str] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        evaluate cross-site segmentation transfer.

        trains on site a, tests on site b. compares performance
        with and without harmonization.

        args:
            site_a_dir: raw data directory for site a (e.g. brats)
            site_b_dir: raw data directory for site b (e.g. upenn)
            harm_a_dir: harmonized data directory for site a
            harm_b_dir: harmonized data directory for site b
            resume_path: directory with checkpoints to resume from
        returns:
            dict with results for all evaluated conditions
        """
        results = {}

        # get subject ids and split
        ids_a = get_subject_ids(site_a_dir)
        ids_b = get_subject_ids(site_b_dir)
        print(f"site a: {len(ids_a)} subjects with segmentations")
        print(f"site b: {len(ids_b)} subjects with segmentations")

        train_a, test_a = split_subjects(ids_a)
        train_b, test_b = split_subjects(ids_b)
        print(f"site a split: {len(train_a)} train, {len(test_a)} test")
        print(f"site b split: {len(train_b)} train, {len(test_b)} test")

        results["data_summary"] = {
            "site_a_total": len(ids_a),
            "site_b_total": len(ids_b),
            "site_a_train": len(train_a),
            "site_a_test": len(test_a),
            "site_b_train": len(train_b),
            "site_b_test": len(test_b),
        }

        # ---- condition 1: train on raw a, test on raw b ----
        print("\n[condition 1] train on raw site a, test on raw site b")
        train_loader_raw_a = self._make_loader(site_a_dir, train_a, shuffle=True)
        test_loader_raw_b = self._make_loader(site_b_dir, test_b, shuffle=False, require_tumor=False)

        resume_raw = None
        if resume_path:
            candidates = sorted(Path(resume_path).glob("raw_a_to_b_*.pt"))
            if candidates:
                resume_raw = str(candidates[-1])

        print("  training u-net on raw site a...")
        model_raw, hist_raw = self.train_segmentation(
            train_loader_raw_a, model_name="raw_a_to_b", resume_path=resume_raw,
        )
        print("  evaluating on raw site b...")
        results["raw_a_to_raw_b"] = self.evaluate_segmentation(model_raw, test_loader_raw_b)
        print(f"  raw baseline dice (mean foreground): "
              f"{results['raw_a_to_raw_b']['dice_mean_foreground_mean']:.4f}")

        # ---- condition 2: train on raw a, test on raw a (upper bound) ----
        print("\n[condition 2] train on raw site a, test on raw site a (within-site)")
        test_loader_raw_a = self._make_loader(site_a_dir, test_a, shuffle=False, require_tumor=False)
        results["raw_a_to_raw_a"] = self.evaluate_segmentation(model_raw, test_loader_raw_a)
        print(f"  within-site dice (mean foreground): "
              f"{results['raw_a_to_raw_a']['dice_mean_foreground_mean']:.4f}")

        # ---- condition 3: train on harmonized a, test on harmonized b ----
        if harm_a_dir and harm_b_dir:
            print("\n[condition 3] train on harmonized site a, test on harmonized site b")

            # harmonized dirs use same subject ids, same seg labels
            train_loader_harm_a = self._make_loader(harm_a_dir, train_a, shuffle=True)
            test_loader_harm_b = self._make_loader(harm_b_dir, test_b, shuffle=False, require_tumor=False)

            resume_harm = None
            if resume_path:
                candidates = sorted(Path(resume_path).glob("harm_a_to_b_*.pt"))
                if candidates:
                    resume_harm = str(candidates[-1])

            print("  training u-net on harmonized site a...")
            model_harm, hist_harm = self.train_segmentation(
                train_loader_harm_a, model_name="harm_a_to_b", resume_path=resume_harm,
            )
            print("  evaluating on harmonized site b...")
            results["harm_a_to_harm_b"] = self.evaluate_segmentation(model_harm, test_loader_harm_b)
            print(f"  harmonized dice (mean foreground): "
                  f"{results['harm_a_to_harm_b']['dice_mean_foreground_mean']:.4f}")

            # ---- condition 4: train on harmonized a, test on harmonized a (within-site) ----
            print("\n[condition 4] train on harmonized site a, test on harmonized site a")
            test_loader_harm_a = self._make_loader(harm_a_dir, test_a, shuffle=False, require_tumor=False)
            results["harm_a_to_harm_a"] = self.evaluate_segmentation(model_harm, test_loader_harm_a)
            print(f"  harmonized within-site dice (mean foreground): "
                  f"{results['harm_a_to_harm_a']['dice_mean_foreground_mean']:.4f}")

            # compute improvement
            raw_dice = results["raw_a_to_raw_b"]["dice_mean_foreground_mean"]
            harm_dice = results["harm_a_to_harm_b"]["dice_mean_foreground_mean"]
            improvement = harm_dice - raw_dice
            results["improvement"] = {
                "dice_mean_foreground_delta": improvement,
                "dice_mean_foreground_relative_pct": (improvement / max(raw_dice, 1e-8)) * 100,
            }
            print(f"\n  harmonization improvement: {improvement:+.4f} "
                  f"({results['improvement']['dice_mean_foreground_relative_pct']:+.1f}%)")

        # ---- reverse direction: train on b, test on a ----
        print("\n[condition 5] train on raw site b, test on raw site a (reverse)")
        train_loader_raw_b = self._make_loader(site_b_dir, train_b, shuffle=True)
        test_loader_raw_a2 = self._make_loader(site_a_dir, test_a, shuffle=False, require_tumor=False)

        resume_rev = None
        if resume_path:
            candidates = sorted(Path(resume_path).glob("raw_b_to_a_*.pt"))
            if candidates:
                resume_rev = str(candidates[-1])

        print("  training u-net on raw site b...")
        model_rev, hist_rev = self.train_segmentation(
            train_loader_raw_b, model_name="raw_b_to_a", resume_path=resume_rev,
        )
        print("  evaluating on raw site a...")
        results["raw_b_to_raw_a"] = self.evaluate_segmentation(model_rev, test_loader_raw_a2)
        print(f"  reverse raw dice (mean foreground): "
              f"{results['raw_b_to_raw_a']['dice_mean_foreground_mean']:.4f}")

        if harm_a_dir and harm_b_dir:
            print("\n[condition 6] train on harmonized site b, test on harmonized site a (reverse)")
            train_loader_harm_b = self._make_loader(harm_b_dir, train_b, shuffle=True)
            test_loader_harm_a2 = self._make_loader(harm_a_dir, test_a, shuffle=False, require_tumor=False)

            resume_rev_harm = None
            if resume_path:
                candidates = sorted(Path(resume_path).glob("harm_b_to_a_*.pt"))
                if candidates:
                    resume_rev_harm = str(candidates[-1])

            print("  training u-net on harmonized site b...")
            model_rev_harm, _ = self.train_segmentation(
                train_loader_harm_b, model_name="harm_b_to_a", resume_path=resume_rev_harm,
            )
            print("  evaluating on harmonized site a...")
            results["harm_b_to_harm_a"] = self.evaluate_segmentation(model_rev_harm, test_loader_harm_a2)
            print(f"  reverse harmonized dice (mean foreground): "
                  f"{results['harm_b_to_harm_a']['dice_mean_foreground_mean']:.4f}")

            # reverse improvement
            raw_rev = results["raw_b_to_raw_a"]["dice_mean_foreground_mean"]
            harm_rev = results["harm_b_to_harm_a"]["dice_mean_foreground_mean"]
            rev_improvement = harm_rev - raw_rev
            results["improvement_reverse"] = {
                "dice_mean_foreground_delta": rev_improvement,
                "dice_mean_foreground_relative_pct": (rev_improvement / max(raw_rev, 1e-8)) * 100,
            }

        return results

    def save_results(self, results: Dict, filename: str = "downstream_results.json"):
        """save evaluation results to json."""
        output_path = self.output_dir / filename
        # convert any numpy types for json serialization
        def _convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        with open(output_path, "w") as f:
            json.dump(_convert(results), f, indent=2, default=str)
        print(f"results saved to {output_path}")


# ============================================================================
# argument parsing
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="downstream task evaluation for harmonization"
    )

    # config file
    parser.add_argument("--config", type=str, default=None,
                        help="path to yaml config file")

    # data directories (raw)
    parser.add_argument("--brats_dir", type=str, default=None,
                        help="path to preprocessed brats data (e.g. preprocessed/brats)")
    parser.add_argument("--upenn_dir", type=str, default=None,
                        help="path to preprocessed upenn data (e.g. preprocessed/upenn)")

    # data directories (harmonized) -- option (a)
    parser.add_argument("--harmonized_brats_dir", type=str, default=None,
                        help="path to pre-generated harmonized brats data")
    parser.add_argument("--harmonized_upenn_dir", type=str, default=None,
                        help="path to pre-generated harmonized upenn data")

    # harmonization model checkpoint -- option (b)
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="path to trained sa-cyclegan-2.5d checkpoint for on-the-fly harmonization")

    # output
    parser.add_argument("--output_dir", type=str, default="results/downstream",
                        help="directory to save results and checkpoints")

    # training hyperparameters
    parser.add_argument("--seg_epochs", type=int, default=50,
                        help="number of training epochs for segmentation u-net")
    parser.add_argument("--seg_lr", type=float, default=1e-3,
                        help="learning rate for segmentation u-net")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training and evaluation")
    parser.add_argument("--image_size", type=int, default=128,
                        help="spatial resolution for resizing slices")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of dataloader workers")
    parser.add_argument("--base_filters", type=int, default=32,
                        help="base filter count for u-net")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_true",
                        help="disable automatic mixed precision")

    # resume
    parser.add_argument("--resume", action="store_true",
                        help="resume from latest checkpoint in output_dir")

    # device
    parser.add_argument("--device", type=str, default="auto",
                        help="device: auto, cuda, mps, cpu")

    return parser.parse_args()


# ============================================================================
# main
# ============================================================================


def main():
    args = parse_args()

    # load config from yaml if provided
    if args.config:
        assert yaml is not None, "pyyaml is required for config loading: pip install pyyaml"
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)
            elif not hasattr(args, k):
                setattr(args, k, v)

    # resolve amp flag
    if args.no_amp:
        args.use_amp = False

    # validate required args
    if not args.brats_dir or not args.upenn_dir:
        print("error: --brats_dir and --upenn_dir are required")
        sys.exit(1)

    # setup
    setup_torch_performance()

    print("=" * 60)
    print("  downstream segmentation evaluation")
    print("=" * 60)
    print(f"brats dir: {args.brats_dir}")
    print(f"upenn dir: {args.upenn_dir}")
    print(f"harmonized brats dir: {args.harmonized_brats_dir}")
    print(f"harmonized upenn dir: {args.harmonized_upenn_dir}")
    print(f"checkpoint dir: {args.checkpoint_dir}")
    print(f"output dir: {args.output_dir}")
    print(f"device: {args.device}")
    print(f"seg epochs: {args.seg_epochs}")
    print(f"seg lr: {args.seg_lr}")
    print(f"batch size: {args.batch_size}")
    print(f"image size: {args.image_size}")
    print(f"amp: {args.use_amp}")
    print(f"resume: {args.resume}")

    # option (b): generate harmonized data on the fly if checkpoint provided
    # but no pre-generated harmonized dirs
    harm_brats = args.harmonized_brats_dir
    harm_upenn = args.harmonized_upenn_dir

    if args.checkpoint_dir and (not harm_brats or not harm_upenn):
        print("\ngenerating harmonized data from checkpoint (option b)...")
        harm_output = Path(args.output_dir) / "harmonized"

        if not harm_brats:
            harm_brats = str(harm_output / "brats")
            generate_harmonized_data(
                checkpoint_path=args.checkpoint_dir,
                source_dir=args.brats_dir,
                output_dir=harm_brats,
                direction="a_to_b",
                device=args.device if args.device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )

        if not harm_upenn:
            harm_upenn = str(harm_output / "upenn")
            generate_harmonized_data(
                checkpoint_path=args.checkpoint_dir,
                source_dir=args.upenn_dir,
                output_dir=harm_upenn,
                direction="b_to_a",
                device=args.device if args.device != "auto" else (
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )

    # create evaluator
    evaluator = DownstreamEvaluator(
        output_dir=args.output_dir,
        device=args.device,
        n_classes=4,
        seg_epochs=args.seg_epochs,
        seg_lr=args.seg_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_amp=args.use_amp,
    )

    # run cross-site transfer evaluation (brats -> upenn)
    print("\n" + "=" * 60)
    print("  cross-site transfer: brats -> upenn")
    print("=" * 60)

    resume_dir = args.output_dir if args.resume else None

    results = evaluator.cross_site_transfer(
        site_a_dir=args.brats_dir,
        site_b_dir=args.upenn_dir,
        harm_a_dir=harm_brats,
        harm_b_dir=harm_upenn,
        resume_path=resume_dir,
    )

    # add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "brats_dir": args.brats_dir,
        "upenn_dir": args.upenn_dir,
        "harmonized_brats_dir": harm_brats,
        "harmonized_upenn_dir": harm_upenn,
        "seg_epochs": args.seg_epochs,
        "seg_lr": args.seg_lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "n_classes": 4,
        "class_names": CLASS_NAMES,
        "region_names": REGION_NAMES,
    }

    # save results
    evaluator.save_results(results, "downstream_results.json")

    # print summary
    print("\n" + "=" * 60)
    print("  summary")
    print("=" * 60)

    def _print_condition(name, key):
        if key in results:
            r = results[key]
            dice_fg = r.get("dice_mean_foreground_mean", 0)
            dice_std = r.get("dice_mean_foreground_std", 0)
            wt = r.get("region_wt_mean", 0)
            tc = r.get("region_tc_mean", 0)
            et = r.get("region_et_mean", 0)
            print(f"  {name}:")
            print(f"    mean foreground dice: {dice_fg:.4f} +/- {dice_std:.4f}")
            print(f"    wt: {wt:.4f}  tc: {tc:.4f}  et: {et:.4f}")

    _print_condition("raw a->b (baseline)", "raw_a_to_raw_b")
    _print_condition("raw a->a (within-site)", "raw_a_to_raw_a")
    _print_condition("harmonized a->b", "harm_a_to_harm_b")
    _print_condition("harmonized a->a", "harm_a_to_harm_a")
    _print_condition("raw b->a (reverse)", "raw_b_to_raw_a")
    _print_condition("harmonized b->a (reverse)", "harm_b_to_harm_a")

    if "improvement" in results:
        imp = results["improvement"]
        print(f"\n  cross-site improvement (a->b): "
              f"{imp['dice_mean_foreground_delta']:+.4f} "
              f"({imp['dice_mean_foreground_relative_pct']:+.1f}%)")
    if "improvement_reverse" in results:
        imp = results["improvement_reverse"]
        print(f"  cross-site improvement (b->a): "
              f"{imp['dice_mean_foreground_delta']:+.4f} "
              f"({imp['dice_mean_foreground_relative_pct']:+.1f}%)")

    print(f"\nresults saved to {args.output_dir}/downstream_results.json")
    print("done.")


if __name__ == "__main__":
    main()
