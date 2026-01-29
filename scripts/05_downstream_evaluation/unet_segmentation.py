#!/usr/bin/env python3
"""
2d u-net segmentation model for brain tumor segmentation.

this model is used for downstream task evaluation to assess the impact
of mri harmonization on segmentation performance.

architecture:
- encoder: 4 downsampling blocks with residual connections
- bottleneck: 2 residual blocks
- decoder: 4 upsampling blocks with skip connections
- output: multi-class segmentation (et, tc, wt) or binary

based on the standard u-net architecture with modifications for
medical image segmentation including:
- instance normalization
- leaky relu activations
- deep supervision (optional)
- attention gates (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock(nn.Module):
    """
    convolutional block with instance norm and activation.

    conv -> inorm -> activation -> conv -> inorm -> activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # residual connection
        if use_residual and in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_residual:
            if self.residual is not None:
                identity = self.residual(identity)
            out = out + identity

        out = self.act2(out)
        return out


class DownBlock(nn.Module):
    """
    encoder block: conv block followed by max pooling.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class UpBlock(nn.Module):
    """
    decoder block: upsample, concat with skip, conv block.
    """

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(in_channels // 2, in_channels // 2, in_channels // 4)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        if self.use_attention:
            skip = self.attention(skip, x)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionGate(nn.Module):
    """
    attention gate for focusing on relevant spatial regions.
    """

    def __init__(self, in_channels_g: int, in_channels_x: int, inter_channels: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, 1),
            nn.InstanceNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, 1),
            nn.InstanceNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # handle size mismatch
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet2D(nn.Module):
    """
    2d u-net for brain tumor segmentation.

    args:
        in_channels: number of input channels (4 for multi-modal mri)
        n_classes: number of output classes
        base_filters: number of filters in first layer
        use_attention: use attention gates in decoder
        deep_supervision: return intermediate outputs for deep supervision
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 4,
        base_filters: int = 32,
        use_attention: bool = False,
        deep_supervision: bool = False
    ):
        super().__init__()

        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        # encoder
        self.enc1 = DownBlock(in_channels, base_filters)
        self.enc2 = DownBlock(base_filters, base_filters * 2)
        self.enc3 = DownBlock(base_filters * 2, base_filters * 4)
        self.enc4 = DownBlock(base_filters * 4, base_filters * 8)

        # bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # decoder
        self.dec4 = UpBlock(base_filters * 16, base_filters * 8, use_attention)
        self.dec3 = UpBlock(base_filters * 8, base_filters * 4, use_attention)
        self.dec2 = UpBlock(base_filters * 4, base_filters * 2, use_attention)
        self.dec1 = UpBlock(base_filters * 2, base_filters, use_attention)

        # output
        self.out_conv = nn.Conv2d(base_filters, n_classes, 1)

        # deep supervision outputs
        if deep_supervision:
            self.ds4 = nn.Conv2d(base_filters * 8, n_classes, 1)
            self.ds3 = nn.Conv2d(base_filters * 4, n_classes, 1)
            self.ds2 = nn.Conv2d(base_filters * 2, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder path
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder path
        x = self.dec4(x, skip4)
        if self.deep_supervision:
            ds4 = self.ds4(x)

        x = self.dec3(x, skip3)
        if self.deep_supervision:
            ds3 = self.ds3(x)

        x = self.dec2(x, skip2)
        if self.deep_supervision:
            ds2 = self.ds2(x)

        x = self.dec1(x, skip1)

        # output
        out = self.out_conv(x)

        if self.deep_supervision:
            # upsample deep supervision outputs to match final size
            ds4 = F.interpolate(ds4, size=out.shape[2:], mode='bilinear', align_corners=True)
            ds3 = F.interpolate(ds3, size=out.shape[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(ds2, size=out.shape[2:], mode='bilinear', align_corners=True)
            return out, ds4, ds3, ds2

        return out


class DiceLoss(nn.Module):
    """
    dice loss for segmentation.

    supports both binary and multi-class segmentation.
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        compute dice loss.

        args:
            pred: predicted logits [b, c, h, w]
            target: ground truth [b, h, w] (class indices) or [b, c, h, w] (one-hot)

        returns:
            dice loss
        """
        # apply softmax to get probabilities
        pred = F.softmax(pred, dim=1)

        # convert target to one-hot if needed
        if target.dim() == 3:
            target = F.one_hot(target.long(), num_classes=pred.shape[1])
            target = target.permute(0, 3, 1, 2).float()

        # flatten
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        # compute dice per class
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # average across classes (excluding background if needed)
        dice_loss = 1 - dice.mean(dim=1)

        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    combined loss: dice + cross entropy.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target.long())
        return self.dice_weight * dice + self.ce_weight * ce


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    smooth: float = 1e-5
) -> torch.Tensor:
    """
    compute dice score per class.

    args:
        pred: predicted logits [b, c, h, w]
        target: ground truth [b, h, w]
        n_classes: number of classes
        smooth: smoothing factor

    returns:
        dice scores [n_classes]
    """
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=1)  # [b, h, w]

    dice_scores = []

    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return torch.tensor(dice_scores)


def compute_hausdorff_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    percentile: float = 95
) -> float:
    """
    compute hausdorff distance at given percentile.

    args:
        pred: binary prediction [h, w]
        target: binary ground truth [h, w]
        percentile: percentile for hausdorff distance

    returns:
        hausdorff distance at percentile
    """
    from scipy.ndimage import distance_transform_edt

    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)

    # handle empty masks
    if not pred_np.any() or not target_np.any():
        return float('inf')

    # compute distance transforms
    pred_dist = distance_transform_edt(~pred_np)
    target_dist = distance_transform_edt(~target_np)

    # get distances from pred boundary to target
    pred_boundary = pred_np ^ ndimage.binary_erosion(pred_np)
    target_boundary = target_np ^ ndimage.binary_erosion(target_np)

    if not pred_boundary.any() or not target_boundary.any():
        return float('inf')

    # distances
    dist_pred_to_target = target_dist[pred_boundary]
    dist_target_to_pred = pred_dist[target_boundary]

    # hausdorff at percentile
    hd_pred = np.percentile(dist_pred_to_target, percentile) if len(dist_pred_to_target) > 0 else 0
    hd_target = np.percentile(dist_target_to_pred, percentile) if len(dist_target_to_pred) > 0 else 0

    return max(hd_pred, hd_target)


# need scipy for hausdorff
from scipy import ndimage
import numpy as np


if __name__ == '__main__':
    # test model
    model = UNet2D(in_channels=4, n_classes=4, base_filters=32, use_attention=True)

    x = torch.randn(2, 4, 128, 128)
    y = model(x)

    print(f'input shape: {x.shape}')
    print(f'output shape: {y.shape}')
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')
