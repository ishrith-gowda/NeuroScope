"""
multi-layer patchnce loss for contrastive unpaired image translation.

implements the patchnce loss from park et al. (eccv 2020) with per-layer
mlp projection heads for multi-scale contrastive learning. designed to
integrate with sa-cyclegan-2.5d as a complementary content preservation
objective alongside cycle consistency.

reference:
    park et al., "contrastive learning for unpaired image-to-image
    translation", eccv 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MLPProjectionHead(nn.Module):
    """
    per-layer mlp projection head for patchnce.

    maps encoder features to a contrastive embedding space. the mlp acts as a
    firewall between the contrastive objective and the generative objective,
    preventing the contrastive loss from directly constraining encoder features.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: feature tensor of shape (b, c, h, w) or (b, n, c)
        returns:
            l2-normalized projected features of shape (b, n, out_channels)
        """
        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x


class PatchNCELoss(nn.Module):
    """
    patch-based noise contrastive estimation loss.

    for each spatial location, the feature vector from the generated image
    (query) should match the corresponding location in the source image
    (positive key) and differ from other locations (negative keys).

    this operates on intermediate encoder features at multiple scales,
    providing content preservation at different abstraction levels.
    """

    def __init__(
        self,
        num_patches: int = 256,
        temperature: float = 0.07,
    ):
        """
        args:
            num_patches: number of spatial locations to sample per layer
            temperature: softmax temperature for contrastive logits
        """
        super().__init__()
        self.num_patches = num_patches
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        compute patchnce loss between query and key features.

        args:
            feat_q: query features (from generated image), shape (b, n, c)
                    already projected and l2-normalized
            feat_k: key features (from source image, positive), shape (b, n, c)
                    already projected and l2-normalized
        returns:
            scalar infonce loss
        """
        b, n, c = feat_q.shape

        # positive logits: dot product at corresponding locations
        l_pos = (feat_q * feat_k).sum(dim=-1, keepdim=True)  # (b, n, 1)
        l_pos = l_pos / self.temperature

        # negative logits: dot product with all other locations (internal negatives)
        l_neg = torch.bmm(feat_q, feat_k.transpose(1, 2))  # (b, n, n)
        l_neg = l_neg / self.temperature

        # mask out positive pairs on the diagonal
        diag_mask = torch.eye(n, device=feat_q.device, dtype=torch.bool)
        l_neg = l_neg.masked_fill(diag_mask.unsqueeze(0), float("-inf"))

        # concatenate: positive at index 0, negatives follow
        logits = torch.cat([l_pos, l_neg], dim=-1)  # (b, n, 1 + n)

        # labels: positive is always at index 0
        labels = torch.zeros(b, n, dtype=torch.long, device=feat_q.device)

        loss = self.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return loss


class MultiLayerPatchNCELoss(nn.Module):
    """
    multi-layer patchnce loss with per-layer mlp projection heads.

    extracts features from multiple encoder layers and computes patchnce
    at each scale. this captures content preservation at both low-level
    (texture, edges) and high-level (semantic structure) abstraction.

    usage:
        # initialize with layer channel dimensions
        nce_loss = multilayerpatchnceloss(layer_channels=[64, 128, 256, 256, 256])

        # during training, extract features from generator encoder
        src_feats = generator.encode(real_a)       # list of feature maps
        gen_feats = generator.encode(fake_b)       # list of feature maps

        # compute loss
        loss = nce_loss(gen_feats, src_feats)
    """

    def __init__(
        self,
        layer_channels: List[int],
        projection_dim: int = 256,
        num_patches: int = 256,
        temperature: float = 0.07,
    ):
        """
        args:
            layer_channels: list of channel dimensions for each encoder layer
            projection_dim: output dimension of mlp projection heads
            num_patches: number of patches to sample per layer
            temperature: softmax temperature
        """
        super().__init__()

        self.num_patches = num_patches

        # per-layer mlp projection heads
        self.mlp_heads = nn.ModuleList([
            MLPProjectionHead(nc, projection_dim)
            for nc in layer_channels
        ])

        # shared patchnce loss
        self.nce_loss = PatchNCELoss(
            num_patches=num_patches,
            temperature=temperature,
        )

    def _sample_patches(
        self,
        feat: torch.Tensor,
        num_patches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        sample spatial locations from a feature map.

        args:
            feat: feature tensor of shape (b, c, h, w)
            num_patches: number of locations to sample
        returns:
            sampled features (b, num_patches, c) and patch indices
        """
        b, c, h, w = feat.shape
        n = h * w

        feat_flat = feat.permute(0, 2, 3, 1).reshape(b, n, c)

        if n <= num_patches:
            return feat_flat, None

        # sample same locations for entire batch (ensures correspondence)
        indices = torch.randperm(n, device=feat.device)[:num_patches]
        indices_expanded = indices.unsqueeze(0).unsqueeze(-1).expand(b, -1, c)
        sampled = feat_flat.gather(1, indices_expanded)

        return sampled, indices

    def forward(
        self,
        query_feats: List[torch.Tensor],
        key_feats: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        compute multi-layer patchnce loss.

        args:
            query_feats: list of feature maps from generated image encoder pass
                         each of shape (b, c_l, h_l, w_l)
            key_feats: list of feature maps from source image encoder pass
                       each of shape (b, c_l, h_l, w_l)
        returns:
            scalar multi-layer patchnce loss (averaged across layers)
        """
        assert len(query_feats) == len(key_feats) == len(self.mlp_heads), (
            f"expected {len(self.mlp_heads)} feature layers, "
            f"got {len(query_feats)} query and {len(key_feats)} key layers"
        )

        total_loss = 0.0
        n_layers = len(self.mlp_heads)

        for i, (feat_q, feat_k, mlp) in enumerate(
            zip(query_feats, key_feats, self.mlp_heads)
        ):
            # sample corresponding patch locations from both feature maps
            b, c, h, w = feat_q.shape
            n = h * w

            feat_q_flat = feat_q.permute(0, 2, 3, 1).reshape(b, n, c)
            feat_k_flat = feat_k.permute(0, 2, 3, 1).reshape(b, n, c)

            if n > self.num_patches:
                indices = torch.randperm(n, device=feat_q.device)[:self.num_patches]
                idx_exp = indices.unsqueeze(0).unsqueeze(-1).expand(b, -1, c)
                feat_q_flat = feat_q_flat.gather(1, idx_exp)
                feat_k_flat = feat_k_flat.gather(1, idx_exp)

            # project through per-layer mlp and l2-normalize
            feat_q_proj = mlp(feat_q_flat)
            feat_k_proj = mlp(feat_k_flat)

            # compute patchnce loss for this layer
            layer_loss = self.nce_loss(feat_q_proj, feat_k_proj)
            total_loss += layer_loss

        return total_loss / n_layers


if __name__ == "__main__":
    print("testing multi-layer patchnce loss...")

    # simulate 5-layer encoder features at decreasing resolutions
    layer_channels = [64, 128, 256, 256, 256]
    resolutions = [128, 64, 32, 32, 32]

    nce = MultiLayerPatchNCELoss(
        layer_channels=layer_channels,
        projection_dim=256,
        num_patches=256,
        temperature=0.07,
    )

    # create dummy features
    query_feats = [torch.randn(2, c, r, r) for c, r in zip(layer_channels, resolutions)]
    key_feats = [torch.randn(2, c, r, r) for c, r in zip(layer_channels, resolutions)]

    loss = nce(query_feats, key_feats)
    print(f"multi-layer patchnce loss: {loss.item():.4f}")

    # test with identical features (should give low loss)
    loss_identical = nce(query_feats, query_feats)
    print(f"patchnce loss (identical features): {loss_identical.item():.4f}")

    # verify loss is lower for identical features
    assert loss_identical.item() < loss.item(), "identical features should have lower loss"
    print("all tests passed")
