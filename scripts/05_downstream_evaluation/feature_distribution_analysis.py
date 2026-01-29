#!/usr/bin/env python3
"""
feature distribution analysis for mri harmonization evaluation.

this module computes comprehensive feature-based metrics to evaluate
harmonization effectiveness by analyzing distribution shifts between domains.

metrics computed:
- frechet inception distance (fid) - using medical imaging features
- maximum mean discrepancy (mmd) with multiple kernels
- kernel inception distance (kid)
- sliced wasserstein distance
- feature-space t-sne and umap visualizations

the key principle: effective harmonization should reduce the distributional
distance between domains in learned feature space.

references:
- heusel et al., "gans trained by a two time-scale update rule converge to
  a local nash equilibrium", neurips 2017 (fid)
- gretton et al., "a kernel two-sample test", jmlr 2012 (mmd)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import linalg, stats
from sklearn.manifold import TSNE
from tqdm import tqdm

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class FeatureExtractor(nn.Module):
    """
    feature extractor for mri images.

    uses a pretrained-style architecture optimized for medical imaging.
    extracts multi-scale features for comprehensive distribution analysis.
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 512):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv2 = self._make_block(64, 64, 2)
        self.conv3 = self._make_block(64, 128, 2, stride=2)
        self.conv4 = self._make_block(128, 256, 2, stride=2)
        self.conv5 = self._make_block(256, 512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, feature_dim)

        self.feature_dim = feature_dim

    def _make_block(self, in_channels: int, out_channels: int,
                   n_blocks: int, stride: int = 1) -> nn.Sequential:
        """create residual block."""
        layers = []

        # first block with potential downsampling
        layers.append(self._make_residual(in_channels, out_channels, stride))

        # remaining blocks
        for _ in range(1, n_blocks):
            layers.append(self._make_residual(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def _make_residual(self, in_channels: int, out_channels: int,
                      stride: int) -> nn.Module:
        """create single residual unit."""
        class ResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, s):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=1, bias=False)
                self.bn1 = nn.InstanceNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
                self.bn2 = nn.InstanceNorm2d(out_ch)

                if s != 1 or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 1, stride=s, bias=False),
                        nn.InstanceNorm2d(out_ch)
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + self.shortcut(x)
                return F.relu(out)

        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def extract_multiscale(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """extract features at multiple scales."""
        features = {}

        x = self.conv1(x)
        x = self.conv2(x)
        features['scale1'] = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = self.conv3(x)
        features['scale2'] = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = self.conv4(x)
        features['scale3'] = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        x = self.conv5(x)
        features['scale4'] = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)

        features['final'] = self.fc(features['scale4'])

        return features


class NiftiFeatureDataset(Dataset):
    """
    dataset for extracting features from nifti volumes.
    """

    def __init__(
        self,
        data_dir: Path,
        modalities: List[str] = ['t1', 't1gd', 't2', 'flair'],
        slice_range: Tuple[int, int] = (50, 110),
        slice_stride: int = 5,
        slice_size: Tuple[int, int] = (128, 128)
    ):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.slice_range = slice_range
        self.slice_stride = slice_stride
        self.slice_size = slice_size

        self.samples = []
        self._index_data()

        print(f'[featuredata] indexed {len(self.samples)} slices from {self.data_dir}')

    def _index_data(self):
        """index all slices."""
        if not self.data_dir.exists():
            print(f'[warning] directory not found: {self.data_dir}')
            return

        for subject_dir in sorted(self.data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            # check modalities
            mod_files = {}
            for mod in self.modalities:
                candidates = list(subject_dir.glob(f'*{mod}*.nii.gz'))
                if not candidates:
                    candidates = list(subject_dir.glob(f'{mod}.nii.gz'))
                if candidates:
                    mod_files[mod] = candidates[0]

            if len(mod_files) != len(self.modalities):
                continue

            # get shape
            import nibabel as nib
            first_mod = list(mod_files.values())[0]
            img = nib.load(str(first_mod))
            n_slices = img.shape[2]

            start = max(self.slice_range[0], 0)
            end = min(self.slice_range[1], n_slices)

            for slice_idx in range(start, end, self.slice_stride):
                self.samples.append((subject_dir, slice_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        subject_dir, slice_idx = self.samples[idx]

        import nibabel as nib
        from skimage.transform import resize

        channels = []
        for mod in self.modalities:
            candidates = list(subject_dir.glob(f'*{mod}*.nii.gz'))
            if not candidates:
                candidates = list(subject_dir.glob(f'{mod}.nii.gz'))

            img = nib.load(str(candidates[0]))
            vol = img.get_fdata()
            slice_data = vol[:, :, slice_idx]

            if slice_data.shape != self.slice_size:
                slice_data = resize(slice_data, self.slice_size, preserve_range=True)

            # normalize
            mask = slice_data > 0
            if mask.sum() > 0:
                mean = slice_data[mask].mean()
                std = slice_data[mask].std() + 1e-8
                slice_data = (slice_data - mean) / std
                slice_data[~mask] = 0

            channels.append(slice_data)

        data = np.stack(channels, axis=0).astype(np.float32)
        return torch.from_numpy(data)


def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    harmonization_model: Optional[nn.Module] = None
) -> np.ndarray:
    """
    extract features from all samples.

    optionally applies harmonization before feature extraction.
    """
    model.eval()
    all_features = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, desc='extracting features'):
            inputs = inputs.to(device)

            if harmonization_model is not None:
                inputs = harmonize_batch(inputs, harmonization_model, device)

            features = model(inputs)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def harmonize_batch(
    inputs: torch.Tensor,
    harmonization_model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """apply harmonization to batch."""
    b, c, h, w = inputs.shape

    # replicate for 2.5d model
    inputs_3slice = inputs.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    inputs_3slice = inputs_3slice.view(b, c * 3, h, w)

    harmonization_model.eval()
    with torch.no_grad():
        harmonized = harmonization_model.G_A2B(inputs_3slice)

    if harmonized.shape[1] == c * 3:
        harmonized = harmonized.view(b, c, 3, h, w)[:, :, 1, :, :]
    elif harmonized.shape[1] != c:
        harmonized = harmonized[:, :c, :, :]

    return harmonized


def compute_fid(
    features_a: np.ndarray,
    features_b: np.ndarray
) -> float:
    """
    compute frechet inception distance between feature distributions.

    fid = ||mu_a - mu_b||^2 + tr(sigma_a + sigma_b - 2*sqrt(sigma_a * sigma_b))

    lower fid indicates more similar distributions.
    """
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)

    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    # add small epsilon for numerical stability
    eps = 1e-6
    sigma_a += np.eye(sigma_a.shape[0]) * eps
    sigma_b += np.eye(sigma_b.shape[0]) * eps

    # mean difference
    diff = mu_a - mu_b
    mean_diff = np.dot(diff, diff)

    # matrix sqrt via eigendecomposition
    covmean, _ = linalg.sqrtm(sigma_a @ sigma_b, disp=False)

    # handle complex numbers from numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # compute fid
    fid = mean_diff + np.trace(sigma_a + sigma_b - 2 * covmean)

    return float(fid)


def compute_kid(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_subsets: int = 100,
    subset_size: int = 1000
) -> Tuple[float, float]:
    """
    compute kernel inception distance with polynomial kernel.

    more robust to sample size than fid.
    returns mean and standard deviation across subsets.
    """
    n_a = len(features_a)
    n_b = len(features_b)

    actual_subset_size = min(subset_size, n_a, n_b)

    def polynomial_kernel(x, y, degree=3, coef0=1, gamma=None):
        """polynomial kernel k(x,y) = (gamma * x.y + coef0)^degree"""
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        return (gamma * np.dot(x, y.T) + coef0) ** degree

    kid_values = []

    for _ in range(n_subsets):
        idx_a = np.random.choice(n_a, actual_subset_size, replace=False)
        idx_b = np.random.choice(n_b, actual_subset_size, replace=False)

        fa = features_a[idx_a]
        fb = features_b[idx_b]

        k_aa = polynomial_kernel(fa, fa)
        k_bb = polynomial_kernel(fb, fb)
        k_ab = polynomial_kernel(fa, fb)

        # unbiased estimator
        m = actual_subset_size
        mmd = (k_aa.sum() - np.trace(k_aa)) / (m * (m - 1)) + \
              (k_bb.sum() - np.trace(k_bb)) / (m * (m - 1)) - \
              2 * k_ab.mean()

        kid_values.append(mmd)

    return float(np.mean(kid_values)), float(np.std(kid_values))


def compute_mmd(
    features_a: np.ndarray,
    features_b: np.ndarray,
    kernel: str = 'rbf',
    sigma: float = 1.0
) -> float:
    """
    compute maximum mean discrepancy between distributions.

    supports rbf and linear kernels.
    """
    n_samples = min(1000, len(features_a), len(features_b))
    idx_a = np.random.choice(len(features_a), n_samples, replace=False)
    idx_b = np.random.choice(len(features_b), n_samples, replace=False)

    fa = features_a[idx_a]
    fb = features_b[idx_b]

    if kernel == 'rbf':
        def k(x, y):
            diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
            return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))
    elif kernel == 'linear':
        def k(x, y):
            return np.dot(x, y.T)
    else:
        raise ValueError(f'unknown kernel: {kernel}')

    k_aa = k(fa, fa).mean()
    k_bb = k(fb, fb).mean()
    k_ab = k(fa, fb).mean()

    mmd = k_aa + k_bb - 2 * k_ab
    return float(mmd)


def compute_sliced_wasserstein(
    features_a: np.ndarray,
    features_b: np.ndarray,
    n_projections: int = 1000
) -> float:
    """
    compute sliced wasserstein distance.

    projects distributions onto random 1d subspaces and computes
    wasserstein distance in each, then averages.
    """
    dim = features_a.shape[1]

    # random projections
    projections = np.random.randn(n_projections, dim)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

    # project features
    proj_a = features_a @ projections.T
    proj_b = features_b @ projections.T

    # compute 1d wasserstein for each projection
    swd = 0
    for i in range(n_projections):
        sorted_a = np.sort(proj_a[:, i])
        sorted_b = np.sort(proj_b[:, i])

        # interpolate to same length
        n = min(len(sorted_a), len(sorted_b))
        idx_a = np.linspace(0, len(sorted_a) - 1, n).astype(int)
        idx_b = np.linspace(0, len(sorted_b) - 1, n).astype(int)

        swd += np.mean(np.abs(sorted_a[idx_a] - sorted_b[idx_b]))

    return float(swd / n_projections)


def compute_distribution_metrics(
    features_a: np.ndarray,
    features_b: np.ndarray
) -> Dict:
    """
    compute all distribution metrics.
    """
    print('[metrics] computing fid...')
    fid = compute_fid(features_a, features_b)

    print('[metrics] computing kid...')
    kid_mean, kid_std = compute_kid(features_a, features_b)

    print('[metrics] computing mmd (rbf)...')
    mmd_rbf = compute_mmd(features_a, features_b, kernel='rbf', sigma=1.0)

    print('[metrics] computing mmd (linear)...')
    mmd_linear = compute_mmd(features_a, features_b, kernel='linear')

    print('[metrics] computing sliced wasserstein...')
    swd = compute_sliced_wasserstein(features_a, features_b)

    # statistical tests
    print('[metrics] computing statistical tests...')

    # multivariate two-sample test using mmd
    # permutation test for significance
    n_perms = 100
    combined = np.vstack([features_a, features_b])
    n_a = len(features_a)
    observed_mmd = mmd_rbf

    perm_mmds = []
    for _ in range(n_perms):
        perm_idx = np.random.permutation(len(combined))
        perm_a = combined[perm_idx[:n_a]]
        perm_b = combined[perm_idx[n_a:]]
        perm_mmds.append(compute_mmd(perm_a, perm_b, kernel='rbf', sigma=1.0))

    p_value = (np.sum(np.array(perm_mmds) >= observed_mmd) + 1) / (n_perms + 1)

    return {
        'fid': fid,
        'kid_mean': kid_mean,
        'kid_std': kid_std,
        'mmd_rbf': mmd_rbf,
        'mmd_linear': mmd_linear,
        'sliced_wasserstein': swd,
        'mmd_permutation_p_value': float(p_value),
    }


def compute_tsne_embedding(
    features: np.ndarray,
    perplexity: int = 30,
    max_iter: int = 1000
) -> np.ndarray:
    """compute t-sne embedding."""
    max_samples = 3000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]

    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=max_iter)
    return tsne.fit_transform(features)


def main():
    parser = argparse.ArgumentParser(
        description='feature distribution analysis for harmonization evaluation'
    )
    parser.add_argument('--domain-a-dir', type=str, required=True,
                       help='path to domain a (brats) data')
    parser.add_argument('--domain-b-dir', type=str, required=True,
                       help='path to domain b (upenn) data')
    parser.add_argument('--harmonization-model', type=str, default=None,
                       help='path to harmonization model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='device')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[featdist] using device: {device}')

    # create datasets
    print('[featdist] loading data...')
    dataset_a = NiftiFeatureDataset(
        data_dir=args.domain_a_dir,
        modalities=['t1', 't1gd', 't2', 'flair'],
        slice_range=(50, 110),
        slice_stride=5
    )
    dataset_b = NiftiFeatureDataset(
        data_dir=args.domain_b_dir,
        modalities=['t1', 't1gd', 't2', 'flair'],
        slice_range=(50, 110),
        slice_stride=5
    )

    loader_a = DataLoader(dataset_a, batch_size=args.batch_size,
                         shuffle=False, num_workers=4)
    loader_b = DataLoader(dataset_b, batch_size=args.batch_size,
                         shuffle=False, num_workers=4)

    # create feature extractor
    feature_extractor = FeatureExtractor(in_channels=4, feature_dim=512)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # extract raw features
    print('[featdist] extracting raw features from domain a...')
    features_a_raw = extract_features(feature_extractor, loader_a, device)

    print('[featdist] extracting raw features from domain b...')
    features_b_raw = extract_features(feature_extractor, loader_b, device)

    print(f'[featdist] extracted {len(features_a_raw)} features from domain a')
    print(f'[featdist] extracted {len(features_b_raw)} features from domain b')

    # compute raw metrics
    print('[featdist] computing raw distribution metrics...')
    raw_metrics = compute_distribution_metrics(features_a_raw, features_b_raw)

    # load harmonization model if provided
    harmonized_metrics = None
    if args.harmonization_model:
        print('[featdist] loading harmonization model...')
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig

        config = SACycleGAN25DConfig()
        harmonization_model = SACycleGAN25D(config)
        checkpoint = torch.load(args.harmonization_model, map_location=device, weights_only=False)

        if 'G_A2B_state_dict' in checkpoint:
            harmonization_model.G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # handle dataparallel wrapped models
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                # remove 'module.' prefix if present
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            harmonization_model.load_state_dict(new_state_dict, strict=False)

        harmonization_model = harmonization_model.to(device)
        harmonization_model.eval()

        # extract harmonized features (harmonize a -> b, then compare with b)
        print('[featdist] extracting harmonized features from domain a...')
        features_a_harmonized = extract_features(
            feature_extractor, loader_a, device, harmonization_model
        )

        print('[featdist] computing harmonized distribution metrics...')
        harmonized_metrics = compute_distribution_metrics(
            features_a_harmonized, features_b_raw
        )

    # compute t-sne embeddings
    print('[featdist] computing t-sne embeddings...')

    # combine features for visualization
    n_a = min(1500, len(features_a_raw))
    n_b = min(1500, len(features_b_raw))

    idx_a = np.random.choice(len(features_a_raw), n_a, replace=False)
    idx_b = np.random.choice(len(features_b_raw), n_b, replace=False)

    combined_raw = np.vstack([features_a_raw[idx_a], features_b_raw[idx_b]])
    labels_raw = np.array([0] * n_a + [1] * n_b)

    tsne_raw = compute_tsne_embedding(combined_raw)

    if harmonized_metrics:
        features_a_harm_sub = features_a_harmonized[idx_a]
        combined_harm = np.vstack([features_a_harm_sub, features_b_raw[idx_b]])
        tsne_harm = compute_tsne_embedding(combined_harm)

    # save results
    results = {
        'raw': raw_metrics,
        'n_samples_a': len(features_a_raw),
        'n_samples_b': len(features_b_raw),
    }

    if harmonized_metrics:
        results['harmonized'] = harmonized_metrics

        # compute improvement (reduction is good)
        results['improvement'] = {
            'fid_reduction': raw_metrics['fid'] - harmonized_metrics['fid'],
            'fid_reduction_percent': 100 * (raw_metrics['fid'] - harmonized_metrics['fid']) / raw_metrics['fid'],
            'kid_reduction': raw_metrics['kid_mean'] - harmonized_metrics['kid_mean'],
            'mmd_reduction': raw_metrics['mmd_rbf'] - harmonized_metrics['mmd_rbf'],
            'swd_reduction': raw_metrics['sliced_wasserstein'] - harmonized_metrics['sliced_wasserstein'],
        }

    with open(output_dir / 'feature_distribution_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # save embeddings
    np.save(output_dir / 'tsne_raw.npy', tsne_raw)
    np.save(output_dir / 'tsne_labels.npy', labels_raw)
    np.save(output_dir / 'features_a_raw.npy', features_a_raw)
    np.save(output_dir / 'features_b_raw.npy', features_b_raw)

    if harmonized_metrics:
        np.save(output_dir / 'tsne_harmonized.npy', tsne_harm)
        np.save(output_dir / 'features_a_harmonized.npy', features_a_harmonized)

    # save feature extractor
    torch.save(feature_extractor.state_dict(), output_dir / 'feature_extractor.pth')

    print('=' * 60)
    print('[featdist] results summary:')
    print(f'  raw fid: {raw_metrics["fid"]:.4f}')
    print(f'  raw kid: {raw_metrics["kid_mean"]:.4f} +/- {raw_metrics["kid_std"]:.4f}')
    print(f'  raw mmd (rbf): {raw_metrics["mmd_rbf"]:.4f}')
    print(f'  raw sliced wasserstein: {raw_metrics["sliced_wasserstein"]:.4f}')

    if harmonized_metrics:
        print()
        print(f'  harmonized fid: {harmonized_metrics["fid"]:.4f}')
        print(f'  harmonized kid: {harmonized_metrics["kid_mean"]:.4f} +/- {harmonized_metrics["kid_std"]:.4f}')
        print(f'  harmonized mmd (rbf): {harmonized_metrics["mmd_rbf"]:.4f}')
        print(f'  harmonized sliced wasserstein: {harmonized_metrics["sliced_wasserstein"]:.4f}')
        print()
        print(f'  fid reduction: {results["improvement"]["fid_reduction"]:.4f} '
              f'({results["improvement"]["fid_reduction_percent"]:.1f}%)')

    print('=' * 60)
    print(f'[featdist] results saved to {output_dir}')


if __name__ == '__main__':
    main()
