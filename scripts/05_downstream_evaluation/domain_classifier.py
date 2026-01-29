#!/usr/bin/env python3
"""
domain classifier for evaluating harmonization effectiveness.

this module implements a domain classification approach to evaluate mri harmonization.
the key insight: if harmonization is effective, a domain classifier should have
difficulty distinguishing between harmonized images from different source domains.

evaluation metrics:
- domain classification accuracy (lower = better harmonization)
- area under roc curve for domain discrimination
- t-sne/umap visualization of feature space before/after harmonization
- feature distance metrics between domains

this is a standard evaluation approach in harmonization literature, complementing
segmentation-based evaluation when segmentation labels are unavailable.
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, f1_score
)
from sklearn.manifold import TSNE
from scipy import stats
from tqdm import tqdm

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DomainClassifier(nn.Module):
    """
    cnn-based domain classifier for distinguishing mri domains.

    architecture: lightweight cnn with global average pooling.
    designed to be discriminative enough to detect domain differences
    but not so powerful that it overfits to minor variations.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 32,
        n_domains: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.features = nn.Sequential(
            # block 1: 128 -> 64
            nn.Conv2d(in_channels, base_filters, 3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, base_filters, 3, padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            # block 2: 64 -> 32
            nn.Conv2d(base_filters, base_filters * 2, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            # block 3: 32 -> 16
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),

            # block 4: 16 -> 8
            nn.Conv2d(base_filters * 4, base_filters * 8, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 8, base_filters * 8, 3, padding=1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 8, base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 4, n_domains),
        )

        self.feature_dim = base_filters * 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """extract feature representation before classifier."""
        features = self.features(x)
        return features.view(features.size(0), -1)


class MRIDomainDataset(Dataset):
    """
    dataset for domain classification.

    loads preprocessed mri slices from two domains (brats, upenn) and
    assigns binary labels for domain classification.
    """

    def __init__(
        self,
        domain_a_dir: Path,
        domain_b_dir: Path,
        split: str = 'train',
        split_file: Optional[Path] = None,
        max_slices_per_subject: int = 10,
        slice_size: Tuple[int, int] = (128, 128)
    ):
        self.domain_a_dir = Path(domain_a_dir)
        self.domain_b_dir = Path(domain_b_dir)
        self.slice_size = slice_size

        self.samples = []
        self.labels = []

        # load samples from both domains
        self._load_domain(self.domain_a_dir, domain_label=0,
                         max_slices=max_slices_per_subject)
        self._load_domain(self.domain_b_dir, domain_label=1,
                         max_slices=max_slices_per_subject)

        print(f'[domaindata] loaded {len(self.samples)} samples '
              f'(domain a: {self.labels.count(0)}, domain b: {self.labels.count(1)})')

    def _load_domain(self, domain_dir: Path, domain_label: int, max_slices: int):
        """load samples from a single domain."""
        if not domain_dir.exists():
            print(f'[warning] domain directory not found: {domain_dir}')
            return

        for subject_dir in sorted(domain_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            # find slice files
            slice_files = sorted(subject_dir.glob('*.npy'))
            if not slice_files:
                # check for nifti files
                slice_files = sorted(subject_dir.glob('*.nii.gz'))

            # limit slices per subject
            slice_files = slice_files[:max_slices]

            for f in slice_files:
                self.samples.append(f)
                self.labels.append(domain_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath = self.samples[idx]
        label = self.labels[idx]

        # load data
        if filepath.suffix == '.npy':
            data = np.load(filepath)
        else:
            # load nifti and extract middle slice
            import nibabel as nib
            img = nib.load(str(filepath))
            vol = img.get_fdata()
            mid_slice = vol.shape[2] // 2
            data = vol[:, :, mid_slice]
            if data.ndim == 2:
                data = data[np.newaxis, ...]

        # ensure correct shape
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # resize if needed
        if data.shape[-2:] != self.slice_size:
            from skimage.transform import resize
            resized = np.zeros((data.shape[0],) + self.slice_size, dtype=np.float32)
            for c in range(data.shape[0]):
                resized[c] = resize(data[c], self.slice_size, preserve_range=True)
            data = resized

        # normalize
        data = self._normalize(data)

        return torch.from_numpy(data.astype(np.float32)), label

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """z-score normalization per channel."""
        for c in range(data.shape[0]):
            mask = data[c] > 0
            if mask.sum() > 0:
                mean = data[c][mask].mean()
                std = data[c][mask].std() + 1e-8
                data[c] = (data[c] - mean) / std
                data[c][~mask] = 0
        return data


class NiftiDomainDataset(Dataset):
    """
    dataset for domain classification using nifti files directly.

    extracts multiple slices from each 3d volume for training.
    """

    def __init__(
        self,
        domain_a_dir: Path,
        domain_b_dir: Path,
        modalities: List[str] = ['t1', 't1gd', 't2', 'flair'],
        slice_range: Tuple[int, int] = (50, 110),
        slice_stride: int = 5,
        slice_size: Tuple[int, int] = (128, 128)
    ):
        self.modalities = modalities
        self.slice_range = slice_range
        self.slice_stride = slice_stride
        self.slice_size = slice_size

        self.samples = []  # list of (subject_dir, slice_idx, domain_label)

        # load from both domains
        self._index_domain(Path(domain_a_dir), domain_label=0)
        self._index_domain(Path(domain_b_dir), domain_label=1)

        # balance domains
        domain_0 = [s for s in self.samples if s[2] == 0]
        domain_1 = [s for s in self.samples if s[2] == 1]
        min_samples = min(len(domain_0), len(domain_1))

        np.random.seed(42)
        domain_0 = [domain_0[i] for i in np.random.permutation(len(domain_0))[:min_samples]]
        domain_1 = [domain_1[i] for i in np.random.permutation(len(domain_1))[:min_samples]]

        self.samples = domain_0 + domain_1
        np.random.shuffle(self.samples)

        print(f'[niftidata] indexed {len(self.samples)} slices '
              f'(domain a: {len(domain_0)}, domain b: {len(domain_1)})')

    def _index_domain(self, domain_dir: Path, domain_label: int):
        """index all slices from a domain."""
        if not domain_dir.exists():
            print(f'[warning] domain directory not found: {domain_dir}')
            return

        for subject_dir in sorted(domain_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            # check if all modalities exist
            mod_files = {}
            for mod in self.modalities:
                candidates = list(subject_dir.glob(f'*{mod}*.nii.gz'))
                if not candidates:
                    candidates = list(subject_dir.glob(f'{mod}.nii.gz'))
                if candidates:
                    mod_files[mod] = candidates[0]

            if len(mod_files) != len(self.modalities):
                continue

            # get volume shape from first modality
            import nibabel as nib
            first_mod = list(mod_files.values())[0]
            img = nib.load(str(first_mod))
            n_slices = img.shape[2]

            # index slices
            start = max(self.slice_range[0], 0)
            end = min(self.slice_range[1], n_slices)

            for slice_idx in range(start, end, self.slice_stride):
                self.samples.append((subject_dir, slice_idx, domain_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        subject_dir, slice_idx, domain_label = self.samples[idx]

        import nibabel as nib
        from skimage.transform import resize

        # load all modalities
        channels = []
        for mod in self.modalities:
            candidates = list(subject_dir.glob(f'*{mod}*.nii.gz'))
            if not candidates:
                candidates = list(subject_dir.glob(f'{mod}.nii.gz'))

            img = nib.load(str(candidates[0]))
            vol = img.get_fdata()
            slice_data = vol[:, :, slice_idx]

            # resize
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
        return torch.from_numpy(data), domain_label


def train_domain_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 10
) -> Dict:
    """
    train domain classifier.

    returns training history and best model state.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_labels, all_probs)

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        print(f'[epoch {epoch+1:03d}] train loss: {train_loss:.4f}, acc: {train_acc:.4f} | '
              f'val loss: {val_loss:.4f}, acc: {val_acc:.4f}, auc: {val_auc:.4f}')

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[training] early stopping at epoch {epoch+1}')
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history['best_val_acc'] = best_val_acc
    return history


def evaluate_domain_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    harmonization_model: Optional[nn.Module] = None
) -> Dict:
    """
    evaluate domain classifier on test data.

    optionally apply harmonization to inputs before classification.
    lower accuracy after harmonization indicates better domain adaptation.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='evaluating'):
            inputs = inputs.to(device)

            # optionally harmonize
            if harmonization_model is not None:
                inputs = harmonize_for_evaluation(inputs, harmonization_model, device)

            outputs = model(inputs)
            features = model.extract_features(inputs)

            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    # compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'features': all_features
    }


def harmonize_for_evaluation(
    inputs: torch.Tensor,
    harmonization_model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    apply harmonization model to inputs for domain classification evaluation.

    handles the conversion from 2d evaluation format to 2.5d model format.
    """
    b, c, h, w = inputs.shape

    # for 2.5d model, replicate slice 3 times
    inputs_3slice = inputs.unsqueeze(2).repeat(1, 1, 3, 1, 1)
    inputs_3slice = inputs_3slice.view(b, c * 3, h, w)

    # harmonize (a -> b)
    harmonization_model.eval()
    with torch.no_grad():
        harmonized = harmonization_model.G_A2B(inputs_3slice)

    # extract center slice
    if harmonized.shape[1] == c * 3:
        harmonized = harmonized.view(b, c, 3, h, w)[:, :, 1, :, :]
    elif harmonized.shape[1] == c:
        pass  # already single slice output
    else:
        # take first c channels
        harmonized = harmonized[:, :c, :, :]

    return harmonized


def compute_feature_statistics(
    features_a: np.ndarray,
    features_b: np.ndarray
) -> Dict:
    """
    compute statistical comparisons between feature distributions.

    metrics:
    - maximum mean discrepancy (mmd)
    - frechet distance (simplified)
    - ks test for each feature dimension
    - cosine similarity of distribution means
    """
    # mean and covariance
    mu_a = np.mean(features_a, axis=0)
    mu_b = np.mean(features_b, axis=0)

    # frechet distance (simplified - just mean difference)
    mean_diff = np.linalg.norm(mu_a - mu_b)

    # cosine similarity
    cosine_sim = np.dot(mu_a, mu_b) / (np.linalg.norm(mu_a) * np.linalg.norm(mu_b) + 1e-8)

    # mmd with rbf kernel (simplified)
    def rbf_kernel(x, y, sigma=1.0):
        diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
        return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))

    # subsample for efficiency
    n_samples = min(500, len(features_a), len(features_b))
    idx_a = np.random.choice(len(features_a), n_samples, replace=False)
    idx_b = np.random.choice(len(features_b), n_samples, replace=False)

    fa_sub = features_a[idx_a]
    fb_sub = features_b[idx_b]

    k_aa = rbf_kernel(fa_sub, fa_sub).mean()
    k_bb = rbf_kernel(fb_sub, fb_sub).mean()
    k_ab = rbf_kernel(fa_sub, fb_sub).mean()
    mmd = k_aa + k_bb - 2 * k_ab

    # ks test on random feature dimensions
    n_dims = min(10, features_a.shape[1])
    dim_indices = np.random.choice(features_a.shape[1], n_dims, replace=False)
    ks_stats = []
    for dim in dim_indices:
        stat, _ = stats.ks_2samp(features_a[:, dim], features_b[:, dim])
        ks_stats.append(stat)

    return {
        'mean_difference': float(mean_diff),
        'cosine_similarity': float(cosine_sim),
        'mmd': float(mmd),
        'mean_ks_statistic': float(np.mean(ks_stats)),
    }


def compute_tsne_embedding(
    features: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute t-sne embedding for visualization.

    subsamples if too many samples for efficiency.
    """
    max_samples = 2000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(features)

    return embedding, labels


def main():
    parser = argparse.ArgumentParser(
        description='domain classification evaluation for harmonization'
    )
    parser.add_argument('--domain-a-dir', type=str, required=True,
                       help='path to domain a (brats) preprocessed data')
    parser.add_argument('--domain-b-dir', type=str, required=True,
                       help='path to domain b (upenn) preprocessed data')
    parser.add_argument('--harmonization-model', type=str, default=None,
                       help='path to trained harmonization model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for results')
    parser.add_argument('--n-epochs', type=int, default=50,
                       help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed')

    args = parser.parse_args()

    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[domaineval] using device: {device}')

    # create dataset
    print('[domaineval] loading data...')
    dataset = NiftiDomainDataset(
        domain_a_dir=args.domain_a_dir,
        domain_b_dir=args.domain_b_dir,
        modalities=['t1', 't1gd', 't2', 'flair'],
        slice_range=(50, 110),
        slice_stride=5
    )

    # split into train/val/test
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)

    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print(f'[domaineval] train: {len(train_indices)}, val: {len(val_indices)}, '
          f'test: {len(test_indices)}')

    # create model
    model = DomainClassifier(in_channels=4, base_filters=32, n_domains=2)
    print(f'[domaineval] model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # train
    print('[domaineval] training domain classifier...')
    history = train_domain_classifier(
        model, train_loader, val_loader, device,
        n_epochs=args.n_epochs, lr=args.lr
    )

    # save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # evaluate on raw data
    print('[domaineval] evaluating on raw data...')
    raw_results = evaluate_domain_classifier(model, test_loader, device)

    print(f'[domaineval] raw data - acc: {raw_results["accuracy"]:.4f}, '
          f'auc: {raw_results["auc"]:.4f}')

    # evaluate with harmonization if model provided
    harmonized_results = None
    if args.harmonization_model:
        print('[domaineval] loading harmonization model...')
        # load harmonization model
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

        print('[domaineval] evaluating on harmonized data...')
        harmonized_results = evaluate_domain_classifier(
            model, test_loader, device, harmonization_model
        )

        print(f'[domaineval] harmonized data - acc: {harmonized_results["accuracy"]:.4f}, '
              f'auc: {harmonized_results["auc"]:.4f}')

    # compute feature statistics
    print('[domaineval] computing feature statistics...')

    # split features by domain
    raw_features_a = raw_results['features'][np.array(raw_results['labels']) == 0]
    raw_features_b = raw_results['features'][np.array(raw_results['labels']) == 1]
    raw_feature_stats = compute_feature_statistics(raw_features_a, raw_features_b)

    if harmonized_results:
        harm_features_a = harmonized_results['features'][np.array(harmonized_results['labels']) == 0]
        harm_features_b = harmonized_results['features'][np.array(harmonized_results['labels']) == 1]
        harm_feature_stats = compute_feature_statistics(harm_features_a, harm_features_b)

    # compute t-sne embeddings
    print('[domaineval] computing t-sne embeddings...')
    raw_tsne, raw_tsne_labels = compute_tsne_embedding(
        raw_results['features'], np.array(raw_results['labels'])
    )

    if harmonized_results:
        harm_tsne, harm_tsne_labels = compute_tsne_embedding(
            harmonized_results['features'], np.array(harmonized_results['labels'])
        )

    # save results
    results = {
        'raw': {
            'accuracy': raw_results['accuracy'],
            'auc': raw_results['auc'],
            'f1': raw_results['f1'],
            'confusion_matrix': raw_results['confusion_matrix'],
            'feature_statistics': raw_feature_stats,
        }
    }

    if harmonized_results:
        results['harmonized'] = {
            'accuracy': harmonized_results['accuracy'],
            'auc': harmonized_results['auc'],
            'f1': harmonized_results['f1'],
            'confusion_matrix': harmonized_results['confusion_matrix'],
            'feature_statistics': harm_feature_stats,
        }

        # improvement metrics (lower is better for harmonization)
        results['improvement'] = {
            'accuracy_reduction': raw_results['accuracy'] - harmonized_results['accuracy'],
            'auc_reduction': raw_results['auc'] - harmonized_results['auc'],
            'mmd_reduction': raw_feature_stats['mmd'] - harm_feature_stats['mmd'],
        }

    with open(output_dir / 'domain_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # save embeddings
    np.save(output_dir / 'raw_tsne_embedding.npy', raw_tsne)
    np.save(output_dir / 'raw_tsne_labels.npy', raw_tsne_labels)

    if harmonized_results:
        np.save(output_dir / 'harmonized_tsne_embedding.npy', harm_tsne)
        np.save(output_dir / 'harmonized_tsne_labels.npy', harm_tsne_labels)

    # save model
    torch.save(model.state_dict(), output_dir / 'domain_classifier.pth')

    print('=' * 60)
    print('[domaineval] results summary:')
    print(f'  raw data accuracy: {raw_results["accuracy"]:.4f}')
    print(f'  raw data auc: {raw_results["auc"]:.4f}')
    if harmonized_results:
        print(f'  harmonized data accuracy: {harmonized_results["accuracy"]:.4f}')
        print(f'  harmonized data auc: {harmonized_results["auc"]:.4f}')
        print(f'  accuracy reduction: {results["improvement"]["accuracy_reduction"]:.4f}')
    print('=' * 60)
    print(f'[domaineval] results saved to {output_dir}')


if __name__ == '__main__':
    main()
