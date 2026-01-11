"""
2.5D MRI Dataset for SA-CycleGAN Training.

Loads 3 adjacent slices from 3D MRI volumes for 2.5D processing,
enabling inter-slice context while maintaining memory efficiency.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable
import numpy as np
import random

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


class MRIDataset25D(Dataset):
    """
    2.5D MRI Dataset for unpaired domain translation.
    
    For each sample, returns 3 adjacent slices stacked with all modalities:
    - Output shape: [12, H, W] = 3 slices x 4 modalities
    
    This enables the model to use inter-slice context for better
    anatomical consistency in the translated output.
    """
    
    MODALITIES = ['t1', 't1gd', 't2', 'flair']  # Standard BraTS modalities
    
    def __init__(
        self,
        root_dir: str,
        slice_range: Tuple[int, int] = (30, 125),  # Central slices with brain
        n_context_slices: int = 1,  # 1 = use slice-1, slice, slice+1
        image_size: Optional[Tuple[int, int]] = (128, 128),  # Resize for memory
        transform: Optional[Callable] = None,
        modalities: Optional[List[str]] = None,
        cache_volumes: bool = False,  # Cache loaded volumes in memory
        precompute_valid_slices: bool = True
    ):
        """
        Args:
            root_dir: Directory containing subject folders
            slice_range: Range of axial slices to use (avoiding empty slices)
            n_context_slices: Number of slices on each side (1 = 3 total slices)
            image_size: Resize images to this size (H, W), None to keep original
            transform: Optional transforms to apply
            modalities: List of modalities to use, defaults to all 4
            cache_volumes: Whether to cache volumes in memory
            precompute_valid_slices: Precompute which slices have brain content
        """
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required: pip install nibabel")
        
        self.root_dir = Path(root_dir)
        self.slice_range = slice_range
        self.n_context = n_context_slices
        self.image_size = image_size
        self.transform = transform
        self.modalities = modalities or self.MODALITIES
        self.cache_volumes = cache_volumes
        
        # Find all valid subjects
        self.subjects = self._find_subjects()
        print(f"Found {len(self.subjects)} subjects in {root_dir}")
        
        # Create index mapping: (subject_idx, slice_idx)
        self.samples = self._create_sample_index()
        print(f"Total samples (slice triplets): {len(self.samples)}")
        
        # Volume cache
        self._cache: Dict[str, np.ndarray] = {}
        
    def _find_subjects(self) -> List[Path]:
        """Find all valid subject directories."""
        subjects = []
        for subj_dir in sorted(self.root_dir.iterdir()):
            if subj_dir.is_dir():
                # Check if all modalities exist
                has_all = all(
                    (subj_dir / f"{mod}.nii.gz").exists() or
                    (subj_dir / f"{subj_dir.name}_{mod}.nii.gz").exists()
                    for mod in self.modalities
                )
                if has_all:
                    subjects.append(subj_dir)
        return subjects
    
    def _create_sample_index(self) -> List[Tuple[int, int]]:
        """Create list of (subject_idx, center_slice_idx) pairs."""
        samples = []
        start = self.slice_range[0] + self.n_context
        end = self.slice_range[1] - self.n_context
        
        for subj_idx in range(len(self.subjects)):
            for slice_idx in range(start, end):
                samples.append((subj_idx, slice_idx))
        
        return samples
    
    def _get_modality_path(self, subj_dir: Path, modality: str) -> Path:
        """Get path to modality file (handles different naming conventions)."""
        # Try simple naming
        simple = subj_dir / f"{modality}.nii.gz"
        if simple.exists():
            return simple
        
        # Try with subject prefix
        prefixed = subj_dir / f"{subj_dir.name}_{modality}.nii.gz"
        if prefixed.exists():
            return prefixed
        
        raise FileNotFoundError(f"Could not find {modality} for {subj_dir.name}")
    
    def _load_volume(self, subj_dir: Path) -> np.ndarray:
        """
        Load all modalities for a subject.
        
        Returns:
            np.ndarray of shape [4, D, H, W] where D is depth (number of slices)
        """
        cache_key = str(subj_dir)
        if self.cache_volumes and cache_key in self._cache:
            return self._cache[cache_key]
        
        modality_volumes = []
        for mod in self.modalities:
            path = self._get_modality_path(subj_dir, mod)
            vol = nib.load(str(path)).get_fdata().astype(np.float32)
            modality_volumes.append(vol)
        
        # Stack: [4, H, W, D] then transpose to [4, D, H, W]
        volume = np.stack(modality_volumes, axis=0)  # [4, H, W, D]
        volume = np.transpose(volume, (0, 3, 1, 2))  # [4, D, H, W]
        
        if self.cache_volumes:
            self._cache[cache_key] = volume
        
        return volume
    
    def _resize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Resize a 2D slice using bilinear interpolation."""
        if self.image_size is None:
            return slice_2d
        
        # Use torch for resizing
        tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            tensor, 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        )
        return resized.squeeze().numpy()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample (3 adjacent slices x 4 modalities).
        
        Returns:
            Dictionary with:
                'image': [12, H, W] tensor (3 slices x 4 modalities)
                'center_slice': [4, H, W] tensor (center slice only, for reference)
                'subject_id': subject identifier
                'slice_idx': center slice index
        """
        subj_idx, center_slice = self.samples[idx]
        subj_dir = self.subjects[subj_idx]
        
        # Load volume [4, D, H, W]
        volume = self._load_volume(subj_dir)
        
        # Extract 3 adjacent slices
        slices = []
        for offset in range(-self.n_context, self.n_context + 1):
            slice_idx = center_slice + offset
            for mod_idx in range(len(self.modalities)):
                slice_2d = volume[mod_idx, slice_idx, :, :]
                slice_2d = self._resize_slice(slice_2d)
                slices.append(slice_2d)
        
        # Stack: [12, H, W] = 3 slices x 4 modalities
        # Order: [s-1_t1, s-1_t1gd, s-1_t2, s-1_flair, s_t1, ..., s+1_flair]
        image = np.stack(slices, axis=0)
        
        # Also get center slice separately for reference
        center_start = self.n_context * len(self.modalities)
        center_end = center_start + len(self.modalities)
        center_image = image[center_start:center_end]
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        center_image = torch.from_numpy(center_image).float()
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            'image': image,
            'center_slice': center_image,
            'subject_id': subj_dir.name,
            'slice_idx': center_slice
        }


class UnpairedMRIDataset25D(Dataset):
    """
    Unpaired dataset for CycleGAN training.
    
    Returns samples from both domains A (BraTS) and B (UPenn)
    without explicit pairing.
    """
    
    def __init__(
        self,
        domain_a_dir: str,
        domain_b_dir: str,
        slice_range: Tuple[int, int] = (30, 125),
        image_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable] = None
    ):
        self.dataset_a = MRIDataset25D(
            root_dir=domain_a_dir,
            slice_range=slice_range,
            image_size=image_size,
            transform=transform
        )
        self.dataset_b = MRIDataset25D(
            root_dir=domain_b_dir,
            slice_range=slice_range,
            image_size=image_size,
            transform=transform
        )
        
        # Match lengths by cycling through smaller dataset
        self.len_a = len(self.dataset_a)
        self.len_b = len(self.dataset_b)
        self.length = max(self.len_a, self.len_b)
        
        print(f"Domain A samples: {self.len_a}")
        print(f"Domain B samples: {self.len_b}")
        print(f"Epoch length: {self.length}")
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sample from each domain (cycling if needed)
        idx_a = idx % self.len_a
        idx_b = idx % self.len_b
        
        sample_a = self.dataset_a[idx_a]
        sample_b = self.dataset_b[idx_b]
        
        return {
            'A': sample_a['image'],           # [12, H, W]
            'B': sample_b['image'],           # [12, H, W]
            'A_center': sample_a['center_slice'],
            'B_center': sample_b['center_slice'],
            'A_subject': sample_a['subject_id'],
            'B_subject': sample_b['subject_id'],
            'A_slice': sample_a['slice_idx'],
            'B_slice': sample_b['slice_idx']
        }


def create_dataloaders(
    brats_dir: str,
    upenn_dir: str,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (128, 128),
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset
    full_dataset = UnpairedMRIDataset25D(
        domain_a_dir=brats_dir,
        domain_b_dir=upenn_dir,
        image_size=image_size
    )
    
    # Split indices
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    import sys
    
    brats_dir = "/Volumes/usb drive/neuroscope/preprocessed/brats"
    upenn_dir = "/Volumes/usb drive/neuroscope/preprocessed/upenn"
    
    print("Testing 2.5D MRI Dataset...")
    print("=" * 60)
    
    dataset = UnpairedMRIDataset25D(
        domain_a_dir=brats_dir,
        domain_b_dir=upenn_dir,
        image_size=(128, 128)
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  A (3 slices x 4 mod): {sample['A'].shape}")
    print(f"  B (3 slices x 4 mod): {sample['B'].shape}")
    print(f"  A center slice: {sample['A_center'].shape}")
    print(f"  A subject: {sample['A_subject']}, slice: {sample['A_slice']}")
    print(f"  B subject: {sample['B_subject']}, slice: {sample['B_slice']}")
