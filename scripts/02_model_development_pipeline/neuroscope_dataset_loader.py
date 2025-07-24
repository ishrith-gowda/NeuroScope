import os
import json
import random
import logging
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import SimpleITK as sitk

def intensity_jitter(x: torch.Tensor) -> torch.Tensor:
    """Applies random intensity jitter to a tensor."""
    return x + 0.1 * (torch.rand_like(x) - 0.5)


def clamp_tensor(x: torch.Tensor) -> torch.Tensor:
    """Clamps tensor values to the [0.0, 1.0] range."""
    return torch.clamp(x, 0.0, 1.0)

def configure_logging():
    """
    Configure logging format and level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def intensity_jitter(x):
    """
    Apply random intensity jitter to a tensor.
    """
    return x + 0.1 * (torch.rand_like(x) - 0.5)

def clamp_tensor(x):
    """
    Clamp tensor values to [0.0, 1.0].
    """
    return torch.clamp(x, 0.0, 1.0)

class NeuroScopeSliceDataset(data.Dataset):
    """
    PyTorch Dataset for 2D axial slices of 4-channel MRI volumes.

    Each sample is a tensor of shape [4, H, W], corresponding to modalities
    (T1, T1GD, T2, FLAIR) for a single axial slice.
    """
    def __init__(
        self,
        base_dir: str,
        metadata_json: str,
        split: str = 'train',
        modalities=('t1.nii.gz','t1gd.nii.gz','t2.nii.gz','flair.nii.gz'),
        transforms=None
    ):
        super().__init__()
        configure_logging()
        logging.info(f"Initializing NeuroScopeSliceDataset(split={split})")
        with open(metadata_json, 'r') as f:
            meta = json.load(f)
        self.items = []
        for section in ('brats','upenn'):
            for sid, info in meta[section]['valid_subjects'].items():
                if info.get('split') != split:
                    continue
                vol_paths = []
                for mod in modalities:
                    match = None
                    for key, path in info.items():
                        if isinstance(path, str) and path.lower().endswith(mod):
                            # Fix USB path if incorrectly rebased
                            path = path.replace(
                                "/Users/IshrithG/Downloads/neuroscope/data/",
                                "/Volumes/USB DRIVE/neuroscope/data/"
                            )
                            if not os.path.exists(path):
                                logging.error("File not found: %s", path)
                            match = path
                            break
                    if match is None or not os.path.exists(match):
                        logging.error("Modality %s not found or path invalid for %s/%s", mod, section, sid)
                        break
                    vol_paths.append(match)
                if len(vol_paths) == len(modalities):
                    self.items.append((section, sid, vol_paths))
        self.transforms = transforms
        self.modalities = modalities
        logging.info(f"Loaded {len(self.items)} subjects for split={split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        section, sid, paths = self.items[idx]
        vols = []
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")
            img = sitk.ReadImage(p)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            vols.append(arr)
        vols = np.stack(vols, axis=0)
        D = vols.shape[1]
        z = random.randint(0, D-1)
        slice4 = vols[:, z, :, :]  # [4,H,W]
        img_tensor = torch.from_numpy(slice4).float()
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        if self.transforms:
            img_tensor = self.transforms(img_tensor)
        return img_tensor

def get_dataloaders(
    base_dir: str,
    metadata_json: str,
    batch_size: int = 16,
    num_workers: int = 0  # safer default for MPS
):
    """
    Create DataLoaders for train/val/test splits.
    Applies random flips and intensity jitter to training data.
    """
    configure_logging()
    if torch.backends.mps.is_available():
        logging.warning("MPS backend detected. Setting num_workers=0 due to macOS multiprocessing limitations.")

    common_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.Lambda(intensity_jitter),
        T.Lambda(clamp_tensor)
    ])

    loaders = {}
    for split in ('train','val','test'):
        ds = NeuroScopeSliceDataset(
            base_dir=base_dir,
            metadata_json=metadata_json,
            split=split,
            transforms=(common_transforms if split=='train' else None)
        )
        loaders[split] = data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split=='train'),
            num_workers=num_workers,
            pin_memory=True
        )
        logging.info(f"Created {split} DataLoader with {len(ds)} samples")
    return loaders

if __name__ == '__main__':
    configure_logging()
    BASE = '/Downloads/neuroscope/data/preprocessed'
    META = os.path.expanduser('~/Downloads/neuroscope/scripts/01_data_preparation_pipeline/neuroscope_dataset_metadata_splits.json')
    loaders = get_dataloaders(BASE, META)
    batch = next(iter(loaders['train']))
    print("Train batch shape:", batch.shape)  # Expect [B,4,H,W]
