import os
import json
import random
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import SimpleITK as sitk


# Public constants
MRI_MODALITIES = ("t1.nii.gz", "t1gd.nii.gz", "t2.nii.gz", "flair.nii.gz")


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def intensity_jitter(x: torch.Tensor) -> torch.Tensor:
    return x + 0.1 * (torch.rand_like(x) - 0.5)


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    # Assume input in [0,1]
    return x * 2.0 - 1.0


class DomainSliceDataset(data.Dataset):
    """Domain‑specific 2D slice dataset.

    Each __getitem__ loads all 4 modalities for one subject (from preprocessed
    directory layout: <base_dir>/<section>/<subject>/<modality>) and returns a
    randomly selected axial slice as float tensor shape [4,H,W] in range [-1,1].
    """

    def __init__(
        self,
        base_dir: str,
        metadata_json: str,
        section: str,
        split: str = "train",
        modalities: Tuple[str, ...] = MRI_MODALITIES,
        slices_per_subject: int = 1,
        transforms: Optional[torch.nn.Module] = None,
        min_modalities_required: int = 4,
    ):
        super().__init__()
        assert section in ("brats", "upenn"), "section must be 'brats' or 'upenn'"
        self.base_dir = base_dir
        self.section = section
        self.split = split
        self.modalities = modalities
        self.transforms = transforms
        self.slices_per_subject = max(1, slices_per_subject)
        self.items: List[Tuple[str, Dict[str, str]]] = []  # (subject_id, modality->path)

        with open(metadata_json, "r") as f:
            meta = json.load(f)

        if section not in meta:
            raise ValueError(f"Section '{section}' not present in metadata JSON")

        subjects_meta = meta[section].get("valid_subjects", {})

        for sid, info in subjects_meta.items():
            if info.get("split") != split:
                continue
            # Build expected preprocessed modality paths instead of relying on raw paths
            modality_paths = {}
            subject_dir = os.path.join(base_dir, section, sid)
            missing = False
            for mod in modalities:
                p = os.path.join(subject_dir, mod)
                if not os.path.isfile(p):
                    missing = True
                    break
                modality_paths[mod] = p
            if not missing and len(modality_paths) >= min_modalities_required:
                self.items.append((sid, modality_paths))

        if not self.items:
            logging.warning(
                "No subjects found for section=%s split=%s in preprocessed dir %s",
                section,
                split,
                base_dir,
            )
        else:
            logging.info(
                "Loaded %d subjects for section=%s split=%s", len(self.items), section, split
            )

    def __len__(self):
        # Optionally amplify dataset length by slices_per_subject to present more slices per epoch
        return len(self.items) * self.slices_per_subject

    def _load_subject_volume(self, idx_subject: int) -> np.ndarray:
        sid, modality_paths = self.items[idx_subject]
        vols: List[np.ndarray] = []
        for mod in self.modalities:
            p = modality_paths.get(mod)
            img = sitk.ReadImage(p)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            vols.append(arr)
        return np.stack(vols, axis=0)  # [4,D,H,W]

    def __getitem__(self, idx: int):
        subj_index = idx // self.slices_per_subject
        vols = self._load_subject_volume(subj_index)
        depth = vols.shape[1]
        z = random.randint(0, depth - 1)
        slice4 = vols[:, z, :, :]
        
        # Convert to tensor and validate
        tensor = torch.from_numpy(slice4).float()
        
        # Validate tensor values
        if not torch.isfinite(tensor).all():
            logging.warning("Non-finite values detected in slice, cleaning...")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=tensor.max(), neginf=tensor.min())
        
        # Ensure input is in [0,1] range (preprocessing should have done this, but safety check)
        tensor = clamp_unit(tensor)
        
        # Convert to [-1,1] for CycleGAN
        tensor = to_minus1_1(tensor)
        
        # Apply transforms if specified
        if self.transforms:
            tensor = self.transforms(tensor)
            
        return tensor


def build_transforms(train: bool = True):
    if train:
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.Lambda(intensity_jitter),  # small jitter in [-0.05,0.05]
                T.Lambda(lambda x: torch.clamp(x, -1.0, 1.0)),
            ]
        )
    return None


def get_cycle_domain_loaders(
    preprocessed_dir: str,
    metadata_json: str,
    batch_size: int = 8,
    num_workers: int = 0,
    slices_per_subject: int = 4,
    seed: int = 42,
):
    """Return separate domain dataloaders for CycleGAN training.

    Returns dict with keys: train_A, train_B, val_A, val_B, (optionally test_A/B)
    """
    configure_logging()
    set_seed(seed)

    if torch.backends.mps.is_available() and num_workers != 0:
        logging.warning("Setting num_workers=0 for MPS compatibility")
        num_workers = 0

    loaders = {}
    for split in ("train", "val", "test"):
        train_flag = split == "train"
        tr = build_transforms(train_flag)
        for section, domain in (("brats", "A"), ("upenn", "B")):
            ds = DomainSliceDataset(
                base_dir=preprocessed_dir,
                metadata_json=metadata_json,
                section=section,
                split=split,
                slices_per_subject=slices_per_subject,
                transforms=tr,
            )
            if len(ds) == 0:
                continue
            key = f"{split}_{domain}"
            loaders[key] = data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=train_flag,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            logging.info(
                "Created DataLoader %s with %d samples (subjects=%d)",
                key,
                len(ds),
                len(ds.items),
            )
    return loaders


if __name__ == "__main__":
    # Simple smoke test (paths may need adjustment for actual environment)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_dir", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    args = parser.parse_args()

    loaders = get_cycle_domain_loaders(
        preprocessed_dir=args.preprocessed_dir, metadata_json=args.metadata_json
    )
    for name, dl in loaders.items():
        batch = next(iter(dl))
        print(name, batch.shape, batch.min().item(), batch.max().item())
