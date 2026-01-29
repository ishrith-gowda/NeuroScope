#!/usr/bin/env python3
"""
prepare segmentation data for downstream task evaluation.

this script prepares brats data with segmentation masks for evaluating
the impact of mri harmonization on downstream tumor segmentation.

brats labels:
- 0: background
- 1: necrotic and non-enhancing tumor core (ncr/net)
- 2: peritumoral edema (ed)
- 4: gd-enhancing tumor (et)

derived regions:
- whole tumor (wt): labels 1 + 2 + 4
- tumor core (tc): labels 1 + 4
- enhancing tumor (et): label 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from tqdm import tqdm


def load_nifti(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    load nifti file and return data with affine matrix.

    args:
        filepath: path to nifti file

    returns:
        tuple of (data array, affine matrix)
    """
    img = nib.load(str(filepath))
    data = img.get_fdata()
    affine = img.affine
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, filepath: Path) -> None:
    """save data as nifti file."""
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(filepath))


def convert_labels_to_regions(seg: np.ndarray) -> Dict[str, np.ndarray]:
    """
    convert brats segmentation labels to clinically relevant regions.

    args:
        seg: segmentation array with brats labels (0, 1, 2, 4)

    returns:
        dict with binary masks for each region
    """
    # brats labels: 0=bg, 1=ncr/net, 2=edema, 4=enhancing
    wt = (seg > 0).astype(np.uint8)  # whole tumor: all non-zero
    tc = ((seg == 1) | (seg == 4)).astype(np.uint8)  # tumor core
    et = (seg == 4).astype(np.uint8)  # enhancing tumor

    return {
        'wt': wt,  # whole tumor
        'tc': tc,  # tumor core
        'et': et,  # enhancing tumor
        'full': seg.astype(np.uint8)  # full multi-class
    }


def extract_slices_with_tumor(
    modalities: Dict[str, np.ndarray],
    segmentation: np.ndarray,
    slice_range: Tuple[int, int] = (30, 125),
    min_tumor_pixels: int = 100
) -> List[Dict]:
    """
    extract 2d slices that contain tumor for training.

    args:
        modalities: dict of modality arrays {name: array}
        segmentation: segmentation array
        slice_range: range of axial slices to consider
        min_tumor_pixels: minimum tumor pixels to include slice

    returns:
        list of dicts with slice data
    """
    slices = []

    start_slice = max(0, slice_range[0])
    end_slice = min(segmentation.shape[2], slice_range[1])

    for z in range(start_slice, end_slice):
        seg_slice = segmentation[:, :, z]
        tumor_pixels = np.sum(seg_slice > 0)

        if tumor_pixels >= min_tumor_pixels:
            slice_data = {
                'slice_idx': z,
                'tumor_pixels': int(tumor_pixels),
            }

            # extract modality slices
            for mod_name, mod_data in modalities.items():
                slice_data[mod_name] = mod_data[:, :, z]

            # extract segmentation
            slice_data['seg'] = seg_slice

            # compute region masks
            regions = convert_labels_to_regions(seg_slice)
            for region_name, region_mask in regions.items():
                slice_data[f'seg_{region_name}'] = region_mask

            slices.append(slice_data)

    return slices


def resize_slice(data: np.ndarray, target_size: Tuple[int, int],
                 is_label: bool = False) -> np.ndarray:
    """
    resize 2d slice to target size.

    args:
        data: 2d array
        target_size: (height, width)
        is_label: if true, use nearest neighbor interpolation

    returns:
        resized array
    """
    order = 0 if is_label else 3  # nearest for labels, cubic for images
    resized = resize(data, target_size, order=order, preserve_range=True,
                    anti_aliasing=not is_label)
    return resized


def normalize_intensity(data: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    normalize intensity using z-score within brain mask.

    args:
        data: image data
        mask: optional brain mask

    returns:
        normalized data
    """
    if mask is None:
        mask = data > 0

    brain_voxels = data[mask]
    if len(brain_voxels) == 0:
        return data

    mean_val = np.mean(brain_voxels)
    std_val = np.std(brain_voxels)

    if std_val < 1e-8:
        return data

    normalized = (data - mean_val) / (std_val + 1e-8)
    normalized[~mask] = 0

    return normalized


def process_brats_subject(
    subject_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (128, 128),
    slice_range: Tuple[int, int] = (30, 125),
    min_tumor_pixels: int = 50
) -> Optional[Dict]:
    """
    process a single brats subject.

    args:
        subject_dir: path to subject directory
        output_dir: output directory for processed data
        target_size: target slice size
        slice_range: axial slice range to extract
        min_tumor_pixels: minimum tumor pixels per slice

    returns:
        dict with subject info or none if failed
    """
    subject_id = subject_dir.name

    # find modality files - brats naming convention
    modality_files = {
        't1': None,
        't1ce': None,  # or t1gd
        't2': None,
        'flair': None,
        'seg': None
    }

    for f in subject_dir.iterdir():
        fname = f.name.lower()
        if f.suffix == '.gz' or f.suffix == '.nii':
            if 't1ce' in fname or 't1gd' in fname or 't1_gd' in fname:
                modality_files['t1ce'] = f
            elif 't1' in fname and 'ce' not in fname and 'gd' not in fname:
                modality_files['t1'] = f
            elif 't2' in fname and 'flair' not in fname:
                modality_files['t2'] = f
            elif 'flair' in fname:
                modality_files['flair'] = f
            elif 'seg' in fname:
                modality_files['seg'] = f

    # check if segmentation exists
    if modality_files['seg'] is None:
        print(f'[warning] no segmentation found for {subject_id}')
        return None

    # check if all modalities exist
    missing_mods = [k for k, v in modality_files.items() if v is None and k != 'seg']
    if missing_mods:
        print(f'[warning] missing modalities for {subject_id}: {missing_mods}')
        return None

    # load all modalities
    modalities = {}
    affine = None
    for mod_name in ['t1', 't1ce', 't2', 'flair']:
        data, aff = load_nifti(modality_files[mod_name])
        modalities[mod_name] = data
        if affine is None:
            affine = aff

    # load segmentation
    seg_data, _ = load_nifti(modality_files['seg'])

    # extract slices with tumor
    slices = extract_slices_with_tumor(
        modalities, seg_data, slice_range, min_tumor_pixels
    )

    if len(slices) == 0:
        print(f'[warning] no tumor slices found for {subject_id}')
        return None

    # create subject output directory
    subject_output = output_dir / subject_id
    subject_output.mkdir(parents=True, exist_ok=True)

    # save processed slices
    subject_info = {
        'subject_id': subject_id,
        'n_slices': len(slices),
        'slice_indices': [],
        'tumor_pixels': []
    }

    for i, slice_data in enumerate(slices):
        slice_idx = slice_data['slice_idx']
        subject_info['slice_indices'].append(slice_idx)
        subject_info['tumor_pixels'].append(slice_data['tumor_pixels'])

        # stack modalities for input (4 channels)
        stacked = np.stack([
            resize_slice(normalize_intensity(slice_data['t1']), target_size),
            resize_slice(normalize_intensity(slice_data['t1ce']), target_size),
            resize_slice(normalize_intensity(slice_data['t2']), target_size),
            resize_slice(normalize_intensity(slice_data['flair']), target_size),
        ], axis=0)

        # resize segmentation
        seg_resized = resize_slice(slice_data['seg'], target_size, is_label=True)

        # resize region masks
        seg_wt = resize_slice(slice_data['seg_wt'], target_size, is_label=True)
        seg_tc = resize_slice(slice_data['seg_tc'], target_size, is_label=True)
        seg_et = resize_slice(slice_data['seg_et'], target_size, is_label=True)

        # save as numpy arrays
        np.save(subject_output / f'slice_{slice_idx:03d}_input.npy', stacked.astype(np.float32))
        np.save(subject_output / f'slice_{slice_idx:03d}_seg.npy', seg_resized.astype(np.uint8))
        np.save(subject_output / f'slice_{slice_idx:03d}_seg_wt.npy', seg_wt.astype(np.uint8))
        np.save(subject_output / f'slice_{slice_idx:03d}_seg_tc.npy', seg_tc.astype(np.uint8))
        np.save(subject_output / f'slice_{slice_idx:03d}_seg_et.npy', seg_et.astype(np.uint8))

    # save subject info
    with open(subject_output / 'info.json', 'w') as f:
        json.dump(subject_info, f, indent=2)

    return subject_info


def create_splits(
    subjects: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    create train/val/test splits.

    args:
        subjects: list of subject ids
        train_ratio: fraction for training
        val_ratio: fraction for validation
        seed: random seed

    returns:
        dict with split assignments
    """
    np.random.seed(seed)

    n = len(subjects)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'train': [subjects[i] for i in train_idx],
        'val': [subjects[i] for i in val_idx],
        'test': [subjects[i] for i in test_idx]
    }


def main():
    parser = argparse.ArgumentParser(
        description='prepare brats segmentation data for downstream evaluation'
    )
    parser.add_argument('--input-dir', type=str, required=True,
                       help='path to raw brats data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for processed data')
    parser.add_argument('--target-size', type=int, nargs=2, default=[128, 128],
                       help='target slice size (h, w)')
    parser.add_argument('--slice-range', type=int, nargs=2, default=[30, 125],
                       help='axial slice range to extract')
    parser.add_argument('--min-tumor-pixels', type=int, default=50,
                       help='minimum tumor pixels per slice')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = tuple(args.target_size)
    slice_range = tuple(args.slice_range)

    output_dir.mkdir(parents=True, exist_ok=True)

    print('[segdata] preparing brats segmentation data')
    print(f'[segdata] input: {input_dir}')
    print(f'[segdata] output: {output_dir}')
    print(f'[segdata] target size: {target_size}')
    print(f'[segdata] slice range: {slice_range}')
    print('=' * 60)

    # find all subjects
    subjects = []
    for d in input_dir.iterdir():
        if d.is_dir():
            subjects.append(d)

    print(f'[segdata] found {len(subjects)} subjects')

    # process subjects
    processed_subjects = []
    total_slices = 0

    for subject_dir in tqdm(subjects, desc='processing subjects'):
        info = process_brats_subject(
            subject_dir, output_dir, target_size, slice_range,
            args.min_tumor_pixels
        )
        if info is not None:
            processed_subjects.append(info['subject_id'])
            total_slices += info['n_slices']

    print(f'[segdata] processed {len(processed_subjects)} subjects')
    print(f'[segdata] total slices: {total_slices}')

    # create splits
    splits = create_splits(
        processed_subjects, args.train_ratio, args.val_ratio, args.seed
    )

    print(f'[segdata] train: {len(splits["train"])} subjects')
    print(f'[segdata] val: {len(splits["val"])} subjects')
    print(f'[segdata] test: {len(splits["test"])} subjects')

    # save splits
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)

    # save metadata
    metadata = {
        'n_subjects': len(processed_subjects),
        'n_slices': total_slices,
        'target_size': target_size,
        'slice_range': slice_range,
        'min_tumor_pixels': args.min_tumor_pixels,
        'splits': {k: len(v) for k, v in splits.items()}
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print('=' * 60)
    print(f'[segdata] done. data saved to {output_dir}')


if __name__ == '__main__':
    main()
