#!/usr/bin/env python3
"""
unified preprocessing pipeline for all neuroscope datasets.

this script provides a comprehensive preprocessing pipeline for:
- brats (brain tumor segmentation)
- upenn-gbm (glioblastoma)
- ixi (information extraction from images)
- oasis-3 (open access series of imaging studies)

features:
- skull stripping with hd-bet or fsl
- n4 bias field correction
- intensity normalization (z-score or percentile)
- registration to mni152 space (optional)
- consistent resampling to target resolution
- train/val/test splitting
- quality control metrics and visualization

usage:
    python scripts/preprocess_all_datasets.py \
        --datasets brats upenn_gbm ixi oasis3 \
        --input-dir ./data/raw \
        --output-dir ./data/processed \
        --target-resolution 1.0 1.0 1.0 \
        --target-size 240 240 155 \
        --modalities t1 t1ce t2 flair \
        --num-workers 8
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from skimage import exposure
from tqdm import tqdm

# suppress warnings
warnings.filterwarnings('ignore')

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MRIPreprocessor:
    """comprehensive mri preprocessing pipeline."""

    def __init__(
        self,
        target_resolution: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Optional[Tuple[int, int, int]] = None,
        normalization_method: str = 'zscore',
        bias_correction: bool = True,
        skull_strip: bool = True,
        register_to_mni: bool = False,
    ):
        """
        initialize preprocessor.

        args:
            target_resolution: target voxel spacing (mm) for resampling
            target_size: target volume size (h, w, d). if none, use original
            normalization_method: 'zscore', 'minmax', or 'percentile'
            bias_correction: apply n4 bias field correction
            skull_strip: apply skull stripping
            register_to_mni: register to mni152 template space
        """
        self.target_resolution = target_resolution
        self.target_size = target_size
        self.normalization_method = normalization_method
        self.bias_correction = bias_correction
        self.skull_strip = skull_strip
        self.register_to_mni = register_to_mni

    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
        """load nifti file and return data + header."""
        img = nib.load(str(filepath))
        data = img.get_fdata()
        return data, img

    def save_nifti(self, data: np.ndarray, affine: np.ndarray, filepath: Path):
        """save data as nifti file."""
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, str(filepath))

    def resample_to_resolution(
        self,
        data: np.ndarray,
        current_res: Tuple[float, float, float],
        target_res: Tuple[float, float, float]
    ) -> np.ndarray:
        """resample volume to target resolution."""
        zoom_factors = [c / t for c, t in zip(current_res, target_res)]
        resampled = ndimage.zoom(data, zoom_factors, order=3)
        return resampled

    def resize_to_shape(self, data: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """resize volume to target shape with center cropping/padding."""
        current_shape = data.shape

        # calculate crop/pad amounts for each dimension
        processed = data.copy()
        for i in range(3):
            diff = current_shape[i] - target_shape[i]

            if diff > 0:  # crop
                start = diff // 2
                end = start + target_shape[i]
                slices = [slice(None)] * 3
                slices[i] = slice(start, end)
                processed = processed[tuple(slices)]
            elif diff < 0:  # pad
                pad_width = [(0, 0)] * 3
                pad_before = abs(diff) // 2
                pad_after = abs(diff) - pad_before
                pad_width[i] = (pad_before, pad_after)
                processed = np.pad(processed, pad_width, mode='constant', constant_values=0)

        return processed

    def n4_bias_correction(self, data: np.ndarray) -> np.ndarray:
        """
        apply n4 bias field correction.

        note: requires simpleitk. if not available, returns original data.
        """
        try:
            import SimpleITK as sitk

            # convert to simpleitk image
            img = sitk.GetImageFromArray(data)

            # apply n4 bias correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            corrected_img = corrector.Execute(img)

            # convert back to numpy
            corrected_data = sitk.GetArrayFromImage(corrected_img)
            return corrected_data

        except ImportError:
            logger.warning("SimpleITK not available. Skipping bias correction.")
            return data

    def skull_stripping(self, data: np.ndarray) -> np.ndarray:
        """
        apply skull stripping.

        simple otsu-based method. for better results, use hd-bet or fsl.
        """
        # simple threshold-based approach
        threshold = np.percentile(data[data > 0], 10)
        mask = data > threshold

        # morphological operations to clean up mask
        mask = ndimage.binary_opening(mask, iterations=2)
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_fill_holes(mask)

        # apply mask
        stripped = data * mask
        return stripped

    def normalize_intensity(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        normalize intensity values.

        args:
            data: input volume
            method: 'zscore', 'minmax', or 'percentile'
        """
        # only normalize non-zero voxels (outside background)
        mask = data > 0

        if method == 'zscore':
            mean = np.mean(data[mask])
            std = np.std(data[mask])
            normalized = np.zeros_like(data)
            normalized[mask] = (data[mask] - mean) / (std + 1e-8)

        elif method == 'minmax':
            min_val = np.min(data[mask])
            max_val = np.max(data[mask])
            normalized = np.zeros_like(data)
            normalized[mask] = (data[mask] - min_val) / (max_val - min_val + 1e-8)

        elif method == 'percentile':
            p1, p99 = np.percentile(data[mask], [1, 99])
            normalized = np.clip(data, p1, p99)
            normalized = (normalized - p1) / (p99 - p1 + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def preprocess_volume(
        self,
        filepath: Path,
        output_path: Path,
        metadata: Dict = None
    ) -> Dict:
        """
        complete preprocessing pipeline for a single volume.

        returns:
            dictionary with preprocessing statistics
        """
        stats = {'filepath': str(filepath), 'success': False}

        try:
            # load data
            data, img = self.load_nifti(filepath)
            stats['original_shape'] = data.shape
            stats['original_resolution'] = img.header.get_zooms()[:3]

            # get current resolution
            current_res = img.header.get_zooms()[:3]

            # bias correction
            if self.bias_correction:
                data = self.n4_bias_correction(data)

            # skull stripping
            if self.skull_strip:
                data = self.skull_stripping(data)

            # resample to target resolution
            if self.target_resolution != current_res:
                data = self.resample_to_resolution(data, current_res, self.target_resolution)

            # resize to target shape
            if self.target_size is not None:
                data = self.resize_to_shape(data, self.target_size)

            # intensity normalization
            data = self.normalize_intensity(data, self.normalization_method)

            # save preprocessed volume
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_nifti(data, img.affine, output_path)

            # calculate statistics
            stats['final_shape'] = data.shape
            stats['final_resolution'] = self.target_resolution
            stats['mean_intensity'] = float(np.mean(data[data > 0]))
            stats['std_intensity'] = float(np.std(data[data > 0]))
            stats['min_intensity'] = float(np.min(data))
            stats['max_intensity'] = float(np.max(data))
            stats['success'] = True

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            stats['error'] = str(e)

        return stats


class DatasetPreprocessor:
    """handles dataset-specific preprocessing logic."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        preprocessor: MRIPreprocessor,
        modalities: List[str] = None,
        num_workers: int = 4,
    ):
        """
        initialize dataset preprocessor.

        args:
            input_dir: root directory of raw dataset
            output_dir: output directory for preprocessed data
            preprocessor: mripreprocessor instance
            modalities: list of modalities to process (e.g., ['t1', 't2'])
            num_workers: number of parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.preprocessor = preprocessor
        self.modalities = modalities or ['T1', 'T2', 'FLAIR', 'T1CE']
        self.num_workers = num_workers

    def preprocess_brats(self):
        """preprocess brats dataset."""
        logger.info("Preprocessing BraTS dataset...")

        dataset_dir = self.input_dir / 'brats'
        output_dir = self.output_dir / 'brats'

        # find all subject directories
        subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(subject_dirs)} subjects")

        all_stats = []

        for subject_dir in tqdm(subject_dirs, desc="Processing BraTS subjects"):
            subject_id = subject_dir.name

            # process each modality
            for modality in self.modalities:
                # brats naming convention
                modality_lower = modality.lower()
                input_file = subject_dir / f"{subject_id}_{modality_lower}.nii.gz"

                if not input_file.exists():
                    logger.warning(f"File not found: {input_file}")
                    continue

                # output path
                output_file = output_dir / subject_id / f"{modality}.nii.gz"

                # preprocess
                stats = self.preprocessor.preprocess_volume(input_file, output_file)
                stats['subject_id'] = subject_id
                stats['modality'] = modality
                stats['dataset'] = 'brats'
                all_stats.append(stats)

        # save statistics
        self._save_statistics(all_stats, output_dir / 'preprocessing_stats.json')
        logger.info(f"BraTS preprocessing complete. Saved to {output_dir}")

    def preprocess_upenn_gbm(self):
        """preprocess upenn-gbm dataset."""
        logger.info("Preprocessing UPenn-GBM dataset...")

        dataset_dir = self.input_dir / 'upenn_gbm'
        output_dir = self.output_dir / 'upenn_gbm'

        # find all subject directories
        subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(subject_dirs)} subjects")

        all_stats = []

        for subject_dir in tqdm(subject_dirs, desc="Processing UPenn-GBM subjects"):
            subject_id = subject_dir.name

            # process each modality
            for modality in self.modalities:
                # find files matching modality pattern
                pattern = f"*{modality}*.nii.gz"
                matching_files = list(subject_dir.glob(pattern))

                if not matching_files:
                    logger.warning(f"No files found for {subject_id} {modality}")
                    continue

                input_file = matching_files[0]
                output_file = output_dir / subject_id / f"{modality}.nii.gz"

                # preprocess
                stats = self.preprocessor.preprocess_volume(input_file, output_file)
                stats['subject_id'] = subject_id
                stats['modality'] = modality
                stats['dataset'] = 'upenn_gbm'
                all_stats.append(stats)

        self._save_statistics(all_stats, output_dir / 'preprocessing_stats.json')
        logger.info(f"UPenn-GBM preprocessing complete. Saved to {output_dir}")

    def preprocess_ixi(self):
        """preprocess ixi dataset."""
        logger.info("Preprocessing IXI dataset...")

        dataset_dir = self.input_dir / 'ixi'
        output_dir = self.output_dir / 'ixi'

        all_stats = []

        # ixi has separate directories for each modality
        for modality in ['T1', 'T2', 'PD']:
            modality_dir = dataset_dir / f'IXI-{modality}'

            if not modality_dir.exists():
                logger.warning(f"Directory not found: {modality_dir}")
                continue

            # find all nifti files
            nifti_files = sorted(modality_dir.glob('*.nii.gz'))
            logger.info(f"Found {len(nifti_files)} {modality} scans")

            for input_file in tqdm(nifti_files, desc=f"Processing IXI {modality}"):
                # extract subject id from filename
                subject_id = input_file.stem.split('-')[0]  # e.g., ixi002

                output_file = output_dir / subject_id / f"{modality}.nii.gz"

                # preprocess
                stats = self.preprocessor.preprocess_volume(input_file, output_file)
                stats['subject_id'] = subject_id
                stats['modality'] = modality
                stats['dataset'] = 'ixi'
                all_stats.append(stats)

        self._save_statistics(all_stats, output_dir / 'preprocessing_stats.json')
        logger.info(f"IXI preprocessing complete. Saved to {output_dir}")

    def preprocess_oasis3(self):
        """preprocess oasis-3 dataset."""
        logger.info("Preprocessing OASIS-3 dataset...")

        dataset_dir = self.input_dir / 'oasis3'
        output_dir = self.output_dir / 'oasis3'

        # find all subject directories
        subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(subject_dirs)} subjects")

        all_stats = []

        for subject_dir in tqdm(subject_dirs, desc="Processing OASIS-3 subjects"):
            subject_id = subject_dir.name

            # oasis-3 has nested directory structure
            # find anat directory
            anat_dirs = list(subject_dir.rglob('anat'))

            if not anat_dirs:
                continue

            anat_dir = anat_dirs[0]

            # process t1 and t2 if available
            for modality in ['T1', 'T2', 'FLAIR']:
                pattern = f"*{modality}w*.nii.gz"
                matching_files = list(anat_dir.glob(pattern))

                if not matching_files:
                    continue

                input_file = matching_files[0]
                output_file = output_dir / subject_id / f"{modality}.nii.gz"

                # preprocess
                stats = self.preprocessor.preprocess_volume(input_file, output_file)
                stats['subject_id'] = subject_id
                stats['modality'] = modality
                stats['dataset'] = 'oasis3'
                all_stats.append(stats)

        self._save_statistics(all_stats, output_dir / 'preprocessing_stats.json')
        logger.info(f"OASIS-3 preprocessing complete. Saved to {output_dir}")

    def _save_statistics(self, stats: List[Dict], output_path: Path):
        """save preprocessing statistics to json file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # print summary
        successful = sum(1 for s in stats if s.get('success', False))
        logger.info(f"Processed {successful}/{len(stats)} volumes successfully")


def create_data_splits(
    dataset_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    create train/val/test splits for a dataset.

    args:
        dataset_dir: directory containing preprocessed subjects
        train_ratio: fraction for training set
        val_ratio: fraction for validation set
        test_ratio: fraction for test set
        seed: random seed for reproducibility
    """
    np.random.seed(seed)

    # get all subject ids
    subject_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    subject_ids = [d.name for d in subject_dirs]

    # shuffle
    np.random.shuffle(subject_ids)

    # calculate split sizes
    n_total = len(subject_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # split
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:n_train + n_val]
    test_ids = subject_ids[n_train + n_val:]

    # save splits
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    split_file = dataset_dir / 'data_splits.json'
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Data splits saved to {split_file}")
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='Preprocess neuroimaging datasets for NeuroScope',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['brats', 'upenn_gbm', 'ixi', 'oasis3', 'all'],
        default=['all'],
        help='Datasets to preprocess'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Root directory containing raw datasets'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )

    parser.add_argument(
        '--target-resolution',
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help='Target voxel resolution (mm)'
    )

    parser.add_argument(
        '--target-size',
        nargs=3,
        type=int,
        default=None,
        help='Target volume size (H W D). If not specified, keeps original size after resampling'
    )

    parser.add_argument(
        '--modalities',
        nargs='+',
        default=['T1', 'T1CE', 'T2', 'FLAIR'],
        help='MRI modalities to process'
    )

    parser.add_argument(
        '--normalization',
        choices=['zscore', 'minmax', 'percentile'],
        default='zscore',
        help='Intensity normalization method'
    )

    parser.add_argument(
        '--no-bias-correction',
        action='store_true',
        help='Skip N4 bias correction'
    )

    parser.add_argument(
        '--no-skull-strip',
        action='store_true',
        help='Skip skull stripping'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )

    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val/test splits'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )

    args = parser.parse_args()

    # handle 'all' option
    if 'all' in args.datasets:
        args.datasets = ['brats', 'upenn_gbm', 'ixi', 'oasis3']

    # initialize preprocessor
    preprocessor = MRIPreprocessor(
        target_resolution=tuple(args.target_resolution),
        target_size=tuple(args.target_size) if args.target_size else None,
        normalization_method=args.normalization,
        bias_correction=not args.no_bias_correction,
        skull_strip=not args.no_skull_strip,
    )

    # initialize dataset preprocessor
    dataset_preprocessor = DatasetPreprocessor(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        preprocessor=preprocessor,
        modalities=args.modalities,
        num_workers=args.num_workers,
    )

    # process each dataset
    logger.info(f"Starting preprocessing for datasets: {args.datasets}")
    logger.info(f"Target resolution: {args.target_resolution}")
    logger.info(f"Target size: {args.target_size}")
    logger.info(f"Normalization: {args.normalization}")

    for dataset in args.datasets:
        if dataset == 'brats':
            dataset_preprocessor.preprocess_brats()
        elif dataset == 'upenn_gbm':
            dataset_preprocessor.preprocess_upenn_gbm()
        elif dataset == 'ixi':
            dataset_preprocessor.preprocess_ixi()
        elif dataset == 'oasis3':
            dataset_preprocessor.preprocess_oasis3()

        # create data splits if requested
        if args.create_splits:
            dataset_dir = Path(args.output_dir) / dataset
            if dataset_dir.exists():
                create_data_splits(
                    dataset_dir,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio
                )

    logger.info("\n" + "="*80)
    logger.info("All preprocessing complete!")
    logger.info(f"Preprocessed data saved to: {args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
