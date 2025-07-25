import os
import json
import logging
import SimpleITK as sitk
import numpy as np
from typing import List, Dict, Tuple


def configure_logging() -> None:
    """
    Configure logging for bias field assessment.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_splits_metadata(path: str) -> Dict:
    """
    Load the metadata JSON with splits and file paths.
    """
    if not os.path.isfile(path):
        logging.error('Metadata file not found: %s', path)
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        data = json.load(f)
    logging.info('Loaded metadata from %s', path)
    return data


def generate_brain_mask(image: sitk.Image) -> sitk.Image:
    """
    Generate a brain mask using Otsu threshold and morphological fill.
    """
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    return sitk.BinaryFillhole(mask)


def compute_bias_residual_metrics(
    image: sitk.Image,
    mask: sitk.Image,
    smoothing_sigma_mm: float = 10.0
) -> Dict[str, float]:
    """
    Estimate bias field presence by comparing original and smoothed intensities.

    Parameters:
        image (sitk.Image): Input MRI volume.
        mask (sitk.Image): Brain mask volume.
        smoothing_sigma_mm (float): Gaussian smoothing sigma in millimeters.

    Returns:
        dict: Metrics with keys 'orig_std', 'residual_std', and 'ratio'.
    """
    # Convert to numpy for statistics
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    values = arr[mask_arr]
    orig_std = float(values.std())

    # Estimate bias field by Gaussian smoothing
    spacing = image.GetSpacing()
    sigma_vox = [smoothing_sigma_mm / sp for sp in spacing]
    smooth = sitk.DiscreteGaussian(image, variance=[sv**2 for sv in sigma_vox])
    smooth_arr = sitk.GetArrayFromImage(smooth).astype(np.float32)

    # Compute residual (anatomical variation)
    resid = arr - smooth_arr
    resid_std = float(resid[mask_arr].std())

    # Ratio: residual variation over original variation
    ratio = resid_std / (orig_std + 1e-8)
    return {'orig_std': orig_std, 'residual_std': resid_std, 'ratio': ratio}


def assess_dataset_bias(
    metadata: Dict,
    section: str,
    modalities: List[str],
    smoothing_sigma_mm: float
) -> Dict:
    """
    Loop over all subjects in a dataset section and compute bias residual metrics for each modality.

    Parameters:
        metadata (dict): Metadata containing valid_subjects info.
        section (str): Section key ('brats' or 'upenn').
        modalities (List[str]): List of modality filename suffixes.
        smoothing_sigma_mm (float): Sigma for Gaussian smoothing in mm.

    Returns:
        dict: Nested dict mapping subject -> modality -> metrics.
    """
    results = {}
    subjects = metadata[section]['valid_subjects']
    for subj_id, info in subjects.items():
        subj_metrics = {}
        for suffix in modalities:
            path = info.get(suffix)
            if not path or not os.path.isfile(path):
                logging.warning('Missing file for %s: %s', subj_id, suffix)
                continue
            logging.info('Assessing bias for %s [%s]', subj_id, suffix)
            img = sitk.ReadImage(path)
            mask = generate_brain_mask(img)
            metrics = compute_bias_residual_metrics(img, mask, smoothing_sigma_mm)
            subj_metrics[suffix] = metrics
        results[subj_id] = subj_metrics
    return results


def main() -> None:
    """
    Main entry to assess bias field correction in neuroscope datasets.

    Writes a JSON with bias residual metrics per subject and modality.
    """
    configure_logging()
    base = os.path.expanduser('~/Downloads/neuroscope')
    meta_path = os.path.join(base, 'scripts', 'neuroscope_dataset_metadata_splits.json')
    metadata = load_splits_metadata(meta_path)

    # Define modalities
    brats_mods = ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
    upenn_mods = ['_T1.nii.gz', '_T1GD.nii.gz', '_T2.nii.gz', '_FLAIR.nii.gz']

    # Smoothing sigma in mm
    sigma_mm = 10.0

    logging.info('Starting bias residual assessment for BraTS')
    brats_results = assess_dataset_bias(metadata, 'brats', brats_mods, sigma_mm)
    logging.info('Starting bias residual assessment for UPenn')
    upenn_results = assess_dataset_bias(metadata, 'upenn', upenn_mods, sigma_mm)

    # Save results
    out = {'brats': brats_results, 'upenn': upenn_results}
    out_path = os.path.join(base, 'scripts', 'neuroscope_bias_assessment.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logging.info('Bias assessment saved to %s', out_path)


if __name__ == '__main__':
    main()
