import os
import json
import time
import logging
import SimpleITK as sitk
import numpy as np
from typing import Dict, Any


def configure_logging() -> None:
    """
    Configure logging for verification script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def verify_subject(
    section: str,
    subj_id: str,
    subj_dir: str,
    target_spacing: tuple
) -> Dict[str, Any]:
    """
    Verify preprocessing of a single subject.

    Checks:
      - Voxel spacing matches target_spacing
      - Intensity ranges [0,1] (with small epsilon)
      - Consistent image shape across modalities
      - Brain mask alignment: fraction of nonzero voxels in T1 mask matches other modalities

    Returns a dict of metrics per modality and subject-level.
    """
    metrics: Dict[str, Any] = {}
    # List NIfTI files
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    imgs = {}
    for f in files:
        path = os.path.join(subj_dir, f)
        img = sitk.ReadImage(path)
        imgs[f] = img
    # Collect spacing and shape
    spacings = {name: img.GetSpacing() for name, img in imgs.items()}
    shapes = {name: img.GetSize() for name, img in imgs.items()}
    # Check spacing
    for name, sp in spacings.items():
        metrics.setdefault('spacing', {})[name] = tuple(round(s,3) for s in sp)
        if not all(abs(sp[i] - target_spacing[i]) < 1e-3 for i in range(3)):
            logging.warning("%s/%s: spacing mismatch in %s: %s", section, subj_id, name, sp)
    # Check shapes
    sizes = list(shapes.values())
    if len(set(sizes)) > 1:
        logging.warning("%s/%s: inconsistent shapes: %s", section, subj_id, shapes)
    metrics['shape'] = shapes
    # Intensity stats and mask alignment
    # Use the first modality as T1 to generate mask
    t1_name = next(n for n in files if 't1' in n.lower() and 'gd' not in n.lower())
    t1_img = imgs[t1_name]
    t1_arr = sitk.GetArrayFromImage(t1_img)
    # Brain mask from T1
    mask = sitk.OtsuThreshold(t1_img, 0, 1, 200)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    mask_fraction = mask_arr.mean()
    metrics['mask_fraction'] = float(mask_fraction)
    # Per modality intensity and mask overlap
    metrics['modalities'] = {}
    for name, img in imgs.items():
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        lumi = float(arr.min())
        hium = float(arr.max())
        p01, p99 = np.percentile(arr, (1,99))
        # Fraction of nonzero voxels within mask
        nonzero = (arr>0)
        overlap = float((nonzero & mask_arr).sum() / (mask_arr.sum() + 1e-8))
        metrics['modalities'][name] = {
            'min': lumi,
            'max': hium,
            'pct1': float(p01),
            'pct99': float(p99),
            'mask_overlap': overlap
        }
        if lumi < -0.01 or hium > 1.01:
            logging.warning("%s/%s: intensity out of [0,1] for %s: min=%.3f, max=%.3f",
                            section, subj_id, name, lumi, hium)
        if overlap < 0.9:
            logging.warning("%s/%s: low mask overlap for %s: %.2f", section, subj_id, name, overlap)
    return metrics


def main() -> None:
    """
    Verify preprocessing for all train subjects and export a JSON report.
    """
    configure_logging()
    # Paths
    local_base = os.path.expanduser('~/Downloads/neuroscope')
    preproc_dir = os.path.join(local_base, 'preprocessed')
    metadata_path = os.path.join(local_base, 'scripts', 'neuroscope_dataset_metadata_splits.json')
    report_path = os.path.join(local_base, 'scripts', 'preprocessing_verification_report.json')
    target_spacing = (1.0, 1.0, 1.0)

    # Load metadata splits
    if not os.path.isfile(metadata_path):
        logging.error("Metadata splits not found: %s", metadata_path)
        return
    with open(metadata_path, 'r') as f:
        meta = json.load(f)

    report: Dict[str, Any] = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'subjects': {}
    }

    # Iterate train subjects from JSON
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') != 'train':
                continue
            subj_dir = os.path.join(preproc_dir, section, sid)
            if not os.path.isdir(subj_dir):
                logging.error("Preprocessed directory missing for %s/%s", section, sid)
                continue
            logging.info("Verifying %s/%s", section, sid)
            metrics = verify_subject(section, sid, subj_dir, target_spacing)
            report['subjects'][f"{section}/{sid}"] = metrics

    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info("Verification report written to %s", report_path)

if __name__ == '__main__':
    main()
