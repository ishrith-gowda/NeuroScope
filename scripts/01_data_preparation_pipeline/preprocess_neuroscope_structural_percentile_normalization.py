import os
import json
import logging
import SimpleITK as sitk
import numpy as np
import time
from typing import List, Tuple, Dict


def configure_logging() -> None:
    """
    Configure logging format and level for preprocessing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_modality_paths(
    data_root: str,
    section: str,
    subj_id: str,
    brats_root: str,
    upenn_root: str
) -> List[str]:
    """
    Locate modality files for a subject in the data directory.

    Returns a list of four modality paths if complete, else empty list.
    """
    if section == 'brats':
        subj_dir = os.path.join(data_root, brats_root, subj_id)
    else:
        subj_dir = os.path.join(data_root, upenn_root, subj_id)
    if not os.path.isdir(subj_dir):
        logging.warning("Subject directory not found: %s", subj_dir)
        return []
    patterns = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
    files = os.listdir(subj_dir)
    lower_map = {fname.lower(): fname for fname in files}
    paths = []
    for pat in patterns:
        match = next((fname for lname, fname in lower_map.items() if lname.endswith(pat)), None)
        if not match:
            logging.warning("Missing modality %s for %s/%s", pat, section, subj_id)
            return []
        paths.append(os.path.join(subj_dir, match))
    return paths


def generate_brain_mask(image: sitk.Image) -> sitk.Image:
    """
    Create a brain mask via Otsu threshold and fill holes.
    """
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    return sitk.BinaryFillhole(mask)


def percentile_normalize(
    image: sitk.Image,
    mask: sitk.Image,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> sitk.Image:
    """
    Clip intensities at given percentiles within mask and rescale to [0,1].
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    vals = arr[mask_arr]
    if vals.size == 0:
        logging.error("Empty mask for percentile normalization")
        return image
    lo, hi = np.percentile(vals, (p_low, p_high))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    norm_img = sitk.GetImageFromArray(arr)
    norm_img.CopyInformation(image)
    return norm_img


def resample_to_isotropic(
    image: sitk.Image,
    spacing: Tuple[float, float, float]
) -> sitk.Image:
    """
    Resample image to target isotropic spacing with linear interpolation.
    """
    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()
    new_size = [int(round(orig_size[i] * (orig_spacing[i] / spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


def process_subject(
    section: str,
    subj_id: str,
    paths: List[str],
    output_dir: str,
    target_spacing: Tuple[float, float, float]
) -> None:
    """
    Preprocess a subject: mask, normalize, resample, and save modalities.

    Logs each step and raises on errors.
    """
    logging.info("[%s/%s] Starting preprocessing", section, subj_id)
    t1 = sitk.ReadImage(paths[0])
    mask = generate_brain_mask(t1)
    subj_out_dir = os.path.join(output_dir, section, subj_id)
    os.makedirs(subj_out_dir, exist_ok=True)
    for path in paths:
        mod = os.path.basename(path)
        logging.info("[%s/%s] Processing modality %s", section, subj_id, mod)
        img = sitk.ReadImage(path)
        norm = percentile_normalize(img, mask)
        resampled = resample_to_isotropic(norm, target_spacing)
        out_path = os.path.join(subj_out_dir, mod)
        sitk.WriteImage(resampled, out_path)
        logging.info("[%s/%s] Saved %s", section, subj_id, mod)
    logging.info("[%s/%s] Completed preprocessing", section, subj_id)


def dump_split_txts(
    json_path: str,
    out_dir: str
) -> None:
    """
    Dump train/val/test subject lists from JSON to text files.

    Each line: '<section>/<subject_id>'.
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)
    splits = {'train': [], 'val': [], 'test': []}
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            sp = info.get('split')
            if sp in splits:
                splits[sp].append(f"{section}/{sid}")
    os.makedirs(out_dir, exist_ok=True)
    for sp, entries in splits.items():
        path = os.path.join(out_dir, f"{sp}_subjects.txt")
        with open(path, 'w') as f:
            for line in entries:
                f.write(line + '\n')
        logging.info("Wrote %d entries to %s", len(entries), path)


def main() -> None:
    """
    Main: preprocess train subjects using JSON splits, then dump .txt lists.

    Implements progress tracking and detailed logging.
    """
    configure_logging()
    start_all = time.time()

    # Raw data on USB
    base_dir = '/Volumes/USB DRIVE/neuroscope'
    data_dir = os.path.join(base_dir, 'data')
    brats_root = os.path.join('BraTS-TCGA-GBM', 'Pre-operative_TCGA_GBM_NIfTI_and_Segmentations')
    upenn_root = os.path.join('PKG - UPENN-GBM-NIfTI', 'UPENN-GBM', 'NIfTI-files', 'images_structural')
    # Local scripts and outputs under home
    local_base = os.path.expanduser('~/Downloads/neuroscope')
    metadata_splits = os.path.join(local_base, 'scripts', 'neuroscope_dataset_metadata_splits.json')
    output_dir = os.path.join(local_base, 'preprocessed')
    txt_out = os.path.join(local_base, 'scripts')
    target_spacing = (1.0, 1.0, 1.0)

    logging.info("Loading JSON splits from %s", metadata_splits)
    if not os.path.isfile(metadata_splits):
        logging.error("Metadata splits file not found: %s", metadata_splits)
        return
    with open(metadata_splits, 'r') as f:
        meta = json.load(f)

    # Build task list from JSON
    tasks = []
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') == 'train':
                tasks.append((section, sid))
    total = len(tasks)
    logging.info("Total train subjects from JSON: %d", total)
    if total == 0:
        logging.error("No train subjects found in JSON.")
        return

    # Resolve paths
    valid_tasks = []
    for section, sid in tasks:
        paths = find_modality_paths(data_dir, section, sid, brats_root, upenn_root)
        if paths:
            valid_tasks.append((section, sid, paths))
    logging.info("Valid subjects to process: %d", len(valid_tasks))

    # Process with progress
    for idx, (section, sid, paths) in enumerate(valid_tasks, 1):
        logging.info("Processing subject %d/%d: %s/%s", idx, len(valid_tasks), section, sid)
        try:
            process_subject(section, sid, paths, output_dir, target_spacing)
        except Exception as e:
            logging.error("Error %s/%s: %s", section, sid, e)

    # Dump .txt files for future
    logging.info("Dumping split .txt files to %s", txt_out)
    dump_split_txts(metadata_splits, txt_out)

    elapsed = time.time() - start_all
    logging.info("All preprocessing finished in %.2f minutes", elapsed/60)
    logging.info("Outputs available in %s", output_dir)

if __name__ == '__main__':
    main()