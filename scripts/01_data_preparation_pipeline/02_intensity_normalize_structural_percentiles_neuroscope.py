import os
import json
import logging
import SimpleITK as sitk
import numpy as np
import time
from typing import List, Tuple, Dict
from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """
    Configure logging format and level for preprocessing.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_modality_paths_fixed(
    section: str,
    subj_id: str
) -> List[str]:
    """
    FIXED VERSION: Locate modality files for a subject, avoiding segmentation masks.
    Uses neuroscope_config.py for path resolution.
    
    Returns a list of four modality paths if complete, else empty list.
    This version specifically excludes segmentation files (_seg.nii.gz).
    """
    if section == 'brats':
        subj_dir = PATHS['raw_data_root'] / PATHS['raw_brats_root'] / subj_id
        # BraTS expected suffixes (avoiding segmentation)
        expected_patterns = ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
    else:
        subj_dir = PATHS['raw_data_root'] / PATHS['raw_upenn_root'] / subj_id
        # UPENN expected suffixes  
        expected_patterns = ['_T1.nii.gz', '_T1GD.nii.gz', '_T2.nii.gz', '_FLAIR.nii.gz']
    
    if not subj_dir.exists():
        logging.warning("subject directory not found: %s", subj_dir)
        return []
    
    files = os.listdir(str(subj_dir))
    # CRITICAL FIX: Exclude segmentation files
    files = [f for f in files if not ('_seg' in f.lower() or 'segmentation' in f.lower())]
    
    paths = []
    for pattern in expected_patterns:
        # Find files that end with the pattern
        matches = [f for f in files if f.endswith(pattern)]
        if not matches:
            logging.warning("missing modality %s for %s/%s in files: %s", 
                          pattern, section, subj_id, files)
            return []
        
        # Take the first match
        paths.append(str(subj_dir / matches[0]))
        logging.info("found %s for %s/%s: %s", pattern, section, subj_id, matches[0])
    
    return paths


def verify_image_is_mri(image_path: str) -> bool:
    """
    Quick verification that an image contains MRI intensities, not segmentation masks.
    
    Returns True if the image appears to be MRI data, False if it looks like a mask.
    """
    try:
        img = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Check if it's binary (0s and 1s only) - likely a mask
        unique_vals = np.unique(arr)
        if len(unique_vals) <= 2 and np.allclose(unique_vals, [0, 1]):
            logging.error("mask detected: %s appears to be binary mask, not MRI data", image_path)
            return False
        
        # Check if it has very few unique values - likely segmentation
        if len(unique_vals) < 10:
            logging.warning("possible mask: %s has only %d unique values", image_path, len(unique_vals))
            return False
        
        # Check if it has reasonable intensity distribution for MRI
        if arr.std() < 0.01:  # Very low variation
            logging.warning("low variation: %s has very low intensity variation", image_path)
            return False
            
        logging.info("verified MRI: %s passes MRI content checks", image_path)
        return True
        
    except Exception as e:
        logging.error("error verifying image %s: %s", image_path, e)
        return False


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
        logging.error("empty mask for percentile normalization")
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


def process_subject_with_verification(
    section: str,
    subj_id: str,
    paths: List[str],
    target_spacing: Tuple[float, float, float]
) -> bool:
    """
    Preprocess a subject with verification: mask, normalize, resample, and save modalities.
    
    Returns True if successful, False if any issues detected.
    """
    logging.info("[%s/%s] starting preprocessing with verification", section, subj_id)
    
    # CRITICAL: Verify all input files are actually MRI data, not masks
    for i, path in enumerate(paths):
        if not verify_image_is_mri(path):
            logging.error("[%s/%s] aborting: input file %d appears to be a mask: %s", 
                         section, subj_id, i, path)
            return False
    
    # All inputs verified as MRI data, proceed with processing
    t1 = sitk.ReadImage(paths[0])
    mask = generate_brain_mask(t1)
    
    # Use neuroscope_config for output directory
    subj_out_dir = PATHS['preprocessed_dir'] / section / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    modality_names = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
    
    for i, path in enumerate(paths):
        mod_name = modality_names[i]
        logging.info("[%s/%s] processing modality %s", section, subj_id, mod_name)
        img = sitk.ReadImage(path)
        norm = percentile_normalize(img, mask)
        resampled = resample_to_isotropic(norm, target_spacing)
        out_path = str(subj_out_dir / mod_name)
        sitk.WriteImage(resampled, out_path)
        
        # Verify output
        if not verify_image_is_mri(out_path):
            logging.error("[%s/%s] error: output file appears to be corrupted: %s", 
                         section, subj_id, out_path)
            return False
            
        logging.info("[%s/%s] successfully saved %s", section, subj_id, mod_name)
    
    logging.info("[%s/%s] completed preprocessing successfully", section, subj_id)
    return True


def dump_split_txts() -> None:
    """
    Dump train/val/test subject lists from JSON to text files.
    Uses neuroscope_config.py for paths.
    """
    with open(str(PATHS['metadata_splits']), 'r') as f:
        meta = json.load(f)
    
    splits = {'train': [], 'val': [], 'test': []}
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            sp = info.get('split')
            if sp in splits:
                splits[sp].append(f"{section}/{sid}")
    
    for sp, entries in splits.items():
        path = PATHS['scripts_dir'] / f"{sp}_subjects.txt"
        with open(str(path), 'w') as f:
            for line in entries:
                f.write(line + '\n')
        logging.info("wrote %d entries to %s", len(entries), path)


def main() -> None:
    """
    Main: preprocess train subjects using JSON splits with verification.
    Uses neuroscope_config.py for all path management.
    """
    configure_logging()
    start_all = time.time()

    # Use neuroscope_config for all paths
    logging.info("using neuroscope_config.py for path management")
    logging.info("  USB root: %s", PATHS['usb_root'])
    logging.info("  raw data: %s", PATHS['raw_data_root'])
    logging.info("  preprocessed output: %s", PATHS['preprocessed_dir'])
    logging.info("  metadata: %s", PATHS['metadata_splits'])
    
    target_spacing = (1.0, 1.0, 1.0)

    logging.info("loading JSON splits from %s", PATHS['metadata_splits'])
    if not PATHS['metadata_splits'].exists():
        logging.error("metadata splits file not found: %s", PATHS['metadata_splits'])
        return
    
    with open(str(PATHS['metadata_splits']), 'r') as f:
        meta = json.load(f)

    # Build task list (process all train subjects)
    tasks = []
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') == 'train':
                tasks.append((section, sid))
    
    total = len(tasks)
    logging.info("total train subjects to process: %d", total)
    
    if total == 0:
        logging.error("no train subjects found in JSON.")
        return

    # Resolve paths and process with verification
    success_count = 0
    for section, sid in tasks:
        paths = find_modality_paths_fixed(section, sid)
        if not paths:
            logging.error("could not find all modalities for %s/%s", section, sid)
            continue
            
        # Process with verification
        if process_subject_with_verification(section, sid, paths, target_spacing):
            success_count += 1
        else:
            logging.error("failed to process %s/%s", section, sid)

    logging.info("successfully processed %d/%d subjects", success_count, total)
    
    # Only dump split files if some subjects were processed successfully
    if success_count > 0:
        logging.info("dumping split .txt files")
        dump_split_txts()

    elapsed = time.time() - start_all
    logging.info("processing finished in %.2f minutes", elapsed/60)
    logging.info("outputs available in %s", PATHS['preprocessed_dir'])


if __name__ == '__main__':
    main()