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


def find_modality_paths_fixed(
    data_root: str,
    section: str,
    subj_id: str,
    brats_root: str,
    upenn_root: str
) -> List[str]:
    """
    FIXED VERSION: Locate modality files for a subject, avoiding segmentation masks.
    
    Returns a list of four modality paths if complete, else empty list.
    This version specifically excludes segmentation files (_seg.nii.gz).
    """
    if section == 'brats':
        subj_dir = os.path.join(data_root, brats_root, subj_id)
        # BraTS expected suffixes (avoiding segmentation)
        expected_patterns = ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
    else:
        subj_dir = os.path.join(data_root, upenn_root, subj_id)
        # UPENN expected suffixes  
        expected_patterns = ['_T1.nii.gz', '_T1GD.nii.gz', '_T2.nii.gz', '_FLAIR.nii.gz']
    
    if not os.path.isdir(subj_dir):
        logging.warning("Subject directory not found: %s", subj_dir)
        return []
    
    files = os.listdir(subj_dir)
    # CRITICAL FIX: Exclude segmentation files
    files = [f for f in files if not ('_seg' in f.lower() or 'segmentation' in f.lower())]
    
    paths = []
    for pattern in expected_patterns:
        # Find files that end with the pattern
        matches = [f for f in files if f.endswith(pattern)]
        if not matches:
            logging.warning("Missing modality %s for %s/%s in files: %s", 
                          pattern, section, subj_id, files)
            return []
        
        # Take the first match
        paths.append(os.path.join(subj_dir, matches[0]))
        logging.info("Found %s for %s/%s: %s", pattern, section, subj_id, matches[0])
    
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
            logging.error("MASK DETECTED: %s appears to be binary mask, not MRI data", image_path)
            return False
        
        # Check if it has very few unique values - likely segmentation
        if len(unique_vals) < 10:
            logging.warning("POSSIBLE MASK: %s has only %d unique values", image_path, len(unique_vals))
            return False
        
        # Check if it has reasonable intensity distribution for MRI
        if arr.std() < 0.01:  # Very low variation
            logging.warning("LOW VARIATION: %s has very low intensity variation", image_path)
            return False
            
        logging.info("VERIFIED MRI: %s passes MRI content checks", image_path)
        return True
        
    except Exception as e:
        logging.error("Error verifying image %s: %s", image_path, e)
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


def process_subject_with_verification(
    section: str,
    subj_id: str,
    paths: List[str],
    output_dir: str,
    target_spacing: Tuple[float, float, float]
) -> bool:
    """
    Preprocess a subject with verification: mask, normalize, resample, and save modalities.
    
    Returns True if successful, False if any issues detected.
    """
    logging.info("[%s/%s] Starting preprocessing with verification", section, subj_id)
    
    # CRITICAL: Verify all input files are actually MRI data, not masks
    for i, path in enumerate(paths):
        if not verify_image_is_mri(path):
            logging.error("[%s/%s] ABORTING: Input file %d appears to be a mask: %s", 
                         section, subj_id, i, path)
            return False
    
    # All inputs verified as MRI data, proceed with processing
    t1 = sitk.ReadImage(paths[0])
    mask = generate_brain_mask(t1)
    subj_out_dir = os.path.join(output_dir, section, subj_id)
    os.makedirs(subj_out_dir, exist_ok=True)
    
    modality_names = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
    
    for i, path in enumerate(paths):
        mod_name = modality_names[i]
        logging.info("[%s/%s] Processing modality %s", section, subj_id, mod_name)
        img = sitk.ReadImage(path)
        norm = percentile_normalize(img, mask)
        resampled = resample_to_isotropic(norm, target_spacing)
        out_path = os.path.join(subj_out_dir, mod_name)
        sitk.WriteImage(resampled, out_path)
        
        # Verify output
        if not verify_image_is_mri(out_path):
            logging.error("[%s/%s] ERROR: Output file appears to be corrupted: %s", 
                         section, subj_id, out_path)
            return False
            
        logging.info("[%s/%s] Successfully saved %s", section, subj_id, mod_name)
    
    logging.info("[%s/%s] Completed preprocessing successfully", section, subj_id)
    return True


def dump_split_txts(
    json_path: str,
    out_dir: str
) -> None:
    """
    Dump train/val/test subject lists from JSON to text files.
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
    Main: preprocess train subjects using JSON splits with verification.
    """
    configure_logging()
    start_all = time.time()

    # ADJUST THESE PATHS TO MATCH YOUR SETUP
    base_dir = '/Volumes/USB DRIVE/neuroscope'  # or your USB drive path
    data_dir = os.path.join(base_dir, 'data')
    brats_root = os.path.join('BraTS-TCGA-GBM', 'Pre-operative_TCGA_GBM_NIfTI_and_Segmentations')
    upenn_root = os.path.join('PKG - UPENN-GBM-NIfTI', 'UPENN-GBM', 'NIfTI-files', 'images_structural')
    
    # Local scripts and outputs
    local_base = os.path.expanduser('~/Downloads/neuroscope')
    metadata_splits = os.path.join(local_base, 'scripts', '01_data_preparation_pipeline', 'neuroscope_dataset_metadata_splits.json')
    output_dir = os.path.join(local_base, 'preprocessed')  # output dir
    txt_out = os.path.join(local_base, 'scripts')
    target_spacing = (1.0, 1.0, 1.0)

    logging.info("Loading JSON splits from %s", metadata_splits)
    if not os.path.isfile(metadata_splits):
        logging.error("Metadata splits file not found: %s", metadata_splits)
        return
    with open(metadata_splits, 'r') as f:
        meta = json.load(f)

    # Build task list (limiting to first few for testing)
    tasks = []
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') == 'train':  # LIMIT FOR TESTING
                tasks.append((section, sid))
    
    total = len(tasks)
    logging.info("Total train subjects to process (testing): %d", total)
    
    if total == 0:
        logging.error("No train subjects found in JSON.")
        return

    # Resolve paths and process with verification
    success_count = 0
    for section, sid in tasks:
        paths = find_modality_paths_fixed(data_dir, section, sid, brats_root, upenn_root)
        if not paths:
            logging.error("Could not find all modalities for %s/%s", section, sid)
            continue
            
        # Process with verification
        if process_subject_with_verification(section, sid, paths, output_dir, target_spacing):
            success_count += 1
        else:
            logging.error("Failed to process %s/%s", section, sid)

    logging.info("Successfully processed %d/%d subjects", success_count, total)
    
    # Only dump split files if some subjects were processed successfully
    if success_count > 0:
        logging.info("Dumping split .txt files to %s", txt_out)
        dump_split_txts(metadata_splits, txt_out)

    elapsed = time.time() - start_all
    logging.info("Processing finished in %.2f minutes", elapsed/60)
    logging.info("Outputs available in %s", output_dir)


if __name__ == '__main__':
    main()