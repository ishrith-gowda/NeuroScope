import os
import json
import logging
import SimpleITK as sitk
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """Configure logging format and level for preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_modality_paths_fixed(section: str, subj_id: str) -> List[str]:
    """
    Locate modality files for a subject, avoiding segmentation masks.
    Uses neuroscope_config.py for path resolution.
    
    Returns a list of four modality paths if complete, else empty list.
    """
    if section == 'brats':
        subj_dir = PATHS['raw_data_root'] / PATHS['raw_brats_root'] / subj_id
        expected_patterns = ['_t1.nii.gz', '_t1Gd.nii.gz', '_t2.nii.gz', '_flair.nii.gz']
    else:
        subj_dir = PATHS['raw_data_root'] / PATHS['raw_upenn_root'] / subj_id
        expected_patterns = ['_T1.nii.gz', '_T1GD.nii.gz', '_T2.nii.gz', '_FLAIR.nii.gz']
    
    if not subj_dir.exists():
        logging.warning("Subject directory not found: %s", subj_dir)
        return []
    
    files = os.listdir(str(subj_dir))
    # Exclude segmentation files
    files = [f for f in files if not ('_seg' in f.lower() or 'segmentation' in f.lower())]
    
    paths = []
    for pattern in expected_patterns:
        matches = [f for f in files if f.endswith(pattern)]
        if not matches:
            logging.warning("Missing modality %s for %s/%s in files: %s", 
                          pattern, section, subj_id, files)
            return []
        paths.append(str(subj_dir / matches[0]))
        logging.debug("Found %s for %s/%s: %s", pattern, section, subj_id, matches[0])
    
    return paths


def verify_image_is_mri(image_path: str) -> bool:
    """
    Quick verification that an image contains MRI intensities, not segmentation masks.
    """
    try:
        img = sitk.ReadImage(image_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        unique_vals = np.unique(arr)
        if len(unique_vals) <= 2 and np.allclose(unique_vals, [0, 1]):
            logging.error("MASK DETECTED: %s appears to be binary mask", image_path)
            return False
        
        if len(unique_vals) < 10:
            logging.warning("POSSIBLE MASK: %s has only %d unique values", image_path, len(unique_vals))
            return False
        
        if arr.std() < 0.01:
            logging.warning("LOW VARIATION: %s has very low intensity variation", image_path)
            return False
            
        return True
        
    except Exception as e:
        logging.error("Error verifying image %s: %s", image_path, e)
        return False


def fast_intensity_normalize(
    image: sitk.Image, 
    background_percentile: float = 5.0,  # More conservative background threshold
    brain_low: float = 1.0, 
    brain_high: float = 99.0
) -> sitk.Image:

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    
    # Step 1: Identify background threshold (exclude bottom background_percentile)
    background_threshold = np.percentile(arr.flatten(), background_percentile)
    
    # Step 2: Get brain tissue voxels (above background threshold)
    brain_mask = arr > background_threshold
    brain_voxels = arr[brain_mask]
    
    if brain_voxels.size == 0:
        logging.warning("No brain voxels found after background exclusion")
        return image
    
    # Step 3: Compute normalization range from brain tissue only
    brain_low_val = np.percentile(brain_voxels, brain_low)
    brain_high_val = np.percentile(brain_voxels, brain_high)
    
    if brain_high_val <= brain_low_val:
        logging.warning("Invalid percentile range: low=%.2f, high=%.2f", brain_low_val, brain_high_val)
        return image
    
    # Step 4: Initialize output array
    arr_normalized = np.zeros_like(arr)
    
    # Step 5: Normalize brain tissue to [0,1]
    brain_intensities = np.clip(arr[brain_mask], brain_low_val, brain_high_val)
    arr_normalized[brain_mask] = (brain_intensities - brain_low_val) / (brain_high_val - brain_low_val)
    
    # Step 6: Background remains at 0 (clean for CycleGAN)
    # arr_normalized[~brain_mask] = 0  # Already initialized to 0
    
    # Create output image
    normalized_img = sitk.GetImageFromArray(arr_normalized)
    normalized_img.CopyInformation(image)
    
    logging.debug("Normalization: bg_thresh=%.2f, brain_range=[%.2f, %.2f] → [0, 1], bg→0", 
                 background_threshold, brain_low_val, brain_high_val)
    
    return normalized_img


def resample_to_isotropic(image: sitk.Image, spacing: Tuple[float, float, float]) -> sitk.Image:
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


def process_subject_fast_intensity_only(
    section: str,
    subj_id: str,
    paths: List[str],
    target_spacing: Tuple[float, float, float],
    enable_resampling: bool = True,
    background_percentile: float = 5.0,
    brain_low: float = 1.0,
    brain_high: float = 99.0
) -> bool:
    start_time = time.time()
    logging.info("[%s/%s] Starting FAST intensity normalization", section, subj_id)
    
    # Verify all input files are MRI data
    for i, path in enumerate(paths):
        if not verify_image_is_mri(path):
            logging.error("[%s/%s] ABORTING: Input file %d appears to be a mask: %s", 
                         section, subj_id, i, path)
            return False
    
    # Create output directory
    subj_out_dir = PATHS['preprocessed_dir'] / section / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each modality with fast intensity normalization
    # Standardized output naming for consistency
    modality_names = ['t1.nii.gz', 't1gd.nii.gz', 't2.nii.gz', 'flair.nii.gz']
    
    for i, (input_path, mod_name) in enumerate(zip(paths, modality_names)):
        mod_start = time.time()
        logging.info("[%s/%s] Processing modality %s", section, subj_id, mod_name)
        
        # Step 1: Load image
        img = sitk.ReadImage(input_path)
        
        # Step 2: Fast intensity normalization with configurable parameters
        normalized_img = fast_intensity_normalize(
            img, 
            background_percentile=background_percentile,
            brain_low=brain_low,
            brain_high=brain_high
        )
        
        # Step 3: Optional isotropic resampling
        if enable_resampling:
            final_img = resample_to_isotropic(normalized_img, target_spacing)
        else:
            final_img = normalized_img
        
        # Step 4: Save with standardized naming
        output_path = str(subj_out_dir / mod_name)
        sitk.WriteImage(final_img, output_path)
        
        # Verify final output
        if not verify_image_is_mri(output_path):
            logging.error("[%s/%s] ERROR: Final output appears corrupted: %s", section, subj_id, mod_name)
            return False
        
        mod_elapsed = time.time() - mod_start
        logging.info("[%s/%s] %s completed in %.1f seconds", section, subj_id, mod_name, mod_elapsed)
    
    total_elapsed = time.time() - start_time
    logging.info("[%s/%s] Fast intensity normalization completed in %.1f seconds", 
                section, subj_id, total_elapsed)
    logging.info("[%s/%s] Ready for CycleGAN training", section, subj_id)
    return True


def process_subjects_from_splits(
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    enable_resampling: bool = True,
    background_percentile: float = 5.0,
    brain_low: float = 1.0,
    brain_high: float = 99.0,
    splits_to_process: List[str] = ['train']  # Can be ['train', 'val', 'test']
) -> None:

    start_all = time.time()
    
    logging.info("=== FAST INTENSITY NORMALIZATION FOR CYCLEGAN ===")
    logging.info("Parameters:")
    logging.info("  Target spacing: %s", target_spacing)
    logging.info("  Resampling enabled: %s", enable_resampling)
    logging.info("  Background percentile: %.1f%%", background_percentile)
    logging.info("  Brain percentiles: %.1f%% - %.1f%%", brain_low, brain_high)
    logging.info("  Splits to process: %s", splits_to_process)
    
    # Load metadata with splits
    if not PATHS['metadata_splits'].exists():
        logging.error("Metadata splits file not found: %s", PATHS['metadata_splits'])
        return
    
    with open(str(PATHS['metadata_splits']), 'r') as f:
        meta = json.load(f)

    # Build task list from specified splits
    tasks = []
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') in splits_to_process:
                tasks.append((section, sid, info.get('split')))
    
    total = len(tasks)
    logging.info("Total subjects to process: %d", total)
    
    if total == 0:
        logging.error("No subjects found in specified splits: %s", splits_to_process)
        return

    # Process subjects with configurable fast intensity normalization
    success_count = 0
    failed_subjects = []
    total_processing_time = 0
    
    for idx, (section, sid, split) in enumerate(tasks, 1):
        logging.info("=== Processing %d/%d: %s/%s (%s split) ===", idx, total, section, sid, split)
        
        paths = find_modality_paths_fixed(section, sid)
        if not paths:
            logging.error("Could not find all modalities for %s/%s", section, sid)
            failed_subjects.append(f"{section}/{sid}")
            continue
            
        # Process with configurable fast intensity normalization
        subject_start = time.time()
        if process_subject_fast_intensity_only(
            section, sid, paths, target_spacing, enable_resampling,
            background_percentile, brain_low, brain_high
        ):
            success_count += 1
            subject_time = time.time() - subject_start
            total_processing_time += subject_time
            avg_time = total_processing_time / success_count
            remaining = total - idx
            eta_minutes = (remaining * avg_time) / 60
            logging.info("SUCCESS: %s/%s (%.1fs) | avg: %.1fs/subject | ETA: %.1f min", 
                        section, sid, subject_time, avg_time, eta_minutes)
        else:
            logging.error("FAILED: %s/%s", section, sid)
            failed_subjects.append(f"{section}/{sid}")

    # Summary
    elapsed_total = time.time() - start_all
    logging.info("=== FAST INTENSITY NORMALIZATION SUMMARY ===")
    logging.info("Successfully processed: %d/%d subjects", success_count, total)
    logging.info("Failed subjects: %d", len(failed_subjects))
    logging.info("Total time: %.1f minutes (%.1f seconds/subject avg)", 
                elapsed_total/60, total_processing_time/max(success_count, 1))
    
    if failed_subjects:
        logging.warning("Failed subjects:")
        for subj in failed_subjects:
            logging.warning("  - %s", subj)
    
    logging.info("Outputs available in %s", PATHS['preprocessed_dir'])


def dump_split_txts() -> None:
    """
    Dump train/val/test subject lists from JSON to text files for convenience.
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
        logging.info("Wrote %d entries to %s", len(entries), path)


def main() -> None:
    configure_logging()
    
    # Process only training subjects by default (for CycleGAN training)
    # You can modify splits_to_process to include ['train', 'val'] or ['train', 'val', 'test']
    process_subjects_from_splits(
        target_spacing=(1.0, 1.0, 1.0),
        enable_resampling=True,
        background_percentile=5.0, 
        brain_low=1.0,
        brain_high=99.0,
        splits_to_process=['train']  # Only train for CycleGAN training
    )
    
    # Generate convenience text files
    dump_split_txts()

if __name__ == '__main__':
    main()