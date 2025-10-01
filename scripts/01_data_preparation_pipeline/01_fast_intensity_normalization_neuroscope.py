import os
import json
import logging
import SimpleITK as sitk
import numpy as np
import time
import argparse
from typing import List, Tuple, Dict, Optional
from neuroscope_preprocessing_config import PATHS
from preprocessing_utils import verify_mri_path


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
        logging.warning("subject directory not found: %s", subj_dir)
        return []
    
    files = os.listdir(str(subj_dir))
    # Exclude segmentation files
    files = [f for f in files if not ('_seg' in f.lower() or 'segmentation' in f.lower())]
    
    paths = []
    for pattern in expected_patterns:
        matches = [f for f in files if f.endswith(pattern)]
        if not matches:
            logging.warning("missing modality %s for %s/%s in files: %s", 
                          pattern, section, subj_id, files)
            return []
        paths.append(str(subj_dir / matches[0]))
        logging.debug("found %s for %s/%s: %s", pattern, section, subj_id, matches[0])
    
    return paths


def fast_intensity_normalize(
    image: sitk.Image, 
    background_percentile: float = 5.0,  # More conservative background threshold
    brain_low: float = 1.0, 
    brain_high: float = 99.0
) -> sitk.Image:
    """
    Fast intensity normalization for CycleGAN preprocessing.
    
    Normalizes brain tissue to [0,1] range while keeping background at 0.
    This ensures clean input for CycleGAN training.
    
    Args:
        image: Input SimpleITK image
        background_percentile: Percentile threshold for background detection
        brain_low: Lower percentile for brain tissue normalization
        brain_high: Upper percentile for brain tissue normalization
        
    Returns:
        Normalized image with brain tissue in [0,1] and background at 0
    """
    try:
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Validate input
        if arr.size == 0:
            logging.error("Empty image array")
            return image
            
        # Check for NaN or infinite values
        if not np.isfinite(arr).all():
            logging.warning("Image contains NaN or infinite values, cleaning...")
            arr = np.nan_to_num(arr, nan=0.0, posinf=arr.max(), neginf=arr.min())
        
        # Step 1: Identify background threshold (exclude bottom background_percentile)
        background_threshold = np.percentile(arr.flatten(), background_percentile)
        
        # Step 2: Get brain tissue voxels (above background threshold)
        brain_mask = arr > background_threshold
        brain_voxels = arr[brain_mask]
        
        if brain_voxels.size == 0:
            logging.warning("no brain voxels found after background exclusion, using global normalization")
            # Fallback: use global normalization
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr_normalized = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr_normalized = np.zeros_like(arr)
        else:
            # Step 3: Compute normalization range from brain tissue only
            brain_low_val = np.percentile(brain_voxels, brain_low)
            brain_high_val = np.percentile(brain_voxels, brain_high)
            
            if brain_high_val <= brain_low_val:
                logging.warning("invalid percentile range: low=%.2f, high=%.2f, using global normalization", 
                               brain_low_val, brain_high_val)
                # Fallback: use global normalization
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max > arr_min:
                    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
                else:
                    arr_normalized = np.zeros_like(arr)
            else:
                # Step 4: Initialize output array
                arr_normalized = np.zeros_like(arr)
                
                # Step 5: Normalize brain tissue to [0,1]
                brain_intensities = np.clip(arr[brain_mask], brain_low_val, brain_high_val)
                arr_normalized[brain_mask] = (brain_intensities - brain_low_val) / (brain_high_val - brain_low_val)
        
        # Step 6: Ensure values are in [0,1] range (safety clamp)
        arr_normalized = np.clip(arr_normalized, 0.0, 1.0)
        
        # Create output image
        normalized_img = sitk.GetImageFromArray(arr_normalized)
        normalized_img.CopyInformation(image)
        
        # Validate output
        output_min, output_max = arr_normalized.min(), arr_normalized.max()
        logging.debug("normalization: bg_thresh=%.2f, brain_range=[%.2f, %.2f] → [%.3f, %.3f]", 
                     background_threshold, brain_low_val if 'brain_low_val' in locals() else 'N/A', 
                     brain_high_val if 'brain_high_val' in locals() else 'N/A', output_min, output_max)
        
        return normalized_img
        
    except Exception as e:
        logging.error("Error in fast_intensity_normalize: %s", str(e))
        return image


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
    logging.info("[%s/%s] starting fast intensity normalization", section, subj_id)
    
    # Verify all input files are MRI data
    for i, path in enumerate(paths):
        if not verify_mri_path(path):
            logging.error("[%s/%s] aborting: input file %d appears to be a mask: %s", 
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
        logging.info("[%s/%s] processing modality %s", section, subj_id, mod_name)
        
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
        if not verify_mri_path(output_path):
            logging.error("[%s/%s] error: final output appears corrupted: %s", section, subj_id, mod_name)
            return False
        
        mod_elapsed = time.time() - mod_start
        logging.info("[%s/%s] %s completed in %.1f seconds", section, subj_id, mod_name, mod_elapsed)
    
    total_elapsed = time.time() - start_time
    logging.info("[%s/%s] fast intensity normalization completed in %.1f seconds", 
                section, subj_id, total_elapsed)
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
    logging.info("parameters:")
    logging.info("  target spacing: %s", target_spacing)
    logging.info("  resampling enabled: %s", enable_resampling)
    logging.info("  background percentile: %.1f%%", background_percentile)
    logging.info("  brain percentiles: %.1f%% - %.1f%%", brain_low, brain_high)
    logging.info("  splits to process: %s", splits_to_process)
    
    # Load metadata with splits
    if not PATHS['metadata_splits'].exists():
        logging.error("metadata splits file not found: %s", PATHS['metadata_splits'])
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
    logging.info("total subjects to process: %d", total)
    
    if total == 0:
        logging.error("no subjects found in specified splits: %s", splits_to_process)
        return

    # Process subjects with configurable fast intensity normalization
    success_count = 0
    failed_subjects = []
    total_processing_time = 0
    
    for idx, (section, sid, split) in enumerate(tasks, 1):
        logging.info("=== processing %d/%d: %s/%s (%s split) ===", idx, total, section, sid, split)
        
        paths = find_modality_paths_fixed(section, sid)
        if not paths:
            logging.error("could not find all modalities for %s/%s", section, sid)
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
            logging.info("success: %s/%s (%.1fs) | avg: %.1fs/subject | eta: %.1f min", 
                        section, sid, subject_time, avg_time, eta_minutes)
        else:
            logging.error("failed: %s/%s", section, sid)
            failed_subjects.append(f"{section}/{sid}")

    # Summary
    elapsed_total = time.time() - start_all
    logging.info("=== FAST INTENSITY NORMALIZATION SUMMARY ===")
    logging.info("successfully processed: %d/%d subjects", success_count, total)
    logging.info("failed subjects: %d", len(failed_subjects))
    logging.info("total time: %.1f minutes (%.1f seconds/subject avg)", 
                elapsed_total/60, total_processing_time/max(success_count, 1))
    
    if failed_subjects:
        logging.warning("failed subjects:")
        for subj in failed_subjects:
            logging.warning("  - %s", subj)
    
    logging.info("outputs available in %s", PATHS['preprocessed_dir'])


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
        logging.info("wrote %d entries to %s", len(entries), path)


def parse_args():
    p = argparse.ArgumentParser(description="Fast intensity normalization for NeuroScope")
    p.add_argument('--target-spacing', type=float, nargs=3, default=(1.0,1.0,1.0))
    p.add_argument('--disable-resample', action='store_true', help='Skip isotropic resampling')
    p.add_argument('--background-percentile', type=float, default=5.0)
    p.add_argument('--brain-low', type=float, default=1.0)
    p.add_argument('--brain-high', type=float, default=99.0)
    p.add_argument('--splits', type=str, default='train,val,test', help='Comma list of splits to process')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    
    # Process only training subjects by default (for CycleGAN training)
    # You can modify splits_to_process to include ['train', 'val'] or ['train', 'val', 'test']
    process_subjects_from_splits(
        target_spacing=tuple(args.target_spacing),
        enable_resampling=not args.disable_resample,
        background_percentile=args.background_percentile,
        brain_low=args.brain_low,
        brain_high=args.brain_high,
        splits_to_process=[s.strip() for s in args.splits.split(',') if s.strip()]
    )
    
    # Generate convenience text files
    dump_split_txts()

if __name__ == '__main__':
    main()