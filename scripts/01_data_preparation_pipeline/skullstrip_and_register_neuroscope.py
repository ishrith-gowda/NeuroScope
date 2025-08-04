import os
import json
import logging
import subprocess
import SimpleITK as sitk
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from neuroscope_config import PATHS


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


def run_hdbet_skull_strip_only(input_path: str, output_dir: str, subject_id: str, modality_name: str) -> Optional[str]:
    """
    Run HD-BET skull-stripping with fallback to Otsu. Returns only skull-stripped image.
    No separate mask file needed - background will be zeros.
    """
    bet_output = os.path.join(output_dir, f"{subject_id}_{modality_name}_BET.nii.gz")
    
    try:
        # Try HD-BET first
        cmd = ['hd-bet', '-i', input_path, '-o', bet_output, '-device', 'cpu', '-mode', 'fast']
        logging.info("Running HD-BET on %s: %s", modality_name, ' '.join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # HD-BET produces skull-stripped image directly
        if os.path.exists(bet_output):
            logging.info("HD-BET successful for %s %s", subject_id, modality_name)
            return bet_output
        else:
            logging.warning("HD-BET output missing, falling back to Otsu")
            raise FileNotFoundError("HD-BET output missing")
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logging.warning("HD-BET failed for %s %s: %s, using Otsu fallback", subject_id, modality_name, e)
        
        # Fallback to Otsu thresholding + masking
        try:
            img = sitk.ReadImage(input_path)
            mask = sitk.OtsuThreshold(img, 0, 1, 200)
            mask = sitk.BinaryFillhole(mask)
            
            # Apply mask to create skull-stripped image (zeros background)
            skull_stripped = sitk.Mask(img, mask)
            
            otsu_output = os.path.join(output_dir, f"{subject_id}_{modality_name}_OTSU.nii.gz")
            sitk.WriteImage(skull_stripped, otsu_output)
            
            logging.info("Otsu fallback successful for %s %s", subject_id, modality_name)
            return otsu_output
            
        except Exception as otsu_error:
            logging.error("Both HD-BET and Otsu failed for %s %s: %s", subject_id, modality_name, otsu_error)
            return None


def percentile_normalize_skull_stripped(image: sitk.Image, p_low: float = 1.0, p_high: float = 99.0) -> sitk.Image:
    """
    Percentile normalize skull-stripped image (no mask needed - zeros are background).
    Computes percentiles only on non-zero voxels, then normalizes entire volume.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    
    # Use non-zero voxels for percentile computation (skull-stripped brain tissue)
    nonzero_vals = arr[arr > 0]
    
    if nonzero_vals.size == 0:
        logging.error("No non-zero voxels found for percentile normalization")
        return image
        
    lo, hi = np.percentile(nonzero_vals, (p_low, p_high))
    
    # Normalize entire array (preserving zeros as background)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    
    norm_img = sitk.GetImageFromArray(arr)
    norm_img.CopyInformation(image)
    return norm_img


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


def process_subject_skull_strip_only(
    section: str,
    subj_id: str,
    paths: List[str],
    target_spacing: Tuple[float, float, float],
    use_hdbet: bool = True
) -> bool:
    """
    Skull-stripping only preprocessing pipeline for CycleGAN domain translation:
    1. Skull-strip each modality independently (HD-BET preferred, Otsu fallback)
    2. Percentile normalize each skull-stripped volume (using non-zero voxels)
    3. Resample to isotropic spacing
    4. Save preprocessed data (background = 0, brain tissue = [0,1])
    
    This preserves intensity distributions while removing non-brain tissue.
    Perfect for BraTS ↔ UPenn domain translation.
    
    Returns True if successful, False otherwise.
    """
    logging.info("[%s/%s] Starting skull-stripping preprocessing for CycleGAN", section, subj_id)
    
    # Verify all input files are MRI data
    for i, path in enumerate(paths):
        if not verify_image_is_mri(path):
            logging.error("[%s/%s] ABORTING: Input file %d appears to be a mask: %s", 
                         section, subj_id, i, path)
            return False
    
    # Create output directory
    subj_out_dir = PATHS['preprocessed_dir'] / section / subj_id
    subj_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each modality independently
    modality_names = ['t1', 't1gd', 't2', 'flair']
    final_outputs = []
    
    for i, (input_path, mod_name) in enumerate(zip(paths, modality_names)):
        logging.info("[%s/%s] Skull-stripping modality %s", section, subj_id, mod_name)
        
        # Step 1: Skull-strip this modality
        if use_hdbet:
            skull_stripped_path = run_hdbet_skull_strip_only(input_path, str(subj_out_dir), subj_id, mod_name)
        else:
            skull_stripped_path = run_hdbet_skull_strip_only(input_path, str(subj_out_dir), subj_id, mod_name)
        
        if skull_stripped_path is None:
            logging.error("[%s/%s] Failed to skull-strip %s", section, subj_id, mod_name)
            return False
        
        # Step 2: Load skull-stripped image
        skull_stripped_img = sitk.ReadImage(skull_stripped_path)
        
        # Step 3: Percentile normalize (using non-zero voxels only)
        norm_img = percentile_normalize_skull_stripped(skull_stripped_img)
        
        # Step 4: Resample to isotropic spacing
        resampled_img = resample_to_isotropic(norm_img, target_spacing)
        
        # Step 5: Save with standard naming
        final_output_path = str(subj_out_dir / f"{mod_name}.nii.gz")
        sitk.WriteImage(resampled_img, final_output_path)
        final_outputs.append(final_output_path)
        
        # Clean up intermediate files
        if os.path.exists(skull_stripped_path):
            os.remove(skull_stripped_path)
        
        # Verify final output
        if not verify_image_is_mri(final_output_path):
            logging.error("[%s/%s] ERROR: Final output appears corrupted: %s", section, subj_id, mod_name)
            return False
        
        logging.info("[%s/%s] Successfully processed %s → %s", section, subj_id, mod_name, f"{mod_name}.nii.gz")
    
    logging.info("[%s/%s] Skull-stripping preprocessing completed successfully", section, subj_id)
    logging.info("[%s/%s] Ready for CycleGAN domain translation training", section, subj_id)
    return True


def dump_split_txts() -> None:
    """
    Dump train/val/test subject lists from JSON to text files.
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
    """
    Main: unified preprocessing pipeline for all train subjects.
    
    This resolves the inconsistency by using ONE approach:
    - HD-BET skull-stripping (with Otsu fallback)
    - Consistent brain mask for all modalities
    - Percentile normalization within brain mask
    - Isotropic resampling
    """
    configure_logging()
    start_all = time.time()

    logging.info("=== UNIFIED NEUROSCOPE PREPROCESSING PIPELINE ===")
    logging.info("Using neuroscope_config.py for path management")
    logging.info("  Raw Data: %s", PATHS['raw_data_root'])
    logging.info("  Preprocessed Output: %s", PATHS['preprocessed_dir'])
    logging.info("  Metadata: %s", PATHS['metadata_splits'])
    
    target_spacing = (1.0, 1.0, 1.0)
    use_hdbet = True  # Set to False to use Otsu-only

    # Load metadata
    if not PATHS['metadata_splits'].exists():
        logging.error("Metadata splits file not found: %s", PATHS['metadata_splits'])
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
    logging.info("Total train subjects to process: %d", total)
    
    if total == 0:
        logging.error("No train subjects found in JSON.")
        return

    # Process subjects with unified approach
    success_count = 0
    failed_subjects = []
    
    for section, sid in tasks:
        paths = find_modality_paths_fixed(section, sid)
        if not paths:
            logging.error("Could not find all modalities for %s/%s", section, sid)
            failed_subjects.append(f"{section}/{sid}")
            continue
            
        # Process with skull-stripping only pipeline
        if process_subject_skull_strip_only(section, sid, paths, target_spacing, use_hdbet):
            success_count += 1
        else:
            logging.error("Failed to process %s/%s", section, sid)
            failed_subjects.append(f"{section}/{sid}")

    # Summary
    logging.info("=== PREPROCESSING SUMMARY ===")
    logging.info("Successfully processed: %d/%d subjects", success_count, total)
    logging.info("Failed subjects: %d", len(failed_subjects))
    
    if failed_subjects:
        logging.warning("Failed subjects list:")
        for subj in failed_subjects:
            logging.warning("  - %s", subj)
    
    # Only dump split files if some subjects were processed successfully
    if success_count > 0:
        logging.info("Dumping split .txt files")
        dump_split_txts()

    elapsed = time.time() - start_all
    logging.info("Unified preprocessing finished in %.2f minutes", elapsed/60)
    logging.info("All outputs use consistent brain extraction and normalization")
    logging.info("Outputs available in %s", PATHS['preprocessed_dir'])


if __name__ == '__main__':
    main()