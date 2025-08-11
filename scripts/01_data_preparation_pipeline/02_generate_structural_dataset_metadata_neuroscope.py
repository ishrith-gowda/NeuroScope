import os
import json
import logging
from pathlib import Path
from glob import glob
from typing import Dict, List, Tuple, Set
import SimpleITK as sitk
import numpy as np
from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """
    Configure the root logger to output debug information with timestamps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def verify_mri_file(file_path: Path) -> bool:
    """
    Verify that a file contains MRI data, not segmentation masks.
    
    Args:
        file_path: Path to the NIfTI file to verify
        
    Returns:
        bool: True if file appears to be MRI data, False if it's likely a mask
    """
    try:
        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # Check if it's binary (0s and 1s only) - likely a mask
        unique_vals = np.unique(arr)
        if len(unique_vals) <= 2 and np.allclose(unique_vals, [0, 1]):
            logging.debug("excluding binary mask file: %s", file_path.name)
            return False
        
        # Check if it has very few unique values - likely segmentation
        if len(unique_vals) < 10:
            logging.debug("axcluding low-variation file (possible mask): %s", file_path.name)
            return False
        
        # Check if it has reasonable intensity distribution for MRI
        if arr.std() < 0.01:  # Very low variation
            logging.debug("axcluding low-variation file: %s", file_path.name)
            return False
            
        return True
        
    except Exception as e:
        logging.warning("could not verify file %s: %s", file_path.name, e)
        return False


def find_modality_files(
    subject_dir: Path,
    modality_suffixes: List[str],
    exclude_patterns: Set[str] = None
) -> Tuple[Dict[str, str], List[str]]:
    """
    Find modality files for a subject, with robust exclusion of segmentation files.
    
    Args:
        subject_dir: Path to subject directory
        modality_suffixes: List of required filename suffixes
        exclude_patterns: Set of patterns to exclude from filenames
        
    Returns:
        Tuple of (found_files_dict, missing_suffixes_list)
    """
    if exclude_patterns is None:
        exclude_patterns = {'_seg', 'segmentation', 'mask', 'label'}
    
    if not subject_dir.exists():
        logging.warning("subject directory does not exist: %s", subject_dir)
        return {}, modality_suffixes.copy()
    
    # Get all NIfTI files in directory
    nifti_files = []
    for pattern in ['*.nii.gz', '*.nii']:
        nifti_files.extend(subject_dir.glob(pattern))
    
    # Filter out segmentation/mask files
    filtered_files = []
    for file_path in nifti_files:
        filename_lower = file_path.name.lower()
        if not any(pattern.lower() in filename_lower for pattern in exclude_patterns):
            if verify_mri_file(file_path):
                filtered_files.append(file_path)
            else:
                logging.debug("excluded non-mri file: %s", file_path.name)
        else:
            logging.debug("excluded segmentation file: %s", file_path.name)
    
    # Match files to required modalities
    found: Dict[str, str] = {}
    missing: List[str] = []
    
    for suffix in modality_suffixes:
        matches = [f for f in filtered_files if f.name.endswith(suffix)]
        if matches:
            # Take the first match if multiple found
            found[suffix] = str(matches[0])
            if len(matches) > 1:
                logging.warning("multiple files found for %s in %s, using: %s", 
                               suffix, subject_dir.name, matches[0].name)
        else:
            missing.append(suffix)
            logging.debug("missing modality %s for subject %s", suffix, subject_dir.name)
    
    return found, missing


def scan_dataset(
    root_dir: Path,
    modality_suffixes: List[str],
    dataset_name: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """
    Scan subject subdirectories under root_dir and check for required modalities.

    Args:
        root_dir: Path to the dataset root containing subject folders
        modality_suffixes: List of filename suffixes for required modalities
        dataset_name: Name of dataset for logging purposes

    Returns:
        Tuple of (valid_subjects_dict, missing_subjects_dict)
    """
    valid: Dict[str, Dict[str, str]] = {}
    missing: Dict[str, List[str]] = {}

    if not root_dir.exists():
        logging.error("%s dataset root directory does not exist: %s", dataset_name, root_dir)
        return valid, missing

    logging.info("scanning %s dataset at: %s", dataset_name, root_dir)
    
    # Get all subdirectories (potential subjects)
    try:
        subject_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    except PermissionError as e:
        logging.error("permission denied accessing %s: %s", root_dir, e)
        return valid, missing
    
    if not subject_dirs:
        logging.warning("no subject directories found in %s", root_dir)
        return valid, missing

    logging.info("found %d potential subject directories in %s", len(subject_dirs), dataset_name)
    
    for subject_dir in sorted(subject_dirs):
        subject_id = subject_dir.name
        logging.debug("processing subject: %s", subject_id)
        
        found_files, missing_mods = find_modality_files(subject_dir, modality_suffixes)
        
        if not missing_mods:
            valid[subject_id] = found_files
            logging.debug("subject %s is complete (%d modalities)", subject_id, len(found_files))
        else:
            missing[subject_id] = missing_mods
            logging.debug("subject %s missing: %s", subject_id, missing_mods)

    logging.info("%s scan complete: %d valid, %d incomplete subjects", 
                 dataset_name, len(valid), len(missing))
    
    return valid, missing


def validate_metadata(metadata: Dict) -> bool:
    """
    Validate the generated metadata structure.
    
    Args:
        metadata: The metadata dictionary to validate
        
    Returns:
        bool: True if metadata is valid, False otherwise
    """
    required_sections = ['brats', 'upenn']
    required_keys = ['valid_subjects', 'missing_subjects']
    
    for section in required_sections:
        if section not in metadata:
            logging.error("missing section: %s", section)
            return False
        
        for key in required_keys:
            if key not in metadata[section]:
                logging.error("missing key '%s' in section '%s'", key, section)
                return False
    
    # Check that we have some valid subjects
    total_valid = sum(len(metadata[section]['valid_subjects']) for section in required_sections)
    if total_valid == 0:
        logging.error("no valid subjects found in any dataset")
        return False
    
    logging.info("metadata validation passed")
    return True


def main() -> None:
    """
    Main entry point for building the neuroscope dataset metadata.

    Scans the BraTS and UPenn structural datasets for subjects with complete modalities,
    validates the data quality, and writes the combined metadata to JSON.
    """
    configure_logging()
    
    logging.info("=== NEUROSCOPE DATASET METADATA GENERATION ===")
    logging.info("Using neuroscope_preprocessing_config.py for path management")

    # Get dataset paths from config
    brats_dir = PATHS['raw_brats_root']
    upenn_dir = PATHS['raw_upenn_root']
    output_path = PATHS['metadata_base']
    
    logging.info("brats dataset path: %s", brats_dir)
    logging.info("upenn dataset path: %s", upenn_dir)
    logging.info("output metadata path: %s", output_path)

    # Define required modalities for each dataset
    brats_modalities = ["_t1.nii.gz", "_t1Gd.nii.gz", "_t2.nii.gz", "_flair.nii.gz"]
    upenn_modalities = ["_T1.nii.gz", "_T1GD.nii.gz", "_T2.nii.gz", "_FLAIR.nii.gz"]

    # Scan datasets
    brats_valid, brats_missing = scan_dataset(brats_dir, brats_modalities, "BraTS")
    upenn_valid, upenn_missing = scan_dataset(upenn_dir, upenn_modalities, "UPenn-GBM")

    # Compile metadata
    metadata = {
        "brats": {
            "valid_subjects": brats_valid,
            "missing_subjects": brats_missing,
            "dataset_info": {
                "total_subjects": len(brats_valid) + len(brats_missing),
                "complete_subjects": len(brats_valid),
                "incomplete_subjects": len(brats_missing),
                "required_modalities": brats_modalities,
                "source_path": str(brats_dir)
            }
        },
        "upenn": {
            "valid_subjects": upenn_valid,
            "missing_subjects": upenn_missing,
            "dataset_info": {
                "total_subjects": len(upenn_valid) + len(upenn_missing),
                "complete_subjects": len(upenn_valid),
                "incomplete_subjects": len(upenn_missing),
                "required_modalities": upenn_modalities,
                "source_path": str(upenn_dir)
            }
        },
        "generation_info": {
            "script_version": "02_generate_structural_dataset_metadata_neuroscope.py v2.0",
            "mri_verification": True,
            "segmentation_exclusion": True
        }
    }

    # Validate metadata before saving
    if not validate_metadata(metadata):
        logging.error("Metadata validation failed. Aborting.")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata JSON
    try:
        with open(output_path, "w") as outfile:
            json.dump(metadata, outfile, indent=2, sort_keys=True)
        logging.info("metadata json successfully written to: %s", output_path)
    except Exception as e:
        logging.error("failed to write metadata json: %s", e)
        return

    # Print summary
    print("\n" + "="*60)
    print("NEUROSCOPE DATASET METADATA GENERATION SUMMARY")
    print("="*60)
    
    for dataset_name, section_key in [("BraTS-TCGA-GBM", "brats"), ("UPenn-GBM", "upenn")]:
        info = metadata[section_key]["dataset_info"]
        print(f"\n{dataset_name}:")
        print(f"  total subjects found:     {info['total_subjects']}")
        print(f"  complete subjects:        {info['complete_subjects']}")
        print(f"  incomplete subjects:      {info['incomplete_subjects']}")
        print(f"  completion rate:          {info['complete_subjects']/max(info['total_subjects'],1)*100:.1f}%")
    
    total_complete = metadata['brats']['dataset_info']['complete_subjects'] + \
                    metadata['upenn']['dataset_info']['complete_subjects']
    
    print(f"\nOVERALL:")
    print(f"  total complete subjects:  {total_complete}")
    print("="*60)

if __name__ == "__main__":
    main()