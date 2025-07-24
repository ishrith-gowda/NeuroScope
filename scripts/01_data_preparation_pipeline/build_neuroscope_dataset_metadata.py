import os
import json
import logging
from glob import glob
from typing import Dict, List, Tuple

# Configure logging
def configure_logging():
    """
    Configure the root logger to output debug information.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def scan_dataset(
    root_dir: str,
    modality_suffixes: List[str]
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[str]]]:
    """
    Scan subject subdirectories under root_dir and check for required modalities.

    Parameters:
        root_dir (str): Path to the dataset root containing subject folders.
        modality_suffixes (List[str]): List of filename suffixes for required modalities.

    Returns:
        valid (Dict[str, Dict[str, str]]): Mapping of subject ID to a dict of {suffix: filepath} for subjects with all modalities present.
        missing (Dict[str, List[str]]): Mapping of subject ID to list of missing modality suffixes.
    """
    valid: Dict[str, Dict[str, str]] = {}
    missing: Dict[str, List[str]] = {}

    if not os.path.isdir(root_dir):
        logging.error("Dataset root directory does not exist: %s", root_dir)
        return valid, missing

    subjects = [
        dirname for dirname in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, dirname))
    ]

    for subject in subjects:
        subject_dir = os.path.join(root_dir, subject)
        found: Dict[str, str] = {}
        missing_mods: List[str] = []

        for suffix in modality_suffixes:
            pattern = os.path.join(subject_dir, f"*{suffix}")
            matches = glob(pattern)
            if matches:
                found[suffix] = matches[0]
            else:
                missing_mods.append(suffix)

        if not missing_mods:
            valid[subject] = found
            logging.debug("Subject %s is valid", subject)
        else:
            missing[subject] = missing_mods
            logging.debug(
                "Subject %s missing modalities: %s", subject, missing_mods
            )

    return valid, missing


def main():
    """
    Main entry point for building the neuroscope dataset metadata.

    Scans the BraTS and UPenn structural datasets for subjects with complete modalities,
    and writes the combined metadata to JSON.
    """
    configure_logging()

    base_data_dir = os.path.expanduser("~/Downloads/neuroscope/data")
    brats_dir = os.path.join(
        base_data_dir,
        "BraTS-TCGA-GBM",
        "Pre-operative_TCGA_GBM_NIfTI_and_Segmentations"
    )
    upenn_dir = os.path.join(
        base_data_dir,
        "PKG - UPENN-GBM-NIfTI",
        "UPENN-GBM",
        "NIfTI-files",
        "images_structural"
    )

    # Define required modalities
    brats_modalities = ["_t1.nii.gz", "_t1Gd.nii.gz", "_t2.nii.gz", "_flair.nii.gz"]
    upenn_modalities = ["_T1.nii.gz", "_T1GD.nii.gz", "_T2.nii.gz", "_FLAIR.nii.gz"]

    logging.info("Scanning BraTS dataset at %s", brats_dir)
    brats_valid, brats_missing = scan_dataset(brats_dir, brats_modalities)
    logging.info(
        "BraTS: %d valid subjects, %d subjects missing modalities",
        len(brats_valid), len(brats_missing)
    )

    logging.info("Scanning UPENN-GBM dataset at %s", upenn_dir)
    upenn_valid, upenn_missing = scan_dataset(upenn_dir, upenn_modalities)
    logging.info(
        "UPENN-GBM: %d valid subjects, %d subjects missing modalities",
        len(upenn_valid), len(upenn_missing)
    )

    # Compile metadata
    metadata = {
        "brats": {
            "valid_subjects": brats_valid,
            "missing_subjects": brats_missing
        },
        "upenn": {
            "valid_subjects": upenn_valid,
            "missing_subjects": upenn_missing
        }
    }

    # Write metadata JSON
    scripts_dir = os.path.expanduser("~/Downloads/neuroscope/scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    output_path = os.path.join(scripts_dir, "neuroscope_dataset_metadata.json")

    with open(output_path, "w") as outfile:
        json.dump(metadata, outfile, indent=2)

    logging.info("Master metadata written to %s", output_path)


if __name__ == "__main__":
    main()
