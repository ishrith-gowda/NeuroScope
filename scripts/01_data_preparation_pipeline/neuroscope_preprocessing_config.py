"""
neuroscope path configuration module

centralized configuration for all data paths to resolve new usb-based setup:
- both project files (neuroscope) and data files (neuroscope (data)) live on the same usb drive.

usage:
    from neuroscope_config import paths
    metadata_path = paths['metadata_splits']
    raw_brats_dir = paths['raw_brats_root']
"""

import os
from pathlib import Path

def get_neuroscope_paths():
    """
    generate standardized paths for neuroscope project.
    
    returns:
        dict: dictionary containing all standardized paths
    """
    # === usb drive root (update if needed) ===
    USB_DRIVE_ROOT = Path("/Volumes/usb drive/")  # macos
    # usb_drive_root = path("/media/usb")        # linux alternative
    # usb_drive_root = path("d:/")               # windows alternative

    # project and data folders on usb
    PROJECT_ROOT = USB_DRIVE_ROOT / "neuroscope"
    DATA_ROOT = USB_DRIVE_ROOT / "neuroscope (data)"
    
    paths = {
        # === usb data paths ===
        'usb_root': USB_DRIVE_ROOT,
        'raw_data_root': DATA_ROOT,
        'raw_brats_root': DATA_ROOT / "BraTS-TCGA-GBM" / "Pre-operative_TCGA_GBM_NIfTI_and_Segmentations",
        'raw_upenn_root': DATA_ROOT / "PKG - UPENN-GBM-NIfTI" / "UPENN-GBM" / "NIfTI-files" / "images_structural",
        'upenn_acquisition_csv': DATA_ROOT / "PKG - UPENN-GBM-NIfTI" / "UPENN-GBM_acquisition.csv",

        # === project paths (on usb) ===
        'local_root': PROJECT_ROOT,
        'scripts_dir': PROJECT_ROOT / "scripts",
        'preprocessed_dir': PROJECT_ROOT / "preprocessed",
        'preprocessed_registered_dir': PROJECT_ROOT / "preprocessed_registered",
        'checkpoints_dir': PROJECT_ROOT / "checkpoints",
        'samples_dir': PROJECT_ROOT / "samples",
        'figures_dir': PROJECT_ROOT / "figures",
        'logs_dir': PROJECT_ROOT / "runs",
        'templates_dir': PROJECT_ROOT / "scripts" / "templates",
        
        # === metadata files ===
        'metadata_base': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "neuroscope_dataset_metadata.json",
        'metadata_enriched': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "neuroscope_dataset_metadata_enriched.json",
        'metadata_splits': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "neuroscope_dataset_metadata_splits.json",
        'bias_assessment': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "neuroscope_bias_assessment.json",
        'slice_bias_assessment': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "neuroscope_slice_bias_assessment.json",
        'preprocessing_report': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "preprocessing_verification_report_v2.json",
        
        # === split files ===
        'train_subjects_txt': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "train_subjects.txt",
        'val_subjects_txt': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "val_subjects.txt",
        'test_subjects_txt': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "test_subjects.txt",
        
        # === templates ===
        'mni_template': PROJECT_ROOT / "scripts" / "01_data_preparation_pipeline" / "templates" / "MNI152_T1_1mm.nii.gz",
    }
    
    return paths

def ensure_directories(paths_dict):
    """
    create necessary project directories if they don't exist.
    """
    local_dirs = [
        'local_root', 'scripts_dir', 'preprocessed_dir', 'preprocessed_registered_dir',
        'checkpoints_dir', 'samples_dir', 'figures_dir', 'logs_dir', 'templates_dir'
    ]
    
    for dir_key in local_dirs:
        if dir_key in paths_dict:
            paths_dict[dir_key].mkdir(parents=True, exist_ok=True)

def validate_usb_access(paths_dict):
    """
    validate that usb drive paths are accessible.
    
    parameters:
        paths_dict (dict): dictionary of paths from get_neuroscope_paths()
        
    returns:
        bool: true if usb drive is accessible, false otherwise
    """
    usb_root = paths_dict['usb_root']
    if not usb_root.exists():
        print(f"warning: usb drive not found at {usb_root}")
        return False
    
    raw_data = paths_dict['raw_data_root']
    if not raw_data.exists():
        print(f"warning: raw data directory not found at {raw_data}")
        return False
        
    return True

# === global paths object ===
PATHS = get_neuroscope_paths()

# auto-create project directories
ensure_directories(PATHS)

# validate usb access (optional - will print warnings if issues found)
validate_usb_access(PATHS)

# === convenience functions ===
def get_raw_subject_dir(section: str, subject_id: str) -> Path:
    """
    get the raw data directory for a specific subject.
    """
    if section == 'brats':
        return PATHS['raw_brats_root'] / subject_id
    elif section == 'upenn':
        return PATHS['raw_upenn_root'] / subject_id
    else:
        raise ValueError(f"Unknown section: {section}")

def get_preprocessed_subject_dir(section: str, subject_id: str) -> Path:
    """
    get the preprocessed data directory for a specific subject.
    """
    return PATHS['preprocessed_dir'] / section / subject_id

if __name__ == "__main__":
    # test the configuration
    print("=== neuroscope path configuration test ===")
    
    print("usb drive paths (raw data):")
    print(f"  usb root: {PATHS['usb_root']}")
    print(f"  raw data: {PATHS['raw_data_root']}")
    print(f"  usb accessible: {validate_usb_access(PATHS)}")
    
    print("project paths (working directory):")
    print(f"  local root: {PATHS['local_root']}")
    print(f"  scripts: {PATHS['scripts_dir']}")
    print(f"  preprocessed: {PATHS['preprocessed_dir']}")
    
    print("key files:")
    print(f"  metadata splits: {PATHS['metadata_splits']}")
    print(f"  mni template: {PATHS['mni_template']}")
    
    print("test subject paths:")
    test_subject = "TCGA-02-0001 (example)"
    print(f"  raw brats subject: {get_raw_subject_dir('brats', test_subject)}")
    print(f"  preprocessed subject: {get_preprocessed_subject_dir('brats', test_subject)}")