"""
NeuroScope Path Configuration Module

Centralized configuration for all data paths to resolve inconsistencies between
USB drive (raw data) and local computer (working directory).

Usage:
    from neuroscope_config import PATHS
    metadata_path = PATHS['metadata_splits']
    raw_brats_dir = PATHS['raw_brats_root']
"""

import os
from pathlib import Path

def get_neuroscope_paths():
    """
    Generate standardized paths for NeuroScope project.
    
    Returns:
        dict: Dictionary containing all standardized paths
    """
    # === USB DRIVE PATHS (Raw Data - Read Only) ===
    # Update this if your USB drive has a different name/path
    USB_DRIVE_ROOT = Path("/Volumes/USB Drive/neuroscope")  # macOS
    # USB_DRIVE_ROOT = Path("/media/usb/neuroscope")        # Linux alternative
    # USB_DRIVE_ROOT = Path("D:/neuroscope")                # Windows alternative
    
    # === LOCAL COMPUTER PATHS (Working Directory) ===
    LOCAL_ROOT = Path.home() / "Downloads" / "neuroscope"
    
    paths = {
        # === RAW DATA PATHS (USB Drive) ===
        'usb_root': USB_DRIVE_ROOT,
        'raw_data_root': USB_DRIVE_ROOT / "data",
        'raw_brats_root': "BraTS-TCGA-GBM/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations",
        'raw_upenn_root': "PKG - UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files/images_structural",
        'upenn_acquisition_csv': USB_DRIVE_ROOT / "data" / "PKG - UPENN-GBM-NIfTI" / "UPENN-GBM_acquisition.csv",
        
        # === WORKING DIRECTORY PATHS (Local Computer) ===
        'local_root': LOCAL_ROOT,
        'scripts_dir': LOCAL_ROOT / "scripts",
        'preprocessed_dir': LOCAL_ROOT / "preprocessed",
        'preprocessed_registered_dir': LOCAL_ROOT / "preprocessed_registered",
        'checkpoints_dir': LOCAL_ROOT / "checkpoints",
        'samples_dir': LOCAL_ROOT / "samples",
        'figures_dir': LOCAL_ROOT / "figures",
        'logs_dir': LOCAL_ROOT / "runs",
        'templates_dir': LOCAL_ROOT / "scripts" / "templates",
        
        # === METADATA FILES ===
        'metadata_base': LOCAL_ROOT / "scripts" / "neuroscope_dataset_metadata.json",
        'metadata_enriched': LOCAL_ROOT / "scripts" / "neuroscope_dataset_metadata_enriched.json",
        'metadata_splits': LOCAL_ROOT / "scripts" / "neuroscope_dataset_metadata_splits.json",
        'bias_assessment': LOCAL_ROOT / "scripts" / "neuroscope_bias_assessment.json",
        'slice_bias_assessment': LOCAL_ROOT / "scripts" / "neuroscope_slice_bias_assessment.json",
        'preprocessing_report': LOCAL_ROOT / "scripts" / "preprocessing_verification_report_v2.json",
        
        # === SPLIT FILES ===
        'train_subjects_txt': LOCAL_ROOT / "scripts" / "train_subjects.txt",
        'val_subjects_txt': LOCAL_ROOT / "scripts" / "val_subjects.txt",
        'test_subjects_txt': LOCAL_ROOT / "scripts" / "test_subjects.txt",
        
        # === TEMPLATES ===
        'mni_template': LOCAL_ROOT / "scripts" / "templates" / "MNI152_T1_1mm.nii.gz",
    }
    
    return paths

def ensure_directories(paths_dict):
    """
    Create necessary local directories if they don't exist.
    Does NOT create USB drive directories (read-only assumption).
    
    Parameters:
        paths_dict (dict): Dictionary of paths from get_neuroscope_paths()
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
    Validate that USB drive paths are accessible.
    
    Parameters:
        paths_dict (dict): Dictionary of paths from get_neuroscope_paths()
        
    Returns:
        bool: True if USB drive is accessible, False otherwise
    """
    usb_root = paths_dict['usb_root']
    if not usb_root.exists():
        print(f"WARNING: USB drive not found at {usb_root}")
        print("Please check that:")
        print("1. USB drive is connected")
        print("2. Path in neuroscope_config.py matches your USB drive mount point")
        return False
    
    raw_data = paths_dict['raw_data_root']
    if not raw_data.exists():
        print(f"WARNING: Raw data directory not found at {raw_data}")
        return False
        
    return True

# === GLOBAL PATHS OBJECT ===
PATHS = get_neuroscope_paths()

# Auto-create local directories
ensure_directories(PATHS)

# Validate USB access (optional - will print warnings if issues found)
validate_usb_access(PATHS)

# === CONVENIENCE FUNCTIONS ===
def get_raw_subject_dir(section: str, subject_id: str) -> Path:
    """
    Get the raw data directory for a specific subject.
    
    Parameters:
        section (str): 'brats' or 'upenn'
        subject_id (str): Subject identifier
        
    Returns:
        Path: Full path to subject directory on USB drive
    """
    if section == 'brats':
        return PATHS['raw_data_root'] / PATHS['raw_brats_root'] / subject_id
    elif section == 'upenn':
        return PATHS['raw_data_root'] / PATHS['raw_upenn_root'] / subject_id
    else:
        raise ValueError(f"Unknown section: {section}")

def get_preprocessed_subject_dir(section: str, subject_id: str) -> Path:
    """
    Get the preprocessed data directory for a specific subject.
    
    Parameters:
        section (str): 'brats' or 'upenn'  
        subject_id (str): Subject identifier
        
    Returns:
        Path: Full path to preprocessed subject directory on local computer
    """
    return PATHS['preprocessed_dir'] / section / subject_id

if __name__ == "__main__":
    # Test the configuration
    print("NeuroScope Path Configuration Test")
    print("=" * 50)
    
    print("\n📁 USB Drive Paths (Raw Data):")
    print(f"  USB Root: {PATHS['usb_root']}")
    print(f"  Raw Data: {PATHS['raw_data_root']}")
    print(f"  USB Accessible: {validate_usb_access(PATHS)}")
    
    print("\n💻 Local Paths (Working Directory):")
    print(f"  Local Root: {PATHS['local_root']}")
    print(f"  Scripts: {PATHS['scripts_dir']}")
    print(f"  Preprocessed: {PATHS['preprocessed_dir']}")
    
    print("\n📄 Key Files:")
    print(f"  Metadata Splits: {PATHS['metadata_splits']}")
    print(f"  MNI Template: {PATHS['mni_template']}")
    
    print("\n🧪 Test Subject Paths:")
    test_subject = "TCGA-02-0001"  # Example
    print(f"  Raw BraTS Subject: {get_raw_subject_dir('brats', test_subject)}")
    print(f"  Preprocessed Subject: {get_preprocessed_subject_dir('brats', test_subject)}")