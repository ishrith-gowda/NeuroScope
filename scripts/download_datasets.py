#!/usr/bin/env python3
"""
Comprehensive Dataset Download Script for NeuroScope.

Downloads and organizes multiple neuroimaging datasets:
- IXI (Information eXtraction from Images) - 600+ healthy brain MRIs
- OASIS-3 (Open Access Series of Imaging Studies) - Aging and dementia
- BraTS (Brain Tumor Segmentation) - Glioblastoma MRIs
- UPenn-GBM - University of Pennsylvania Glioblastoma dataset

Usage:
    python scripts/download_datasets.py --datasets ixi oasis brats --output-dir ./data/raw
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Handles downloading and organizing medical imaging datasets."""

    DATASET_INFO = {
        'ixi': {
            'name': 'IXI Dataset',
            'description': '600+ T1, T2, PD, MRA, DTI scans from 3 London hospitals',
            'url': 'https://brain-development.org/ixi-dataset/',
            'size': '~70GB',
            'download_method': 'manual',  # Requires registration
            'modalities': ['T1', 'T2', 'PD', 'MRA', 'DTI'],
            'subjects': 600,
        },
        'oasis3': {
            'name': 'OASIS-3',
            'description': 'Longitudinal neuroimaging, clinical, and cognitive dataset',
            'url': 'https://www.oasis-brains.org/',
            'size': '~2TB (full), ~100GB (subset)',
            'download_method': 'aws',  # Available via AWS S3
            'modalities': ['T1', 'T2', 'FLAIR', 'ASL', 'SWI'],
            'subjects': 1098,
        },
        'brats': {
            'name': 'BraTS (TCGA-GBM)',
            'description': 'Brain Tumor Segmentation Challenge data',
            'url': 'https://www.med.upenn.edu/cbica/brats2021/',
            'size': '~10GB',
            'download_method': 'registration',  # Requires Synapse registration
            'modalities': ['T1', 'T1CE', 'T2', 'FLAIR'],
            'subjects': '~500',
        },
        'upenn_gbm': {
            'name': 'UPenn-GBM',
            'description': 'University of Pennsylvania Glioblastoma dataset',
            'url': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642',
            'size': '~50GB',
            'download_method': 'nbia',  # NBIA Data Retriever
            'modalities': ['T1', 'T1CE', 'T2', 'FLAIR'],
            'subjects': 630,
        }
    }

    def __init__(self, output_dir: Path):
        """Initialize downloader.

        Args:
            output_dir: Root directory for downloaded datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_dataset_info(self, dataset: str):
        """Print information about a dataset."""
        info = self.DATASET_INFO.get(dataset.lower())
        if not info:
            logger.error(f"Unknown dataset: {dataset}")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"Dataset: {info['name']}")
        logger.info(f"Description: {info['description']}")
        logger.info(f"URL: {info['url']}")
        logger.info(f"Size: {info['size']}")
        logger.info(f"Modalities: {', '.join(info['modalities'])}")
        logger.info(f"Subjects: {info['subjects']}")
        logger.info(f"Download Method: {info['download_method']}")
        logger.info(f"{'='*80}\n")

    def download_ixi(self):
        """Download IXI dataset.

        Note: IXI requires manual download from the website.
        This function provides instructions.
        """
        logger.info("Downloading IXI Dataset...")
        self.print_dataset_info('ixi')

        output_path = self.output_dir / 'ixi'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        IXI Dataset Download Instructions:

        1. Visit: https://brain-development.org/ixi-dataset/
        2. Download the following archives:
           - IXI-T1.tar (T1-weighted images)
           - IXI-T2.tar (T2-weighted images)
           - IXI-PD.tar (Proton Density images)
           - IXI-MRA.tar (MR Angiography)
           - IXI.xls (Demographics spreadsheet)

        3. Extract all archives to: {output_path}

        4. After download, run:
           python scripts/preprocess_ixi.py --input-dir {output_path}

        Expected structure:
        {output_path}/
        ├── IXI-T1/
        ├── IXI-T2/
        ├── IXI-PD/
        ├── IXI-MRA/
        └── IXI.xls
        """

        logger.info(instructions.format(output_path=output_path))

        # Create a README with instructions
        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def download_oasis3(self, subset: bool = True):
        """Download OASIS-3 dataset from AWS S3.

        Args:
            subset: If True, download only T1-weighted images (smaller subset)
        """
        logger.info("Downloading OASIS-3 Dataset...")
        self.print_dataset_info('oasis3')

        output_path = self.output_dir / 'oasis3'
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if AWS CLI is installed
        try:
            subprocess.run(['aws', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("AWS CLI not installed. Install with: pip install awscli")
            logger.error("Then configure with: aws configure")
            return

        # OASIS-3 is available as AWS Open Data
        bucket = 's3://openneuro/ds004513'  # Example - actual bucket may differ

        if subset:
            logger.info("Downloading T1-weighted subset (~100GB)...")
            # Download only T1 scans
            cmd = f"aws s3 sync {bucket} {output_path} --no-sign-request --exclude '*' --include '*/anat/*T1w.nii.gz'"
        else:
            logger.info("Downloading full dataset (~2TB)...")
            cmd = f"aws s3 sync {bucket} {output_path} --no-sign-request"

        logger.info(f"Running: {cmd}")
        logger.info("This may take several hours depending on your internet connection...")

        # Execute download
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"+ OASIS-3 downloaded successfully to: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nAlternative: Visit https://www.oasis-brains.org/ and download manually")

    def download_brats(self):
        """Download BraTS dataset.

        Note: Requires Synapse registration.
        """
        logger.info("Downloading BraTS Dataset...")
        self.print_dataset_info('brats')

        output_path = self.output_dir / 'brats'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        BraTS Dataset Download Instructions:

        1. Create account at: https://www.synapse.org/

        2. Request access to BraTS challenge data:
           https://www.synapse.org/#!Synapse:syn51156910

        3. Install Synapse Python client:
           pip install synapseclient

        4. Download using Python:
           ```python
           import synapseclient
           syn = synapseclient.Synapse()
           syn.login('username', 'password')

           # Download BraTS 2021 Training Data
           syn.get('syn51514105', downloadLocation='{output_path}')
           ```

        5. After download, extract and run:
           python scripts/preprocess_brats.py --input-dir {output_path}

        Expected structure:
        {output_path}/
        ├── BraTS2021_00000/
        │   ├── BraTS2021_00000_t1.nii.gz
        │   ├── BraTS2021_00000_t1ce.nii.gz
        │   ├── BraTS2021_00000_t2.nii.gz
        │   ├── BraTS2021_00000_flair.nii.gz
        │   └── BraTS2021_00000_seg.nii.gz
        ├── BraTS2021_00001/
        └── ...
        """

        logger.info(instructions.format(output_path=output_path))

        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def download_upenn_gbm(self):
        """Download UPenn-GBM dataset from TCIA.

        Note: Requires NBIA Data Retriever.
        """
        logger.info("Downloading UPenn-GBM Dataset...")
        self.print_dataset_info('upenn_gbm')

        output_path = self.output_dir / 'upenn_gbm'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        UPenn-GBM Dataset Download Instructions:

        1. Visit The Cancer Imaging Archive:
           https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642

        2. Download and install NBIA Data Retriever:
           https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

        3. Download the manifest file from the website

        4. Open NBIA Data Retriever and load the manifest

        5. Set download location to: {output_path}

        6. After download completes, convert DICOM to NIfTI:
           python scripts/convert_dicom_to_nifti.py --input-dir {output_path}

        7. Then run preprocessing:
           python scripts/preprocess_upenn_gbm.py --input-dir {output_path}
        """

        logger.info(instructions.format(output_path=output_path))

        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def create_download_script_summary(self, datasets: List[str]):
        """Create a summary document of all datasets to download."""
        summary_path = self.output_dir / 'DATASETS_SUMMARY.md'

        content = """# NeuroScope Datasets Summary

This document summarizes all datasets used in the NeuroScope project.

"""

        for dataset in datasets:
            info = self.DATASET_INFO.get(dataset.lower(), {})
            if info:
                content += f"""
## {info.get('name', dataset)}

- **Description**: {info.get('description', 'N/A')}
- **URL**: {info.get('url', 'N/A')}
- **Size**: {info.get('size', 'N/A')}
- **Modalities**: {', '.join(info.get('modalities', []))}
- **Subjects**: {info.get('subjects', 'N/A')}
- **Download Method**: {info.get('download_method', 'N/A')}

---
"""

        content += """
## Dataset Usage in Paper

### Main Results (2D SA-CycleGAN):
- **Training**: BraTS-TCGA (multi-institutional glioblastoma)
- **Target**: UPenn-GBM (single-site glioblastoma)
- **Validation**: IXI (healthy brain MRI)

### Generalization Experiments:
- **Cross-pathology**: IXI (healthy) → OASIS-3 (dementia)
- **Cross-dataset**: Train on BraTS, test on IXI and OASIS-3

### Ablation Studies:
- All conducted on BraTS ↔ UPenn-GBM

## Total Storage Requirements

- **Minimum** (BraTS + UPenn-GBM): ~60GB
- **Recommended** (+ IXI subset): ~130GB
- **Full** (+ OASIS-3 full): ~2.1TB

## Preprocessing Pipeline

After downloading all datasets, run the unified preprocessing:

```bash
python scripts/preprocess_all_datasets.py \\
    --datasets brats upenn_gbm ixi oasis3 \\
    --output-dir ./data/processed \\
    --modalities T1 T1CE T2 FLAIR \\
    --target-resolution 1.0 1.0 1.0 \\
    --target-size 240 240 155
```

This will create a unified dataset structure suitable for training.
"""

        with open(summary_path, 'w') as f:
            f.write(content)

        logger.info(f"\n+ Dataset summary created: {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Download neuroimaging datasets for NeuroScope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download IXI dataset (manual)
    python scripts/download_datasets.py --datasets ixi --output-dir ./data/raw

    # Download OASIS-3 subset via AWS
    python scripts/download_datasets.py --datasets oasis3 --output-dir ./data/raw --oasis-subset

    # Get info for all datasets
    python scripts/download_datasets.py --info-only
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['ixi', 'oasis3', 'brats', 'upenn_gbm', 'all'],
        default=['all'],
        help='Datasets to download'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/raw',
        help='Output directory for datasets'
    )

    parser.add_argument(
        '--oasis-subset',
        action='store_true',
        help='Download only OASIS-3 T1 subset (~100GB vs ~2TB)'
    )

    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Only print dataset information without downloading'
    )

    args = parser.parse_args()

    # Handle 'all' option
    if 'all' in args.datasets:
        args.datasets = ['ixi', 'oasis3', 'brats', 'upenn_gbm']

    # Initialize downloader
    downloader = DatasetDownloader(Path(args.output_dir))

    # Info only mode
    if args.info_only:
        for dataset in args.datasets:
            downloader.print_dataset_info(dataset)
        return

    # Download datasets
    logger.info(f"Starting dataset downloads to: {args.output_dir}")
    logger.info(f"Datasets selected: {', '.join(args.datasets)}\n")

    for dataset in args.datasets:
        if dataset == 'ixi':
            downloader.download_ixi()
        elif dataset == 'oasis3':
            downloader.download_oasis3(subset=args.oasis_subset)
        elif dataset == 'brats':
            downloader.download_brats()
        elif dataset == 'upenn_gbm':
            downloader.download_upenn_gbm()

    # Create summary document
    downloader.create_download_script_summary(args.datasets)

    logger.info("\n" + "="*80)
    logger.info("Dataset download setup complete!")
    logger.info(f"Follow the instructions in {args.output_dir}/*/DOWNLOAD_INSTRUCTIONS.md")
    logger.info(f"See {args.output_dir}/DATASETS_SUMMARY.md for overview")
    logger.info("="*80)


if __name__ == '__main__':
    main()
