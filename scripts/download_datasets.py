#!/usr/bin/env python3
"""
comprehensive dataset download script for neuroscope.

downloads and organizes multiple neuroimaging datasets:
- ixi (information extraction from images) - 600+ healthy brain mris
- oasis-3 (open access series of imaging studies) - aging and dementia
- brats (brain tumor segmentation) - glioblastoma mris
- upenn-gbm - university of pennsylvania glioblastoma dataset

usage:
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

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """handles downloading and organizing medical imaging datasets."""

    DATASET_INFO = {
        'ixi': {
            'name': 'IXI Dataset',
            'description': '600+ T1, T2, PD, MRA, DTI scans from 3 London hospitals',
            'url': 'https://brain-development.org/ixi-dataset/',
            'size': '~70GB',
            'download_method': 'manual',  # requires registration
            'modalities': ['T1', 'T2', 'PD', 'MRA', 'DTI'],
            'subjects': 600,
        },
        'oasis3': {
            'name': 'OASIS-3',
            'description': 'Longitudinal neuroimaging, clinical, and cognitive dataset',
            'url': 'https://www.oasis-brains.org/',
            'size': '~2TB (full), ~100GB (subset)',
            'download_method': 'aws',  # available via aws s3
            'modalities': ['T1', 'T2', 'FLAIR', 'ASL', 'SWI'],
            'subjects': 1098,
        },
        'brats': {
            'name': 'BraTS (TCGA-GBM)',
            'description': 'Brain Tumor Segmentation Challenge data',
            'url': 'https://www.med.upenn.edu/cbica/brats2021/',
            'size': '~10GB',
            'download_method': 'registration',  # requires synapse registration
            'modalities': ['T1', 'T1CE', 'T2', 'FLAIR'],
            'subjects': '~500',
        },
        'upenn_gbm': {
            'name': 'UPenn-GBM',
            'description': 'University of Pennsylvania Glioblastoma dataset',
            'url': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642',
            'size': '~50GB',
            'download_method': 'nbia',  # nbia data retriever
            'modalities': ['T1', 'T1CE', 'T2', 'FLAIR'],
            'subjects': 630,
        }
    }

    def __init__(self, output_dir: Path):
        """initialize downloader.

        args:
            output_dir: root directory for downloaded datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_dataset_info(self, dataset: str):
        """print information about a dataset."""
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
        """download ixi dataset.

        note: ixi requires manual download from the website.
        this function provides instructions.
        """
        logger.info("Downloading IXI Dataset...")
        self.print_dataset_info('ixi')

        output_path = self.output_dir / 'ixi'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        ixi dataset download instructions:

        1. visit: https://brain-development.org/ixi-dataset/
        2. download the following archives:
           - ixi-t1.tar (t1-weighted images)
           - ixi-t2.tar (t2-weighted images)
           - ixi-pd.tar (proton density images)
           - ixi-mra.tar (mr angiography)
           - ixi.xls (demographics spreadsheet)

        3. extract all archives to: {output_path}

        4. after download, run:
           python scripts/preprocess_ixi.py --input-dir {output_path}

        expected structure:
        {output_path}/
        ├── ixi-t1/
        ├── ixi-t2/
        ├── ixi-pd/
        ├── ixi-mra/
        └── ixi.xls
        """

        logger.info(instructions.format(output_path=output_path))

        # create a readme with instructions
        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def download_oasis3(self, subset: bool = True):
        """download oasis-3 dataset from aws s3.

        args:
            subset: if true, download only t1-weighted images (smaller subset)
        """
        logger.info("Downloading OASIS-3 Dataset...")
        self.print_dataset_info('oasis3')

        output_path = self.output_dir / 'oasis3'
        output_path.mkdir(parents=True, exist_ok=True)

        # check if aws cli is installed
        try:
            subprocess.run(['aws', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("AWS CLI not installed. Install with: pip install awscli")
            logger.error("Then configure with: aws configure")
            return

        # oasis-3 is available as aws open data
        bucket = 's3://openneuro/ds004513'  # example - actual bucket may differ

        if subset:
            logger.info("Downloading T1-weighted subset (~100GB)...")
            # download only t1 scans
            cmd = f"aws s3 sync {bucket} {output_path} --no-sign-request --exclude '*' --include '*/anat/*T1w.nii.gz'"
        else:
            logger.info("Downloading full dataset (~2TB)...")
            cmd = f"aws s3 sync {bucket} {output_path} --no-sign-request"

        logger.info(f"Running: {cmd}")
        logger.info("This may take several hours depending on your internet connection...")

        # execute download
        try:
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"+ OASIS-3 downloaded successfully to: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            logger.info("\nAlternative: Visit https://www.oasis-brains.org/ and download manually")

    def download_brats(self):
        """download brats dataset.

        note: requires synapse registration.
        """
        logger.info("Downloading BraTS Dataset...")
        self.print_dataset_info('brats')

        output_path = self.output_dir / 'brats'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        brats dataset download instructions:

        1. create account at: https://www.synapse.org/

        2. request access to brats challenge data:
           https://www.synapse.org/#!synapse:syn51156910

        3. install synapse python client:
           pip install synapseclient

        4. download using python:
           ```python
           import synapseclient
           syn = synapseclient.synapse()
           syn.login('username', 'password')

           # download brats 2021 training data
           syn.get('syn51514105', downloadlocation='{output_path}')
           ```

        5. after download, extract and run:
           python scripts/preprocess_brats.py --input-dir {output_path}

        expected structure:
        {output_path}/
        ├── brats2021_00000/
        │   ├── brats2021_00000_t1.nii.gz
        │   ├── brats2021_00000_t1ce.nii.gz
        │   ├── brats2021_00000_t2.nii.gz
        │   ├── brats2021_00000_flair.nii.gz
        │   └── brats2021_00000_seg.nii.gz
        ├── brats2021_00001/
        └── ...
        """

        logger.info(instructions.format(output_path=output_path))

        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def download_upenn_gbm(self):
        """download upenn-gbm dataset from tcia.

        note: requires nbia data retriever.
        """
        logger.info("Downloading UPenn-GBM Dataset...")
        self.print_dataset_info('upenn_gbm')

        output_path = self.output_dir / 'upenn_gbm'
        output_path.mkdir(parents=True, exist_ok=True)

        instructions = """
        upenn-gbm dataset download instructions:

        1. visit the cancer imaging archive:
           https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageid=70225642

        2. download and install nbia data retriever:
           https://wiki.cancerimagingarchive.net/display/nbia/downloading+tcia+images

        3. download the manifest file from the website

        4. open nbia data retriever and load the manifest

        5. set download location to: {output_path}

        6. after download completes, convert dicom to nifti:
           python scripts/convert_dicom_to_nifti.py --input-dir {output_path}

        7. then run preprocessing:
           python scripts/preprocess_upenn_gbm.py --input-dir {output_path}
        """

        logger.info(instructions.format(output_path=output_path))

        readme_path = output_path / 'DOWNLOAD_INSTRUCTIONS.md'
        with open(readme_path, 'w') as f:
            f.write(instructions.format(output_path=output_path))

        logger.info(f"Instructions saved to: {readme_path}")

    def create_download_script_summary(self, datasets: List[str]):
        """create a summary document of all datasets to download."""
        summary_path = self.output_dir / 'DATASETS_SUMMARY.md'

        content = """# neuroscope datasets summary

this document summarizes all datasets used in the neuroscope project.

"""

        for dataset in datasets:
            info = self.DATASET_INFO.get(dataset.lower(), {})
            if info:
                content += f"""
## {info.get('name', dataset)}

- **description**: {info.get('description', 'n/a')}
- **url**: {info.get('url', 'n/a')}
- **size**: {info.get('size', 'n/a')}
- **modalities**: {', '.join(info.get('modalities', []))}
- **subjects**: {info.get('subjects', 'n/a')}
- **download method**: {info.get('download_method', 'n/a')}

---
"""

        content += """
## dataset usage in paper

### main results (2d sa-cyclegan):
- **training**: brats-tcga (multi-institutional glioblastoma)
- **target**: upenn-gbm (single-site glioblastoma)
- **validation**: ixi (healthy brain mri)

### generalization experiments:
- **cross-pathology**: ixi (healthy) → oasis-3 (dementia)
- **cross-dataset**: train on brats, test on ixi and oasis-3

### ablation studies:
- all conducted on brats ↔ upenn-gbm

## total storage requirements

- **minimum** (brats + upenn-gbm): ~60gb
- **recommended** (+ ixi subset): ~130gb
- **full** (+ oasis-3 full): ~2.1tb

## preprocessing pipeline

after downloading all datasets, run the unified preprocessing:

```bash
python scripts/preprocess_all_datasets.py \\
    --datasets brats upenn_gbm ixi oasis3 \\
    --output-dir ./data/processed \\
    --modalities t1 t1ce t2 flair \\
    --target-resolution 1.0 1.0 1.0 \\
    --target-size 240 240 155
```

this will create a unified dataset structure suitable for training.
"""

        with open(summary_path, 'w') as f:
            f.write(content)

        logger.info(f"\n+ Dataset summary created: {summary_path}")


def main():
    """main execution function."""
    parser = argparse.ArgumentParser(
        description='Download neuroimaging datasets for NeuroScope',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    # download ixi dataset (manual)
    python scripts/download_datasets.py --datasets ixi --output-dir ./data/raw

    # download oasis-3 subset via aws
    python scripts/download_datasets.py --datasets oasis3 --output-dir ./data/raw --oasis-subset

    # get info for all datasets
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

    # handle 'all' option
    if 'all' in args.datasets:
        args.datasets = ['ixi', 'oasis3', 'brats', 'upenn_gbm']

    # initialize downloader
    downloader = DatasetDownloader(Path(args.output_dir))

    # info only mode
    if args.info_only:
        for dataset in args.datasets:
            downloader.print_dataset_info(dataset)
        return

    # download datasets
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

    # create summary document
    downloader.create_download_script_summary(args.datasets)

    logger.info("\n" + "="*80)
    logger.info("Dataset download setup complete!")
    logger.info(f"Follow the instructions in {args.output_dir}/*/DOWNLOAD_INSTRUCTIONS.md")
    logger.info(f"See {args.output_dir}/DATASETS_SUMMARY.md for overview")
    logger.info("="*80)


if __name__ == '__main__':
    main()
