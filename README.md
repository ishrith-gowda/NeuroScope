# NeuroScope
Domain-aware standardization of multimodal glioma MRI; CycleGAN‑based framework for standardizing multi‑institutional glioblastoma MRI scans (T1, T1ce, T2, FLAIR) across different scanner protocols.

## Table of Contents
1. Overview
2. Features
3. Directory structure
4. Requirements
5. Installation
6. Data preparation
7. Preprocessing pipeline
8. Training
9. Evaluation & visualization
10. Usage examples
11. Contributing
12. License

## Overview
NeuroScope tackles scanner‑protocol heterogeneity in glioblastoma MRI by learning an unsupervised image‑to‑image translation between BraTS (TCGA‑GBM) and UPenn‑GBM datasets. The CycleGAN operates on four‑channel 2D axial slices (T1, T1ce, T2, FLAIR) to produce harmonized volumes for downstream radiomic analysis.

## Features
- Domain splitting into BraTS (domain A) and UPenn (domain B)
- Preprocessing: skull‑stripping, percentile intensity normalization, 1 mm isotropic resampling
- CycleGAN with ResNet‑based generators (nine residual blocks) and 70×70 PatchGAN discriminators
- Losses: adversarial (MSE), cycle‑consistency (L1), identity (L1)
- TensorBoard logging: loss curves, weight histograms, sample grids
- Visualization: sample translation montages; SSIM/PSNR histograms; model summary and parameter counts

## Requirements
- Python 3.9 or later
- PyTorch 1.11 or later (CUDA or MPS support)
- torchvision, SimpleITK, matplotlib, numpy, pandas
- TensorBoard
- ANTs or HD‑BET (for optional skull‑stripping)
- NBIA Data Retriever or Aspera Connect (for dataset download)

## Installation
1. Clone the repo: git clone [https://github.com/ishrith-gowda/NeuroScope.git](https://github.com/ishrith-gowda/NeuroScope.git) and cd NeuroScope
2. Create and activate a virtual environment: python -m venv venv; source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. (Optional) Install HD‑BET: pip install hd-bet
5. Place the MNI152 template at \~/Templates/MNI152\_T1\_1mm.nii.gz

## Data preparation
1. Download BraTS (TCGA‑GBM) and UPenn‑GBM data via TCIA.
2. Organize under data/: data/BraTS-TCGA-GBM and data/UPENN-GBM.

## Preprocessing pipeline
Run:
- python 01\_data\_preparation\_pipeline/enrich\_metadata.py
- python 01\_data\_preparation\_pipeline/preprocess\_intensity.py
- python 01\_data\_preparation\_pipeline/verify\_preprocessing.py
These build a unified metadata JSON, normalize intensities, resample to 1 mm, and verify spacing/intensity/mask overlap.

## Training
Run:
python 02\_model\_development\_pipeline/train\_cyclegan.py<br/>
\--data\_root /path/to/data/preprocessed<br/>
\--meta\_json /path/to/scripts/neuroscope\_dataset\_metadata\_splits.json<br/>
\--n\_epochs 100<br/>
\--batch\_size 8<br/>
\--lr 2e-4<br/>
\--decay\_epoch 50<br/>
\--checkpoint\_interval 10<br/>
\--sample\_interval 200<br/>
\--log\_interval 50<br/>
Models are saved in checkpoints/, sample images in samples/, and TensorBoard logs in \~/Downloads/neuroscope/runs. Launch TensorBoard with tensorboard --logdir \~/Downloads/neuroscope/runs.

## Evaluation & visualization
After training, run:
python 02\_model\_development\_pipeline/visualize\_cyclegan\_training.py
Generates loss curves, SSIM/PSNR histograms, translation grids, and model summaries in figures/.

## Usage examples
Inference on new scans:
- Load a full model: model = torch.load('checkpoints/full\_G\_A2B\_100.pt')\['architecture']
- model.eval(); output = model(input\_tensor) for each slice

Contributing
1. Fork the repo and create a feature branch
2. Commit and push changes
3. Open a pull request with tests and PEP8 compliance
