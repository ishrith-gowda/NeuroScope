import os
import json
import logging
import subprocess
import SimpleITK as sitk
from typing import List, Tuple
from neuroscope_config import PATHS

def configure_logging():
    """
    Configure logging format and level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_train_subjects(json_path: str) -> List[Tuple[str, str]]:
    """
    Load train subjects (section, subject_id) from metadata JSON.
    """
    if not os.path.isfile(json_path):
        logging.error("Metadata JSON not found: %s", json_path)
        return []
    with open(json_path, 'r') as f:
        meta = json.load(f)
    subjects = []
    for section in ('brats', 'upenn'):
        for sid, info in meta.get(section, {}).get('valid_subjects', {}).items():
            if info.get('split') == 'train':
                subjects.append((section, sid))
    logging.info("Loaded %d train subjects from %s", len(subjects), json_path)
    return subjects

def run_hdbet(input_path: str, output_path: str, device: str = 'cpu') -> str:
    """
    Run HD-BET skull-stripping CLI. Returns mask path.

    Requires 'hd-bet' installed and in PATH.
    """
    cmd = [
        'hd-bet',
        '-i', input_path,
        '-o', output_path,
        '-device', device,
        '-mode', 'fast'
    ]
    logging.info("Running HD-BET: %s", ' '.join(cmd))
    subprocess.run(cmd, check=True)
    # HD-BET outputs *_BET.nii.gz and *_BET_mask.nii.gz
    mask_path = output_path.replace('.nii', '_mask.nii')
    if not os.path.isfile(mask_path):
        # Fallback to gz
        mask_path = mask_path + '.gz'
    return mask_path

def register_to_mni(
    moving_path: str,
    fixed_path: str,
    out_moving: str,
    transform_out: str
) -> sitk.Transform:
    """
    Register moving image to fixed MNI template and write resampled moving image.
    Saves transform to transform_out (optional).
    """
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    # Initial alignment
    init_tx = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    # Registration
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(init_tx, inPlace=False)
    logging.info("Starting registration of %s to %s", moving_path, fixed_path)
    tx = reg.Execute(fixed, moving)
    logging.info("Registration completed: final metric %f", reg.GetMetricValue())
    # Resample moving
    resampled = sitk.Resample(
        moving, fixed, tx,
        sitk.sitkLinear, 0.0,
        moving.GetPixelID()
    )
    sitk.WriteImage(resampled, out_moving)
    # Optionally save transform parameters
    sitk.WriteTransform(tx, transform_out)
    return tx

def apply_transform(
    input_path: str,
    fixed: sitk.Image,
    tx: sitk.Transform,
    out_path: str
) -> None:
    """
    Apply existing transform to an image and save.
    """
    img = sitk.ReadImage(input_path, sitk.sitkFloat32)
    res = sitk.Resample(img, fixed, tx, sitk.sitkLinear, 0.0, img.GetPixelID())
    sitk.WriteImage(res, out_path)

def main():
    configure_logging()
    
    # Use standardized paths
    preprocessed_dir = PATHS['preprocessed_dir']
    metadata_path = PATHS['metadata_splits']
    mni_template = PATHS['mni_template']
    output_root = PATHS['preprocessed_registered_dir']
    device = 'cpu'  # or 'cuda'
    
    logging.info("Using paths:")
    logging.info("  Preprocessed data: %s", preprocessed_dir)
    logging.info("  Metadata: %s", metadata_path)
    logging.info("  MNI template: %s", mni_template)
    logging.info("  Output directory: %s", output_root)
    
    # Check if MNI template exists
    if not mni_template.exists():
        logging.error("MNI template not found at %s", mni_template)
        logging.error("Please place MNI152_T1_1mm.nii.gz in the templates directory")
        return
    
    subjects = load_train_subjects(str(metadata_path))
    
    for section, sid in subjects:
        # Prepare dirs
        pre_dir = preprocessed_dir / section / sid
        out_dir = output_root / section / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not pre_dir.exists():
            logging.warning("Preprocessed directory not found for %s/%s", section, sid)
            continue
        
        # Identify T1
        files = os.listdir(str(pre_dir))
        t1_files = [f for f in files if 't1.nii' in f.lower() and 'gd' not in f.lower()]
        
        if not t1_files:
            logging.warning("No T1 file found for %s/%s", section, sid)
            continue
            
        t1 = t1_files[0]
        t1_path = str(pre_dir / t1)
        
        # HD-BET
        bet_output = str(out_dir / f"{sid}_BET.nii.gz")
        try:
            mask_path = run_hdbet(t1_path, bet_output, device)
        except subprocess.CalledProcessError as e:
            logging.error("HD-BET failed for %s/%s: %s", section, sid, e)
            continue
        
        # Registration
        reg_out = str(out_dir / f"{sid}_registered.nii.gz")
        tx_out = str(out_dir / f"{sid}_to_MNI.tfm")
        
        try:
            tx = register_to_mni(bet_output, str(mni_template), reg_out, tx_out)
            fixed_img = sitk.ReadImage(str(mni_template), sitk.sitkFloat32)
            
            # Apply to other modalities
            for f in files:
                if f == t1 or not f.endswith('.nii'):
                    continue
                inp = str(pre_dir / f)
                outf = str(out_dir / f.replace('.nii.gz', '_MNI.nii.gz'))
                apply_transform(inp, fixed_img, tx, outf)
                logging.info("Applied transform to %s", f)
                
        except Exception as e:
            logging.error("Registration failed for %s/%s: %s", section, sid, e)
            continue
    
    logging.info("Skull-stripping and registration complete for all train subjects.")

if __name__ == '__main__':
    main()