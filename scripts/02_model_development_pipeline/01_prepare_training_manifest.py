import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import SimpleITK as sitk

# Import PATHS from preprocessing config (sibling folder)
HERE = Path(__file__).resolve().parent
PREP_DIR = HERE.parent / '01_data_preparation_pipeline'
if str(PREP_DIR) not in sys.path:
    sys.path.insert(0, str(PREP_DIR))
import neuroscope_preprocessing_config as npc  # type: ignore
PATHS = npc.PATHS


def setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='[%(levelname)s] %(message)s')


def list_subjects_for_split(metadata_path: Path, split: str) -> Dict[str, List[str]]:
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    out: Dict[str, List[str]] = {"brats": [], "upenn": []}
    for section in ("brats", "upenn"):
        for sid, info in meta.get(section, {}).get("valid_subjects", {}).items():
            if info.get("split") == split:
                out[section].append(sid)
    return out


def verify_modalities_and_depth(preprocessed_dir: Path, section: str, sid: str) -> Tuple[bool, int]:
    subj_dir = preprocessed_dir / section / sid
    required = ("t1.nii.gz", "t1gd.nii.gz", "t2.nii.gz", "flair.nii.gz")
    try:
        sizes = []
        for mod in required:
            p = subj_dir / mod
            if not p.exists():
                return False, 0
            img = sitk.ReadImage(str(p))
            arr_size = img.GetSize()  # (W,H,D)
            sizes.append(arr_size[2])
        # sanity: all modalities same depth
        if len(set(sizes)) != 1:
            logging.warning("depth mismatch for %s/%s: %s", section, sid, sizes)
        return True, sizes[0]
    except Exception as e:
        logging.warning("failed to read %s/%s due to %s", section, sid, e)
        return False, 0


def build_manifest(splits: List[str], verbose: bool = False) -> Dict:
    setup_logging(verbose)
    pre_dir = PATHS['preprocessed_dir']
    meta_path = PATHS['metadata_splits']
    if not pre_dir.exists():
        raise FileNotFoundError(f"preprocessed dir not found: {pre_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata splits not found: {meta_path}")

    manifest = {"splits": {}, "root": str(pre_dir)}
    for split in splits:
        manifest["splits"][split] = {"brats": [], "upenn": []}
        subjects = list_subjects_for_split(meta_path, split)
        for section in ("brats", "upenn"):
            for sid in subjects[section]:
                ok, depth = verify_modalities_and_depth(pre_dir, section, sid)
                if ok:
                    manifest["splits"][split][section].append({
                        "subject": sid,
                        "depth": depth,
                        "path": str(pre_dir / section / sid)
                    })

    # quick summaries
    summary = {}
    for split in splits:
        summary[split] = {
            "brats_subjects": len(manifest["splits"][split]["brats"]),
            "upenn_subjects": len(manifest["splits"][split]["upenn"]),
        }
    manifest["summary"] = summary
    return manifest


def write_manifest_files(manifest: Dict) -> Tuple[Path, Path]:
    out_dir = Path(PATHS['scripts_dir']) / '02_model_development_pipeline'
    out_dir.mkdir(parents=True, exist_ok=True)
    full = out_dir / 'neuroscope_training_manifest.json'
    summ = out_dir / 'neuroscope_training_manifest_summary.json'
    with open(full, 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(summ, 'w') as f:
        json.dump(manifest.get('summary', {}), f, indent=2)
    logging.info("wrote manifest: %s", full)
    logging.info("wrote summary: %s", summ)
    return full, summ


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare training manifest for CycleGAN")
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of splits to include')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    manifest = build_manifest(splits, verbose=args.verbose)
    write_manifest_files(manifest)


if __name__ == '__main__':
    main()
