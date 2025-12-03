import argparse
import json
import logging
import sys
import time
import datetime
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
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def list_subjects_for_split(metadata_path: Path, split: str) -> Dict[str, List[str]]:
    logging.info(f"Loading subjects for split: {split} from {metadata_path}")
    start_time = time.time()
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    out: Dict[str, List[str]] = {"brats": [], "upenn": []}
    for section in ("brats", "upenn"):
        for sid, info in meta.get(section, {}).get("valid_subjects", {}).items():
            if info.get("split") == split:
                out[section].append(sid)
    
    elapsed = time.time() - start_time
    logging.info(f"Found {len(out['brats'])} BRATS subjects and {len(out['upenn'])} UPENN subjects for split '{split}' in {elapsed:.2f}s")
    return out


def verify_modalities_and_depth(preprocessed_dir: Path, section: str, sid: str) -> Tuple[bool, int]:
    subj_dir = preprocessed_dir / section / sid
    required = ("t1.nii.gz", "t1gd.nii.gz", "t2.nii.gz", "flair.nii.gz")
    logging.debug(f"Verifying subject {section}/{sid} in {subj_dir}")
    
    try:
        sizes = []
        missing_files = []
        
        for mod in required:
            p = subj_dir / mod
            if not p.exists():
                missing_files.append(mod)
                logging.debug(f"Missing file for {section}/{sid}: {mod}")
                continue
                
            try:
                img = sitk.ReadImage(str(p))
                arr_size = img.GetSize()  # (W,H,D)
                sizes.append(arr_size[2])
                logging.debug(f"Read {mod} for {section}/{sid}, depth: {arr_size[2]}")
            except Exception as e:
                logging.warning(f"Error reading {mod} for {section}/{sid}: {e}")
                missing_files.append(mod)
        
        if missing_files:
            logging.warning(f"Subject {section}/{sid} missing files: {', '.join(missing_files)}")
            return False, 0
            
        # sanity: all modalities same depth
        if len(set(sizes)) != 1:
            logging.warning("Depth mismatch for %s/%s: %s", section, sid, sizes)
        
        return True, sizes[0]
    except Exception as e:
        logging.warning("Failed to read %s/%s due to %s", section, sid, e)
        return False, 0


def build_manifest(splits: List[str], verbose: bool = False) -> Dict:
    start_time = time.time()
    setup_logging(verbose)
    logging.info(f"Starting manifest build for splits: {', '.join(splits)}")
    
    pre_dir = PATHS['preprocessed_dir']
    meta_path = PATHS['metadata_splits']
    
    logging.info(f"Using preprocessed directory: {pre_dir}")
    logging.info(f"Using metadata splits file: {meta_path}")
    
    if not pre_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found: {pre_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata splits not found: {meta_path}")

    manifest = {"splits": {}, "root": str(pre_dir)}
    total_subjects_processed = 0
    total_subjects_valid = 0
    last_update_time = time.time()
    
    for split in splits:
        split_start_time = time.time()
        logging.info(f"Processing split: {split}")
        manifest["splits"][split] = {"brats": [], "upenn": []}
        subjects = list_subjects_for_split(meta_path, split)
        
        total_to_process = sum(len(subjects[section]) for section in ("brats", "upenn"))
        processed_count = 0
        
        for section in ("brats", "upenn"):
            section_start = time.time()
            section_valid = 0
            section_total = len(subjects[section])
            
            logging.info(f"Processing {section_total} subjects from {section} dataset for {split} split")
            
            for i, sid in enumerate(subjects[section]):
                processed_count += 1
                total_subjects_processed += 1
                
                # Print progress update every 10 subjects or if 5+ seconds passed since last update
                current_time = time.time()
                if i % 10 == 0 or current_time - last_update_time >= 5:
                    progress_pct = (processed_count / total_to_process) * 100
                    elapsed = current_time - start_time
                    logging.info(f"Progress: {processed_count}/{total_to_process} ({progress_pct:.1f}%) - Processing {section}/{sid} ({i+1}/{section_total})")
                    last_update_time = current_time
                
                ok, depth = verify_modalities_and_depth(pre_dir, section, sid)
                if ok:
                    section_valid += 1
                    total_subjects_valid += 1
                    manifest["splits"][split][section].append({
                        "subject": sid,
                        "depth": depth,
                        "path": str(pre_dir / section / sid)
                    })
            
            section_time = time.time() - section_start
            logging.info(f"Completed {section} for {split}: {section_valid}/{section_total} valid subjects in {section_time:.2f}s")
        
        split_time = time.time() - split_start_time
        logging.info(f"Completed split {split} processing in {split_time:.2f}s")

    # quick summaries
    summary = {}
    for split in splits:
        summary[split] = {
            "brats_subjects": len(manifest["splits"][split]["brats"]),
            "upenn_subjects": len(manifest["splits"][split]["upenn"]),
            "total_subjects": len(manifest["splits"][split]["brats"]) + len(manifest["splits"][split]["upenn"])
        }
        
    manifest["summary"] = summary
    
    total_time = time.time() - start_time
    logging.info(f"Manifest building complete: {total_subjects_valid}/{total_subjects_processed} valid subjects in {total_time:.2f}s")
    
    return manifest


def write_manifest_files(manifest: Dict) -> Tuple[Path, Path]:
    logging.info("Writing manifest files...")
    start_time = time.time()
    
    out_dir = Path(PATHS['scripts_dir']) / '02_model_development_pipeline'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    full = out_dir / 'neuroscope_training_manifest.json'
    summ = out_dir / 'neuroscope_training_manifest_summary.json'
    
    logging.info(f"Writing full manifest to: {full}")
    with open(full, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logging.info(f"Writing summary manifest to: {summ}")
    with open(summ, 'w') as f:
        json.dump(manifest.get('summary', {}), f, indent=2)
    
    elapsed = time.time() - start_time
    logging.info(f"Manifest files written successfully in {elapsed:.2f}s")
    
    # Print summary information
    logging.info("=== Manifest Summary ===")
    for split, info in manifest.get('summary', {}).items():
        logging.info(f"Split '{split}': {info.get('total_subjects', 0)} total subjects")
        logging.info(f"  - BRATS: {info.get('brats_subjects', 0)} subjects")
        logging.info(f"  - UPENN: {info.get('upenn_subjects', 0)} subjects")
    
    return full, summ


def parse_args():
    ap = argparse.ArgumentParser(description="Prepare training manifest for CycleGAN")
    ap.add_argument('--splits', type=str, default='train,val', help='Comma list of splits to include')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    
    # Show start message with timestamp
    print(f"=== NeuroScope Training Manifest Preparation ===")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    print(f"Processing splits: {', '.join(splits)}")
    
    manifest = build_manifest(splits, verbose=args.verbose)
    full_path, summary_path = write_manifest_files(manifest)
    
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    
    print(f"\n=== Processing Complete ===")
    print(f"Total time: {int(minutes)} minutes {seconds:.2f} seconds")
    print(f"Full manifest: {full_path}")
    print(f"Summary manifest: {summary_path}")
    
    logging.info(f"Training manifest preparation completed successfully in {total_time:.2f}s")


if __name__ == '__main__':
    main()
