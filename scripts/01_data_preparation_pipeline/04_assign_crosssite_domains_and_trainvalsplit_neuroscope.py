import os
import json
import logging
import random
from typing import Dict, List, Tuple, Any
from pathlib import Path
import time

from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """
    configure logging format and level for domain assignment and splitting.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def validate_enriched_metadata(metadata: Dict[str, Any]) -> bool:
    """
    validate the enriched metadata structure before processing.
    
    args:
        metadata: the loaded metadata dictionary
        
    returns:
        bool: true if metadata is valid, false otherwise
    """
    required_sections = ['brats', 'upenn']
    required_keys = ['valid_subjects', 'missing_subjects', 'dataset_info']
    
    for section in required_sections:
        if section not in metadata:
            logging.error("missing required section: %s", section)
            return False
        
        for key in required_keys:
            if key not in metadata[section]:
                logging.error("missing required key '%s' in section '%s'", key, section)
                return False
        
        # check that we have valid subjects to work with
        valid_subjects = metadata[section]['valid_subjects']
        if not isinstance(valid_subjects, dict):
            logging.error("'valid_subjects' in section '%s' is not a dictionary", section)
            return False
    
    # check total subject counts
    brats_count = len(metadata['brats']['valid_subjects'])
    upenn_count = len(metadata['upenn']['valid_subjects'])
    total_subjects = brats_count + upenn_count
    
    if total_subjects == 0:
        logging.error("no valid subjects found in any dataset")
        return False
    
    logging.info("metadata validation passed: %d brats + %d upenn = %d total subjects", 
                brats_count, upenn_count, total_subjects)
    return True


def load_enriched_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    load and validate the enriched neuroscope dataset metadata.
    
    args:
        metadata_path: path to the enriched metadata json file
        
    returns:
        dict: loaded and validated metadata
        
    raises:
        filenotfounderror: if metadata file doesn't exist
        valueerror: if metadata is invalid
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"enriched metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info("loaded enriched metadata from %s", metadata_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json in metadata file: {e}")
    
    if not validate_enriched_metadata(metadata):
        raise ValueError("metadata validation failed")
    
    return metadata


def assign_domain_labels(metadata: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    assign domain labels to subjects and collect subject lists.
    
    brats subjects -> domain a (source domain)
    upenn subjects -> domain b (target domain)
    
    args:
        metadata: the metadata dictionary to modify
        
    returns:
        tuple of (brats_subject_ids, upenn_subject_ids)
    """
    logging.info("assigning domain labels...")
    
    # process brats subjects (domain a)
    brats_subjects = []
    brats_valid = metadata['brats']['valid_subjects']
    for subj_id, subj_info in brats_valid.items():
        subj_info['domain'] = 'A'
        subj_info['domain_name'] = 'BraTS-TCGA-GBM'
        brats_subjects.append(subj_id)
    
    # process upenn subjects (domain b)  
    upenn_subjects = []
    upenn_valid = metadata['upenn']['valid_subjects']
    for subj_id, subj_info in upenn_valid.items():
        subj_info['domain'] = 'B'
        subj_info['domain_name'] = 'UPenn-GBM'
        upenn_subjects.append(subj_id)
    
    logging.info("domain assignment complete:")
    logging.info("  domain a (brats): %d subjects", len(brats_subjects))
    logging.info("  domain b (upenn): %d subjects", len(upenn_subjects))
    
    return brats_subjects, upenn_subjects


def stratified_split_subjects(
    subject_ids: List[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Dict[str, str]:
    """
    split subjects into train/validation/test sets with specified proportions.
    
    args:
        subject_ids: list of subject identifiers
        train_frac: fraction for training split (default: 0.7)
        val_frac: fraction for validation split (default: 0.15)
        test_frac: fraction for test split (default: 0.15)
        seed: random seed for reproducibility (default: 42)
        
    returns:
        dict mapping subject_id to split label ('train', 'val', 'test')
        
    raises:
        valueerror: if fractions don't sum to 1.0
    """
    # validate input fractions
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total_frac:.6f}")
    
    if not subject_ids:
        logging.warning("empty subject list provided for splitting")
        return {}
    
    total = len(subject_ids)
    
    # set random seed for reproducibility
    random.seed(seed)
    shuffled_ids = subject_ids.copy()
    random.shuffle(shuffled_ids)
    
    # calculate split sizes
    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    n_test = total - n_train - n_val  # ensure remainder goes to test
    
    # assign splits
    split_assignments = {}
    for idx, subj_id in enumerate(shuffled_ids):
        if idx < n_train:
            split_assignments[subj_id] = 'train'
        elif idx < n_train + n_val:
            split_assignments[subj_id] = 'val'
        else:
            split_assignments[subj_id] = 'test'
    
    logging.info("split %d subjects: %d train, %d val, %d test (seed=%d)", 
                total, n_train, n_val, n_test, seed)
    
    return split_assignments


def apply_splits_to_metadata(
    metadata: Dict[str, Any],
    brats_splits: Dict[str, str],
    upenn_splits: Dict[str, str]
) -> Dict[str, Any]:
    """
    apply split assignments to the metadata structure.
    
    args:
        metadata: the metadata dictionary to modify
        brats_splits: split assignments for brats subjects
        upenn_splits: split assignments for upenn subjects
        
    returns:
        updated metadata dictionary
    """
    logging.info("applying split assignments to metadata...")
    
    # apply brats splits
    for subj_id, split in brats_splits.items():
        if subj_id in metadata['brats']['valid_subjects']:
            metadata['brats']['valid_subjects'][subj_id]['split'] = split
        else:
            logging.warning("brats subject %s not found in valid_subjects", subj_id)
    
    # apply upenn splits
    for subj_id, split in upenn_splits.items():
        if subj_id in metadata['upenn']['valid_subjects']:
            metadata['upenn']['valid_subjects'][subj_id]['split'] = split
        else:
            logging.warning("upenn subject %s not found in valid_subjects", subj_id)
    
    # add split summary to metadata
    metadata['split_info'] = {
        'train_fraction': 0.7,
        'val_fraction': 0.15,
        'test_fraction': 0.15,
        'random_seed': 42,
        'split_method': 'stratified_by_domain'
    }
    
    return metadata


def generate_split_summary(metadata: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    generate summary statistics for the train/val/test splits.
    
    args:
        metadata: metadata with split assignments
        
    returns:
        dictionary with split counts by domain and overall
    """
    summary = {
        'brats': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'upenn': {'train': 0, 'val': 0, 'test': 0, 'total': 0},
        'overall': {'train': 0, 'val': 0, 'test': 0, 'total': 0}
    }
    
    for section in ['brats', 'upenn']:
        for subj_id, subj_info in metadata[section]['valid_subjects'].items():
            split = subj_info.get('split', 'unknown')
            if split in summary[section]:
                summary[section][split] += 1
                summary[section]['total'] += 1
                summary['overall'][split] += 1
                summary['overall']['total'] += 1
    
    return summary


def save_split_text_files(
    metadata: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    save subject lists for each split to text files for easy access.
    
    args:
        metadata: metadata with split assignments
        output_dir: directory to save split files
    """
    logging.info("generating split text files...")
    
    # collect subjects by split
    splits = {'train': [], 'val': [], 'test': []}
    
    for section in ['brats', 'upenn']:
        for subj_id, subj_info in metadata[section]['valid_subjects'].items():
            split = subj_info.get('split')
            if split and split in splits:
                # use section/subject_id format for consistency
                entry = f"{section}/{subj_id}"
                splits[split].append(entry)
    
    # write split files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split, subjects in splits.items():
        file_path = output_dir / f"{split}_subjects.txt"
        try:
            with open(file_path, 'w') as f:
                for subject in sorted(subjects):  # sort for consistency
                    f.write(subject + '\n')
            logging.info("wrote %d %s subjects to %s", len(subjects), split, file_path)
        except Exception as e:
            logging.error("failed to write %s split file: %s", split, e)


def save_metadata_with_splits(
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """
    save the metadata with domain and split assignments.
    
    args:
        metadata: complete metadata dictionary
        output_path: path to save the final metadata json
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        
        logging.info("final metadata with splits saved to: %s", output_path)
        
    except Exception as e:
        logging.error("failed to save metadata: %s", e)
        raise


def print_split_summary(summary: Dict[str, Dict[str, int]]) -> None:
    """
    print a formatted summary of the train/val/test splits.
    
    args:
        summary: split summary from generate_split_summary()
    """
    print("\n" + "="*70)
    print("neuroscope domain assignment & train/val/test split summary")
    print("="*70)
    
    # domain-specific summaries
    for domain, domain_name in [('brats', 'BraTS-TCGA-GBM'), ('upenn', 'UPenn-GBM')]:
        counts = summary[domain]
        total = counts['total']
        
        print(f"\n{domain_name}:")
        print(f"  train:      {counts['train']:3d} subjects ({counts['train']/max(total,1)*100:.1f}%)")
        print(f"  validation: {counts['val']:3d} subjects ({counts['val']/max(total,1)*100:.1f}%)")
        print(f"  test:       {counts['test']:3d} subjects ({counts['test']/max(total,1)*100:.1f}%)")
        print(f"  total:      {total:3d} subjects")
    
    # overall summary
    overall = summary['overall']
    total = overall['total']
    
    print(f"\noverall:")
    print(f"  train:      {overall['train']:3d} subjects ({overall['train']/max(total,1)*100:.1f}%)")
    print(f"  validation: {overall['val']:3d} subjects ({overall['val']/max(total,1)*100:.1f}%)")
    print(f"  test:       {overall['test']:3d} subjects ({overall['test']/max(total,1)*100:.1f}%)")
    print(f"  total:      {total:3d} subjects")
    
    print(f"\ndomain assignment:")
    print(f"  domain a (brats): {summary['brats']['total']} subjects")
    print(f"  domain b (upenn): {summary['upenn']['total']} subjects")
    
    print("="*70)


def main() -> None:
    """
    main function to assign domains and generate train/val/test splits.
    """
    start_time = time.time()
    configure_logging()
    
    logging.info("=== NEUROSCOPE DOMAIN ASSIGNMENT & SPLIT GENERATION ===")
    logging.info("using neuroscope_preprocessing_config.py for path management")
    
    # define input/output paths using centralized config
    enriched_metadata_path = PATHS['metadata_enriched']
    output_metadata_path = PATHS['metadata_splits']
    split_files_dir = PATHS['scripts_dir']
    
    logging.info("input:  %s", enriched_metadata_path)
    logging.info("output: %s", output_metadata_path)
    logging.info("splits: %s", split_files_dir)
    
    try:
        # step 1: load and validate enriched metadata
        logging.info("step 1: loading enriched metadata...")
        metadata = load_enriched_metadata(enriched_metadata_path)
        
        # step 2: assign domain labels
        logging.info("step 2: assigning domain labels...")
        brats_subjects, upenn_subjects = assign_domain_labels(metadata)
        
        # step 3: generate train/val/test splits per domain
        logging.info("step 3: generating train/val/test splits...")
        split_config = {
            'train_frac': 0.7,
            'val_frac': 0.15, 
            'test_frac': 0.15,
            'seed': 42
        }
        
        brats_splits = stratified_split_subjects(brats_subjects, **split_config)
        upenn_splits = stratified_split_subjects(upenn_subjects, **split_config)
        
        # step 4: apply splits to metadata
        logging.info("step 4: applying splits to metadata...")
        metadata = apply_splits_to_metadata(metadata, brats_splits, upenn_splits)
        
        # step 5: generate summary statistics
        logging.info("step 5: generating summary statistics...")
        summary = generate_split_summary(metadata)
        
        # step 6: save final metadata with splits
        logging.info("step 6: saving metadata with splits...")
        save_metadata_with_splits(metadata, output_metadata_path)
        
        # step 7: save convenience split text files
        logging.info("step 7: saving split text files...")
        save_split_text_files(metadata, split_files_dir)
        
        # step 8: display summary
        elapsed_time = time.time() - start_time
        print_split_summary(summary)
        
        logging.info("domain assignment and splitting completed successfully in %.2f seconds", elapsed_time)
        
    except Exception as e:
        logging.error("script failed: %s", e)
        raise


if __name__ == '__main__':
    main()