import os
import json
import logging
import random
from typing import Dict, List, Tuple


def configure_logging() -> None:
    """
    Configure the root logger to output informational messages.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_json(path: str) -> Dict:
    """
    Load a JSON file and return its contents as a Python dictionary.

    Parameters:
        path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.isfile(path):
        logging.error("JSON file not found at %s", path)
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    logging.info("Loaded JSON metadata from %s", path)
    return data


def save_json(data: Dict, path: str) -> None:
    """
    Save a Python dictionary as pretty-formatted JSON to the given path.

    Parameters:
        data (dict): The data to serialize.
        path (str): Path where the JSON should be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info("Saved JSON metadata to %s", path)


def assign_domain_and_collect(
    metadata: Dict,
    section: str,
    domain_label: str
) -> List[str]:
    """
    Assign a domain label to each valid subject in a metadata section and
    collect the list of subject keys.

    Parameters:
        metadata (dict): The full metadata dictionary.
        section (str): Section key in metadata ('brats' or 'upenn').
        domain_label (str): Domain label to assign ('A' or 'B').

    Returns:
        List[str]: List of subject IDs processed in this section.
    """
    valid = metadata[section]['valid_subjects']
    subject_ids: List[str] = []
    for subj_id, subj_info in valid.items():
        subj_info['domain'] = domain_label
        subject_ids.append(subj_id)
    logging.info(
        "Assigned domain '%s' to %d subjects in section '%s'", 
        domain_label, len(subject_ids), section
    )
    return subject_ids


def stratified_split(
    subject_ids: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int
) -> Dict[str, str]:
    """
    Split a list of subject IDs into train/val/test with stratified proportions.

    Parameters:
        subject_ids (List[str]): List of subject identifiers.
        train_frac (float): Fraction of subjects for training split.
        val_frac (float): Fraction for validation split.
        test_frac (float): Fraction for test split.
        seed (int): Random seed for reproducibility.

    Returns:
        Dict[str, str]: Mapping subject_id to split label ('train', 'val', 'test').
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"
    total = len(subject_ids)
    random.seed(seed)
    shuffled = subject_ids.copy()
    random.shuffle(shuffled)

    n_train = int(total * train_frac)
    n_val = int(total * val_frac)
    # Ensure test gets the remainder
    n_test = total - n_train - n_val

    splits: Dict[str, str] = {}
    for idx, subj_id in enumerate(shuffled):
        if idx < n_train:
            splits[subj_id] = 'train'
        elif idx < n_train + n_val:
            splits[subj_id] = 'val'
        else:
            splits[subj_id] = 'test'

    logging.info(
        "Split %d subjects into %d train, %d val, %d test (seed=%d)",
        total, n_train, n_val, n_test, seed
    )
    return splits


def write_split_lists(
    split_map: Dict[str, str],
    section: str,
    out_dir: str
) -> None:
    """
    Write subject IDs for each split into text files.

    Parameters:
        split_map (Dict[str, str]): Mapping of subject_id to split name.
        section (str): Dataset section name ('brats' or 'upenn').
        out_dir (str): Directory where split text files will be saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    lists: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    for subj_id, split in split_map.items():
        entry = f"{section}/{subj_id}"
        lists[split].append(entry)

    for split, subjects in lists.items():
        path = os.path.join(out_dir, f"{split}_subjects.txt")
        with open(path, 'w') as f:
            for line in subjects:
                f.write(line + '\n')
        logging.info("Wrote %d %s subjects to %s", len(subjects), split, path)


def main() -> None:
    """
    Main script to assign domains and generate train/val/test splits for neuroscope.
    """
    configure_logging()

    scripts_dir = os.path.expanduser('~/Downloads/neuroscope/scripts')
    enriched_path = os.path.join(scripts_dir, 'neuroscope_dataset_metadata_enriched.json')
    output_metadata = os.path.join(scripts_dir, 'neuroscope_dataset_metadata_splits.json')
    split_files_dir = scripts_dir

    # Load enriched metadata
    metadata = load_json(enriched_path)

    # Assign domains and collect subject lists
    brats_ids = assign_domain_and_collect(metadata, 'brats', 'A')
    upenn_ids = assign_domain_and_collect(metadata, 'upenn', 'B')

    # Define split fractions and seed
    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    seed = 42

    # Generate splits per domain
    brats_splits = stratified_split(brats_ids, train_frac, val_frac, test_frac, seed)
    upenn_splits = stratified_split(upenn_ids, train_frac, val_frac, test_frac, seed)

    # Annotate metadata with split labels
    for subj_id, split in brats_splits.items():
        metadata['brats']['valid_subjects'][subj_id]['split'] = split
    for subj_id, split in upenn_splits.items():
        metadata['upenn']['valid_subjects'][subj_id]['split'] = split

    # Save updated metadata
    save_json(metadata, output_metadata)

    # Write train/val/test lists
    write_split_lists(brats_splits, 'brats', split_files_dir)
    write_split_lists(upenn_splits, 'upenn', split_files_dir)

if __name__ == '__main__':
    main()
