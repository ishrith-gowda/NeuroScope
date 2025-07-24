import os
import json
import logging

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info("Saved rebased metadata to %s", path)

def rebase_paths(metadata, old_root, new_root):
    """
    Walk through every valid_subject entry and replace old_root with new_root.
    """
    for section in ['brats', 'upenn']:
        subjects = metadata[section]['valid_subjects']
        for subj_id, info in subjects.items():
            for key, val in list(info.items()):
                if isinstance(val, str) and val.startswith(old_root):
                    new_path = val.replace(old_root, new_root, 1)
                    info[key] = new_path
    return metadata

def main():
    configure_logging()

    # Adjust these to match your environment
    scripts_dir = "/Volumes/USB Drive/neuroscope/scripts"
    old_meta = os.path.join(scripts_dir, "neuroscope_dataset_metadata_splits.json")
    new_meta = os.path.join(scripts_dir, "neuroscope_dataset_metadata_rebased.json")

    # The old data root you used when generating metadata
    old_root = os.path.expanduser("~/neuroscope/data")
    # The new data root on your flash drive
    new_root = "/Volumes/USB Drive/neuroscope/data"

    logging.info("Loading original metadata from %s", old_meta)
    metadata = load_json(old_meta)

    logging.info("Rebasing paths from %s to %s", old_root, new_root)
    metadata = rebase_paths(metadata, old_root, new_root)

    save_json(metadata, new_meta)

if __name__ == "__main__":
    main()
