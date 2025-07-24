import os
import json
import logging
from typing import Dict, Any
import pandas as pd


def configure_logging() -> None:
    """
    Configure logging for the enrichment script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load the existing neuroscope dataset metadata JSON.

    Parameters:
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        metadata (dict): Parsed JSON as Python dict.
    """
    if not os.path.isfile(metadata_path):
        logging.error("Metadata file not found: %s", metadata_path)
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    logging.info("Loaded metadata from %s", metadata_path)
    return metadata


def load_acquisition_data(csv_path: str) -> pd.DataFrame:
    """
    Load UPenn acquisition parameters CSV into a DataFrame.

    Parameters:
        csv_path (str): Path to the acquisition CSV file.

    Returns:
        df (DataFrame): Pandas DataFrame with acquisition data.
    """
    if not os.path.isfile(csv_path):
        logging.error("Acquisition CSV not found: %s", csv_path)
        raise FileNotFoundError(f"Acquisition CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logging.info("Loaded acquisition data from %s", csv_path)
    return df


def enrich_upenn_metadata(
    metadata: Dict[str, Any],
    acquisition_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Enrich the UPenn 'valid_subjects' entries with acquisition metadata.

    Parameters:
        metadata (dict): Original dataset metadata dictionary.
        acquisition_df (DataFrame): Acquisition parameters DataFrame.

    Returns:
        enriched_metadata (dict): Metadata with added fields under upenn valid_subjects.
    """
    upenn_section = metadata.get('upenn', {})
    valid_subjects = upenn_section.get('valid_subjects', {})

    for subject_id, subj_info in valid_subjects.items():
        # CSV ID matches subject_id
        row = acquisition_df[acquisition_df['ID'] == subject_id]
        if row.empty:
            logging.warning("No acquisition data for subject %s", subject_id)
            continue
        # Assume single match
        row = row.iloc[0]

        # Extract relevant fields
        vendor = row.get('Manufacturer', None)
        model = row.get('Model', None)
        field_strength = row.get('Magnetic Field Strength', None)

        # Add to metadata
        subj_info['vendor'] = vendor
        subj_info['scanner_model'] = model
        subj_info['field_strength'] = float(field_strength) if pd.notnull(field_strength) else None
        logging.debug(
            "Enriched %s: vendor=%s, model=%s, field_strength=%s",
            subject_id, vendor, model, field_strength
        )

    metadata['upenn']['valid_subjects'] = valid_subjects
    return metadata


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    """
    Save enriched metadata dictionary to a JSON file.

    Parameters:
        metadata (dict): Metadata to save.
        output_path (str): Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info("Enriched metadata written to %s", output_path)


def main() -> None:
    """
    Main function to enrich neuroscope metadata with UPenn acquisition parameters.

    Steps:
    1. Load existing metadata JSON.
    2. Load acquisition parameters CSV.
    3. Enrich UPenn valid_subjects with vendor, scanner model, and field strength.
    4. Save enriched metadata to new JSON.
    """
    configure_logging()

    # Define paths
    scripts_dir = os.path.expanduser('~/Downloads/neuroscope/scripts')
    data_dir = os.path.expanduser('~/Downloads/neuroscope/data/PKG - UPENN-GBM-NIfTI')
    metadata_path = os.path.join(scripts_dir, 'neuroscope_dataset_metadata.json')
    acquisition_csv = os.path.join(data_dir, 'UPENN-GBM_acquisition.csv')
    output_path = os.path.join(scripts_dir, 'neuroscope_dataset_metadata_enriched.json')

    # Execute
    metadata = load_metadata(metadata_path)
    acquisition_df = load_acquisition_data(acquisition_csv)
    enriched_metadata = enrich_upenn_metadata(metadata, acquisition_df)
    save_metadata(enriched_metadata, output_path)


if __name__ == '__main__':
    main()
