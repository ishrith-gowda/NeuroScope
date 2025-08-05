import os
import json
import logging
from typing import Dict, Any
import pandas as pd

from neuroscope_preprocessing_config import PATHS  # ✅ NEW import

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
    """
    upenn_section = metadata.get('upenn', {})
    valid_subjects = upenn_section.get('valid_subjects', {})

    for subject_id, subj_info in valid_subjects.items():
        row = acquisition_df[acquisition_df['ID'] == subject_id]
        if row.empty:
            logging.warning("No acquisition data for subject %s", subject_id)
            continue

        row = row.iloc[0]
        vendor = row.get('Manufacturer', None)
        model = row.get('Model', None)
        field_strength = row.get('Magnetic Field Strength', None)

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
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info("Enriched metadata written to %s", output_path)


def main() -> None:
    """
    Main function to enrich neuroscope metadata with UPenn acquisition parameters.
    """
    configure_logging()
    metadata_path = PATHS['metadata_base']
    acquisition_csv_path = PATHS['upenn_acquisition_csv']
    enriched_output_path = PATHS['metadata_enriched']

    metadata = load_metadata(metadata_path)
    acquisition_df = load_acquisition_data(acquisition_csv_path)
    enriched_metadata = enrich_upenn_metadata(metadata, acquisition_df)
    save_metadata(enriched_metadata, enriched_output_path)


if __name__ == '__main__':
    main()