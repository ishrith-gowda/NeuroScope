import os
import json
import logging
from typing import Dict, Any, Optional, Set
import pandas as pd
from pathlib import Path

from neuroscope_preprocessing_config import PATHS


def configure_logging() -> None:
    """
    Configure logging for the enrichment script.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load the existing neuroscope dataset metadata JSON with validation.
    
    Args:
        metadata_path: Path to the metadata JSON file
        
    Returns:
        Dict containing the loaded metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not metadata_path.exists():
        logging.error("metadata file not found: %s", metadata_path)
        raise FileNotFoundError(f"metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info("loaded metadata from: %s", metadata_path)
        
        # Validate basic structure
        required_sections = ['brats', 'upenn']
        for section in required_sections:
            if section not in metadata:
                raise ValueError(f"missing required section: {section}")
            if 'valid_subjects' not in metadata[section]:
                raise ValueError(f"missing 'valid_subjects' in section: {section}")
        
        logging.info("metadata structure validated")
        return metadata
        
    except json.JSONDecodeError as e:
        logging.error("invalid json in metadata file: %s", e)
        raise
    except Exception as e:
        logging.error("error loading metadata: %s", e)
        raise


def load_and_validate_acquisition_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate UPenn acquisition parameters CSV.
    
    Args:
        csv_path: Path to the acquisition CSV file
        
    Returns:
        Validated pandas DataFrame
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        logging.error("acquisition csv not found: %s", csv_path)
        raise FileNotFoundError(f"acquisition csv not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        logging.info("loaded acquisition csv: %s (%d rows)", csv_path, len(df))
        
        # Check required columns
        required_cols = ['ID']  # Only ID is truly required
        optional_cols = ['Manufacturer', 'Model', 'Magnetic Field Strength']
        
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"missing required columns: {missing_required}")
        
        missing_optional = [col for col in optional_cols if col not in df.columns]
        if missing_optional:
            logging.warning("missing optional columns: %s", missing_optional)
        
        # Log available columns
        logging.info("available columns: %s", list(df.columns))
        
        # Validate ID column
        if df['ID'].isna().any():
            na_count = df['ID'].isna().sum()
            logging.warning("found %d rows with missing ids, removing them", na_count)
            df = df.dropna(subset=['ID'])
        
        # Check for duplicate IDs
        duplicates = df['ID'].duplicated().sum()
        if duplicates > 0:
            logging.warning("found %d duplicate ids in csv", duplicates)
            df = df.drop_duplicates(subset=['ID'], keep='first')
        
        logging.info("csv validation complete: %d valid rows", len(df))
        return df
        
    except Exception as e:
        logging.error("error loading/validating csv: %s", e)
        raise


def clean_and_validate_field_strength(value: Any) -> Optional[float]:
    """
    Clean and validate magnetic field strength values.
    
    Args:
        value: Raw field strength value from CSV
        
    Returns:
        Cleaned float value or None if invalid
    """
    if pd.isna(value):
        return None
    
    try:
        # Convert to string and clean
        str_val = str(value).strip().lower()
        
        # Handle common formats like "3.0T", "1.5 tesla", etc.
        str_val = str_val.replace('t', '').replace('tesla', '').strip()
        
        float_val = float(str_val)
        
        # Validate reasonable range for MRI field strengths
        if 0.1 <= float_val <= 20.0:  # Reasonable range for MRI scanners
            return float_val
        else:
            logging.debug("field strength outside reasonable range: %s", float_val)
            return None
            
    except (ValueError, TypeError):
        logging.debug("could not parse field strength: %s", value)
        return None


def enrich_brats_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich BraTS metadata with known/inferred acquisition parameters.
    
    Args:
        metadata: The metadata dictionary to enrich
        
    Returns:
        Updated metadata dictionary
    """
    brats_section = metadata.get('brats', {})
    valid_subjects = brats_section.get('valid_subjects', {})
    
    enriched_count = 0
    
    for subject_id, subj_info in valid_subjects.items():
        # BraTS-TCGA data is from TCGA, typically acquired on various scanners
        # We can add some general information based on TCGA dataset characteristics
        subj_info['dataset_source'] = 'TCGA-GBM'
        subj_info['vendor'] = 'Mixed'  # TCGA includes multiple vendors
        subj_info['scanner_model'] = 'Mixed'  # Multiple models
        subj_info['field_strength'] = None  # Unknown, mixed field strengths
        subj_info['acquisition_site'] = 'Multiple TCGA Sites'
        subj_info['notes'] = 'Multi-institutional TCGA dataset with heterogeneous acquisition parameters'
        
        enriched_count += 1
    
    metadata['brats']['valid_subjects'] = valid_subjects
    logging.info("enriched %d brats subjects with dataset information", enriched_count)
    
    return metadata


def enrich_upenn_metadata(
    metadata: Dict[str, Any],
    acquisition_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Enrich UPenn metadata with acquisition parameters from CSV.
    
    Args:
        metadata: The metadata dictionary to enrich
        acquisition_df: DataFrame with acquisition parameters
        
    Returns:
        Updated metadata dictionary
    """
    upenn_section = metadata.get('upenn', {})
    valid_subjects = upenn_section.get('valid_subjects', {})
    
    enriched_count = 0
    missing_count = 0
    
    # Get set of available IDs in CSV for efficient lookup
    csv_ids: Set[str] = set(acquisition_df['ID'].astype(str))
    
    for subject_id, subj_info in valid_subjects.items():
        # Add dataset source information
        subj_info['dataset_source'] = 'UPenn-GBM'
        subj_info['acquisition_site'] = 'University of Pennsylvania'
        
        # Try to find matching acquisition data
        matching_rows = acquisition_df[acquisition_df['ID'].astype(str) == str(subject_id)]
        
        if matching_rows.empty:
            logging.debug("no acquisition data found for upenn subject: %s", subject_id)
            subj_info['vendor'] = None
            subj_info['scanner_model'] = None
            subj_info['field_strength'] = None
            subj_info['notes'] = 'Acquisition parameters not available'
            missing_count += 1
            continue
        
        # Use first matching row if multiple found
        if len(matching_rows) > 1:
            logging.warning("multiple acquisition rows found for %s, using first", subject_id)
        
        row = matching_rows.iloc[0]
        
        # Extract and clean acquisition parameters
        vendor = row.get('Manufacturer', None)
        model = row.get('Model', None)
        field_strength_raw = row.get('Magnetic Field Strength', None)
        
        # Clean vendor
        if pd.notna(vendor):
            vendor = str(vendor).strip()
            if vendor.lower() in ['', 'unknown', 'n/a', 'na']:
                vendor = None
        else:
            vendor = None
        
        # Clean model
        if pd.notna(model):
            model = str(model).strip()
            if model.lower() in ['', 'unknown', 'n/a', 'na']:
                model = None
        else:
            model = None
        
        # Clean field strength
        field_strength = clean_and_validate_field_strength(field_strength_raw)
        
        # Update subject info
        subj_info['vendor'] = vendor
        subj_info['scanner_model'] = model
        subj_info['field_strength'] = field_strength
        
        if vendor or model or field_strength:
            enriched_count += 1
            logging.debug("enriched %s: vendor=%s, model=%s, field=%.1fT", 
                         subject_id, vendor, model, field_strength or 0.0)
        else:
            missing_count += 1
            subj_info['notes'] = 'Acquisition parameters present but could not be parsed'
    
    metadata['upenn']['valid_subjects'] = valid_subjects
    
    logging.info("enriched %d upenn subjects with acquisition data", enriched_count)
    if missing_count > 0:
        logging.warning("%d upenn subjects missing or invalid acquisition data", missing_count)
    
    return metadata


def add_enrichment_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add metadata about the enrichment process.
    
    Args:
        metadata: The metadata dictionary to update
        
    Returns:
        Updated metadata with enrichment information
    """
    if 'generation_info' not in metadata:
        metadata['generation_info'] = {}
    
    metadata['generation_info'].update({
        'enrichment_script': '04_enrich_metadata_with_site_modality_neuroscope.py v2.0',
        'enrichment_features': [
            'upenn_acquisition_parameters',
            'brats_dataset_annotation',
            'field_strength_validation',
            'vendor_model_cleaning'
        ]
    })
    
    return metadata


def validate_enriched_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate the enriched metadata structure and content.
    
    Args:
        metadata: The enriched metadata to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Check basic structure
        for section in ['brats', 'upenn']:
            if section not in metadata:
                logging.error("missing section: %s", section)
                return False
            
            valid_subjects = metadata[section].get('valid_subjects', {})
            if not valid_subjects:
                logging.error("no valid subjects in section: %s", section)
                return False
            
            # Check that enrichment fields were added
            sample_subject = next(iter(valid_subjects.values()))
            required_fields = ['dataset_source', 'vendor', 'scanner_model', 'field_strength']
            
            for field in required_fields:
                if field not in sample_subject:
                    logging.error("missing enrichment field '%s' in section %s", field, section)
                    return False
        
        logging.info("enriched metadata validation passed")
        return True
        
    except Exception as e:
        logging.error("metadata validation error: %s", e)
        return False


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """
    Save enriched metadata dictionary to a JSON file with validation.
    
    Args:
        metadata: The metadata dictionary to save
        output_path: Path where to save the JSON file
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with proper formatting
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        
        logging.info("enriched metadata successfully written to: %s", output_path)
        
        # Verify file was written correctly
        file_size = output_path.stat().st_size
        logging.info("output file size: %.2f KB", file_size / 1024)
        
    except Exception as e:
        logging.error("failed to save enriched metadata: %s", e)
        raise


def print_enrichment_summary(metadata: Dict[str, Any]) -> None:
    """
    Print a comprehensive summary of the enrichment results.
    
    Args:
        metadata: The enriched metadata dictionary
    """
    print("\n" + "="*60)
    print("NEUROSCOPE METADATA ENRICHMENT SUMMARY")
    print("="*60)
    
    for dataset_name, section_key in [("BraTS-TCGA-GBM", "brats"), ("UPenn-GBM", "upenn")]:
        valid_subjects = metadata[section_key]['valid_subjects']
        total_subjects = len(valid_subjects)
        
        # Count subjects with acquisition data
        with_vendor = sum(1 for s in valid_subjects.values() if s.get('vendor'))
        with_model = sum(1 for s in valid_subjects.values() if s.get('scanner_model'))
        with_field = sum(1 for s in valid_subjects.values() if s.get('field_strength'))
        
        print(f"\n{dataset_name}:")
        print(f"  total subjects:           {total_subjects}")
        print(f"  with vendor info:         {with_vendor} ({with_vendor/total_subjects*100:.1f}%)")
        print(f"  with scanner model:       {with_model} ({with_model/total_subjects*100:.1f}%)")
        print(f"  with field strength:      {with_field} ({with_field/total_subjects*100:.1f}%)")
        
        if section_key == 'upenn' and with_field > 0:
            field_strengths = [s['field_strength'] for s in valid_subjects.values() 
                             if s.get('field_strength') is not None]
            unique_fields = sorted(set(field_strengths))
            print(f"  field strengths found:    {unique_fields}")
    
    print("="*60)


def main() -> None:
    """
    Main function to enrich neuroscope metadata with acquisition parameters.
    """
    configure_logging()
    
    logging.info("=== NEUROSCOPE METADATA ENRICHMENT ===")
    logging.info("using neuroscope_preprocessing_config.py for path management")
    
    # Get paths from config
    metadata_path = PATHS['metadata_base']
    acquisition_csv_path = PATHS['upenn_acquisition_csv']
    enriched_output_path = PATHS['metadata_enriched']
    
    logging.info("input metadata: %s", metadata_path)
    logging.info("upenn acquisition csv: %s", acquisition_csv_path)
    logging.info("output metadata: %s", enriched_output_path)
    
    try:
        # Load and validate inputs
        metadata = load_metadata(metadata_path)
        acquisition_df = load_and_validate_acquisition_data(acquisition_csv_path)
        
        # Enrich both datasets
        metadata = enrich_brats_metadata(metadata)
        metadata = enrich_upenn_metadata(metadata, acquisition_df)
        
        # Add enrichment metadata
        metadata = add_enrichment_metadata(metadata)
        
        # Validate enriched metadata
        if not validate_enriched_metadata(metadata):
            logging.error("enrichment validation failed; aborting.")
            return
        
        # Save enriched metadata
        save_metadata(metadata, enriched_output_path)
        
        # Print summary
        print_enrichment_summary(metadata)
        
        logging.info("metadata enrichment completed successfully")
        
    except Exception as e:
        logging.error("metadata enrichment failed: %s", e)
        raise


if __name__ == '__main__':
    main()