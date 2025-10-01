import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import SimpleITK as sitk

# ---- Intensity / MRI helpers -------------------------------------------------

def is_probable_mri_image(image: sitk.Image) -> bool:
    try:
        arr = sitk.GetArrayFromImage(image).astype(np.float32)
        unique_vals = np.unique(arr)
        # Reject near-binary or extremely low variation
        if len(unique_vals) <= 4 and np.all(np.isin(unique_vals, [0, 1, 2, 3])):
            return False
        if arr.std() < 0.005:
            return False
        return True
    except Exception:
        return False

def verify_mri_path(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        img = sitk.ReadImage(str(path))
        return is_probable_mri_image(img)
    except Exception:
        return False

# ---- Brain mask utilities (unified) -----------------------------------------

def generate_brain_mask(image: sitk.Image, background_threshold: float = 0.01) -> sitk.Image:
    """Unified robust brain mask creation used across scripts.
    Strategy: Otsu with validation -> fallback threshold -> morphological cleanup.
    """
    arr = sitk.GetArrayFromImage(image)
    try:
        otsu = sitk.OtsuThreshold(image, 0, 1, 256)
        otsu_arr = sitk.GetArrayFromImage(otsu)
        ratio = otsu_arr.sum() / max(1, otsu_arr.size)
        if 0.08 < ratio < 0.85:  # plausible brain occupancy
            mask = otsu
        else:
            raise ValueError("Otsu mask volume outside expected range")
    except Exception:
        simple = (arr > background_threshold).astype(np.uint8)
        mask = sitk.GetImageFromArray(simple)
        mask.CopyInformation(image)
    # Cleanup
    try:
        mask = sitk.BinaryFillhole(mask)
        mask = sitk.BinaryOpeningByReconstruction(mask, [2, 2, 2])
    except Exception:
        pass
    return mask

# ---- JSON helpers -----------------------------------------------------------

def write_json_with_schema(data: Dict[str, Any], path: Path, schema: Optional[Dict[str, Any]] = None, summary: bool = False) -> None:
    """Write JSON after optional lightweight schema validation.
    Schema (if provided) format: {'required_keys': [...]} for shallow validation.
    If summary=True, trims large subtrees where possible.
    """
    if schema:
        missing = [k for k in schema.get('required_keys', []) if k not in data]
        if missing:
            logging.warning("JSON missing expected top-level keys: %s", missing)
    if summary:
        data = generate_summary_view(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    logging.info("Wrote JSON: %s (summary=%s)", path, summary)


def generate_summary_view(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(data)
    # Drop or shrink heavy fields if present
    for heavy_key in ['detailed_results', 'processing_details', 'detailed_diagnoses']:
        if heavy_key in summary:
            summary[heavy_key] = f"omitted_in_summary (see full file for {heavy_key})"
    return summary

# ---- N4 support utilities ---------------------------------------------------

def evaluate_bias_need(image: sitk.Image, mask: sitk.Image) -> float:
    """Return median slice CV to decide if N4 is warranted."""
    arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    cvs = []
    for z in range(arr.shape[0]):
        sm = mask_arr[z]
        if sm.sum() > 50:
            vals = arr[z][sm]
            m = vals.mean()
            if m > 0:
                cvs.append(vals.std() / m)
    if not cvs:
        return 999.0
    return float(np.median(cvs))


def acceptable_n4_change(orig_stats: Dict[str, float], corr_stats: Dict[str, float]) -> bool:
    """Decide if corrected stats are acceptable relative to original."""
    try:
        range_ratio = corr_stats['range'] / max(1e-6, orig_stats['range'])
        mean_ratio = corr_stats['mean'] / max(1e-6, orig_stats['mean'])
        if not (0.4 < range_ratio < 2.5):
            return False
        if not (0.4 < mean_ratio < 2.0):
            return False
        return True
    except KeyError:
        return False

# ---- Small stat helpers -----------------------------------------------------

def basic_intensity_stats(image: sitk.Image, mask: Optional[sitk.Image] = None) -> Dict[str, float]:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    if mask is not None:
        m = sitk.GetArrayFromImage(mask).astype(bool)
        arr = arr[m]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'range': 0.0}
    return {'mean': float(arr.mean()), 'std': float(arr.std()), 'range': float(arr.max() - arr.min())}
