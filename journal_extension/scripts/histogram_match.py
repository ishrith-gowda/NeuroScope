#!/usr/bin/env python3
"""histogram-matching baseline (non-learned harmonization) for the exp #3/#4 comparison.

the simplest classical harmonization baseline: match each source (upenn) volume's intensity
distribution to a fixed reference (brats) volume, per modality, over brain (non-zero) voxels, and
write the result in our per-subject BraTS layout so the SAME frozen nnU-Net / downstream eval can
score it alongside the learned arms. gives reviewers the "does a trivial non-learned method do
better/worse than the GAN?" comparison the finding needs.

reference = a fixed brats TRAIN subject (first id in the train list), so no test leakage. matching is
done on non-zero voxels only (the brain), leaving background at 0, then clamped to [0,1].

usage: histogram_match.py --src <upenn_dir> --ref_dir <brats_dir> --ref_id <brats_subj> \
         --ids <upenn_ids.txt> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.exposure import match_histograms

MODS = ["t1", "t1gd", "t2", "flair"]


def _load(p: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def _match_masked(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """match src's brain-voxel histogram to ref's brain-voxel histogram; background stays 0."""
    out = np.zeros_like(src, dtype=np.float32)
    sm = src > 0
    rm = ref > 0
    if sm.sum() == 0 or rm.sum() == 0:
        return src.astype(np.float32)
    matched = match_histograms(src[sm].reshape(-1, 1), ref[rm].reshape(-1, 1)).reshape(-1)
    out[sm] = np.clip(matched, 0.0, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="histogram-matching harmonization baseline")
    ap.add_argument("--src", required=True, help="source cohort dir (e.g. upenn)")
    ap.add_argument("--ref_dir", required=True, help="reference cohort dir (e.g. brats)")
    ap.add_argument("--ref_id", default="", help="reference subject id (default: first in ref_dir)")
    ap.add_argument("--ids", required=True, help="source subject ids to match, one per line")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref_subj = (
        Path(a.ref_dir) / a.ref_id if a.ref_id else Path(sorted(glob.glob(a.ref_dir + "/*"))[0])
    )
    refs = {m: _load(ref_subj / f"{m}.nii.gz") for m in MODS}
    print(f"reference: {ref_subj.name}")

    with open(a.ids) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    aff = np.eye(4)
    n = 0
    for sid in ids:
        s = Path(a.src) / sid
        if not (s / "seg.nii.gz").exists():
            continue
        o = Path(a.out) / sid
        o.mkdir(parents=True, exist_ok=True)
        for m in MODS:
            matched = _match_masked(_load(s / f"{m}.nii.gz"), refs[m])
            nib.save(nib.Nifti1Image(matched, aff), str(o / f"{m}.nii.gz"))
        nib.save(nib.load(str(s / "seg.nii.gz")), str(o / "seg.nii.gz"))
        n += 1
    print(f"histogram-matched {n} subjects -> {a.out}")


if __name__ == "__main__":
    main()
