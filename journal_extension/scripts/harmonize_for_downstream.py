#!/usr/bin/env python3
"""
correct harmonization for the downstream segmentation eval.

the original eval_downstream.generate_harmonized_data corrupted the input to the
generator three ways vs the training pipeline:
  (1) normalization: it fed [-1,1] (x/vmax*2-1); the model trains/expects [0,1]
      (the on-disk preprocessed data is already [0,1]).
  (2) modality order: it used [t1,t1gd,flair,t2]; training uses [t1,t1gd,t2,flair].
  (3) 2.5d interleaving: it was modality-major ([t1x3,t1gdx3,...]); training is
      slice-major ([s-1:(t1,t1gd,t2,flair), s:..., s+1:...]).
any one corrupts the harmonized output; together they very likely manufactured
the -19% dice "regression" reported in ext d.

this script harmonizes by reusing the LITERAL training dataset (MRIDataset25D),
so channel order, normalization, and 2.5d interleaving are correct by
construction. output is clamped to [0,1] (the data domain; the generator output
is ~[0,1] with minor negative artifacts). harmonized volumes (128x128, [0,1])
are saved per modality with the correct filenames, plus a nearest-resized seg,
so eval_downstream's BraTSSliceDataset can consume them via --harmonized_*_dir.

usage:
    python harmonize_for_downstream.py --checkpoint <ext_a_ckpt> \
        --site_dir ~/neuroscope/preprocessed/upenn --direction B2A \
        --output_dir ~/neuroscope/experiments/ext_a_downstream/harm_upenn_to_brats
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from neuroscope.models.architectures.sa_cyclegan_25d import (  # noqa: E402
    SACycleGAN25DConfig, create_model,
)
from neuroscope.data.datasets.dataset_25d import MRIDataset25D  # noqa: E402

# output channel order MUST match MRIDataset25D.MODALITIES (training)
MOD_OUT = ["t1", "t1gd", "t2", "flair"]


def load_generator(ckpt_path: str, device: str):
    cfg = SACycleGAN25DConfig(ngf=64, ndf=64, n_residual_blocks=9,
                              attention_layers=(3, 4, 5), nce_feature_layers=(2, 5))
    m = create_model(cfg).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for p in ("_orig_mod.", "module."):
            nk = nk.replace(p, "")
        cleaned[nk] = v
    missing, unexpected = m.load_state_dict(cleaned, strict=False)
    print(f"loaded generator: missing={len(missing)} unexpected={len(unexpected)} epoch={ck.get('epoch','?')}")
    return m


@torch.no_grad()
def harmonize_site(model, direction: str, site_dir: str, out_dir: str,
                   device: str, batch: int = 32, slice_range=(30, 125)):
    import nibabel as nib
    from skimage.transform import resize as skresize

    G = model.G_A2B if direction == "A2B" else model.G_B2A
    ds = MRIDataset25D(root_dir=site_dir, slice_range=slice_range,
                       image_size=(128, 128), cache_volumes=True)

    by_subj = defaultdict(list)
    for i, (subj_idx, sl) in enumerate(ds.samples):
        by_subj[subj_idx].append((i, sl))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)

    for n, (subj_idx, items) in enumerate(by_subj.items()):
        subj = ds.subjects[subj_idx]
        if not (subj / "seg.nii.gz").exists():
            continue  # downstream eval only uses seg-bearing subjects; skip the rest
        vol = ds._load_volume(subj)          # [4, d, h, w]
        D = vol.shape[1]
        hv = np.zeros((4, 128, 128, D), dtype=np.float32)

        imgs, slis = [], []

        def flush():
            if not imgs:
                return
            x = torch.stack(imgs).to(device)             # [b,12,128,128]
            y = G(x).clamp(0, 1).cpu().numpy()           # [b,4,128,128]
            for j, sl in enumerate(slis):
                hv[:, :, :, sl] = y[j]
            imgs.clear()
            slis.clear()

        for i, sl in items:
            imgs.append(ds[i]["image"])
            slis.append(sl)
            if len(imgs) >= batch:
                flush()
        flush()

        # fill out-of-range edge slices with the nearest harmonized slice
        in_range = sorted(sl for _, sl in items)
        if in_range:
            lo, hi = in_range[0], in_range[-1]
            for s in range(0, lo):
                hv[:, :, :, s] = hv[:, :, :, lo]
            for s in range(hi + 1, D):
                hv[:, :, :, s] = hv[:, :, :, hi]

        osub = out / subj.name
        osub.mkdir(exist_ok=True)
        for ci, mod in enumerate(MOD_OUT):
            nib.save(nib.Nifti1Image(hv[ci], affine), str(osub / f"{mod}.nii.gz"))

        # nearest-resized seg at 128, original label values preserved (remap done downstream)
        seg = nib.load(str(subj / "seg.nii.gz")).get_fdata()
        segr = np.zeros((128, 128, seg.shape[2]), dtype=np.int16)
        for s in range(seg.shape[2]):
            segr[:, :, s] = skresize(seg[:, :, s], (128, 128), order=0,
                                     preserve_range=True, anti_aliasing=False).astype(np.int16)
        nib.save(nib.Nifti1Image(segr, affine), str(osub / "seg.nii.gz"))

        if (n + 1) % 25 == 0:
            print(f"  harmonized {n + 1}/{len(by_subj)} subjects")

    print(f"done: {len(by_subj)} subjects -> {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--site_dir", required=True)
    p.add_argument("--direction", required=True, choices=["A2B", "B2A"],
                   help="A2B = harmonize this site with G_A2B; B2A = with G_B2A")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_generator(args.checkpoint, dev)
    harmonize_site(model, args.direction, args.site_dir, args.output_dir, dev, batch=args.batch)
