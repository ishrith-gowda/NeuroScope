#!/usr/bin/env python3
"""
corrected cross-site downstream segmentation eval (the clean §11 protocol).

reuses eval_downstream.py's proven components (BraTSSliceDataset, 2D U-Net,
train_segmentation, evaluate_segmentation -> per-class + WT/TC/ET dice + HD95,
subject-level), but runs the canonical, interpretable harmonization experiment:

  train a tumor segmenter on site A (raw); evaluate it on
    (i)  site B raw                      -> cross-site baseline (domain gap)
    (ii) site B harmonized to A (G_B2A)  -> gap reduced by harmonization
  harmonization HELPS iff dice(ii) > dice(i). plus the reverse (train B, test A).

the harmonized dirs MUST be produced by harmonize_for_downstream.py (correct
channel order / [0,1] norm / slice-major 2.5d), NOT the buggy
eval_downstream.generate_harmonized_data.

usage:
    python eval_downstream_corrected.py \
      --brats_dir ~/neuroscope/preprocessed/brats \
      --upenn_dir ~/neuroscope/preprocessed/upenn \
      --harm_upenn_to_brats <dir from harmonize_for_downstream --site upenn --dir B2A> \
      --harm_brats_to_upenn <dir from harmonize_for_downstream --site brats --dir A2B> \
      --output_dir ~/neuroscope/experiments/ext_a_downstream/lambda0.5 \
      --seg_epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_downstream import (
    DownstreamEvaluator,
    get_subject_ids,
    setup_torch_performance,
    split_subjects,
)

FG = "dice_mean_foreground_mean"
WT = "region_wt_mean"


def _summary(r):
    return {
        k: r.get(k)
        for k in (
            FG,
            "dice_mean_foreground_std",
            WT,
            "region_tc_mean",
            "region_et_mean",
            "hd95_mean",
            "hd95_std",
        )
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brats_dir", required=True)
    p.add_argument("--upenn_dir", required=True)
    p.add_argument("--harm_upenn_to_brats", required=True, help="upenn harmonized to brats (G_B2A)")
    p.add_argument("--harm_brats_to_upenn", required=True, help="brats harmonized to upenn (G_A2B)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seg_epochs", type=int, default=40)
    p.add_argument("--seg_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=12)
    args = p.parse_args()

    setup_torch_performance()
    ev = DownstreamEvaluator(
        output_dir=args.output_dir,
        n_classes=4,
        seg_epochs=args.seg_epochs,
        seg_lr=args.seg_lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=128,
        use_amp=True,
    )

    ids_a = get_subject_ids(args.brats_dir)
    ids_b = get_subject_ids(args.upenn_dir)
    train_a, test_a = split_subjects(ids_a)
    train_b, test_b = split_subjects(ids_b)
    print(
        f"brats: {len(ids_a)} ({len(train_a)} train / {len(test_a)} test) | "
        f"upenn: {len(ids_b)} ({len(train_b)} train / {len(test_b)} test)"
    )

    results: dict = {
        "splits": {
            "brats_train": len(train_a),
            "brats_test": len(test_a),
            "upenn_train": len(train_b),
            "upenn_test": len(test_b),
        }
    }

    # ---- forward: train on raw brats (A), test raw upenn vs upenn->brats ----
    print("\n[A] training segmenter on raw brats ...")
    S_A, _ = ev.train_segmentation(
        ev._make_loader(args.brats_dir, train_a, shuffle=True), model_name="segA"
    )
    print("[A] eval within-site (brats test) ...")
    results["A_on_rawA"] = _summary(
        ev.evaluate_segmentation(
            S_A, ev._make_loader(args.brats_dir, test_a, shuffle=False, require_tumor=False)
        )
    )
    print("[A] eval cross-site raw (upenn) ...")
    results["A_on_rawB"] = _summary(
        ev.evaluate_segmentation(
            S_A, ev._make_loader(args.upenn_dir, test_b, shuffle=False, require_tumor=False)
        )
    )
    print("[A] eval cross-site harmonized (upenn->brats) ...")
    results["A_on_harmB"] = _summary(
        ev.evaluate_segmentation(
            S_A,
            ev._make_loader(args.harm_upenn_to_brats, test_b, shuffle=False, require_tumor=False),
        )
    )

    # ---- reverse: train on raw upenn (B), test raw brats vs brats->upenn ----
    print("\n[B] training segmenter on raw upenn ...")
    S_B, _ = ev.train_segmentation(
        ev._make_loader(args.upenn_dir, train_b, shuffle=True), model_name="segB"
    )
    print("[B] eval within-site (upenn test) ...")
    results["B_on_rawB"] = _summary(
        ev.evaluate_segmentation(
            S_B, ev._make_loader(args.upenn_dir, test_b, shuffle=False, require_tumor=False)
        )
    )
    print("[B] eval cross-site raw (brats) ...")
    results["B_on_rawA"] = _summary(
        ev.evaluate_segmentation(
            S_B, ev._make_loader(args.brats_dir, test_a, shuffle=False, require_tumor=False)
        )
    )
    print("[B] eval cross-site harmonized (brats->upenn) ...")
    results["B_on_harmA"] = _summary(
        ev.evaluate_segmentation(
            S_B,
            ev._make_loader(args.harm_brats_to_upenn, test_a, shuffle=False, require_tumor=False),
        )
    )

    # ---- harmonization deltas (the headline) ----
    def delta(harm, raw):
        d = results[harm][FG] - results[raw][FG]
        return {
            "raw": results[raw][FG],
            "harmonized": results[harm][FG],
            "delta": d,
            "relative_pct": 100 * d / max(results[raw][FG], 1e-8),
        }

    results["harmonization_effect"] = {
        "A_to_B_direction (train brats, test upenn)": delta("A_on_harmB", "A_on_rawB"),
        "B_to_A_direction (train upenn, test brats)": delta("B_on_harmA", "B_on_rawA"),
    }
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "seg_epochs": args.seg_epochs,
        "protocol": "train-raw-test-raw-vs-harmonized",
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "downstream_corrected.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n==== DOWNSTREAM (corrected) ====")
    for k, v in results["harmonization_effect"].items():
        print(
            f"  {k}: raw {v['raw']:.4f} -> harmonized {v['harmonized']:.4f} "
            f"({v['delta']:+.4f}, {v['relative_pct']:+.1f}%)"
        )
    print(
        f"  within-site upper bounds: brats {results['A_on_rawA'][FG]:.4f}, "
        f"upenn {results['B_on_rawB'][FG]:.4f}"
    )
    print(f"saved -> {out / 'downstream_corrected.json'}")


if __name__ == "__main__":
    main()
