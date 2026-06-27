#!/usr/bin/env python3
"""
paired significance + base-baseline summary for ext a (tier c rigor).

reads the seeded, per-subject harmonization eval jsons and reports:
  - base sa-cyclegan vs the lambda arms (FID_A2B, struct ssim, domain-clf);
  - paired wilcoxon on per-subject structure-preservation SSIM: hybrid (0.5) vs
    cycle-only (0.0), and hybrid vs base, over their common subjects.

usage:
    python paired_stats.py --eval_dir ~/neuroscope/experiments/ext_a_eval \
        --out ~/neuroscope/experiments/ext_a_results/paired_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SK = "masked_windowed_ssim_structure_A2B"


def load(eval_dir: Path, label: str):
    p = eval_dir / f"eval_{label}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def persubj(d):
    return d.get(SK, {}).get("per_subject", {}) if d else {}


def paired(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    av = np.array([a[s] for s in common])
    bv = np.array([b[s] for s in common])
    p = float(wilcoxon(av, bv).pvalue) if len(common) > 1 and np.any(av != bv) else float("nan")
    return {
        "n": len(common),
        "mean_hybrid": float(av.mean()),
        "mean_other": float(bv.mean()),
        "delta": float((av - bv).mean()),
        "wilcoxon_p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ed = Path(args.eval_dir)

    labels = ["0.0", "0.1", "0.5", "1.0", "2.0"]
    arms = {L: load(ed, f"lambda{L}") for L in labels}
    base = load(ed, "base")

    def summ(d):
        if not d:
            return None
        return {
            "fid_A2B": d.get("fid_A2B_avg"),
            "struct_ssim": d[SK]["mean"],
            "domain_clf_harm": d["domain_classifier"]["acc_harmonized"],
        }

    res = {
        "baseline": summ(base),
        "arms": {L: summ(arms[L]) for L in labels},
        "paired_struct_ssim": {},
    }
    h = persubj(arms["0.5"])
    if h and persubj(arms["0.0"]):
        res["paired_struct_ssim"]["hybrid_vs_cycleonly"] = paired(h, persubj(arms["0.0"]))
    if h and persubj(base):
        res["paired_struct_ssim"]["hybrid_vs_base"] = paired(h, persubj(base))

    Path(args.out).write_text(json.dumps(res, indent=2, default=str))

    print("=== base vs arms (FID_A2B / struct-ssim / domain-clf harm) ===")
    for name, s in [("base", res["baseline"]), *[(f"l{L}", res["arms"][L]) for L in labels]]:
        if s:
            fid = s["fid_A2B"]
            fid_s = f"{fid:.1f}" if fid is not None else "--"
            print(
                f"  {name:5s}: FID {fid_s:>6s}  struct {s['struct_ssim']:.3f}  domclf {s['domain_clf_harm']:.3f}"
            )
    print("=== paired struct-ssim wilcoxon ===")
    for k, v in res["paired_struct_ssim"].items():
        print(
            f"  {k}: n={v['n']} hybrid {v['mean_hybrid']:.3f} vs {v['mean_other']:.3f} "
            f"(Δ={v['delta']:+.3f}, p={v['wilcoxon_p']:.2e})"
        )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
