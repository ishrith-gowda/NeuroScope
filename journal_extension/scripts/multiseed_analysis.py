#!/usr/bin/env python3
"""
2-seed analysis for ext a: combine the seed-42 and seed-1 arms into mean+-std per
lambda (FID_A2B, struct ssim, domain-clf), an error-bar sweep figure, and the
optimum verdict -- is lambda=0.5 uniquely best, or a flat 0.5-1.0 region?

reads <eval_dir>/eval_lambda{L}.json (seed 42) and eval_lambda{L}_seed1.json,
writes fig_ext_a_multiseed.{pdf,png}, table_ext_a_multiseed.tex, multiseed.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SK = "masked_windowed_ssim_structure_A2B"
LAMS = ["0.0", "0.1", "0.5", "1.0", "2.0"]  # strings matching the eval json filenames


def load(eval_dir: Path, label: str):
    p = eval_dir / f"eval_{label}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    ed = Path(args.eval_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = {}
    for L in LAMS:
        s42 = load(ed, f"lambda{L}")
        s1 = load(ed, f"lambda{L}_seed1")
        if not (s42 and s1):
            continue
        fid = [s42["fid_A2B_avg"], s1["fid_A2B_avg"]]
        ss = [s42[SK]["mean"], s1[SK]["mean"]]
        dc = [s42["domain_classifier"]["acc_harmonized"], s1["domain_classifier"]["acc_harmonized"]]
        rows[float(L)] = {
            "fid_mean": float(np.mean(fid)),
            "fid_std": float(np.std(fid)),
            "ssim_mean": float(np.mean(ss)),
            "ssim_std": float(np.std(ss)),
            "dc_mean": float(np.mean(dc)),
            "fid_seeds": fid,
            "ssim_seeds": ss,
        }

    keys = list(rows)
    fm = [rows[x]["fid_mean"] for x in keys]
    fs = [rows[x]["fid_std"] for x in keys]
    sm = [rows[x]["ssim_mean"] for x in keys]
    ss_ = [rows[x]["ssim_std"] for x in keys]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.errorbar(keys, fm, yerr=fs, marker="s", capsize=4, color="#1D4ED8")
    a1.set_xlabel(r"$\lambda_{\mathrm{NCE}}$")
    a1.set_title(r"FID$_{A\to B}$ (mean$\pm$std, 2 seeds)")
    a1.grid(alpha=0.3)
    a2.errorbar(keys, sm, yerr=ss_, marker="^", capsize=4, color="#059669")
    a2.set_xlabel(r"$\lambda_{\mathrm{NCE}}$")
    a2.set_title(r"struct.\ SSIM (mean$\pm$std, 2 seeds)")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_ext_a_multiseed.pdf")
    fig.savefig(out / "fig_ext_a_multiseed.png", dpi=150)
    plt.close(fig)

    best = min(rows, key=lambda x: rows[x]["fid_mean"])
    bm, bs = rows[best]["fid_mean"], rows[best]["fid_std"]
    flat = [x for x in rows if (rows[x]["fid_mean"] - bm) <= (rows[x]["fid_std"] + bs)]

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\lambda_{\mathrm{NCE}}$ & FID$_{A\to B}$ & struct.\ SSIM & dom.-clf \\",
        r"\midrule",
    ]
    for x in keys:
        r = rows[x]
        bold = x == best
        cell = f"{r['fid_mean']:.1f}$\\pm${r['fid_std']:.1f}"
        cell = r"\textbf{" + cell + "}" if bold else cell
        lines.append(
            f"{x:g} & {cell} & {r['ssim_mean']:.3f}$\\pm${r['ssim_std']:.3f} & {r['dc_mean']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_ext_a_multiseed.tex").write_text("\n".join(lines) + "\n")
    (out / "multiseed.json").write_text(json.dumps(rows, indent=2))

    print("=== 2-seed FID_A2B (mean+-std) | struct ssim ===")
    for x in keys:
        r = rows[x]
        print(
            f"  lambda{x:g}: FID {r['fid_mean']:.1f}+-{r['fid_std']:.1f}  "
            f"struct {r['ssim_mean']:.3f}+-{r['ssim_std']:.3f}"
        )
    print(
        f"best FID: lambda{best:g} ({bm:.1f}); statistically-flat region: {[f'{x:g}' for x in flat]}"
    )
    print(f"wrote fig_ext_a_multiseed + table -> {out}")


if __name__ == "__main__":
    main()
