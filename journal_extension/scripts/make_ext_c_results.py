#!/usr/bin/env python3
"""
ext c results: turn the multi-domain eval jsons into a paper-ready NxN FID
heatmap + a zero-shot panel + a summary table.

reads <eval_dir>/{matrix_fid,domain_confusion,zeroshot}.json and writes
fig_ext_c_matrix.{pdf,png} and table_ext_c.tex to <out_dir>.

usage:
    python make_ext_c_results.py --eval_dir ~/neuroscope/experiments/ext_c/eval_full \
        --out_dir ~/neuroscope/experiments/ext_c/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p: Path):
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

    m = load(ed / "matrix_fid.json")
    dc = load(ed / "domain_confusion.json")
    zs = load(ed / "zeroshot.json")
    names = [s.replace("_", "-") for s in m["domain_names"]]
    mat = np.array(m["matrix_rows_target_cols_source"])
    n = len(names)
    off = mat[~np.eye(n, dtype=bool)]

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    im = ax.imshow(mat, cmap="viridis_r")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    ax.set_title(r"$N{\times}N$ FID (G: source$\to$target)")
    thr = mat.mean()
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] > thr else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)

    zk = [k.replace("_", "-") for k in zs]
    zv = [zs[k] for k in zs]
    axz.bar(range(len(zk)), zv, color="#7C3AED")
    axz.axhline(off.mean(), ls="--", color="gray", label=f"trained cross-site ({off.mean():.0f})")
    axz.set_xticks(range(len(zk)))
    axz.set_xticklabels(zk, rotation=45, ha="right", fontsize=8)
    axz.set_ylabel(r"FID $\to$ tss-06")
    axz.set_title("zero-shot (held-out sites)")
    axz.legend(fontsize=7)
    axz.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_ext_c_matrix.pdf")
    fig.savefig(out / "fig_ext_c_matrix.png", dpi=150)
    plt.close(fig)

    diag = float(np.diag(mat).mean())
    offm = float(off.mean())
    zmean = float(np.mean(zv))
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"metric & value \\",
        r"\midrule",
        f"within-site FID (diagonal) & {diag:.1f} \\\\",
        f"cross-site FID (off-diagonal) & {offm:.1f} \\\\",
        f"zero-shot FID (held-out site) & {zmean:.1f} \\\\",
        f"site-clf acc (real$\\to$harm.; chance {dc['chance']:.2f}) & "
        f"{dc['real_val_acc']:.3f}$\\to${dc['harmonized_to_ref_acc_wrt_original']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (out / "table_ext_c.tex").write_text("\n".join(lines) + "\n")
    print(
        f"within {diag:.1f} | cross {offm:.1f} | zero-shot {zmean:.1f} | "
        f"site-clf {dc['real_val_acc']:.3f}->{dc['harmonized_to_ref_acc_wrt_original']:.3f} "
        f"(chance {dc['chance']})"
    )
    print(f"wrote fig_ext_c_matrix + table_ext_c -> {out}")


if __name__ == "__main__":
    main()
