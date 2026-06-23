#!/usr/bin/env python3
"""
turn the corrected-eval JSONs into paper-ready ext a results: latex tables +
comparison figures. graceful with partial data (run repeatedly as sweep arms
land). reads:
  - <eval_dir>/eval_lambda*.json  (harmonization metrics per lambda)
  - <downstream_json>             (corrected cross-site dice/hd95; optional)
writes booktabs tables (.tex) and figures (.pdf/.png) to <out_dir>.

usage (on node or locally against cluster_backup):
    python generate_ext_a_results.py \
      --eval_dir ~/neuroscope/experiments/ext_a_eval \
      --downstream_json ~/neuroscope/experiments/ext_a_downstream/lambda0.5/downstream_corrected.json \
      --out_dir ~/neuroscope/experiments/ext_a_results
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODS = ["FLAIR", "T1", "T1ce", "T2"]


def load_eval_arms(eval_dir: Path):
    arms = {}
    for f in sorted(glob.glob(str(eval_dir / "eval_lambda*.json"))):
        m = re.search(r"lambda([0-9.]+)\.json", f)
        if not m:
            continue
        with open(f) as fh:
            arms[float(m.group(1))] = json.load(fh)
    return dict(sorted(arms.items()))


def fmt(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def kid_avg(r, direction):
    """mean KID over the 4 modalities for a direction (A2B/B2A), x1000 for readability."""
    km = r.get("kid_mean", {})
    vals = [km[f"{direction}_{m}"] for m in MODS if f"{direction}_{m}" in km]
    return 1000 * sum(vals) / len(vals) if vals else None


def write_harmonization_table(arms, out_dir: Path):
    if not arms:
        return
    best_lam = min(arms, key=lambda L: arms[L].get("fid_A2B_avg", 1e9))  # FID-optimal arm
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$\lambda_{\mathrm{NCE}}$ & dom.-clf raw$\to$harm & FID$_{A\to B}$ & "
        r"FID$_{B\to A}$ & KID$_{A\to B}(\times10^3)$ & struct.\ SSIM & cycle SSIM \\",
        r"\midrule",
    ]
    for L, r in arms.items():
        dc = r["domain_classifier"]
        cells = [
            f"{L:g}",
            f"{fmt(dc['acc_raw'])}$\\to${fmt(dc['acc_harmonized'])}",
            fmt(r.get("fid_A2B_avg"), 1),
            fmt(r.get("fid_B2A_avg"), 1),
            fmt(kid_avg(r, "A2B"), 1),
            fmt(r["masked_windowed_ssim_structure_A2B"]["mean"]),
            fmt(r["masked_windowed_ssim_cycle_A2B"]["mean"]),
        ]
        if best_lam == L:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_ext_a_harmonization.tex").write_text("\n".join(lines) + "\n")
    print(
        f"  wrote table_ext_a_harmonization.tex ({len(arms)} arms; FID-optimal lambda={best_lam:g})"
    )


def write_downstream_table(ds, out_dir: Path):
    if not ds or "harmonization_effect" not in ds:
        return
    FG = "dice_mean_foreground_mean"
    HD = "hd95_mean"
    cond = {
        "A_to_B_direction": ("A_on_rawB", "A_on_harmB"),
        "B_to_A_direction": ("B_on_rawA", "B_on_harmA"),
    }
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"direction & raw Dice & harm.\ Dice & $\Delta$ & raw HD95 & harm HD95 \\",
        r"\midrule",
    ]
    for k, v in ds["harmonization_effect"].items():
        short = k.split(" ")[0]
        rc, hc = cond.get(short, (None, None))
        rhd = fmt(ds.get(rc, {}).get(HD), 1) if rc else "--"
        hhd = fmt(ds.get(hc, {}).get(HD), 1) if hc else "--"
        lines.append(
            f"{short} & {fmt(v['raw'])} & {fmt(v['harmonized'])} & "
            f"{v['delta']:+.3f} & {rhd} & {hhd} \\\\"
        )
    # within-site upper bounds
    if "A_on_rawA" in ds:
        lines += [
            r"\midrule",
            f"within-site brats & \\multicolumn{{5}}{{c}}{{{fmt(ds['A_on_rawA'][FG])}}} \\\\",
            f"within-site upenn & \\multicolumn{{5}}{{c}}{{{fmt(ds['B_on_rawB'][FG])}}} \\\\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_ext_a_downstream.tex").write_text("\n".join(lines) + "\n")
    print("  wrote table_ext_a_downstream.tex")


def fig_lambda_sweep(arms, out_dir: Path):
    if len(arms) < 2:
        print(f"  (lambda-sweep figure needs >=2 arms; have {len(arms)})")
        return
    lams = list(arms.keys())
    dc = [arms[L]["domain_classifier"]["acc_harmonized"] for L in lams]
    fid = [arms[L].get("fid_A2B_avg") for L in lams]
    ssim = [arms[L]["masked_windowed_ssim_cycle_A2B"]["mean"] for L in lams]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(lams, dc, "o-", color="#B45309")
    axes[0].axhline(0.5, ls="--", color="gray", lw=1, label="chance")
    axes[0].set_title("domain-clf acc (harmonized)")
    axes[0].set_xlabel(r"$\lambda_{NCE}$")
    axes[0].legend()
    axes[1].plot(lams, fid, "s-", color="#1D4ED8")
    axes[1].set_title(r"FID$_{A\to B}$ (lower better)")
    axes[1].set_xlabel(r"$\lambda_{NCE}$")
    axes[2].plot(lams, ssim, "^-", color="#059669")
    axes[2].set_title("masked windowed SSIM (cycle)")
    axes[2].set_xlabel(r"$\lambda_{NCE}$")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_a_lambda_sweep.pdf")
    fig.savefig(out_dir / "fig_ext_a_lambda_sweep.png", dpi=150)
    plt.close(fig)
    print("  wrote fig_ext_a_lambda_sweep.{pdf,png}")


def fig_downstream(ds, out_dir: Path):
    if not ds or "harmonization_effect" not in ds:
        return
    dirs = list(ds["harmonization_effect"].keys())
    raw = [ds["harmonization_effect"][d]["raw"] for d in dirs]
    harm = [ds["harmonization_effect"][d]["harmonized"] for d in dirs]
    x = range(len(dirs))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar([i - 0.2 for i in x], raw, 0.4, label="raw (cross-site)", color="#9CA3AF")
    ax.bar([i + 0.2 for i in x], harm, 0.4, label="harmonized", color="#059669")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d.split("_")[0] for d in dirs])
    ax.set_ylabel("cross-site Dice (mean fg)")
    ax.set_title("downstream segmentation: raw vs harmonized")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ext_a_downstream.pdf")
    fig.savefig(out_dir / "fig_ext_a_downstream.png", dpi=150)
    plt.close(fig)
    print("  wrote fig_ext_a_downstream.{pdf,png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True)
    p.add_argument("--downstream_json", default=None)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arms = load_eval_arms(Path(args.eval_dir))
    ds = None
    if args.downstream_json and Path(args.downstream_json).exists():
        with open(args.downstream_json) as f:
            ds = json.load(f)

    print(
        f"loaded {len(arms)} harmonization arm(s): {[f'{L:g}' for L in arms]}; downstream={'yes' if ds else 'no'}"
    )
    write_harmonization_table(arms, out)
    write_downstream_table(ds, out)
    fig_lambda_sweep(arms, out)
    fig_downstream(ds, out)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
