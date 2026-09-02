#!/usr/bin/env python3
"""publication figure with error bars: the fidelity(FID)-vs-utility(mIoU) plane, multi-seed.

reads benchmark_seeds.json (from aggregate_seeds.py): deterministic conditions (raw/colormatch/cyclegan)
plus SDEdit strength points with mean/std over seeds. draws the SDEdit strength frontier with x/y error
bars, the empty-prompt ablation, and the deterministic reference points. same story as plot_frontier.py
but with the seed variability made explicit -- the version for the paper.

usage: plot_frontier_seeds.py --seeds-json <benchmark_seeds.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-json", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.dirname(a.seeds_json)
    with open(a.seeds_json) as f:
        d = json.load(f)

    gan, diff, nonl, base = "#bb463c", "#0e7c7b", "#2f7d5b", "#586170"
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11, "figure.dpi": 200}
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    sd = [r for r in d.get("sdedit", []) if not r["empty_prompt"] and r["FID_mean"] is not None]
    sd.sort(key=lambda r: r["strength"])
    if sd:
        xs = [r["FID_mean"] for r in sd]
        ys = [r["mIoU_mean"] for r in sd]
        xe = [r["FID_std"] or 0.0 for r in sd]
        ye = [r["mIoU_std"] or 0.0 for r in sd]
        ax.plot(xs, ys, "-", color=diff, lw=1.5, zorder=2, alpha=0.9)
        ax.errorbar(
            xs,
            ys,
            xerr=xe,
            yerr=ye,
            fmt="o",
            color=diff,
            ecolor=diff,
            elinewidth=1.1,
            capsize=3,
            ms=7,
            mec="white",
            mew=1.1,
            zorder=3,
            label=f"SDEdit (diffusion), strength sweep (n={sd[0]['n_seeds']} seeds)",
        )
        for r in sd:
            ax.annotate(
                f"{r['strength']:.2f}",
                (r["FID_mean"], r["mIoU_mean"]),
                textcoords="offset points",
                xytext=(8, -2),
                fontsize=8.5,
                color=diff,
            )

    for r in d.get("sdedit", []):
        if r["empty_prompt"] and r["FID_mean"] is not None:
            ax.errorbar(
                [r["FID_mean"]],
                [r["mIoU_mean"]],
                xerr=[r["FID_std"] or 0.0],
                yerr=[r["mIoU_std"] or 0.0],
                fmt="D",
                mfc="none",
                mec=diff,
                ecolor=diff,
                capsize=3,
                ms=8,
                mew=1.5,
                zorder=3,
                label=f"SDEdit {r['strength']:.2f}, empty prompt (ablation)",
            )

    det = d.get("deterministic", {})
    marks = {
        "cyclegan": (gan, "*", "CycleGAN (adversarial)", 240),
        "colormatch": (nonl, "s", "color match (non-learned)", 150),
        "raw": (base, "o", "raw GTA5 (no translation)", 150),
    }
    for tag, (color, mk, lbl, sz) in marks.items():
        if tag in det and det[tag].get("FID") is not None:
            ax.scatter(
                [det[tag]["FID"]],
                [det[tag]["mIoU"]],
                s=sz,
                marker=mk,
                color=color,
                edgecolor="white",
                lw=1.3,
                zorder=4,
                label=lbl,
            )

    cn = d.get("controlnet")
    if cn and cn.get("FID_mean") is not None:
        ax.errorbar(
            [cn["FID_mean"]],
            [cn["mIoU_mean"]],
            xerr=[cn["FID_std"] or 0.0],
            yerr=[cn["mIoU_std"] or 0.0],
            fmt="^",
            color="#6a4c93",
            ecolor="#6a4c93",
            capsize=3,
            ms=12,
            mec="white",
            mew=1.2,
            zorder=4,
            label=f"ControlNet-Canny (structure-preserving, n={cn['n_seeds']})",
        )

    if d.get("raw_mIoU") is not None:
        ax.axhline(d["raw_mIoU"], color=base, lw=0.7, ls=":", alpha=0.6, zorder=1)

    ax.annotate(
        "ideal\n(low FID, high mIoU)\n— unreached by learned",
        xy=(0.03, 0.9),
        xycoords="axes fraction",
        fontsize=8.2,
        color=nonl,
        va="top",
        style="italic",
    )
    ax.set_xlabel("FID  (fidelity to Cityscapes — lower is better  ←)")
    ax.set_ylabel("mIoU  (downstream utility, frozen SegFormer — higher is better  ↑)")
    ax.set_title(
        "Fidelity is not utility: GTA5→Cityscapes (mean ± std over seeds)",
        fontsize=11.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8.4, loc="lower right")
    ax.grid(True, color="#00000010", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"frontier_seeds.{ext}"), bbox_inches="tight")
    print(f"wrote frontier_seeds.pdf / .png -> {out}")


if __name__ == "__main__":
    main()
