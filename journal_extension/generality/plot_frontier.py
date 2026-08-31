#!/usr/bin/env python3
"""headline figure for the generality paper: the fidelity(FID)-vs-utility(mIoU) plane.

learned translation (CycleGAN + the SDEdit strength frontier) trades utility for fidelity and never enters
the high-utility region; the non-learned baseline and raw sit at high utility. the SDEdit points, connected
by strength, trace the tradeoff continuously -- the visual core of "fidelity is not utility".

usage: plot_frontier.py --results <results_n200 dir> [--out <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results: str) -> dict[str, dict]:
    out = {}
    for p in glob.glob(os.path.join(results, "*.json")):
        if os.path.basename(p).startswith("benchmark"):
            continue
        with open(p) as f:
            d = json.load(f)
        if d.get("FID") is not None and d.get("mIoU") is not None:
            out[d.get("tag", os.path.splitext(os.path.basename(p))[0])] = d
    return out


def strength_of(tag: str) -> float | None:
    for part in tag.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            return int(part[1:]) / 100
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.results
    d = load(a.results)

    gan, diff, nonl, base = "#bb463c", "#0e7c7b", "#2f7d5b", "#586170"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 11,
            "axes.linewidth": 0.9,
            "figure.dpi": 200,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    # sdedit strength frontier (target prompt), connected by increasing strength
    sd = sorted(
        [
            (strength_of(t), v)
            for t, v in d.items()
            if t.startswith("sdedit") and "empty" not in t and strength_of(t)
        ],
        key=lambda x: x[0] or 0.0,
    )
    if sd:
        xs = [v["FID"] for _, v in sd]
        ys = [v["mIoU"] for _, v in sd]
        ax.plot(xs, ys, "-", color=diff, lw=1.6, zorder=2, alpha=0.9)
        ax.scatter(
            xs,
            ys,
            s=70,
            color=diff,
            edgecolor="white",
            lw=1.2,
            zorder=3,
            label="SDEdit (diffusion), strength sweep",
        )
        for s, v in sd:
            ax.annotate(
                f"{s:.2f}",
                (v["FID"], v["mIoU"]),
                textcoords="offset points",
                xytext=(7, -3),
                fontsize=8.5,
                color=diff,
            )

    # empty-prompt ablation point, if present
    if "sdedit_s055_empty" in d:
        v = d["sdedit_s055_empty"]
        ax.scatter(
            [v["FID"]],
            [v["mIoU"]],
            s=80,
            marker="D",
            facecolor="none",
            edgecolor=diff,
            lw=1.6,
            zorder=3,
            label="SDEdit 0.55, empty prompt (ablation)",
        )

    def pt(tag, color, marker, label, size=150):
        if tag in d:
            v = d[tag]
            ax.scatter(
                [v["FID"]],
                [v["mIoU"]],
                s=size,
                marker=marker,
                color=color,
                edgecolor="white",
                lw=1.3,
                zorder=4,
                label=label,
            )

    pt("cyclegan", gan, "*", "CycleGAN (adversarial)", size=240)
    pt("colormatch", nonl, "s", "color match (non-learned)")
    pt("raw", base, "o", "raw GTA5 (no translation)")

    # "better" guide: ideal is high mIoU + low FID (upper-left)
    if "raw" in d:
        ax.axhline(d["raw"]["mIoU"], color=base, lw=0.7, ls=":", alpha=0.6, zorder=1)
        ax.annotate(
            "raw utility",
            (ax.get_xlim()[1], d["raw"]["mIoU"]),
            color=base,
            fontsize=8,
            va="bottom",
            ha="right",
        )

    ax.set_xlabel("FID  (fidelity to Cityscapes — lower is better  ←)")
    ax.set_ylabel("mIoU  (downstream utility, frozen SegFormer — higher is better  ↑)")
    ax.set_title(
        "Fidelity is not utility: GTA5→Cityscapes\nlearned translation trades utility for fidelity; non-learned preserves it",
        fontsize=11.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8.6, loc="lower right", ncol=1)
    # the ideal corner (high utility + high fidelity) is empty -- no learned method reaches it
    ax.annotate(
        "ideal\n(low FID, high mIoU)\n— unreached by learned",
        xy=(0.03, 0.9),
        xycoords="axes fraction",
        fontsize=8.2,
        color=nonl,
        va="top",
        ha="left",
        style="italic",
    )
    ax.grid(True, color="#00000010", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"frontier.{ext}"), bbox_inches="tight")
    print(f"wrote frontier.pdf / frontier.png -> {out}")
    print("points:", {t: (round(v["FID"], 1), round(v["mIoU"], 3)) for t, v in d.items()})


if __name__ == "__main__":
    main()
