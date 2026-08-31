#!/usr/bin/env python3
"""aggregate per-condition generality results into the headline fidelity-vs-utility table + fid/miou frontier.

reads every <tag>.json ({tag,mIoU,FID,n_images,...}) in a results dir and emits:
  - benchmark_table.md   (human-readable, sorted by FID)
  - benchmark_table.tex  (LaTeX booktabs for the manuscript)
  - benchmark_combined.json (machine-readable, incl. frontier points)

usage: aggregate_generality.py --results <dir> [--out <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# condition family + display label; anything else is inferred as a diffusion strength point
FAMILY = {
    "raw": ("baseline", "raw GTA5 (no translation)"),
    "colormatch": ("non-learned", "color match (histogram)"),
    "cyclegan": ("learned-GAN", "CycleGAN (adversarial)"),
}


def _family(tag: str) -> tuple[str, str]:
    if tag in FAMILY:
        return FAMILY[tag]
    if tag.startswith("sdedit"):
        empty = "empty" in tag
        s = ""
        for part in tag.split("_"):
            if part.startswith("s") and part[1:].isdigit():
                s = f"{int(part[1:]) / 100:.2f}"
        lbl = f"SDEdit (diffusion) strength {s}" + (", empty prompt" if empty else "")
        return "learned-diffusion", lbl
    return "other", tag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.results

    rows = []
    for p in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        if os.path.basename(p).startswith("benchmark"):
            continue
        with open(p) as f:
            d = json.load(f)
        tag = d.get("tag") or os.path.splitext(os.path.basename(p))[0]
        fam, lbl = _family(tag)
        rows.append(
            {
                "tag": tag,
                "family": fam,
                "label": lbl,
                "FID": d.get("FID"),
                "mIoU": d.get("mIoU"),
                "n_images": d.get("n_images"),
            }
        )

    raw = next((r for r in rows if r["tag"] == "raw"), None)
    raw_miou = raw["mIoU"] if raw else None
    for r in rows:
        r["dmIoU_vs_raw"] = (
            (r["mIoU"] - raw_miou) if (raw_miou is not None and r["mIoU"] is not None) else None
        )

    rows.sort(key=lambda r: (r["FID"] is None, r["FID"] if r["FID"] is not None else 0.0))

    # markdown
    md = [
        "# generality benchmark: fidelity (FID) vs downstream utility (frozen-SegFormer mIoU)",
        "",
    ]
    md.append(
        "GTA5->Cityscapes, N per condition as listed, same frozen SegFormer-b4 evaluator + clean-fid."
    )
    md.append("")
    md.append("| condition | family | FID (down) | mIoU (up) | dmIoU vs raw | N |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        fid = f"{r['FID']:.2f}" if r["FID"] is not None else "-"
        miou = f"{r['mIoU']:.4f}" if r["mIoU"] is not None else "-"
        dm = f"{r['dmIoU_vs_raw']:+.4f}" if r["dmIoU_vs_raw"] is not None else "-"
        md.append(
            f"| {r['label']} | {r['family']} | {fid} | {miou} | {dm} | {r.get('n_images', '-')} |"
        )
    md.append("")
    md.append(
        "**reading:** learned translation (GAN + diffusion) lowers FID but drops mIoU; the non-learned "
        "baseline preserves mIoU. no diffusion strength should pareto-dominate the non-learned point."
    )
    with open(os.path.join(out, "benchmark_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # latex
    tex = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"condition & family & FID $\downarrow$ & mIoU $\uparrow$ & $\Delta$mIoU \\",
        r"\midrule",
    ]
    for r in rows:
        fid = f"{r['FID']:.2f}" if r["FID"] is not None else "-"
        miou = f"{r['mIoU']:.4f}" if r["mIoU"] is not None else "-"
        dm = f"{r['dmIoU_vs_raw']:+.4f}" if r["dmIoU_vs_raw"] is not None else "-"
        tex.append(f"{r['label']} & {r['family']} & {fid} & {miou} & {dm} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "benchmark_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    frontier = sorted(
        [
            {"strength": r["label"], "FID": r["FID"], "mIoU": r["mIoU"]}
            for r in rows
            if r["family"] == "learned-diffusion"
        ],
        key=lambda x: (x["FID"] is None, x["FID"] or 0.0),
    )
    with open(os.path.join(out, "benchmark_combined.json"), "w") as f:
        json.dump({"rows": rows, "diffusion_frontier": frontier, "raw_mIoU": raw_miou}, f, indent=2)

    print("\n".join(md))
    print(f"\nwrote benchmark_table.md / .tex / benchmark_combined.json -> {out}")


if __name__ == "__main__":
    main()
