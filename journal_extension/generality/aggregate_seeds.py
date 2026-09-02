#!/usr/bin/env python3
"""aggregate multi-seed generality results into a mean +/- std frontier (publication-grade error bars).

per-seed eval jsons are tagged: raw / colormatch / cyclegan (deterministic, seed-independent) and
sdedit_s{SS}_seed{K} / sdedit_s{SS}_empty_seed{K} (diffusion, strength SS/100, prompt target/empty, seed K).
groups the SDEdit points by (strength, empty?) across seeds and reports mean +/- std of FID and mIoU.

emits benchmark_seeds.md (human table) and benchmark_seeds.json (machine, for the error-bar figure).

usage: aggregate_seeds.py --results <dir> [--out <dir>]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

DET = {"raw": "baseline", "colormatch": "non-learned", "cyclegan": "learned-GAN"}
SD = re.compile(r"^sdedit_s(\d+)(_empty)?_seed(\d+)$")
CN = re.compile(r"^controlnet_seed(\d+)$")  # structure-preserving diffusion baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.results

    det: dict[str, dict] = {}
    groups: dict[tuple[float, bool], dict[str, list[float]]] = {}
    cn: dict[str, list[float]] = {"FID": [], "mIoU": []}
    for p in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.startswith("benchmark"):
            continue
        with open(p) as f:
            d = json.load(f)
        fid, miou = d.get("FID"), d.get("mIoU")
        if base in DET:
            det[base] = {
                "family": DET[base],
                "FID": fid,
                "mIoU": miou,
                "n_images": d.get("n_images"),
            }
            continue
        if CN.match(base):
            if fid is not None:
                cn["FID"].append(fid)
            if miou is not None:
                cn["mIoU"].append(miou)
            continue
        m = SD.match(base)
        if not m:
            continue
        strength = int(m.group(1)) / 100
        empty = m.group(2) is not None
        g = groups.setdefault((strength, empty), {"FID": [], "mIoU": []})
        if fid is not None:
            g["FID"].append(fid)
        if miou is not None:
            g["mIoU"].append(miou)

    raw_miou = det.get("raw", {}).get("mIoU")

    def stat(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return None, None
        return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)

    sd_rows = []
    for (strength, empty), g in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        fm, fs = stat(g["FID"])
        mm, ms = stat(g["mIoU"])
        sd_rows.append(
            {
                "strength": strength,
                "empty_prompt": empty,
                "FID_mean": fm,
                "FID_std": fs,
                "mIoU_mean": mm,
                "mIoU_std": ms,
                "n_seeds": max(len(g["FID"]), len(g["mIoU"])),
            }
        )

    cfm, cfs = stat(cn["FID"])
    cmm, cms = stat(cn["mIoU"])
    cn_row = (
        {
            "FID_mean": cfm,
            "FID_std": cfs,
            "mIoU_mean": cmm,
            "mIoU_std": cms,
            "n_seeds": max(len(cn["FID"]), len(cn["mIoU"])),
        }
        if (cn["FID"] or cn["mIoU"])
        else None
    )

    # markdown
    md = ["# generality benchmark (multi-seed): fidelity vs utility, mean +/- std", ""]
    md.append(
        "GTA5->Cityscapes, N per condition, frozen SegFormer-b4 + clean-fid; SDEdit over seeds."
    )
    md.append("")
    md.append("| condition | family | FID (down) | mIoU (up) | seeds |")
    md.append("|---|---|---:|---:|---:|")
    for tag in ("raw", "colormatch", "cyclegan"):
        if tag in det:
            v = det[tag]
            fid = f"{v['FID']:.2f}" if v["FID"] is not None else "-"
            miou = f"{v['mIoU']:.4f}" if v["mIoU"] is not None else "-"
            md.append(f"| {tag} | {v['family']} | {fid} | {miou} | 1 (det.) |")
    for r in sd_rows:
        lbl = f"SDEdit 0.{int(r['strength'] * 100):02d}" + (" empty" if r["empty_prompt"] else "")
        fid = f"{r['FID_mean']:.2f} ± {r['FID_std']:.2f}" if r["FID_mean"] is not None else "-"
        miou = f"{r['mIoU_mean']:.4f} ± {r['mIoU_std']:.4f}" if r["mIoU_mean"] is not None else "-"
        md.append(f"| {lbl} | learned-diffusion | {fid} | {miou} | {r['n_seeds']} |")
    if cn_row:
        fid = (
            f"{cn_row['FID_mean']:.2f} ± {cn_row['FID_std']:.2f}"
            if cn_row["FID_mean"] is not None
            else "-"
        )
        miou = (
            f"{cn_row['mIoU_mean']:.4f} ± {cn_row['mIoU_std']:.4f}"
            if cn_row["mIoU_mean"] is not None
            else "-"
        )
        md.append(
            f"| ControlNet-Canny (structure-preserving) | learned-diffusion+struct | {fid} | {miou} | {cn_row['n_seeds']} |"
        )
    with open(os.path.join(out, "benchmark_seeds.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    with open(os.path.join(out, "benchmark_seeds.json"), "w") as f:
        json.dump(
            {"deterministic": det, "sdedit": sd_rows, "controlnet": cn_row, "raw_mIoU": raw_miou},
            f,
            indent=2,
        )

    print("\n".join(md))
    print(f"\nwrote benchmark_seeds.md / benchmark_seeds.json -> {out}")


if __name__ == "__main__":
    main()
