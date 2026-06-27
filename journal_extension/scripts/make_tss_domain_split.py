#!/usr/bin/env python3
"""
build the tcga tissue-source-site (tss) multi-domain split for ext c.

brats/tcga-gbm subject ids encode the contributing institution as the 2-digit
tss code in tcga-XX-YYYY. we form n_domains real multi-institutional domains from
the largest tss sites and hold out the smaller sites for zero-shot transfer.

writes:
  domain_split.json      domain_name -> [subject ids]   (consumed by train_multi_domain)
  heldout_zeroshot.json  domain_name -> [subject ids]   (held-out sites)
  multi_domain_tss.yaml  training config for train_multi_domain.py

usage:
    python make_tss_domain_split.py --brats_dir ~/neuroscope/preprocessed/brats \
        --out_dir ~/neuroscope/experiments/ext_c --n_domains 4
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

TSS_RE = re.compile(r"TCGA-([0-9A-Za-z]+)-")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--brats_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_domains", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    brats = Path(args.brats_dir).expanduser()
    by_tss: dict[str, list[str]] = defaultdict(list)
    for d in sorted(brats.iterdir()):
        if not d.is_dir():
            continue
        m = TSS_RE.match(d.name)
        if m:
            by_tss[m.group(1)].append(d.name)

    ranked = sorted(by_tss.items(), key=lambda kv: len(kv[1]), reverse=True)
    domains = ranked[: args.n_domains]
    heldout = ranked[args.n_domains :]

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    split = {f"tss_{code}": ids for code, ids in domains}
    (out / "domain_split.json").write_text(json.dumps(split, indent=2))
    (out / "heldout_zeroshot.json").write_text(
        json.dumps({f"tss_{code}": ids for code, ids in heldout}, indent=2)
    )

    domain_names = list(split.keys())
    cfg = {
        "data_dirs": {name: str(brats) for name in domain_names},
        "domain_names": domain_names,
        "domain_split_file": str(out / "domain_split.json"),
        "n_domains": len(domain_names),
        "epochs": args.epochs,
        "batch_size": 16,
        "image_size": 128,
        "lambda_cls": 1.0,
        "lambda_rec": 10.0,
        "num_workers": 8,
        "output_dir": str(out / "runs"),
        "experiment_name": "ext_c_multidomain_tss4",
    }
    (out / "multi_domain_tss.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("domains:", [(n, len(i)) for n, i in split.items()])
    print("heldout (zero-shot):", [(f"tss_{c}", len(i)) for c, i in heldout])
    print(f"wrote {out / 'domain_split.json'}, {out / 'multi_domain_tss.yaml'}")


if __name__ == "__main__":
    main()
