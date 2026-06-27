"""
statistical analysis of extension a's per-slice test results.

operates on results/patchnce/test_results_lambda*.json and produces:

  - wilcoxon signed-rank tests for each metric, all 6 lambda pairs
  - bonferroni-corrected p-values across the 6 comparisons
  - bootstrapped 95 percent confidence intervals on the mean difference
  - cohen's d effect sizes
  - a single latex table summarising all pairwise comparisons
  - a json artifact with full numeric output

usage:
    python statistical_tests.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

LAMBDAS: list[float] = [0.1, 0.5, 1.0, 2.0]
METRICS: list[str] = [
    "ssim_A2B",
    "ssim_B2A",
    "ssim_global_A2B",
    "ssim_global_B2A",
    "psnr_A2B",
    "psnr_B2A",
    "mae_A2B",
    "mae_B2A",
]


def load_per_slice(results_dir: Path) -> dict[float, dict[str, np.ndarray]]:
    out: dict[float, dict[str, np.ndarray]] = {}
    for lam in LAMBDAS:
        path = results_dir / f"test_results_lambda{lam}.json"
        if not path.exists():
            continue
        with open(path) as f:
            payload = json.load(f)
        per_slice = payload.get("per_slice", {})
        out[lam] = {
            m: np.asarray(per_slice.get(m, []), dtype=np.float64) for m in METRICS if m in per_slice
        }
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    s = float(diff.std(ddof=1))
    return float(diff.mean() / s) if s > 0 else 0.0


def bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    samples = rng.choice(n, size=(n_boot, n), replace=True)
    means = diff[samples].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def pairwise_compare(per_slice: dict[float, dict[str, np.ndarray]]) -> list[dict]:
    """run all pairwise comparisons across lambdas for every metric."""
    results: list[dict] = []
    pairs = list(itertools.combinations(sorted(per_slice.keys()), 2))
    for metric in METRICS:
        for la, lb in pairs:
            a = per_slice[la].get(metric)
            b = per_slice[lb].get(metric)
            if a is None or b is None or len(a) == 0 or len(b) == 0 or len(a) != len(b):
                continue
            try:
                w_stat, w_p = stats.wilcoxon(a, b, zero_method="pratt", method="approx")
            except ValueError:
                w_stat, w_p = float("nan"), 1.0
            t_stat, t_p = stats.ttest_rel(a, b)
            d = cohens_d(a, b)
            ci_lo, ci_hi = bootstrap_ci(a - b)
            results.append(
                {
                    "metric": metric,
                    "lambda_a": la,
                    "lambda_b": lb,
                    "n": len(a),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_diff": float((a - b).mean()),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    "ttest_stat": float(t_stat),
                    "ttest_p": float(t_p),
                    "cohens_d": d,
                    "ci95_lower": ci_lo,
                    "ci95_upper": ci_hi,
                }
            )
    return results


def apply_bonferroni(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """add bonferroni-corrected significance flag based on within-metric pairs."""
    by_metric: dict[str, list[dict]] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r)
    for _metric, group in by_metric.items():
        n = len(group)
        adjusted = alpha / max(n, 1)
        for r in group:
            r["bonferroni_alpha"] = adjusted
            r["significant_bonferroni"] = bool(r["wilcoxon_p"] < adjusted)
    return rows


def write_latex(rows: list[dict], out_path: Path) -> None:
    """compact pairwise table for the manuscript."""
    primary_metrics = [
        ("ssim_global_A2B", "SSIM (A$\\to$B, global)"),
        ("ssim_global_B2A", "SSIM (B$\\to$A, global)"),
        ("psnr_A2B", "PSNR (A$\\to$B, dB)"),
        ("psnr_B2A", "PSNR (B$\\to$A, dB)"),
    ]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Pairwise statistical comparison across $\\lambda_{\\mathrm{NCE}}$ "
        "values on the held-out test set (5{,}265 slices). "
        "Wilcoxon signed-rank $p$-values are reported with Bonferroni correction across "
        "the six pairs per metric ($\\alpha_{\\mathrm{adj}}{=}0.0083$). "
        "Significant differences are marked $\\dagger$.}"
    )
    lines.append("\\label{tab:patchnce_pairwise_significance}")
    lines.append("\\begin{tabular}{l l c c c c c}")
    lines.append("\\toprule")
    lines.append(
        "Metric & Pair & Mean A & Mean B & $\\Delta$ Mean & Cohen's $d$ & Wilcoxon $p$ \\\\"
    )
    lines.append("\\midrule")
    for metric_key, metric_name in primary_metrics:
        for r in rows:
            if r["metric"] != metric_key:
                continue
            sig = "$\\dagger$" if r.get("significant_bonferroni") else ""
            lines.append(
                f"{metric_name} & "
                f"$\\lambda${r['lambda_a']} vs $\\lambda${r['lambda_b']} & "
                f"{r['mean_a']:.4f} & {r['mean_b']:.4f} & "
                f"{r['mean_diff']:+.4f} & {r['cohens_d']:+.3f} & "
                f"{r['wilcoxon_p']:.2e}{sig} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    res_dir = repo_root / "journal_extension" / "results" / "patchnce"
    fig_dir = repo_root / "journal_extension" / "figures"

    per_slice = load_per_slice(res_dir)
    if not per_slice:
        print("no per-slice test results found.")
        return

    print(f"loaded per-slice arrays for {len(per_slice)} lambda runs.")

    rows = pairwise_compare(per_slice)
    rows = apply_bonferroni(rows)

    out_path_json = res_dir / "patchnce_statistical_tests.json"
    out_path_json.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_path_json}")

    out_path_tex = fig_dir / "table_patchnce_statistics.tex"
    write_latex(rows, out_path_tex)
    print(f"wrote {out_path_tex}")

    primary = [r for r in rows if r["metric"] == "ssim_global_A2B"]
    print("\nsummary (ssim_global_A2B, A->B direction):")
    for r in primary:
        flag = " *bonferroni-sig*" if r.get("significant_bonferroni") else ""
        print(
            f"  lambda={r['lambda_a']} vs lambda={r['lambda_b']}: "
            f"d={r['cohens_d']:+.3f}, mean_diff={r['mean_diff']:+.4f}, "
            f"p_wilcoxon={r['wilcoxon_p']:.2e}{flag}"
        )


if __name__ == "__main__":
    main()
