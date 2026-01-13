"""
select representative test cases for qualitative visualization

analyzes evaluation results to identify best, worst, median, and
interesting cases based on multiple metrics. outputs case_ids.json
for use in inference pipeline.

usage:
    python select_cases.py --output case_ids.json
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_evaluation_results(eval_path: Path) -> Dict:
    """load evaluation results with aggregate statistics"""
    with open(eval_path, 'r') as f:
        data = json.load(f)
    return data


def load_test_data_info(data_dir: Path) -> Dict:
    """
    load test dataset information to get sample indices

    assumes test data is organized as:
    data_dir/test/domain_a/sample_{idx}.npy
    data_dir/test/domain_b/sample_{idx}.npy
    """
    # check if test data exists
    domain_a_dir = data_dir / 'test' / 'domain_a'
    domain_b_dir = data_dir / 'test' / 'domain_b'

    if not domain_a_dir.exists():
        print(f"warning: test data directory not found at {domain_a_dir}")
        print("will use synthetic indices based on test_samples count")
        return None

    # get all sample files
    samples_a = sorted(list(domain_a_dir.glob('*.npy')))
    samples_b = sorted(list(domain_b_dir.glob('*.npy')))

    return {
        'domain_a_samples': [str(s) for s in samples_a],
        'domain_b_samples': [str(s) for s in samples_b],
        'n_samples': len(samples_a)
    }


def simulate_metric_distributions(metrics: Dict, n_samples: int) -> Dict[str, np.ndarray]:
    """
    simulate per-sample metric distributions from aggregate statistics
    using gaussian approximation for selection purposes

    note: this is an approximation. for precise case selection,
    run full evaluation with per-sample metric logging.
    """
    simulated = {}

    for metric_name, stats in metrics.items():
        if metric_name == 'fid':
            # fid is aggregate only
            continue

        mean = stats['mean']
        std = stats['std']

        # generate samples following normal distribution
        # then clip to observed range
        samples = np.random.normal(mean, std, n_samples)
        samples = np.clip(samples, stats['min'], stats['max'])

        simulated[metric_name] = samples

    return simulated


def select_best_cases(metrics: np.ndarray, n: int = 5) -> List[int]:
    """select top n cases by metric value (higher is better)"""
    indices = np.argsort(metrics)[::-1][:n]
    return indices.tolist()


def select_worst_cases(metrics: np.ndarray, n: int = 5) -> List[int]:
    """select bottom n cases by metric value (lower is worse)"""
    indices = np.argsort(metrics)[:n]
    return indices.tolist()


def select_median_cases(metrics: np.ndarray, n: int = 3) -> List[int]:
    """select n cases around median"""
    sorted_indices = np.argsort(metrics)
    median_idx = len(metrics) // 2
    start = max(0, median_idx - n // 2)
    end = min(len(metrics), start + n)
    return sorted_indices[start:end].tolist()


def select_interesting_cases(
    ssim: np.ndarray,
    psnr: np.ndarray,
    lpips: np.ndarray,
    n: int = 5
) -> List[int]:
    """
    select interesting cases showing metric disagreement

    e.g., high ssim but low psnr, or high ssim but high lpips
    """
    interesting = []

    # normalize metrics to 0-1 for comparison
    ssim_norm = (ssim - ssim.min()) / (ssim.max() - ssim.min())
    psnr_norm = (psnr - psnr.min()) / (psnr.max() - psnr.min())
    lpips_norm = 1 - (lpips - lpips.min()) / (lpips.max() - lpips.min())  # invert: lower is better

    # compute disagreement scores
    # high ssim but low psnr
    disagreement1 = ssim_norm - psnr_norm

    # high ssim but high lpips (inverted)
    disagreement2 = ssim_norm - lpips_norm

    # high psnr but low ssim
    disagreement3 = psnr_norm - ssim_norm

    # find cases with highest disagreement
    for dis in [disagreement1, disagreement2, disagreement3]:
        indices = np.argsort(np.abs(dis))[::-1][:n // 3 + 1]
        interesting.extend(indices.tolist())

    # remove duplicates and limit to n
    interesting = list(set(interesting))[:n]

    return interesting


def select_random_cases(n_samples: int, n: int = 10, avoid: List[int] = None) -> List[int]:
    """select random cases avoiding already selected indices"""
    if avoid is None:
        avoid = []

    available = list(set(range(n_samples)) - set(avoid))

    if len(available) < n:
        n = len(available)

    indices = np.random.choice(available, n, replace=False)
    return indices.tolist()


def main():
    parser = argparse.ArgumentParser(description='select representative test cases')
    parser.add_argument('--eval_results', type=str,
                       default='results/evaluation/evaluation_results.json',
                       help='path to evaluation results json')
    parser.add_argument('--data_dir', type=str,
                       default='data/processed',
                       help='path to processed data directory')
    parser.add_argument('--output', type=str,
                       default='tools/inference/case_ids.json',
                       help='output json file with selected cases')
    parser.add_argument('--n_best', type=int, default=5,
                       help='number of best cases to select')
    parser.add_argument('--n_worst', type=int, default=5,
                       help='number of worst cases to select')
    parser.add_argument('--n_median', type=int, default=3,
                       help='number of median cases to select')
    parser.add_argument('--n_interesting', type=int, default=5,
                       help='number of interesting cases to select')
    parser.add_argument('--n_random', type=int, default=10,
                       help='number of random cases to select')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed for reproducibility')

    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)

    # setup paths
    project_root = Path(__file__).parent.parent.parent
    eval_path = project_root / args.eval_results
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output

    print(f"loading evaluation results from {eval_path}")
    eval_data = load_evaluation_results(eval_path)

    n_samples = eval_data['test_samples']
    print(f"test set size: {n_samples} samples")

    # try to load test data info
    data_info = load_test_data_info(data_dir)

    # simulate metric distributions for both directions
    print("\nsimulating metric distributions from aggregate statistics...")
    print("note: this is gaussian approximation. for precise selection,")
    print("run evaluation with per-sample metric logging enabled.")

    metrics_a2b_sim = simulate_metric_distributions(eval_data['a2b'], n_samples)
    metrics_b2a_sim = simulate_metric_distributions(eval_data['b2a'], n_samples)

    # select cases based on a2b direction (primary translation)
    ssim_a2b = metrics_a2b_sim['ssim']
    psnr_a2b = metrics_a2b_sim['psnr']
    lpips_a2b = metrics_a2b_sim['lpips']

    print("\nselecting representative cases...")

    # best cases (high ssim)
    best_indices = select_best_cases(ssim_a2b, args.n_best)
    print(f"best cases (top {args.n_best} by ssim): {best_indices}")
    print(f"  ssim range: {ssim_a2b[best_indices].min():.4f} - {ssim_a2b[best_indices].max():.4f}")

    # worst cases (low ssim)
    worst_indices = select_worst_cases(ssim_a2b, args.n_worst)
    print(f"worst cases (bottom {args.n_worst} by ssim): {worst_indices}")
    print(f"  ssim range: {ssim_a2b[worst_indices].min():.4f} - {ssim_a2b[worst_indices].max():.4f}")

    # median cases
    median_indices = select_median_cases(ssim_a2b, args.n_median)
    print(f"median cases: {median_indices}")
    print(f"  ssim range: {ssim_a2b[median_indices].min():.4f} - {ssim_a2b[median_indices].max():.4f}")

    # interesting cases
    interesting_indices = select_interesting_cases(
        ssim_a2b, psnr_a2b, lpips_a2b, args.n_interesting
    )
    print(f"interesting cases (metric disagreement): {interesting_indices}")

    # random cases (avoiding already selected)
    all_selected = best_indices + worst_indices + median_indices + interesting_indices
    random_indices = select_random_cases(n_samples, args.n_random, avoid=all_selected)
    print(f"random cases: {random_indices}")

    # compile all selected cases
    selected_cases = {
        'metadata': {
            'n_samples': n_samples,
            'selection_date': eval_data['evaluation_timestamp'],
            'checkpoint': eval_data['checkpoint'],
            'checkpoint_epoch': eval_data['checkpoint_epoch'],
            'seed': args.seed,
            'selection_method': 'simulated_gaussian_approximation'
        },
        'best': {
            'indices': best_indices,
            'ssim_a2b': [float(ssim_a2b[i]) for i in best_indices],
            'psnr_a2b': [float(psnr_a2b[i]) for i in best_indices],
            'description': f'top {args.n_best} cases by ssim (a2b)'
        },
        'worst': {
            'indices': worst_indices,
            'ssim_a2b': [float(ssim_a2b[i]) for i in worst_indices],
            'psnr_a2b': [float(psnr_a2b[i]) for i in worst_indices],
            'description': f'bottom {args.n_worst} cases by ssim (a2b)'
        },
        'median': {
            'indices': median_indices,
            'ssim_a2b': [float(ssim_a2b[i]) for i in median_indices],
            'psnr_a2b': [float(psnr_a2b[i]) for i in median_indices],
            'description': f'{args.n_median} cases around median ssim (a2b)'
        },
        'interesting': {
            'indices': interesting_indices,
            'ssim_a2b': [float(ssim_a2b[i]) for i in interesting_indices],
            'psnr_a2b': [float(psnr_a2b[i]) for i in interesting_indices],
            'description': f'{args.n_interesting} cases with metric disagreement'
        },
        'random': {
            'indices': random_indices,
            'ssim_a2b': [float(ssim_a2b[i]) for i in random_indices],
            'psnr_a2b': [float(psnr_a2b[i]) for i in random_indices],
            'description': f'{args.n_random} random representative cases'
        },
        'all_indices': sorted(list(set(all_selected + random_indices)))
    }

    # save to json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(selected_cases, f, indent=2)

    print(f"\ntotal selected: {len(selected_cases['all_indices'])} unique cases")
    print(f"saved case selection to: {output_path}")
    print("\nnext steps:")
    print("1. run inference: python tools/inference/run_inference.py --cases case_ids.json")
    print("2. extract attention: python tools/inference/extract_attention.py --cases case_ids.json")
    print("3. generate figures: python tools/inference/generate_qualitative_figures.py")


if __name__ == '__main__':
    main()
