#!/usr/bin/env python3
"""
ablation study: baseline cyclegan vs sa-cyclegan with attention mechanisms.

this script performs a rigorous comparison between:
1. baseline cyclegan (no attention)
2. sa-cyclegan (self-attention + cbam)

metrics computed:
- ssim (structural similarity index)
- psnr (peak signal-to-noise ratio)
- cycle consistency error
- identity preservation error
- per-modality analysis (t1, t1ce, t2, flair)

statistical analysis:
- paired t-tests for significance
- effect size (cohen's d)
- confidence intervals

usage:
    python ablation_study.py \
        --baseline-checkpoint /path/to/baseline/checkpoint_best.pth \
        --attention-checkpoint /path/to/sa_cyclegan/checkpoint_best.pth \
        --output-dir ./results/ablation
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuroscope.models.architectures.sa_cyclegan_25d import SACycleGAN25D, SACycleGAN25DConfig
from neuroscope.models.architectures.baseline_cyclegan_25d import BaselineCycleGAN25D, BaselineCycleGAN25DConfig
from neuroscope.data.datasets.dataset_25d import create_dataloaders


class AblationStudy:
    """
    comprehensive ablation study comparing baseline vs attention-based cyclegan.
    """

    def __init__(
        self,
        baseline_checkpoint: str,
        attention_checkpoint: str,
        brats_dir: str,
        upenn_dir: str,
        output_dir: str,
        device: str = 'cuda',
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        self.baseline_path = Path(baseline_checkpoint)
        self.attention_path = Path(attention_checkpoint)
        self.brats_dir = Path(brats_dir)
        self.upenn_dir = Path(upenn_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"[ablation] device: {self.device}")
        print(f"[ablation] baseline checkpoint: {self.baseline_path}")
        print(f"[ablation] attention checkpoint: {self.attention_path}")

        # modality names for per-channel analysis
        self.modalities = ['T1', 'T1CE', 'T2', 'FLAIR']

    def load_baseline_model(self) -> Tuple[nn.Module, int]:
        """load baseline cyclegan model."""
        print("[ablation] loading baseline model...")
        checkpoint = torch.load(
            self.baseline_path,
            map_location=self.device,
            weights_only=False
        )

        config = BaselineCycleGAN25DConfig(
            ngf=64,
            ndf=64,
            n_residual_blocks=9,
        )

        model = BaselineCycleGAN25D(config)

        # handle dataparallel state dict - components were individually wrapped
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # remove .module. from component-level wrapping (e.g., G_A2B.module.xxx -> G_A2B.xxx)
            new_key = k.replace('.module.', '.')
            # also handle if whole model was wrapped
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        epoch = checkpoint.get('epoch', 100)
        print(f"[ablation] baseline loaded: epoch {epoch}")

        return model, epoch

    def load_attention_model(self) -> Tuple[nn.Module, int]:
        """load sa-cyclegan model with attention."""
        print("[ablation] loading attention model...")
        checkpoint = torch.load(
            self.attention_path,
            map_location=self.device,
            weights_only=False
        )

        config = SACycleGAN25DConfig(
            ngf=64,
            ndf=64,
            n_residual_blocks=9,
        )

        model = SACycleGAN25D(config)

        # handle dataparallel state dict - components were individually wrapped
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # remove .module. from component-level wrapping (e.g., G_A2B.module.xxx -> G_A2B.xxx)
            new_key = k.replace('.module.', '.')
            # also handle if whole model was wrapped
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        epoch = checkpoint.get('epoch', 100)
        print(f"[ablation] attention model loaded: epoch {epoch}")

        return model, epoch

    def load_test_data(self) -> DataLoader:
        """load test dataset."""
        print("[ablation] loading test data...")
        _, _, test_loader = create_dataloaders(
            brats_dir=str(self.brats_dir),
            upenn_dir=str(self.upenn_dir),
            batch_size=self.batch_size,
            image_size=(128, 128),
            num_workers=self.num_workers,
        )
        print(f"[ablation] test batches: {len(test_loader)}")
        return test_loader

    def compute_cycle_metrics(
        self,
        model: nn.Module,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        center_A: torch.Tensor,
        center_B: torch.Tensor,
    ) -> Dict[str, float]:
        """
        compute cycle consistency and translation metrics.

        returns:
            dict with ssim, psnr for both directions and cycle consistency
        """
        with torch.no_grad():
            # forward translations
            fake_B = model.G_A2B(real_A)
            fake_A = model.G_B2A(real_B)

            # create 3-slice input for cycle
            fake_B_3slice = fake_B.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_B_3slice = fake_B_3slice.view(fake_B.size(0), -1, fake_B.size(2), fake_B.size(3))
            fake_A_3slice = fake_A.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            fake_A_3slice = fake_A_3slice.view(fake_A.size(0), -1, fake_A.size(2), fake_A.size(3))

            # cycle reconstructions
            rec_A = model.G_B2A(fake_B_3slice)
            rec_B = model.G_A2B(fake_A_3slice)

            # identity mappings
            identity_A = model.G_B2A(real_A)
            identity_B = model.G_A2B(real_B)

        # convert to numpy for metric computation
        center_A_np = center_A.cpu().numpy()
        center_B_np = center_B.cpu().numpy()
        rec_A_np = rec_A.cpu().numpy()
        rec_B_np = rec_B.cpu().numpy()
        identity_A_np = identity_A.cpu().numpy()
        identity_B_np = identity_B.cpu().numpy()

        metrics = {
            'cycle_ssim_A': [],
            'cycle_ssim_B': [],
            'cycle_psnr_A': [],
            'cycle_psnr_B': [],
            'identity_ssim_A': [],
            'identity_ssim_B': [],
        }

        # per-modality metrics
        for m_idx, modality in enumerate(self.modalities):
            metrics[f'cycle_ssim_A_{modality}'] = []
            metrics[f'cycle_ssim_B_{modality}'] = []

        # compute metrics for each sample in batch
        for i in range(center_A_np.shape[0]):
            # cycle consistency ssim
            for c in range(4):  # 4 modalities
                ssim_A = ssim_func(
                    center_A_np[i, c],
                    rec_A_np[i, c],
                    data_range=center_A_np[i, c].max() - center_A_np[i, c].min()
                )
                ssim_B = ssim_func(
                    center_B_np[i, c],
                    rec_B_np[i, c],
                    data_range=center_B_np[i, c].max() - center_B_np[i, c].min()
                )
                metrics[f'cycle_ssim_A_{self.modalities[c]}'].append(ssim_A)
                metrics[f'cycle_ssim_B_{self.modalities[c]}'].append(ssim_B)

            # overall cycle ssim (mean across modalities)
            cycle_ssim_A = np.mean([
                ssim_func(center_A_np[i, c], rec_A_np[i, c],
                         data_range=center_A_np[i, c].max() - center_A_np[i, c].min())
                for c in range(4)
            ])
            cycle_ssim_B = np.mean([
                ssim_func(center_B_np[i, c], rec_B_np[i, c],
                         data_range=center_B_np[i, c].max() - center_B_np[i, c].min())
                for c in range(4)
            ])

            metrics['cycle_ssim_A'].append(cycle_ssim_A)
            metrics['cycle_ssim_B'].append(cycle_ssim_B)

            # cycle psnr
            mse_A = np.mean((center_A_np[i] - rec_A_np[i]) ** 2)
            mse_B = np.mean((center_B_np[i] - rec_B_np[i]) ** 2)
            psnr_A = 10 * np.log10(1.0 / (mse_A + 1e-10))
            psnr_B = 10 * np.log10(1.0 / (mse_B + 1e-10))

            metrics['cycle_psnr_A'].append(psnr_A)
            metrics['cycle_psnr_B'].append(psnr_B)

            # identity ssim
            identity_ssim_A = np.mean([
                ssim_func(center_A_np[i, c], identity_A_np[i, c],
                         data_range=center_A_np[i, c].max() - center_A_np[i, c].min())
                for c in range(4)
            ])
            identity_ssim_B = np.mean([
                ssim_func(center_B_np[i, c], identity_B_np[i, c],
                         data_range=center_B_np[i, c].max() - center_B_np[i, c].min())
                for c in range(4)
            ])

            metrics['identity_ssim_A'].append(identity_ssim_A)
            metrics['identity_ssim_B'].append(identity_ssim_B)

        return metrics

    def run_evaluation(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
        """
        run full evaluation on test set.

        returns:
            dict containing all metrics for this model
        """
        print(f"[ablation] evaluating {model_name}...")

        all_metrics = {
            'cycle_ssim_A': [],
            'cycle_ssim_B': [],
            'cycle_psnr_A': [],
            'cycle_psnr_B': [],
            'identity_ssim_A': [],
            'identity_ssim_B': [],
        }

        # per-modality metrics
        for modality in self.modalities:
            all_metrics[f'cycle_ssim_A_{modality}'] = []
            all_metrics[f'cycle_ssim_B_{modality}'] = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"evaluating {model_name}"):
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                center_A = batch['A_center'].to(self.device)
                center_B = batch['B_center'].to(self.device)

                batch_metrics = self.compute_cycle_metrics(
                    model, real_A, real_B, center_A, center_B
                )

                for key, values in batch_metrics.items():
                    all_metrics[key].extend(values)

        # compute summary statistics
        summary = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values),
            }
            summary[f'{key}_values'] = values.tolist()

        return summary

    def compute_statistical_tests(
        self,
        baseline_results: Dict,
        attention_results: Dict
    ) -> Dict:
        """
        compute statistical significance tests between models.

        returns:
            dict with p-values, effect sizes, and confidence intervals
        """
        print("[ablation] computing statistical tests...")

        stats_results = {}

        metrics_to_test = [
            'cycle_ssim_A', 'cycle_ssim_B',
            'cycle_psnr_A', 'cycle_psnr_B',
            'identity_ssim_A', 'identity_ssim_B',
        ]

        # add per-modality metrics
        for modality in self.modalities:
            metrics_to_test.append(f'cycle_ssim_A_{modality}')
            metrics_to_test.append(f'cycle_ssim_B_{modality}')

        for metric in metrics_to_test:
            baseline_vals = np.array(baseline_results[f'{metric}_values'])
            attention_vals = np.array(attention_results[f'{metric}_values'])

            # paired t-test
            t_stat, p_value = stats.ttest_rel(attention_vals, baseline_vals)

            # effect size (cohen's d for paired samples)
            diff = attention_vals - baseline_vals
            cohens_d = np.mean(diff) / np.std(diff)

            # 95% confidence interval for the difference
            se = stats.sem(diff)
            ci_low = np.mean(diff) - 1.96 * se
            ci_high = np.mean(diff) + 1.96 * se

            # improvement percentage
            baseline_mean = np.mean(baseline_vals)
            attention_mean = np.mean(attention_vals)
            improvement_pct = ((attention_mean - baseline_mean) / baseline_mean) * 100

            stats_results[metric] = {
                'baseline_mean': float(baseline_mean),
                'baseline_std': float(np.std(baseline_vals)),
                'attention_mean': float(attention_mean),
                'attention_std': float(np.std(attention_vals)),
                'difference': float(attention_mean - baseline_mean),
                'improvement_pct': float(improvement_pct),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'ci_95_low': float(ci_low),
                'ci_95_high': float(ci_high),
                'significant': bool(p_value < 0.05),
            }

        return stats_results

    def generate_latex_table(self, stats_results: Dict) -> str:
        """generate latex table for publication."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study: Baseline CycleGAN vs SA-CycleGAN with Attention}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Metric & Baseline & SA-CycleGAN & $\Delta$ & $p$-value & Cohen's $d$ \\",
            r"\midrule",
        ]

        main_metrics = [
            ('cycle_ssim_A', 'Cycle SSIM (A)'),
            ('cycle_ssim_B', 'Cycle SSIM (B)'),
            ('cycle_psnr_A', 'Cycle PSNR (A)'),
            ('cycle_psnr_B', 'Cycle PSNR (B)'),
            ('identity_ssim_A', 'Identity SSIM (A)'),
            ('identity_ssim_B', 'Identity SSIM (B)'),
        ]

        for key, name in main_metrics:
            r = stats_results[key]
            sig = r'$^{*}$' if r['significant'] else ''
            lines.append(
                f"{name} & {r['baseline_mean']:.4f}$\\pm${r['baseline_std']:.4f} & "
                f"{r['attention_mean']:.4f}$\\pm${r['attention_std']:.4f} & "
                f"{r['difference']:+.4f}{sig} & {r['p_value']:.4f} & {r['cohens_d']:.3f} \\\\"
            )

        lines.extend([
            r"\midrule",
            r"\multicolumn{6}{l}{\textit{Per-Modality Cycle SSIM (A$\rightarrow$B$\rightarrow$A)}} \\",
        ])

        for modality in self.modalities:
            key = f'cycle_ssim_A_{modality}'
            r = stats_results[key]
            sig = r'$^{*}$' if r['significant'] else ''
            lines.append(
                f"\\quad {modality} & {r['baseline_mean']:.4f} & "
                f"{r['attention_mean']:.4f} & "
                f"{r['difference']:+.4f}{sig} & {r['p_value']:.4f} & {r['cohens_d']:.3f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\multicolumn{6}{l}{\footnotesize $^{*}$ Significant at $p < 0.05$} \\",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return '\n'.join(lines)

    def run(self):
        """run complete ablation study."""
        print("=" * 60)
        print("ablation study: baseline vs attention cyclegan")
        print("=" * 60)

        # load models
        baseline_model, baseline_epoch = self.load_baseline_model()
        attention_model, attention_epoch = self.load_attention_model()

        # load test data
        test_loader = self.load_test_data()

        # run evaluations
        baseline_results = self.run_evaluation(baseline_model, test_loader, "baseline")
        attention_results = self.run_evaluation(attention_model, test_loader, "attention")

        # statistical tests
        stats_results = self.compute_statistical_tests(baseline_results, attention_results)

        # generate latex table
        latex_table = self.generate_latex_table(stats_results)

        # save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # save detailed results
        results = {
            'timestamp': timestamp,
            'baseline_checkpoint': str(self.baseline_path),
            'attention_checkpoint': str(self.attention_path),
            'baseline_epoch': baseline_epoch,
            'attention_epoch': attention_epoch,
            'baseline_results': {k: v for k, v in baseline_results.items() if not k.endswith('_values')},
            'attention_results': {k: v for k, v in attention_results.items() if not k.endswith('_values')},
            'statistical_tests': stats_results,
        }

        results_path = self.output_dir / f"ablation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[ablation] results saved to: {results_path}")

        # save latex table
        latex_path = self.output_dir / f"ablation_table_{timestamp}.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"[ablation] latex table saved to: {latex_path}")

        # print summary
        print("\n" + "=" * 60)
        print("ablation study results summary")
        print("=" * 60)
        print(f"\n{'metric':<25} {'baseline':>12} {'attention':>12} {'diff':>10} {'p-value':>10}")
        print("-" * 70)

        for metric in ['cycle_ssim_A', 'cycle_ssim_B', 'cycle_psnr_A', 'cycle_psnr_B']:
            r = stats_results[metric]
            sig = '*' if r['significant'] else ''
            print(f"{metric:<25} {r['baseline_mean']:>12.4f} {r['attention_mean']:>12.4f} "
                  f"{r['difference']:>+10.4f}{sig} {r['p_value']:>10.4f}")

        print("\n" + "=" * 60)
        print("per-modality cycle ssim (a->b->a)")
        print("=" * 60)
        for modality in self.modalities:
            r = stats_results[f'cycle_ssim_A_{modality}']
            sig = '*' if r['significant'] else ''
            print(f"{modality:<10} baseline: {r['baseline_mean']:.4f}  attention: {r['attention_mean']:.4f}  "
                  f"diff: {r['difference']:+.4f}{sig}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='ablation study comparing baseline and attention cyclegan',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--baseline-checkpoint', type=str, required=True,
                        help='path to baseline cyclegan checkpoint')
    parser.add_argument('--attention-checkpoint', type=str, required=True,
                        help='path to sa-cyclegan checkpoint')
    parser.add_argument('--brats-dir', type=str,
                        default='/home/cc/neuroscope/preprocessed/brats',
                        help='path to brats data')
    parser.add_argument('--upenn-dir', type=str,
                        default='/home/cc/neuroscope/preprocessed/upenn',
                        help='path to upenn data')
    parser.add_argument('--output-dir', type=str,
                        default='./results/ablation',
                        help='output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device for computation')

    args = parser.parse_args()

    study = AblationStudy(
        baseline_checkpoint=args.baseline_checkpoint,
        attention_checkpoint=args.attention_checkpoint,
        brats_dir=args.brats_dir,
        upenn_dir=args.upenn_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    study.run()


if __name__ == '__main__':
    main()
