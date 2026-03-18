"""
ablation study framework.

systematic evaluation of model components
with statistical validation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import itertools


@dataclass
class AblationConfig:
    """configuration for a single ablation variant."""
    name: str
    description: str
    modifications: Dict[str, Any]
    expected_impact: str = "unknown"  # positive, negative, neutral


class AblationStudy:
    """
    comprehensive ablation study framework.
    
    supports:
    - single-component ablations
    - multi-component ablations
    - interaction studies
    - statistical comparison
    """
    
    def __init__(
        self,
        base_config: Any,
        output_dir: str,
        n_trials: int = 3
    ):
        """
        args:
            base_config: base experiment configuration
            output_dir: output directory
            n_trials: number of trials per ablation
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        
        self.ablations: List[AblationConfig] = []
        self.results: Dict[str, List[Dict]] = {}
    
    def add_ablation(self, ablation: AblationConfig):
        """add an ablation configuration."""
        self.ablations.append(ablation)
    
    def add_component_ablations(self):
        """add standard component ablations."""
        components = [
            AblationConfig(
                name="no_attention",
                description="Without self-attention mechanism",
                modifications={"model.use_attention": False},
                expected_impact="negative"
            ),
            AblationConfig(
                name="no_perceptual",
                description="Without perceptual loss",
                modifications={"loss.use_perceptual": False},
                expected_impact="negative"
            ),
            AblationConfig(
                name="no_contrastive",
                description="Without contrastive loss",
                modifications={"loss.use_contrastive": False},
                expected_impact="negative"
            ),
            AblationConfig(
                name="no_tumor_preservation",
                description="Without tumor preservation loss",
                modifications={"loss.use_tumor_preservation": False},
                expected_impact="negative"
            ),
            AblationConfig(
                name="no_ssim",
                description="Without MS-SSIM loss",
                modifications={"loss.use_ssim": False},
                expected_impact="negative"
            ),
        ]
        
        for ablation in components:
            self.add_ablation(ablation)
    
    def add_interaction_study(
        self,
        components: List[str],
        max_combination: int = 2
    ):
        """
        add interaction studies between components.
        
        args:
            components: list of component keys
            max_combination: maximum number of components to combine
        """
        for r in range(2, min(len(components), max_combination) + 1):
            for combo in itertools.combinations(components, r):
                name = "no_" + "_and_".join(
                    c.split('.')[-1] for c in combo
                )
                
                modifications = {c: False for c in combo}
                
                self.add_ablation(AblationConfig(
                    name=name,
                    description=f"Without: {', '.join(combo)}",
                    modifications=modifications,
                    expected_impact="negative"
                ))
    
    def apply_modifications(
        self,
        config: Any,
        modifications: Dict[str, Any]
    ) -> Any:
        """
        apply modifications to configuration.
        
        args:
            config: base configuration
            modifications: dict of path -> value
            
        returns:
            modified configuration
        """
        import copy
        
        modified = copy.deepcopy(config)
        
        for path, value in modifications.items():
            parts = path.split('.')
            obj = modified
            
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            setattr(obj, parts[-1], value)
        
        return modified
    
    def run(self, runner_class=None) -> Dict:
        """
        run complete ablation study.
        
        args:
            runner_class: experiment runner class
            
        returns:
            results dictionary
        """
        from .runner import ExperimentRunner
        
        runner_class = runner_class or ExperimentRunner
        
        # run baseline
        print("=" * 60)
        print("running baseline configuration")
        print("=" * 60)
        
        self.results['baseline'] = []
        for trial in range(self.n_trials):
            config = self._modify_for_trial(self.base_config, trial)
            runner = runner_class(
                config,
                self.output_dir / 'baseline' / f'trial_{trial}'
            )
            runner.setup()
            result = runner.run()
            self.results['baseline'].append(result)
        
        # run ablations
        for ablation in self.ablations:
            print("=" * 60)
            print(f"running ablation: {ablation.name}")
            print(f"description: {ablation.description}")
            print("=" * 60)
            
            self.results[ablation.name] = []
            
            for trial in range(self.n_trials):
                config = self.apply_modifications(
                    self.base_config,
                    ablation.modifications
                )
                config = self._modify_for_trial(config, trial)
                
                runner = runner_class(
                    config,
                    self.output_dir / ablation.name / f'trial_{trial}'
                )
                runner.setup()
                result = runner.run()
                self.results[ablation.name].append(result)
        
        # analyze and report
        analysis = self._analyze_results()
        self._save_report(analysis)
        
        return analysis
    
    def _modify_for_trial(self, config: Any, trial: int) -> Any:
        """modify config for specific trial (different seed)."""
        import copy
        modified = copy.deepcopy(config)
        modified.training.seed = config.training.seed + trial
        return modified
    
    def _analyze_results(self) -> Dict:
        """
        analyze ablation results with statistical tests.
        
        returns:
            analysis dictionary
        """
        from scipy import stats
        import numpy as np
        
        analysis = {
            'baseline': {},
            'ablations': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # extract baseline metrics
        baseline_metrics = self._extract_metrics('baseline')
        analysis['baseline'] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for metric, values in baseline_metrics.items()
        }
        
        # analyze each ablation
        for ablation in self.ablations:
            ablation_metrics = self._extract_metrics(ablation.name)
            
            analysis['ablations'][ablation.name] = {
                'description': ablation.description,
                'expected_impact': ablation.expected_impact,
                'metrics': {}
            }
            
            for metric, values in ablation_metrics.items():
                baseline_values = baseline_metrics.get(metric, [])
                
                if len(baseline_values) >= 2 and len(values) >= 2:
                    # paired t-test
                    t_stat, p_value = stats.ttest_ind(
                        baseline_values, values
                    )
                    
                    # effect size (cohen's d)
                    pooled_std = np.sqrt(
                        (np.std(baseline_values)**2 + np.std(values)**2) / 2
                    )
                    cohens_d = (
                        np.mean(baseline_values) - np.mean(values)
                    ) / (pooled_std + 1e-8)
                    
                    analysis['ablations'][ablation.name]['metrics'][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'delta': float(np.mean(values) - np.mean(baseline_values)),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': p_value < 0.05
                    }
        
        # rank ablations by impact
        analysis['rankings'] = self._rank_ablations(analysis)
        
        return analysis
    
    def _extract_metrics(self, name: str) -> Dict[str, List[float]]:
        """extract metrics from results."""
        metrics = {}
        
        for result in self.results.get(name, []):
            if 'test_metrics' in result:
                for metric, value in result['test_metrics'].items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
        
        return metrics
    
    def _rank_ablations(self, analysis: Dict) -> Dict:
        """rank ablations by their impact on metrics."""
        rankings = {}
        
        for metric in ['ssim', 'psnr']:
            impacts = []
            
            for name, data in analysis['ablations'].items():
                if metric in data.get('metrics', {}):
                    delta = data['metrics'][metric]['delta']
                    impacts.append((name, delta))
            
            # sort by absolute impact (most impactful first)
            impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            rankings[metric] = impacts
        
        return rankings
    
    def _save_report(self, analysis: Dict):
        """save analysis report."""
        # json report
        path = self.output_dir / 'ablation_analysis.json'
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # generate markdown report
        self._generate_markdown_report(analysis)
    
    def _generate_markdown_report(self, analysis: Dict):
        """generate markdown report."""
        lines = [
            "# Ablation Study Report",
            "",
            "## Baseline Performance",
            ""
        ]
        
        # baseline table
        lines.extend([
            "| Metric | Mean | Std |",
            "|--------|------|-----|"
        ])
        
        for metric, stats in analysis['baseline'].items():
            lines.append(
                f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} |"
            )
        
        lines.extend(["", "## Ablation Results", ""])
        
        # ablation table
        lines.extend([
            "| Ablation | SSIM Δ | PSNR Δ | Significant |",
            "|----------|--------|--------|-------------|"
        ])
        
        for name, data in analysis['ablations'].items():
            ssim_delta = data['metrics'].get('ssim', {}).get('delta', 0)
            psnr_delta = data['metrics'].get('psnr', {}).get('delta', 0)
            sig = "Yes" if data['metrics'].get('ssim', {}).get('significant', False) else "No"
            
            lines.append(
                f"| {name} | {ssim_delta:+.4f} | {psnr_delta:+.2f} | {sig} |"
            )
        
        lines.extend(["", "## Component Rankings", ""])
        
        for metric, ranking in analysis.get('rankings', {}).items():
            lines.append(f"### {metric.upper()} Impact")
            lines.append("")
            for i, (name, delta) in enumerate(ranking, 1):
                lines.append(f"{i}. **{name}**: {delta:+.4f}")
            lines.append("")
        
        # save
        path = self.output_dir / 'ablation_report.md'
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


def run_ablation_suite(
    base_config: Any,
    output_dir: str,
    n_trials: int = 3,
    include_interactions: bool = True
) -> Dict:
    """
    run complete ablation study suite.
    
    args:
        base_config: base experiment configuration
        output_dir: output directory
        n_trials: number of trials per ablation
        include_interactions: include interaction studies
        
    returns:
        analysis results
    """
    study = AblationStudy(base_config, output_dir, n_trials)
    
    # add standard ablations
    study.add_component_ablations()
    
    # add interactions
    if include_interactions:
        study.add_interaction_study([
            "loss.use_perceptual",
            "loss.use_contrastive",
            "model.use_attention"
        ])
    
    return study.run()
