"""
Ablation Study Framework.

Systematic evaluation of model components
with statistical validation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import itertools


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""
    name: str
    description: str
    modifications: Dict[str, Any]
    expected_impact: str = "unknown"  # positive, negative, neutral


class AblationStudy:
    """
    Comprehensive ablation study framework.
    
    Supports:
    - Single-component ablations
    - Multi-component ablations
    - Interaction studies
    - Statistical comparison
    """
    
    def __init__(
        self,
        base_config: Any,
        output_dir: str,
        n_trials: int = 3
    ):
        """
        Args:
            base_config: Base experiment configuration
            output_dir: Output directory
            n_trials: Number of trials per ablation
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        
        self.ablations: List[AblationConfig] = []
        self.results: Dict[str, List[Dict]] = {}
    
    def add_ablation(self, ablation: AblationConfig):
        """Add an ablation configuration."""
        self.ablations.append(ablation)
    
    def add_component_ablations(self):
        """Add standard component ablations."""
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
        Add interaction studies between components.
        
        Args:
            components: List of component keys
            max_combination: Maximum number of components to combine
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
        Apply modifications to configuration.
        
        Args:
            config: Base configuration
            modifications: Dict of path -> value
            
        Returns:
            Modified configuration
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
        Run complete ablation study.
        
        Args:
            runner_class: Experiment runner class
            
        Returns:
            Results dictionary
        """
        from .runner import ExperimentRunner
        
        runner_class = runner_class or ExperimentRunner
        
        # Run baseline
        print("=" * 60)
        print("Running baseline configuration")
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
        
        # Run ablations
        for ablation in self.ablations:
            print("=" * 60)
            print(f"Running ablation: {ablation.name}")
            print(f"Description: {ablation.description}")
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
        
        # Analyze and report
        analysis = self._analyze_results()
        self._save_report(analysis)
        
        return analysis
    
    def _modify_for_trial(self, config: Any, trial: int) -> Any:
        """Modify config for specific trial (different seed)."""
        import copy
        modified = copy.deepcopy(config)
        modified.training.seed = config.training.seed + trial
        return modified
    
    def _analyze_results(self) -> Dict:
        """
        Analyze ablation results with statistical tests.
        
        Returns:
            Analysis dictionary
        """
        from scipy import stats
        import numpy as np
        
        analysis = {
            'baseline': {},
            'ablations': {},
            'statistical_tests': {},
            'rankings': {}
        }
        
        # Extract baseline metrics
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
        
        # Analyze each ablation
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
                    # Paired t-test
                    t_stat, p_value = stats.ttest_ind(
                        baseline_values, values
                    )
                    
                    # Effect size (Cohen's d)
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
        
        # Rank ablations by impact
        analysis['rankings'] = self._rank_ablations(analysis)
        
        return analysis
    
    def _extract_metrics(self, name: str) -> Dict[str, List[float]]:
        """Extract metrics from results."""
        metrics = {}
        
        for result in self.results.get(name, []):
            if 'test_metrics' in result:
                for metric, value in result['test_metrics'].items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
        
        return metrics
    
    def _rank_ablations(self, analysis: Dict) -> Dict:
        """Rank ablations by their impact on metrics."""
        rankings = {}
        
        for metric in ['ssim', 'psnr']:
            impacts = []
            
            for name, data in analysis['ablations'].items():
                if metric in data.get('metrics', {}):
                    delta = data['metrics'][metric]['delta']
                    impacts.append((name, delta))
            
            # Sort by absolute impact (most impactful first)
            impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            rankings[metric] = impacts
        
        return rankings
    
    def _save_report(self, analysis: Dict):
        """Save analysis report."""
        # JSON report
        path = self.output_dir / 'ablation_analysis.json'
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(analysis)
    
    def _generate_markdown_report(self, analysis: Dict):
        """Generate markdown report."""
        lines = [
            "# Ablation Study Report",
            "",
            "## Baseline Performance",
            ""
        ]
        
        # Baseline table
        lines.extend([
            "| Metric | Mean | Std |",
            "|--------|------|-----|"
        ])
        
        for metric, stats in analysis['baseline'].items():
            lines.append(
                f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} |"
            )
        
        lines.extend(["", "## Ablation Results", ""])
        
        # Ablation table
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
        
        # Save
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
    Run complete ablation study suite.
    
    Args:
        base_config: Base experiment configuration
        output_dir: Output directory
        n_trials: Number of trials per ablation
        include_interactions: Include interaction studies
        
    Returns:
        Analysis results
    """
    study = AblationStudy(base_config, output_dir, n_trials)
    
    # Add standard ablations
    study.add_component_ablations()
    
    # Add interactions
    if include_interactions:
        study.add_interaction_study([
            "loss.use_perceptual",
            "loss.use_contrastive",
            "model.use_attention"
        ])
    
    return study.run()
