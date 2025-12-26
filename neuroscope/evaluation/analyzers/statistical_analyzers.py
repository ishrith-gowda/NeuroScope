"""
Result Analyzers.

Analysis frameworks for comprehensive evaluation
of harmonization results across modalities and regions.
"""

from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json


@dataclass
class AnalysisResult:
    """Result from analysis."""
    name: str
    metrics: Dict[str, float]
    per_sample: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'metrics': self.metrics,
            'per_sample': self.per_sample,
            'metadata': self.metadata
        }
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ModalityAnalyzer:
    """
    Analyze results per MRI modality.
    
    Breaks down performance across T1, T1ce, T2, FLAIR.
    """
    
    MODALITIES = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    def __init__(self, metrics_fn: callable = None):
        """
        Args:
            metrics_fn: Function to compute metrics
        """
        self.metrics_fn = metrics_fn
        self.results_per_modality: Dict[str, List[Dict]] = {
            m: [] for m in self.MODALITIES
        }
    
    def add_result(
        self,
        modality: str,
        original: np.ndarray,
        harmonized: np.ndarray,
        reference: np.ndarray = None
    ):
        """
        Add a result for a specific modality.
        
        Args:
            modality: Modality name (T1, T1ce, T2, FLAIR)
            original: Original image
            harmonized: Harmonized image
            reference: Reference image (if available)
        """
        if modality not in self.MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        
        if self.metrics_fn is not None:
            if reference is not None:
                metrics = self.metrics_fn(harmonized, reference)
            else:
                metrics = self.metrics_fn(original, harmonized)
        else:
            # Default metrics
            metrics = {
                'mse': float(np.mean((original - harmonized) ** 2)),
                'mae': float(np.mean(np.abs(original - harmonized))),
                'correlation': float(np.corrcoef(
                    original.flatten(), harmonized.flatten()
                )[0, 1])
            }
        
        self.results_per_modality[modality].append(metrics)
    
    def analyze(self) -> Dict[str, AnalysisResult]:
        """
        Analyze results across all modalities.
        
        Returns:
            Dict of modality -> AnalysisResult
        """
        results = {}
        
        for modality, metric_list in self.results_per_modality.items():
            if not metric_list:
                continue
            
            # Aggregate metrics
            aggregated = {}
            per_sample = {}
            
            metric_names = metric_list[0].keys()
            
            for name in metric_names:
                values = [m[name] for m in metric_list]
                aggregated[f'{name}_mean'] = float(np.mean(values))
                aggregated[f'{name}_std'] = float(np.std(values))
                per_sample[name] = values
            
            results[modality] = AnalysisResult(
                name=f'{modality}_analysis',
                metrics=aggregated,
                per_sample=per_sample,
                metadata={'n_samples': len(metric_list)}
            )
        
        return results
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics per modality."""
        results = self.analyze()
        
        summary = {}
        for modality, result in results.items():
            summary[modality] = {
                k: v for k, v in result.metrics.items()
                if k.endswith('_mean')
            }
        
        return summary


class RegionAnalyzer:
    """
    Analyze results per tumor region.
    
    Evaluates performance in enhancing tumor, necrotic core, and edema.
    """
    
    REGIONS = ['enhancing', 'necrotic', 'edema', 'whole_tumor', 'background']
    
    def __init__(self, metrics_fn: callable = None):
        """
        Args:
            metrics_fn: Function to compute metrics
        """
        self.metrics_fn = metrics_fn
        self.results_per_region: Dict[str, List[Dict]] = {
            r: [] for r in self.REGIONS
        }
    
    def add_result(
        self,
        original: np.ndarray,
        harmonized: np.ndarray,
        segmentation: np.ndarray,
        label_mapping: Dict[str, int] = None
    ):
        """
        Add a result with segmentation mask.
        
        Args:
            original: Original image
            harmonized: Harmonized image
            segmentation: Segmentation mask
            label_mapping: Mapping from region name to label value
        """
        if label_mapping is None:
            label_mapping = {
                'background': 0,
                'necrotic': 1,
                'edema': 2,
                'enhancing': 4,
                'whole_tumor': [1, 2, 4]
            }
        
        for region, label in label_mapping.items():
            if isinstance(label, list):
                mask = np.isin(segmentation, label)
            else:
                mask = segmentation == label
            
            if mask.sum() == 0:
                continue
            
            orig_region = original[mask]
            harm_region = harmonized[mask]
            
            metrics = {
                'mse': float(np.mean((orig_region - harm_region) ** 2)),
                'mae': float(np.mean(np.abs(orig_region - harm_region))),
                'correlation': float(np.corrcoef(
                    orig_region.flatten(), harm_region.flatten()
                )[0, 1]) if len(orig_region) > 1 else 1.0,
                'volume': float(mask.sum())
            }
            
            self.results_per_region[region].append(metrics)
    
    def analyze(self) -> Dict[str, AnalysisResult]:
        """Analyze results across all regions."""
        results = {}
        
        for region, metric_list in self.results_per_region.items():
            if not metric_list:
                continue
            
            aggregated = {}
            per_sample = {}
            
            metric_names = metric_list[0].keys()
            
            for name in metric_names:
                values = [m[name] for m in metric_list]
                aggregated[f'{name}_mean'] = float(np.mean(values))
                aggregated[f'{name}_std'] = float(np.std(values))
                per_sample[name] = values
            
            results[region] = AnalysisResult(
                name=f'{region}_analysis',
                metrics=aggregated,
                per_sample=per_sample,
                metadata={'n_samples': len(metric_list)}
            )
        
        return results


class AblationAnalyzer:
    """
    Ablation study analysis framework.
    
    Systematically evaluates component contributions.
    """
    
    def __init__(self, baseline_name: str = 'full_model'):
        """
        Args:
            baseline_name: Name of the full model configuration
        """
        self.baseline_name = baseline_name
        self.configurations: Dict[str, Dict[str, List[float]]] = {}
    
    def add_configuration(
        self,
        name: str,
        removed_components: List[str],
        metrics: Dict[str, float]
    ):
        """
        Add a configuration result.
        
        Args:
            name: Configuration name
            removed_components: List of removed component names
            metrics: Evaluation metrics
        """
        if name not in self.configurations:
            self.configurations[name] = {
                'removed': removed_components,
                'metrics': []
            }
        
        self.configurations[name]['metrics'].append(metrics)
    
    def analyze(self) -> AnalysisResult:
        """
        Analyze ablation study results.
        
        Returns:
            AnalysisResult with component contributions
        """
        if self.baseline_name not in self.configurations:
            raise ValueError(f"Baseline {self.baseline_name} not found")
        
        baseline_metrics = self.configurations[self.baseline_name]['metrics']
        baseline_means = {}
        
        for metric_name in baseline_metrics[0].keys():
            values = [m[metric_name] for m in baseline_metrics]
            baseline_means[metric_name] = np.mean(values)
        
        # Compute contributions
        contributions = {}
        
        for config_name, config_data in self.configurations.items():
            if config_name == self.baseline_name:
                continue
            
            config_metrics = config_data['metrics']
            config_means = {}
            
            for metric_name in config_metrics[0].keys():
                values = [m[metric_name] for m in config_metrics]
                config_means[metric_name] = np.mean(values)
            
            # Contribution = baseline - ablated
            contrib = {}
            for metric_name in baseline_means.keys():
                diff = baseline_means[metric_name] - config_means[metric_name]
                contrib[metric_name] = diff
            
            removed = config_data['removed']
            key = '_'.join(removed) if removed else config_name
            contributions[key] = contrib
        
        return AnalysisResult(
            name='ablation_analysis',
            metrics=contributions,
            metadata={
                'baseline': self.baseline_name,
                'baseline_metrics': baseline_means,
                'n_configurations': len(self.configurations)
            }
        )
    
    def get_component_ranking(
        self,
        metric_name: str = 'ssim'
    ) -> List[Tuple[str, float]]:
        """
        Rank components by contribution to specific metric.
        
        Args:
            metric_name: Metric to use for ranking
            
        Returns:
            List of (component, contribution) sorted by contribution
        """
        analysis = self.analyze()
        
        ranking = []
        for component, metrics in analysis.metrics.items():
            if metric_name in metrics:
                ranking.append((component, metrics[metric_name]))
        
        return sorted(ranking, key=lambda x: x[1], reverse=True)


class CrossDatasetAnalyzer:
    """
    Cross-dataset generalization analysis.
    
    Evaluates how well models trained on one dataset
    perform on another.
    """
    
    def __init__(self, source_dataset: str, target_datasets: List[str]):
        """
        Args:
            source_dataset: Training dataset name
            target_datasets: List of evaluation dataset names
        """
        self.source_dataset = source_dataset
        self.target_datasets = target_datasets
        
        self.results: Dict[str, Dict[str, List[float]]] = {
            ds: {} for ds in target_datasets
        }
    
    def add_result(
        self,
        target_dataset: str,
        metrics: Dict[str, float]
    ):
        """
        Add evaluation result on target dataset.
        
        Args:
            target_dataset: Target dataset name
            metrics: Evaluation metrics
        """
        if target_dataset not in self.target_datasets:
            raise ValueError(f"Unknown target dataset: {target_dataset}")
        
        for metric_name, value in metrics.items():
            if metric_name not in self.results[target_dataset]:
                self.results[target_dataset][metric_name] = []
            self.results[target_dataset][metric_name].append(value)
    
    def analyze(self) -> AnalysisResult:
        """
        Analyze cross-dataset performance.
        
        Returns:
            AnalysisResult with generalization metrics
        """
        aggregated = {}
        per_dataset = {}
        
        for dataset, metrics in self.results.items():
            per_dataset[dataset] = {}
            
            for metric_name, values in metrics.items():
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                
                aggregated[f'{dataset}_{metric_name}_mean'] = mean_val
                aggregated[f'{dataset}_{metric_name}_std'] = std_val
                per_dataset[dataset][metric_name] = values
        
        # Compute generalization gap
        if len(self.target_datasets) >= 2:
            source_idx = self.target_datasets.index(self.source_dataset) \
                if self.source_dataset in self.target_datasets else 0
            
            for metric_name in list(self.results.values())[0].keys():
                source_perf = np.mean(
                    self.results[self.target_datasets[source_idx]][metric_name]
                )
                
                for i, dataset in enumerate(self.target_datasets):
                    if i == source_idx:
                        continue
                    target_perf = np.mean(self.results[dataset][metric_name])
                    gap = source_perf - target_perf
                    aggregated[f'gap_{self.source_dataset}_to_{dataset}_{metric_name}'] = gap
        
        return AnalysisResult(
            name='cross_dataset_analysis',
            metrics=aggregated,
            per_sample=per_dataset,
            metadata={
                'source_dataset': self.source_dataset,
                'target_datasets': self.target_datasets
            }
        )
    
    def get_generalization_score(
        self,
        metric_name: str = 'ssim'
    ) -> float:
        """
        Compute overall generalization score.
        
        Higher score = better generalization.
        
        Args:
            metric_name: Metric to use
            
        Returns:
            Generalization score (0-1)
        """
        values = []
        
        for dataset, metrics in self.results.items():
            if metric_name in metrics:
                values.append(np.mean(metrics[metric_name]))
        
        if len(values) < 2:
            return 1.0
        
        # Score based on consistency across datasets
        mean_perf = np.mean(values)
        std_perf = np.std(values)
        
        # Higher mean, lower std = better generalization
        cv = std_perf / (mean_perf + 1e-8)  # Coefficient of variation
        score = mean_perf * (1 - cv)
        
        return float(np.clip(score, 0, 1))
