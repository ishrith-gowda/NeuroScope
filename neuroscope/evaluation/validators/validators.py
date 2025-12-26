"""
Model Validators.

Cross-validation and validation frameworks for
robust model evaluation.
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class ValidationResult:
    """Result from validation run."""
    fold: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    best_epoch: int = 0
    training_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of all validation folds."""
    n_folds: int
    metric_means: Dict[str, float]
    metric_stds: Dict[str, float]
    fold_results: List[ValidationResult]
    best_fold: int = 0
    
    def get_summary_string(self) -> str:
        """Get human-readable summary."""
        lines = [f"Validation Summary ({self.n_folds} folds)"]
        lines.append("-" * 40)
        
        for metric, mean in self.metric_means.items():
            std = self.metric_stds.get(metric, 0)
            lines.append(f"{metric}: {mean:.4f} ± {std:.4f}")
        
        return "\n".join(lines)


class BaseValidator(ABC):
    """Base class for validators."""
    
    @abstractmethod
    def get_splits(
        self,
        data_indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get train/validation splits.
        
        Args:
            data_indices: Array of data indices
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        pass
    
    @abstractmethod
    def validate(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        data_indices: np.ndarray
    ) -> ValidationSummary:
        """
        Run validation.
        
        Args:
            train_fn: Function to train model
            eval_fn: Function to evaluate model
            data_indices: Data indices
            
        Returns:
            ValidationSummary
        """
        pass


class CrossValidator(BaseValidator):
    """
    K-Fold Cross-Validation.
    
    Standard cross-validation with optional stratification.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        stratified: bool = False
    ):
        """
        Args:
            n_folds: Number of folds
            shuffle: Shuffle before splitting
            random_state: Random seed
            stratified: Stratified sampling
        """
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified
    
    def get_splits(
        self,
        data_indices: np.ndarray,
        labels: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get K-fold splits.
        
        Args:
            data_indices: Array of data indices
            labels: Optional labels for stratification
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_samples = len(data_indices)
        indices = data_indices.copy()
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        if self.stratified and labels is not None:
            return self._stratified_splits(indices, labels)
        
        # Regular K-fold
        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds)
        fold_sizes[:n_samples % self.n_folds] += 1
        
        splits = []
        current = 0
        
        for fold_size in fold_sizes:
            val_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])
            splits.append((train_indices, val_indices))
            current += fold_size
        
        return splits
    
    def _stratified_splits(
        self,
        indices: np.ndarray,
        labels: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits."""
        unique_labels = np.unique(labels)
        
        # Group indices by label
        label_indices = {
            label: indices[labels == label]
            for label in unique_labels
        }
        
        # Split each group
        splits = [
            (np.array([], dtype=int), np.array([], dtype=int))
            for _ in range(self.n_folds)
        ]
        
        for label, group_indices in label_indices.items():
            group_splits = CrossValidator(
                n_folds=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state,
                stratified=False
            ).get_splits(group_indices)
            
            for i, (train_idx, val_idx) in enumerate(group_splits):
                splits[i] = (
                    np.concatenate([splits[i][0], train_idx]),
                    np.concatenate([splits[i][1], val_idx])
                )
        
        return splits
    
    def validate(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        data_indices: np.ndarray,
        labels: np.ndarray = None
    ) -> ValidationSummary:
        """
        Run K-fold cross-validation.
        
        Args:
            train_fn: Function(train_indices) -> model
            eval_fn: Function(model, val_indices) -> metrics
            data_indices: Array of data indices
            labels: Optional labels for stratification
            
        Returns:
            ValidationSummary
        """
        import time
        
        splits = self.get_splits(data_indices, labels)
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(splits):
            start_time = time.time()
            
            # Train
            model, train_metrics = train_fn(train_indices)
            
            # Evaluate
            val_metrics = eval_fn(model, val_indices)
            
            training_time = time.time() - start_time
            
            result = ValidationResult(
                fold=fold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                training_time=training_time
            )
            fold_results.append(result)
        
        # Aggregate results
        return self._aggregate_results(fold_results)
    
    def _aggregate_results(
        self,
        fold_results: List[ValidationResult]
    ) -> ValidationSummary:
        """Aggregate fold results."""
        all_metrics = {}
        
        for result in fold_results:
            for metric, value in result.val_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        metric_means = {
            k: float(np.mean(v)) for k, v in all_metrics.items()
        }
        metric_stds = {
            k: float(np.std(v)) for k, v in all_metrics.items()
        }
        
        # Find best fold (highest mean metric)
        primary_metric = list(all_metrics.keys())[0]
        best_fold = int(np.argmax(all_metrics[primary_metric]))
        
        return ValidationSummary(
            n_folds=len(fold_results),
            metric_means=metric_means,
            metric_stds=metric_stds,
            fold_results=fold_results,
            best_fold=best_fold
        )


class HoldoutValidator(BaseValidator):
    """
    Holdout Validation.
    
    Simple train/validation/test split.
    """
    
    def __init__(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Shuffle before splitting
            random_state: Random seed
        """
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.random_state = random_state
    
    def get_splits(
        self,
        data_indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get train/val split."""
        return [self.get_train_val_test_split(data_indices)[:2]]
    
    def get_train_val_test_split(
        self,
        data_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/validation/test split.
        
        Args:
            data_indices: Array of data indices
            
        Returns:
            Tuple of (train, val, test) indices
        """
        n_samples = len(data_indices)
        indices = data_indices.copy()
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        n_test = int(n_samples * self.test_ratio)
        n_val = int(n_samples * self.val_ratio)
        n_train = n_samples - n_test - n_val
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        return train_indices, val_indices, test_indices
    
    def validate(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        data_indices: np.ndarray
    ) -> ValidationSummary:
        """
        Run holdout validation.
        
        Args:
            train_fn: Training function
            eval_fn: Evaluation function
            data_indices: Data indices
            
        Returns:
            ValidationSummary
        """
        import time
        
        train_indices, val_indices, test_indices = \
            self.get_train_val_test_split(data_indices)
        
        start_time = time.time()
        
        model, train_metrics = train_fn(train_indices)
        val_metrics = eval_fn(model, val_indices)
        test_metrics = eval_fn(model, test_indices) if len(test_indices) > 0 else None
        
        training_time = time.time() - start_time
        
        result = ValidationResult(
            fold=0,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            training_time=training_time
        )
        
        return ValidationSummary(
            n_folds=1,
            metric_means=val_metrics,
            metric_stds={k: 0.0 for k in val_metrics.keys()},
            fold_results=[result],
            best_fold=0
        )


class TemporalValidator(BaseValidator):
    """
    Temporal Validation.
    
    Time-based splitting for temporal data.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: int = None
    ):
        """
        Args:
            n_splits: Number of splits
            gap: Gap between train and test
            test_size: Fixed test size
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
    
    def get_splits(
        self,
        data_indices: np.ndarray,
        timestamps: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get temporal splits.
        
        Args:
            data_indices: Data indices (assumed sorted by time)
            timestamps: Optional timestamps for sorting
            
        Returns:
            List of (train_indices, val_indices)
        """
        if timestamps is not None:
            # Sort by timestamp
            sort_idx = np.argsort(timestamps)
            data_indices = data_indices[sort_idx]
        
        n_samples = len(data_indices)
        test_size = self.test_size or n_samples // (self.n_splits + 1)
        
        splits = []
        
        for i in range(self.n_splits):
            train_end = n_samples - (self.n_splits - i) * test_size - self.gap
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if train_end <= 0 or test_end > n_samples:
                continue
            
            train_indices = data_indices[:train_end]
            test_indices = data_indices[test_start:test_end]
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def validate(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        data_indices: np.ndarray,
        timestamps: np.ndarray = None
    ) -> ValidationSummary:
        """
        Run temporal validation.
        
        Args:
            train_fn: Training function
            eval_fn: Evaluation function
            data_indices: Data indices
            timestamps: Timestamps
            
        Returns:
            ValidationSummary
        """
        import time
        
        splits = self.get_splits(data_indices, timestamps)
        fold_results = []
        
        for fold, (train_indices, val_indices) in enumerate(splits):
            start_time = time.time()
            
            model, train_metrics = train_fn(train_indices)
            val_metrics = eval_fn(model, val_indices)
            
            training_time = time.time() - start_time
            
            result = ValidationResult(
                fold=fold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                training_time=training_time
            )
            fold_results.append(result)
        
        # Aggregate
        all_metrics = {}
        for result in fold_results:
            for metric, value in result.val_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        metric_means = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        metric_stds = {k: float(np.std(v)) for k, v in all_metrics.items()}
        
        return ValidationSummary(
            n_folds=len(fold_results),
            metric_means=metric_means,
            metric_stds=metric_stds,
            fold_results=fold_results
        )


class NestedCrossValidator:
    """
    Nested Cross-Validation.
    
    Inner loop for hyperparameter tuning,
    outer loop for unbiased evaluation.
    """
    
    def __init__(
        self,
        outer_folds: int = 5,
        inner_folds: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            outer_folds: Outer CV folds
            inner_folds: Inner CV folds
            random_state: Random seed
        """
        self.outer_cv = CrossValidator(
            n_folds=outer_folds,
            random_state=random_state
        )
        self.inner_cv = CrossValidator(
            n_folds=inner_folds,
            random_state=random_state + 1
        )
    
    def validate(
        self,
        train_fn: Callable,
        eval_fn: Callable,
        hyperparams_fn: Callable,
        data_indices: np.ndarray
    ) -> ValidationSummary:
        """
        Run nested cross-validation.
        
        Args:
            train_fn: Training function(indices, hyperparams) -> model
            eval_fn: Evaluation function
            hyperparams_fn: Hyperparameter search function
            data_indices: Data indices
            
        Returns:
            ValidationSummary
        """
        outer_splits = self.outer_cv.get_splits(data_indices)
        fold_results = []
        
        for fold, (train_indices, test_indices) in enumerate(outer_splits):
            # Inner loop: hyperparameter tuning
            best_hyperparams = hyperparams_fn(
                train_fn, eval_fn, train_indices, self.inner_cv
            )
            
            # Outer loop: train with best hyperparams, test
            model, train_metrics = train_fn(train_indices, best_hyperparams)
            test_metrics = eval_fn(model, test_indices)
            
            result = ValidationResult(
                fold=fold,
                train_metrics=train_metrics,
                val_metrics=test_metrics,
                metadata={'best_hyperparams': best_hyperparams}
            )
            fold_results.append(result)
        
        return self.outer_cv._aggregate_results(fold_results)
