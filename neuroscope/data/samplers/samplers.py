"""
custom samplers for training data.

provides balanced and stratified sampling strategies for
handling imbalanced datasets in medical imaging.
"""

from typing import List, Optional, Iterator, Dict, Sequence
import random
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset


class BalancedSampler(Sampler[int]):
    """
    balanced sampler that ensures equal sampling from each class/domain.
    
    useful for domain adaptation where source and target domains
    may have different sizes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        labels: Optional[List[int]] = None,
        samples_per_class: Optional[int] = None,
        replacement: bool = True
    ):
        """
        args:
            dataset: dataset to sample from
            labels: class/domain labels for each sample. if none, tries to
                   extract from dataset.
            samples_per_class: number of samples per class. if none, uses
                              the size of the smallest class.
            replacement: whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        
        # get labels
        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = self._extract_labels()
        
        # find unique classes and their indices
        self.classes = np.unique(self.labels)
        self.class_indices = {
            cls: np.where(self.labels == cls)[0]
            for cls in self.classes
        }
        
        # determine samples per class
        if samples_per_class is not None:
            self.samples_per_class = samples_per_class
        else:
            min_class_size = min(len(indices) for indices in self.class_indices.values())
            self.samples_per_class = min_class_size
        
        self.num_samples = self.samples_per_class * len(self.classes)
    
    def _extract_labels(self) -> np.ndarray:
        """try to extract labels from dataset."""
        labels = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if isinstance(item, dict):
                if 'label' in item:
                    labels.append(item['label'])
                elif 'domain' in item:
                    labels.append(hash(item['domain']) % 1000)
                else:
                    labels.append(0)
            else:
                labels.append(0)
        return np.array(labels)
    
    def __iter__(self) -> Iterator[int]:
        indices = []
        
        for cls in self.classes:
            class_indices = self.class_indices[cls]
            
            if self.replacement or len(class_indices) >= self.samples_per_class:
                sampled = np.random.choice(
                    class_indices,
                    size=self.samples_per_class,
                    replace=self.replacement
                )
            else:
                # repeat indices if not enough samples
                repeats = self.samples_per_class // len(class_indices) + 1
                expanded = np.tile(class_indices, repeats)
                sampled = expanded[:self.samples_per_class]
            
            indices.extend(sampled)
        
        # shuffle
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler[int]):
    """
    weighted random sampler with customizable weights.
    
    can handle both class-based and sample-based weighting.
    """
    
    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None
    ):
        """
        args:
            weights: weight for each sample
            num_samples: number of samples to draw
            replacement: whether to sample with replacement
            generator: optional random generator
        """
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator
        )
        yield from indices.tolist()
    
    def __len__(self) -> int:
        return self.num_samples
    
    @classmethod
    def from_labels(
        cls,
        labels: Sequence[int],
        num_samples: Optional[int] = None
    ) -> 'WeightedRandomSampler':
        """
        create sampler from class labels with inverse frequency weighting.
        
        args:
            labels: class label for each sample
            num_samples: number of samples to draw (default: len(labels))
        """
        labels = np.array(labels)
        class_counts = np.bincount(labels)
        
        # inverse frequency weighting
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        if num_samples is None:
            num_samples = len(labels)
        
        return cls(sample_weights, num_samples)


class DomainBalancedSampler(Sampler[int]):
    """
    sampler for balanced sampling across domains.
    
    specifically designed for domain adaptation scenarios
    with multiple source/target domains.
    """
    
    def __init__(
        self,
        domain_datasets: Dict[str, Dataset],
        samples_per_domain: Optional[int] = None,
        strategy: str = 'uniform'  # 'uniform', 'proportional', 'inverse'
    ):
        """
        args:
            domain_datasets: dictionary mapping domain names to datasets
            samples_per_domain: samples per domain per epoch
            strategy: sampling strategy
        """
        self.domain_datasets = domain_datasets
        self.domains = list(domain_datasets.keys())
        self.strategy = strategy
        
        # compute domain sizes
        self.domain_sizes = {
            domain: len(dataset)
            for domain, dataset in domain_datasets.items()
        }
        
        # determine samples per domain
        if samples_per_domain is not None:
            self.samples_per_domain = samples_per_domain
        elif strategy == 'uniform':
            self.samples_per_domain = min(self.domain_sizes.values())
        else:
            self.samples_per_domain = max(self.domain_sizes.values())
        
        # compute domain weights
        if strategy == 'uniform':
            self.domain_weights = {d: 1.0 for d in self.domains}
        elif strategy == 'proportional':
            total = sum(self.domain_sizes.values())
            self.domain_weights = {
                d: size / total for d, size in self.domain_sizes.items()
            }
        elif strategy == 'inverse':
            inv_sizes = {d: 1.0 / s for d, s in self.domain_sizes.items()}
            total = sum(inv_sizes.values())
            self.domain_weights = {d: w / total for d, w in inv_sizes.items()}
        
        # build global index mapping
        self._build_index_mapping()
    
    def _build_index_mapping(self):
        """build mapping from global index to (domain, local_index)."""
        self.index_mapping = []
        offset = 0
        
        for domain in self.domains:
            size = self.domain_sizes[domain]
            for i in range(size):
                self.index_mapping.append((domain, i, offset + i))
            offset += size
        
        self.total_samples = len(self.index_mapping)
    
    def __iter__(self) -> Iterator[int]:
        indices = []
        
        for domain in self.domains:
            domain_size = self.domain_sizes[domain]
            
            # sample from this domain
            if domain_size >= self.samples_per_domain:
                domain_indices = np.random.choice(
                    domain_size,
                    size=self.samples_per_domain,
                    replace=False
                )
            else:
                domain_indices = np.random.choice(
                    domain_size,
                    size=self.samples_per_domain,
                    replace=True
                )
            
            # convert to global indices
            offset = sum(
                self.domain_sizes[d]
                for d in self.domains[:self.domains.index(domain)]
            )
            global_indices = domain_indices + offset
            indices.extend(global_indices)
        
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return self.samples_per_domain * len(self.domains)
    
    def get_domain_for_index(self, global_idx: int) -> str:
        """get domain name for a global index."""
        return self.index_mapping[global_idx][0]


class StratifiedSampler(Sampler[int]):
    """
    stratified sampler that maintains class distribution in batches.
    
    useful for maintaining consistent class ratios within each batch.
    """
    
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        args:
            labels: class label for each sample
            batch_size: batch size
            shuffle: whether to shuffle within strata
            drop_last: whether to drop incomplete batches
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # group indices by class
        self.classes = np.unique(self.labels)
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist()
            for cls in self.classes
        }
        
        # compute samples per class per batch
        class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        total = sum(class_counts.values())
        
        self.class_batch_sizes = {
            cls: max(1, round(batch_size * count / total))
            for cls, count in class_counts.items()
        }
        
        # adjust to match batch size
        while sum(self.class_batch_sizes.values()) != batch_size:
            diff = batch_size - sum(self.class_batch_sizes.values())
            if diff > 0:
                cls = max(class_counts, key=class_counts.get)
                self.class_batch_sizes[cls] += 1
            else:
                cls = min(class_counts, key=class_counts.get)
                self.class_batch_sizes[cls] = max(1, self.class_batch_sizes[cls] - 1)
    
    def __iter__(self) -> Iterator[int]:
        # shuffle within each class
        if self.shuffle:
            for cls in self.classes:
                random.shuffle(self.class_indices[cls])
        
        # create iterators for each class
        class_iters = {
            cls: iter(self.class_indices[cls])
            for cls in self.classes
        }
        
        batches = []
        while True:
            batch = []
            for cls in self.classes:
                cls_batch_size = self.class_batch_sizes[cls]
                
                for _ in range(cls_batch_size):
                    try:
                        batch.append(next(class_iters[cls]))
                    except StopIteration:
                        if self.shuffle:
                            random.shuffle(self.class_indices[cls])
                        class_iters[cls] = iter(self.class_indices[cls])
                        try:
                            batch.append(next(class_iters[cls]))
                        except StopIteration:
                            break
            
            if len(batch) < self.batch_size:
                if not self.drop_last and batch:
                    batches.extend(batch)
                break
            
            if self.shuffle:
                random.shuffle(batch)
            batches.extend(batch)
        
        return iter(batches)
    
    def __len__(self) -> int:
        total = len(self.labels)
        if self.drop_last:
            return (total // self.batch_size) * self.batch_size
        return total


class SubsetRandomSampler(Sampler[int]):
    """
    random sampler from a subset of indices.
    
    useful for train/val splits or selecting specific samples.
    """
    
    def __init__(
        self,
        indices: Sequence[int],
        generator: Optional[torch.Generator] = None
    ):
        """
        args:
            indices: sequence of indices to sample from
            generator: optional random generator
        """
        self.indices = list(indices)
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        if self.generator is not None:
            perm = torch.randperm(len(self.indices), generator=self.generator)
            indices = [self.indices[i] for i in perm]
        else:
            indices = self.indices.copy()
            random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @classmethod
    def random_split(
        cls,
        dataset_size: int,
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ) -> tuple:
        """
        create train/val samplers from random split.
        
        args:
            dataset_size: total number of samples
            train_ratio: ratio for training set
            seed: random seed for reproducibility
            
        returns:
            tuple of (train_sampler, val_sampler)
        """
        if seed is not None:
            random.seed(seed)
        
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        split_idx = int(dataset_size * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        return cls(train_indices), cls(val_indices)
