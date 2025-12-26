"""
DataLoader Factories and Utilities.

Provides convenient factories for creating DataLoaders with
proper configuration for training, validation, and testing.
"""

from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import multiprocessing


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader creation."""
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False
    
    def __post_init__(self):
        # Auto-configure based on system
        if self.num_workers == -1:
            self.num_workers = min(4, multiprocessing.cpu_count() // 2)
        
        # Disable persistent workers if no workers
        if self.num_workers == 0:
            self.persistent_workers = False


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Create DataLoader with sensible defaults.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if sampler is provided)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
        sampler: Optional custom sampler
        collate_fn: Optional custom collation function
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep workers alive between epochs
        
    Returns:
        Configured DataLoader
    """
    # Handle worker configuration
    if num_workers == -1:
        num_workers = min(4, multiprocessing.cpu_count() // 2)
    
    # Disable features incompatible with no workers
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False
    
    # Shuffle and sampler are mutually exclusive
    if sampler is not None:
        shuffle = False
    
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'persistent_workers': persistent_workers and num_workers > 0,
    }
    
    if sampler is not None:
        loader_kwargs['sampler'] = sampler
    
    if collate_fn is not None:
        loader_kwargs['collate_fn'] = collate_fn
    
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    
    return DataLoader(dataset, **loader_kwargs)


def create_train_loader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    sampler: Optional[Sampler] = None,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create DataLoader optimized for training.
    
    Features:
        - Shuffling enabled (unless custom sampler provided)
        - Drop last batch to avoid batch norm issues
        - Pin memory for faster GPU transfer
        - Persistent workers for faster epoch transitions
    """
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )


def create_val_loader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create DataLoader optimized for validation.
    
    Features:
        - No shuffling for reproducibility
        - Larger batch size (no gradient storage needed)
        - Keep all samples (no drop_last)
    """
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=False
    )


def create_test_loader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 2,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Create DataLoader optimized for testing/inference.
    
    Features:
        - Small batch size for per-sample metrics
        - No shuffling for reproducibility
        - Fewer workers (less overhead)
    """
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=1,
        persistent_workers=False
    )


class InfiniteDataLoader:
    """
    DataLoader that cycles indefinitely.
    
    Useful for training loops that use iteration count
    instead of epoch count.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 4,
        sampler: Optional[Sampler] = None,
        collate_fn: Optional[Callable] = None
    ):
        """
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            num_workers: Number of worker processes
            sampler: Optional custom sampler
            collate_fn: Optional custom collation function
        """
        self.dataset = dataset
        self.loader = create_train_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=collate_fn
        )
        self._iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.loader)
        
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        
        return batch
    
    def __len__(self):
        return len(self.loader)
    
    def reset(self):
        """Reset the iterator."""
        self._iterator = None


class PrefetchDataLoader:
    """
    DataLoader wrapper with GPU prefetching for faster training.
    
    Prefetches the next batch to GPU while current batch is being processed.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device = None
    ):
        """
        Args:
            loader: DataLoader to wrap
            device: Device to prefetch to (default: current CUDA device)
        """
        self.loader = loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    
    def __iter__(self):
        loader_iter = iter(self.loader)
        
        # Prefetch first batch
        try:
            batch = next(loader_iter)
            batch = self._to_device(batch)
        except StopIteration:
            return
        
        for next_batch in loader_iter:
            # Start prefetch of next batch
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = self._to_device(next_batch)
            else:
                next_batch = self._to_device(next_batch)
            
            # Yield current batch
            yield batch
            
            # Wait for prefetch to complete
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)
            
            batch = next_batch
        
        # Yield last batch
        yield batch
    
    def _to_device(self, batch: Any) -> Any:
        """Recursively move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        else:
            return batch
    
    def __len__(self):
        return len(self.loader)


class MultiDomainLoader:
    """
    Loader for multiple domains with synchronized iteration.
    
    Yields batches from multiple domains simultaneously.
    """
    
    def __init__(
        self,
        domain_datasets: Dict[str, Dataset],
        batch_size: int = 8,
        num_workers: int = 2,
        strategy: str = 'zip'  # 'zip' or 'cycle'
    ):
        """
        Args:
            domain_datasets: Dictionary mapping domain names to datasets
            batch_size: Batch size per domain
            num_workers: Workers per loader
            strategy: 'zip' for parallel iteration, 'cycle' for infinite cycling
        """
        self.domains = list(domain_datasets.keys())
        self.strategy = strategy
        
        self.loaders = {
            domain: create_train_loader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
            for domain, dataset in domain_datasets.items()
        }
    
    def __iter__(self):
        if self.strategy == 'zip':
            # Parallel iteration
            iterators = {domain: iter(loader) for domain, loader in self.loaders.items()}
            
            while True:
                batch = {}
                try:
                    for domain in self.domains:
                        batch[domain] = next(iterators[domain])
                except StopIteration:
                    break
                yield batch
        
        else:  # cycle
            # Infinite cycling
            iterators = {domain: iter(loader) for domain, loader in self.loaders.items()}
            
            while True:
                batch = {}
                for domain in self.domains:
                    try:
                        batch[domain] = next(iterators[domain])
                    except StopIteration:
                        iterators[domain] = iter(self.loaders[domain])
                        batch[domain] = next(iterators[domain])
                yield batch
    
    def __len__(self):
        if self.strategy == 'zip':
            return min(len(loader) for loader in self.loaders.values())
        else:
            return max(len(loader) for loader in self.loaders.values())


def collate_medical_images(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collation function for medical image batches.
    
    Handles variable-sized images and optional segmentation masks.
    """
    collated = {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch if key in item]
        
        if not values:
            continue
        
        if isinstance(values[0], torch.Tensor):
            # Stack tensors
            try:
                collated[key] = torch.stack(values)
            except RuntimeError:
                # Variable sizes - keep as list
                collated[key] = values
        elif isinstance(values[0], np.ndarray):
            # Convert numpy arrays
            try:
                collated[key] = torch.stack([torch.from_numpy(v) for v in values])
            except RuntimeError:
                collated[key] = [torch.from_numpy(v) for v in values]
        elif isinstance(values[0], (int, float)):
            # Numeric values
            collated[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            # String values
            collated[key] = values
        else:
            # Keep as list
            collated[key] = values
    
    return collated


# Import numpy for collate function
import numpy as np
