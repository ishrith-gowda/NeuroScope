"""Utilities for working with paired MRI volumes."""

from pathlib import Path
import json
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any

from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class MRIPairLoader:
    """Loader for paired MRI volumes for tasks like image translation.
    
    This class handles loading and preprocessing paired volumes from different
    directories, ensuring matching filenames and consistent preprocessing.
    """
    
    def __init__(
        self,
        domain_a_dir: Union[str, Path],
        domain_b_dir: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        preprocessing_pipeline: Optional[Any] = None,
        transform_a: Optional[Any] = None,
        transform_b: Optional[Any] = None,
        paired: bool = True,
    ):
        """Initialize MRIPairLoader.
        
        Args:
            domain_a_dir: Directory with domain A volumes.
            domain_b_dir: Directory with domain B volumes.
            file_pattern: Glob pattern for input files.
            preprocessing_pipeline: Optional preprocessing pipeline.
            transform_a: Optional transforms for domain A.
            transform_b: Optional transforms for domain B.
            paired: Whether the data is paired (same filenames in both domains).
        """
        import glob
        
        self.domain_a_dir = Path(domain_a_dir)
        self.domain_b_dir = Path(domain_b_dir)
        self.file_pattern = file_pattern
        self.preprocessing_pipeline = preprocessing_pipeline
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.paired = paired
        
        # List files in both domains
        self.domain_a_files = sorted(glob.glob(str(self.domain_a_dir / self.file_pattern)))
        self.domain_b_files = sorted(glob.glob(str(self.domain_b_dir / self.file_pattern)))
        
        # Check for paired data
        if paired:
            # Extract filenames without directories
            a_filenames = [os.path.basename(f) for f in self.domain_a_files]
            b_filenames = [os.path.basename(f) for f in self.domain_b_files]
            
            # Find common filenames
            common_filenames = set(a_filenames).intersection(set(b_filenames))
            
            if not common_filenames:
                logger.warning("No common filenames found between domains. Check directories or set paired=False.")
            else:
                # Keep only matched files
                self.domain_a_files = [f for f in self.domain_a_files if os.path.basename(f) in common_filenames]
                self.domain_b_files = [f for f in self.domain_b_files if os.path.basename(f) in common_filenames]
                
                # Sort to ensure same order
                self.domain_a_files = sorted(self.domain_a_files, key=lambda x: os.path.basename(x))
                self.domain_b_files = sorted(self.domain_b_files, key=lambda x: os.path.basename(x))
    
    def __len__(self) -> int:
        """Return number of paired samples."""
        return len(self.domain_a_files)
    
    def load_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load a pair of volumes by index.
        
        Args:
            index: Index of the pair to load.
            
        Returns:
            Tuple of (domain_a_volume, domain_b_volume, filename).
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} items")
        
        # Get file paths
        a_file = self.domain_a_files[index]
        
        if self.paired:
            b_file = self.domain_b_files[index]
        else:
            # For unpaired data, randomly sample from domain B
            b_file = self.domain_b_files[np.random.randint(0, len(self.domain_b_files))]
        
        # Load volumes
        a_volume = self._load_volume(a_file)
        b_volume = self._load_volume(b_file)
        
        # Apply preprocessing if available
        if self.preprocessing_pipeline:
            a_volume = self.preprocessing_pipeline.preprocess(a_volume)
            b_volume = self.preprocessing_pipeline.preprocess(b_volume)
        
        # Apply domain-specific transforms if available
        if self.transform_a:
            a_volume = self.transform_a(a_volume)
        
        if self.transform_b:
            b_volume = self.transform_b(b_volume)
        
        # Return volumes and filename (for reference)
        return a_volume, b_volume, os.path.basename(a_file)
    
    def load_domain_a(self, index: int) -> Tuple[np.ndarray, str]:
        """Load a volume from domain A.
        
        Args:
            index: Index of the volume to load.
            
        Returns:
            Tuple of (domain_a_volume, filename).
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} items")
        
        # Get file path
        a_file = self.domain_a_files[index]
        
        # Load volume
        a_volume = self._load_volume(a_file)
        
        # Apply preprocessing if available
        if self.preprocessing_pipeline:
            a_volume = self.preprocessing_pipeline.preprocess(a_volume)
        
        # Apply domain-specific transform if available
        if self.transform_a:
            a_volume = self.transform_a(a_volume)
        
        # Return volume and filename (for reference)
        return a_volume, os.path.basename(a_file)
    
    def load_domain_b(self, index: int) -> Tuple[np.ndarray, str]:
        """Load a volume from domain B.
        
        Args:
            index: Index of the volume to load.
            
        Returns:
            Tuple of (domain_b_volume, filename).
        """
        if index >= len(self.domain_b_files):
            raise IndexError(f"Index {index} out of range for domain B with {len(self.domain_b_files)} items")
        
        # Get file path
        b_file = self.domain_b_files[index]
        
        # Load volume
        b_volume = self._load_volume(b_file)
        
        # Apply preprocessing if available
        if self.preprocessing_pipeline:
            b_volume = self.preprocessing_pipeline.preprocess(b_volume)
        
        # Apply domain-specific transform if available
        if self.transform_b:
            b_volume = self.transform_b(b_volume)
        
        # Return volume and filename (for reference)
        return b_volume, os.path.basename(b_file)
    
    def _load_volume(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load a volume from file.
        
        Args:
            file_path: Path to volume file.
            
        Returns:
            Volume as numpy array.
        """
        if str(file_path).endswith(".nii") or str(file_path).endswith(".nii.gz"):
            try:
                import nibabel as nib
                nii_img = nib.load(str(file_path))
                return np.asarray(nii_img.dataobj)
            except ImportError:
                raise ImportError("nibabel is required for loading NIfTI files")
        elif str(file_path).endswith(".npy"):
            return np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


class MRIPairDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for paired MRI volumes.
    
    This class extends MRIPairLoader to provide a PyTorch Dataset interface,
    making it compatible with PyTorch DataLoader for batch processing.
    """
    
    def __init__(
        self,
        domain_a_dir: Union[str, Path],
        domain_b_dir: Union[str, Path],
        file_pattern: str = "*.nii.gz",
        preprocessing_pipeline: Optional[Any] = None,
        transform_a: Optional[Any] = None,
        transform_b: Optional[Any] = None,
        paired: bool = True,
        load_into_memory: bool = False,
    ):
        """Initialize MRIPairDataset.
        
        Args:
            domain_a_dir: Directory with domain A volumes.
            domain_b_dir: Directory with domain B volumes.
            file_pattern: Glob pattern for input files.
            preprocessing_pipeline: Optional preprocessing pipeline.
            transform_a: Optional transforms for domain A.
            transform_b: Optional transforms for domain B.
            paired: Whether the data is paired (same filenames in both domains).
            load_into_memory: Whether to load all data into memory.
        """
        # Initialize loader
        self.loader = MRIPairLoader(
            domain_a_dir=domain_a_dir,
            domain_b_dir=domain_b_dir,
            file_pattern=file_pattern,
            preprocessing_pipeline=preprocessing_pipeline,
            transform_a=None,  # We'll apply transforms during __getitem__
            transform_b=None,
            paired=paired,
        )
        
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.load_into_memory = load_into_memory
        
        # Preload data if requested
        self.preloaded_data = None
        if load_into_memory:
            self.preloaded_data = []
            for i in range(len(self.loader)):
                a_volume, b_volume, filename = self.loader.load_pair(i)
                self.preloaded_data.append((a_volume, b_volume, filename))
    
    def __len__(self) -> int:
        """Return number of paired samples."""
        return len(self.loader)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a paired sample by index.
        
        Args:
            index: Index of the pair to load.
            
        Returns:
            Dictionary with 'A', 'B', and 'filename' keys.
        """
        # Load data
        if self.preloaded_data:
            a_volume, b_volume, filename = self.preloaded_data[index]
        else:
            a_volume, b_volume, filename = self.loader.load_pair(index)
        
        # Apply transforms if available
        if self.transform_a:
            a_volume = self.transform_a(a_volume)
        
        if self.transform_b:
            b_volume = self.transform_b(b_volume)
        
        # Convert to torch tensors if not already
        if not isinstance(a_volume, torch.Tensor):
            a_volume = torch.tensor(a_volume, dtype=torch.float32)
        
        if not isinstance(b_volume, torch.Tensor):
            b_volume = torch.tensor(b_volume, dtype=torch.float32)
        
        # Return as dictionary
        return {
            'A': a_volume,
            'B': b_volume,
            'filename': filename,
        }


def create_paired_dataset_splits(
    domain_a_dir: Union[str, Path],
    domain_b_dir: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    file_pattern: str = "*.nii.gz",
    paired: bool = True,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create train/val/test splits for paired dataset.
    
    Args:
        domain_a_dir: Directory with domain A volumes.
        domain_b_dir: Directory with domain B volumes.
        output_dir: Output directory for split files.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        file_pattern: Glob pattern for input files.
        paired: Whether the data is paired (same filenames in both domains).
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary with filenames for each split.
    """
    import glob
    import random
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # List files in both domains
    domain_a_files = sorted(glob.glob(str(Path(domain_a_dir) / file_pattern)))
    domain_b_files = sorted(glob.glob(str(Path(domain_b_dir) / file_pattern)))
    
    # Extract filenames without directories
    a_filenames = [os.path.basename(f) for f in domain_a_files]
    b_filenames = [os.path.basename(f) for f in domain_b_files]
    
    if paired:
        # Find common filenames
        common_filenames = sorted(list(set(a_filenames).intersection(set(b_filenames))))
        
        if not common_filenames:
            logger.warning("No common filenames found between domains. Check directories or set paired=False.")
            return {}
        
        # Shuffle filenames
        random.shuffle(common_filenames)
        
        # Calculate split sizes
        total = len(common_filenames)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        test_size = total - train_size - val_size
        
        # Create splits
        train_files = common_filenames[:train_size]
        val_files = common_filenames[train_size:train_size + val_size]
        test_files = common_filenames[train_size + val_size:]
        
    else:
        # Independently shuffle each domain
        random.shuffle(a_filenames)
        random.shuffle(b_filenames)
        
        # Calculate split sizes for domain A
        total_a = len(a_filenames)
        train_size_a = int(total_a * train_ratio)
        val_size_a = int(total_a * val_ratio)
        test_size_a = total_a - train_size_a - val_size_a
        
        # Calculate split sizes for domain B
        total_b = len(b_filenames)
        train_size_b = int(total_b * train_ratio)
        val_size_b = int(total_b * val_ratio)
        test_size_b = total_b - train_size_b - val_size_b
        
        # Create splits for domain A
        train_files_a = a_filenames[:train_size_a]
        val_files_a = a_filenames[train_size_a:train_size_a + val_size_a]
        test_files_a = a_filenames[train_size_a + val_size_a:]
        
        # Create splits for domain B
        train_files_b = b_filenames[:train_size_b]
        val_files_b = b_filenames[train_size_b:train_size_b + val_size_b]
        test_files_b = b_filenames[train_size_b + val_size_b:]
        
        # Store separately for unpaired data
        train_files_dict = {"A": train_files_a, "B": train_files_b}
        val_files_dict = {"A": val_files_a, "B": val_files_b}
        test_files_dict = {"A": test_files_a, "B": test_files_b}
        
        train_files = train_files_dict
        val_files = val_files_dict
        test_files = test_files_dict
    
    # Prepare output data
    split_data = {
        "paired": paired,
        "domain_a_dir": str(domain_a_dir),
        "domain_b_dir": str(domain_b_dir),
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    
    # Save splits to JSON file
    with open(output_dir / "dataset_splits.json", "w") as f:
        json.dump(split_data, f, indent=4)
    
    logger.info(f"Dataset splits saved to {output_dir / 'dataset_splits.json'}")
    
    if paired:
        logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)} samples")
    else:
        logger.info(f"Train: {len(train_files_dict['A'])} (A), {len(train_files_dict['B'])} (B) samples")
        logger.info(f"Val: {len(val_files_dict['A'])} (A), {len(val_files_dict['B'])} (B) samples")
        logger.info(f"Test: {len(test_files_dict['A'])} (A), {len(test_files_dict['B'])} (B) samples")
    
    return split_data