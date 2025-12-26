"""
I/O Utilities.

File handling, checkpoint management, and configuration loading
for medical imaging data.
"""

from typing import Optional, Dict, List, Any, Union, Tuple
from pathlib import Path
import json
import shutil
import re
import torch
import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files matching pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    Copy file to destination.
    
    Args:
        src: Source file
        dst: Destination
        overwrite: Allow overwriting
        
    Returns:
        Destination path
    """
    src, dst = Path(src), Path(dst)
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    
    return dst


# NIfTI Handling

def load_nifti(
    path: Union[str, Path],
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, Any]:
    """
    Load NIfTI file.
    
    Args:
        path: Path to NIfTI file
        dtype: Data type
        
    Returns:
        Tuple of (data array, affine matrix)
    """
    import nibabel as nib
    
    img = nib.load(str(path))
    data = img.get_fdata().astype(dtype)
    affine = img.affine
    
    return data, affine


def save_nifti(
    data: np.ndarray,
    path: Union[str, Path],
    affine: np.ndarray = None
):
    """
    Save data as NIfTI file.
    
    Args:
        data: Data array
        path: Output path
        affine: Affine transformation matrix
    """
    import nibabel as nib
    
    if affine is None:
        affine = np.eye(4)
    
    img = nib.Nifti1Image(data, affine)
    
    ensure_dir(Path(path).parent)
    nib.save(img, str(path))


def load_nifti_as_tensor(
    path: Union[str, Path],
    device: str = 'cpu',
    add_batch_dim: bool = True,
    add_channel_dim: bool = True
) -> torch.Tensor:
    """
    Load NIfTI file as PyTorch tensor.
    
    Args:
        path: Path to NIfTI file
        device: Device to load to
        add_batch_dim: Add batch dimension
        add_channel_dim: Add channel dimension
        
    Returns:
        Tensor
    """
    data, _ = load_nifti(path)
    tensor = torch.from_numpy(data).float().to(device)
    
    if add_channel_dim and tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    
    return tensor


# Checkpoint Management

def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Dict[str, float] = None,
    config: Dict = None,
    **kwargs
) -> Path:
    """
    Save training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        metrics: Evaluation metrics
        config: Configuration
        **kwargs: Additional items to save
        
    Returns:
        Saved checkpoint path
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'metrics': metrics or {},
        'config': config or {}
    }
    
    if model is not None:
        if hasattr(model, 'module'):
            # DataParallel/DistributedDataParallel
            checkpoint['model_state_dict'] = model.module.state_dict()
        else:
            checkpoint['model_state_dict'] = model.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module = None,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: str = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load to
        strict: Strict state dict loading
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        if hasattr(model, 'module'):
            model.module.load_state_dict(
                checkpoint['model_state_dict'], strict=strict
            )
        else:
            model.load_state_dict(
                checkpoint['model_state_dict'], strict=strict
            )
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    pattern: str = "checkpoint_epoch_*.pth"
) -> Optional[Path]:
    """
    Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Checkpoint filename pattern
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Extract epoch/step numbers and sort
    def extract_number(path):
        match = re.search(r'(\d+)', path.stem)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=extract_number, reverse=True)
    return checkpoints[0]


def save_cyclegan_checkpoint(
    path: Union[str, Path],
    G_A2B: torch.nn.Module,
    G_B2A: torch.nn.Module,
    D_A: torch.nn.Module,
    D_B: torch.nn.Module,
    opt_G: torch.optim.Optimizer = None,
    opt_D: torch.optim.Optimizer = None,
    epoch: int = 0,
    **kwargs
) -> Path:
    """
    Save CycleGAN-specific checkpoint.
    
    Args:
        path: Checkpoint path
        G_A2B: Generator A to B
        G_B2A: Generator B to A
        D_A: Discriminator A
        D_B: Discriminator B
        opt_G: Generator optimizer
        opt_D: Discriminator optimizer
        epoch: Current epoch
        **kwargs: Additional items
        
    Returns:
        Saved checkpoint path
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    checkpoint = {
        'epoch': epoch,
        'G_A2B_state_dict': G_A2B.state_dict(),
        'G_B2A_state_dict': G_B2A.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
    }
    
    if opt_G is not None:
        checkpoint['opt_G_state_dict'] = opt_G.state_dict()
    
    if opt_D is not None:
        checkpoint['opt_D_state_dict'] = opt_D.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    return path


def load_cyclegan_checkpoint(
    path: Union[str, Path],
    G_A2B: torch.nn.Module = None,
    G_B2A: torch.nn.Module = None,
    D_A: torch.nn.Module = None,
    D_B: torch.nn.Module = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load CycleGAN-specific checkpoint.
    
    Args:
        path: Checkpoint path
        G_A2B: Generator A to B
        G_B2A: Generator B to A
        D_A: Discriminator A
        D_B: Discriminator B
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    if G_A2B is not None and 'G_A2B_state_dict' in checkpoint:
        G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
    
    if G_B2A is not None and 'G_B2A_state_dict' in checkpoint:
        G_B2A.load_state_dict(checkpoint['G_B2A_state_dict'])
    
    if D_A is not None and 'D_A_state_dict' in checkpoint:
        D_A.load_state_dict(checkpoint['D_A_state_dict'])
    
    if D_B is not None and 'D_B_state_dict' in checkpoint:
        D_B.load_state_dict(checkpoint['D_B_state_dict'])
    
    return checkpoint


# Configuration Handling

def load_config(
    path: Union[str, Path],
    format: str = 'auto'
) -> Dict[str, Any]:
    """
    Load configuration file.
    
    Args:
        path: Config file path
        format: 'yaml', 'json', or 'auto'
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if format == 'auto':
        format = path.suffix.lstrip('.')
    
    with open(path, 'r') as f:
        if format in ['yaml', 'yml']:
            import yaml
            return yaml.safe_load(f)
        elif format == 'json':
            return json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")


def save_config(
    config: Dict[str, Any],
    path: Union[str, Path],
    format: str = 'auto'
):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        path: Output path
        format: 'yaml', 'json', or 'auto'
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    if format == 'auto':
        format = path.suffix.lstrip('.')
    
    with open(path, 'w') as f:
        if format in ['yaml', 'yml']:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        elif format == 'json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
