"""
i/o utilities.

file handling, checkpoint management, and configuration loading
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
    ensure directory exists.
    
    args:
        path: directory path
        
    returns:
        path object
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
    list files matching pattern.
    
    args:
        directory: directory to search
        pattern: glob pattern
        recursive: search recursively
        
    returns:
        list of matching file paths
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
    copy file to destination.
    
    args:
        src: source file
        dst: destination
        overwrite: allow overwriting
        
    returns:
        destination path
    """
    src, dst = Path(src), Path(dst)
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists: {dst}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    
    return dst


# nifti handling

def load_nifti(
    path: Union[str, Path],
    dtype: np.dtype = np.float32
) -> Tuple[np.ndarray, Any]:
    """
    load nifti file.
    
    args:
        path: path to nifti file
        dtype: data type
        
    returns:
        tuple of (data array, affine matrix)
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
    save data as nifti file.
    
    args:
        data: data array
        path: output path
        affine: affine transformation matrix
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
    load nifti file as pytorch tensor.
    
    args:
        path: path to nifti file
        device: device to load to
        add_batch_dim: add batch dimension
        add_channel_dim: add channel dimension
        
    returns:
        tensor
    """
    data, _ = load_nifti(path)
    tensor = torch.from_numpy(data).float().to(device)
    
    if add_channel_dim and tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)
    
    return tensor


# checkpoint management

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
    save training checkpoint.
    
    args:
        path: checkpoint path
        model: model to save
        optimizer: optimizer state
        scheduler: scheduler state
        epoch: current epoch
        step: current step
        metrics: evaluation metrics
        config: configuration
        **kwargs: additional items to save
        
    returns:
        saved checkpoint path
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
            # dataparallel/distributeddataparallel
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
    load training checkpoint.
    
    args:
        path: checkpoint path
        model: model to load weights into
        optimizer: optimizer to load state into
        scheduler: scheduler to load state into
        device: device to load to
        strict: strict state dict loading
        
    returns:
        checkpoint dictionary
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
    get the latest checkpoint from a directory.
    
    args:
        checkpoint_dir: directory containing checkpoints
        pattern: checkpoint filename pattern
        
    returns:
        path to latest checkpoint or none
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        return None
    
    # extract epoch/step numbers and sort
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
    save cyclegan-specific checkpoint.
    
    args:
        path: checkpoint path
        g_a2b: generator a to b
        g_b2a: generator b to a
        d_a: discriminator a
        d_b: discriminator b
        opt_g: generator optimizer
        opt_d: discriminator optimizer
        epoch: current epoch
        **kwargs: additional items
        
    returns:
        saved checkpoint path
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
    load cyclegan-specific checkpoint.
    
    args:
        path: checkpoint path
        g_a2b: generator a to b
        g_b2a: generator b to a
        d_a: discriminator a
        d_b: discriminator b
        device: device to load to
        
    returns:
        checkpoint dictionary
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


# configuration handling

def load_config(
    path: Union[str, Path],
    format: str = 'auto'
) -> Dict[str, Any]:
    """
    load configuration file.
    
    args:
        path: config file path
        format: 'yaml', 'json', or 'auto'
        
    returns:
        configuration dictionary
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
    save configuration to file.
    
    args:
        config: configuration dictionary
        path: output path
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
    deep merge two configuration dictionaries.
    
    args:
        base: base configuration
        override: override configuration
        
    returns:
        merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
