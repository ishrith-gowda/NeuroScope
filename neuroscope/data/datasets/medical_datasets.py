"""
comprehensive medical imaging dataset implementations.

provides dataset classes for major neuroimaging datasets:
- ixi (information extraction from images)
- oasis (open access series of imaging studies)
- adni (alzheimer's disease neuroimaging initiative)
- abide (autism brain imaging data exchange)
- hcp (human connectome project)
- tcga-gbm (the cancer genome atlas glioblastoma)

all datasets support both 2d slice-wise and 3d volumetric loading.
"""

from typing import List, Optional, Dict, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import abstractmethod
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import torchio as tio
    HAS_TORCHIO = True
except ImportError:
    HAS_TORCHIO = False


@dataclass
class DatasetStats:
    """statistics for a dataset."""
    n_subjects: int
    n_slices: int
    modalities: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]
    resolution: Tuple[float, float, float]


class BaseMedicalDataset(Dataset):
    """
    base class for medical imaging datasets.
    
    provides common functionality for loading, preprocessing,
    and augmentation of neuroimaging data.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        mode: str = '2d',  # '2d', '3d', or '2.5d'
        slice_axis: int = 2,
        slice_range: Optional[Tuple[int, int]] = None,
        cache_data: bool = False,
        return_metadata: bool = False
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.modalities = modalities or self._default_modalities()
        self.transform = transform
        self.mode = mode
        self.slice_axis = slice_axis
        self.slice_range = slice_range or self._default_slice_range()
        self.cache_data = cache_data
        self.return_metadata = return_metadata
        
        self._cache = {}
        self.samples = self._discover_samples()
    
    @abstractmethod
    def _default_modalities(self) -> List[str]:
        """return default modalities for dataset."""
        pass
    
    @abstractmethod
    def _default_slice_range(self) -> Tuple[int, int]:
        """return default slice range."""
        pass
    
    @abstractmethod
    def _discover_samples(self) -> List[Dict[str, Any]]:
        """discover and return list of sample metadata."""
        pass
    
    @abstractmethod
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        """load a single modality volume."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        
        # check cache
        cache_key = f"{sample['subject_id']}_{sample.get('slice_idx', 'vol')}"
        if self.cache_data and cache_key in self._cache:
            item = self._cache[cache_key].copy()
        else:
            item = self._load_sample(sample)
            if self.cache_data:
                self._cache[cache_key] = item.copy()
        
        # apply transforms
        if self.transform is not None:
            item = self.transform(item)
        
        return item
    
    def _load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """load a sample (slice or volume)."""
        modality_data = []
        
        for modality in self.modalities:
            volume = self._load_volume(sample, modality)
            
            if self.mode == '2d':
                slice_idx = sample['slice_idx']
                if self.slice_axis == 0:
                    data = volume[slice_idx, :, :]
                elif self.slice_axis == 1:
                    data = volume[:, slice_idx, :]
                else:
                    data = volume[:, :, slice_idx]
                modality_data.append(data)
            else:
                modality_data.append(volume)
        
        # stack modalities
        if self.mode == '2d':
            image = np.stack(modality_data, axis=0).astype(np.float32)
        else:
            image = np.stack(modality_data, axis=0).astype(np.float32)
        
        item = {
            'image': torch.from_numpy(image),
            'subject_id': sample['subject_id']
        }
        
        if self.mode == '2d':
            item['slice_idx'] = sample['slice_idx']
        
        if self.return_metadata:
            item['metadata'] = sample
        
        return item


class IXIDataset(BaseMedicalDataset):
    """
    ixi dataset - information extraction from images.
    
    multi-site brain mri dataset with t1, t2, pd, mra, and dti.
    sites: guys, hh (hammersmith hospital), iop (institute of psychiatry)
    
    reference: https://brain-development.org/ixi-dataset/
    """
    
    SITES = ['Guys', 'HH', 'IOP']
    AVAILABLE_MODALITIES = ['t1', 't2', 'pd', 'mra', 'dti']
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        sites: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        **kwargs
    ):
        self.sites = sites or self.SITES
        super().__init__(root_dir, modalities, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1', 't2']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (30, 130)
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        # ixi naming convention: ixi{id}-{site}-{session}_{modality}.nii.gz
        for nifti_file in sorted(self.root_dir.glob('*.nii.gz')):
            parts = nifti_file.stem.replace('.nii', '').split('-')
            if len(parts) >= 2:
                subject_id = parts[0]
                site = parts[1] if len(parts) > 1 else 'unknown'
                
                if site in self.sites:
                    # check if we have all required modalities
                    has_all = True
                    modality_paths = {}
                    
                    for mod in self.modalities:
                        pattern = f"*{subject_id}*{mod.upper()}*.nii.gz"
                        matches = list(self.root_dir.glob(pattern))
                        if matches:
                            modality_paths[mod] = matches[0]
                        else:
                            has_all = False
                            break
                    
                    if has_all and subject_id not in [s.get('subject_id') for s in samples]:
                        if self.mode == '2d':
                            for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                                samples.append({
                                    'subject_id': subject_id,
                                    'site': site,
                                    'modality_paths': modality_paths,
                                    'slice_idx': slice_idx
                                })
                        else:
                            samples.append({
                                'subject_id': subject_id,
                                'site': site,
                                'modality_paths': modality_paths
                            })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required for NIfTI loading")


class OASISDataset(BaseMedicalDataset):
    """
    oasis dataset - open access series of imaging studies.
    
    longitudinal brain mri dataset with cross-sectional and 
    longitudinal collections. includes subjects with dementia.
    
    reference: https://www.oasis-brains.org/
    """
    
    COLLECTIONS = ['OASIS-1', 'OASIS-2', 'OASIS-3']
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        collection: str = 'OASIS-1',
        include_longitudinal: bool = False,
        **kwargs
    ):
        self.collection = collection
        self.include_longitudinal = include_longitudinal
        super().__init__(root_dir, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (50, 150)
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        # oasis directory structure varies by collection
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            
            # find t1 volume
            t1_patterns = [
                '*T1w*.nii.gz',
                '*mpr*.nii.gz',
                '*anat*.nii.gz'
            ]
            
            t1_path = None
            for pattern in t1_patterns:
                matches = list(subject_dir.rglob(pattern))
                if matches:
                    t1_path = matches[0]
                    break
            
            if t1_path:
                modality_paths = {'t1': t1_path}
                
                if self.mode == '2d':
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'slice_idx': slice_idx
                        })
                else:
                    samples.append({
                        'subject_id': subject_id,
                        'modality_paths': modality_paths
                    })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required")


class ADNIDataset(BaseMedicalDataset):
    """
    adni dataset - alzheimer's disease neuroimaging initiative.
    
    longitudinal study with multiple mri protocols across sites.
    includes cognitively normal, mci, and ad subjects.
    
    reference: https://adni.loni.usc.edu/
    """
    
    PROTOCOLS = ['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO']
    DIAGNOSES = ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        protocol: Optional[str] = None,
        diagnosis: Optional[List[str]] = None,
        baseline_only: bool = True,
        **kwargs
    ):
        self.protocol = protocol
        self.diagnosis = diagnosis
        self.baseline_only = baseline_only
        super().__init__(root_dir, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (40, 160)
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        # adni uses bids-like structure
        for subject_dir in sorted(self.root_dir.glob('sub-*')):
            subject_id = subject_dir.name
            
            # find sessions
            session_dirs = list(subject_dir.glob('ses-*'))
            if self.baseline_only and session_dirs:
                session_dirs = [sorted(session_dirs)[0]]
            elif not session_dirs:
                session_dirs = [subject_dir]
            
            for session_dir in session_dirs:
                anat_dir = session_dir / 'anat'
                if not anat_dir.exists():
                    anat_dir = session_dir
                
                t1_files = list(anat_dir.glob('*T1w*.nii.gz'))
                if t1_files:
                    modality_paths = {'t1': t1_files[0]}
                    
                    if self.mode == '2d':
                        for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                            samples.append({
                                'subject_id': subject_id,
                                'session': session_dir.name,
                                'modality_paths': modality_paths,
                                'slice_idx': slice_idx
                            })
                    else:
                        samples.append({
                            'subject_id': subject_id,
                            'session': session_dir.name,
                            'modality_paths': modality_paths
                        })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required")


class ABIDEDataset(BaseMedicalDataset):
    """
    abide dataset - autism brain imaging data exchange.
    
    large-scale multi-site dataset for autism research.
    includes resting-state fmri and structural mri.
    
    reference: http://fcon_1000.projects.nitrc.org/indi/abide/
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        collection: str = 'ABIDE_I',  # or 'abide_ii'
        sites: Optional[List[str]] = None,
        diagnosis: Optional[str] = None,  # 'asd' or 'tc'
        **kwargs
    ):
        self.collection = collection
        self.sites = sites
        self.selected_diagnosis = diagnosis
        super().__init__(root_dir, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (40, 140)
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            
            # find anatomical scan
            anat_patterns = ['*mprage*.nii.gz', '*T1*.nii.gz', '*anat*.nii.gz']
            
            t1_path = None
            for pattern in anat_patterns:
                matches = list(subject_dir.rglob(pattern))
                if matches:
                    t1_path = matches[0]
                    break
            
            if t1_path:
                modality_paths = {'t1': t1_path}
                
                if self.mode == '2d':
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'slice_idx': slice_idx
                        })
                else:
                    samples.append({
                        'subject_id': subject_id,
                        'modality_paths': modality_paths
                    })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required")


class HCPDataset(BaseMedicalDataset):
    """
    hcp dataset - human connectome project.
    
    high-quality, high-resolution brain imaging data
    with extensive multi-modal acquisitions.
    
    reference: https://www.humanconnectome.org/
    """
    
    AVAILABLE_MODALITIES = ['t1', 't2', 'dwi', 'bold']
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: Optional[List[str]] = None,
        resolution: str = '0.7mm',  # or '1.25mm'
        **kwargs
    ):
        self.resolution = resolution
        super().__init__(root_dir, modalities, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1', 't2']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (50, 200)  # hcp has higher resolution
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            
            # hcp structure
            t1_dir = subject_dir / 'T1w'
            if not t1_dir.exists():
                t1_dir = subject_dir
            
            modality_paths = {}
            has_all = True
            
            for mod in self.modalities:
                if mod == 't1':
                    pattern = '*T1w*.nii.gz'
                elif mod == 't2':
                    pattern = '*T2w*.nii.gz'
                else:
                    pattern = f'*{mod}*.nii.gz'
                
                matches = list(t1_dir.rglob(pattern))
                if matches:
                    modality_paths[mod] = matches[0]
                else:
                    has_all = False
                    break
            
            if has_all:
                if self.mode == '2d':
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'slice_idx': slice_idx
                        })
                else:
                    samples.append({
                        'subject_id': subject_id,
                        'modality_paths': modality_paths
                    })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required")


class TCGAGBMDataset(BaseMedicalDataset):
    """
    tcga-gbm dataset - the cancer genome atlas glioblastoma.
    
    multi-institutional glioblastoma imaging data with
    genomic annotations.
    
    reference: https://wiki.cancerimagingarchive.net/
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        include_genomics: bool = False,
        tumor_grades: Optional[List[str]] = None,
        **kwargs
    ):
        self.include_genomics = include_genomics
        self.tumor_grades = tumor_grades
        super().__init__(root_dir, **kwargs)
    
    def _default_modalities(self) -> List[str]:
        return ['t1', 't1ce', 't2', 'flair']
    
    def _default_slice_range(self) -> Tuple[int, int]:
        return (30, 130)
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_id = subject_dir.name
            modality_paths = {}
            has_all = True
            
            modality_patterns = {
                't1': ['*T1*pre*.nii.gz', '*T1*.nii.gz'],
                't1ce': ['*T1*post*.nii.gz', '*T1*Gd*.nii.gz', '*T1*ce*.nii.gz'],
                't2': ['*T2*.nii.gz'],
                'flair': ['*FLAIR*.nii.gz', '*flair*.nii.gz']
            }
            
            for mod in self.modalities:
                found = False
                for pattern in modality_patterns.get(mod, [f'*{mod}*.nii.gz']):
                    matches = list(subject_dir.rglob(pattern))
                    if matches:
                        modality_paths[mod] = matches[0]
                        found = True
                        break
                
                if not found:
                    has_all = False
                    break
            
            if has_all:
                # look for segmentation
                seg_matches = list(subject_dir.rglob('*seg*.nii.gz'))
                seg_path = seg_matches[0] if seg_matches else None
                
                if self.mode == '2d':
                    for slice_idx in range(self.slice_range[0], self.slice_range[1]):
                        samples.append({
                            'subject_id': subject_id,
                            'modality_paths': modality_paths,
                            'segmentation_path': seg_path,
                            'slice_idx': slice_idx
                        })
                else:
                    samples.append({
                        'subject_id': subject_id,
                        'modality_paths': modality_paths,
                        'segmentation_path': seg_path
                    })
        
        return samples
    
    def _load_volume(self, sample: Dict[str, Any], modality: str) -> np.ndarray:
        path = sample['modality_paths'][modality]
        if HAS_NIBABEL:
            return nib.load(str(path)).get_fdata()
        raise RuntimeError("nibabel required")


class VolumetricDataset(Dataset):
    """
    generic 3d volumetric dataset.
    
    loads full 3d volumes with optional patching for
    memory-efficient training.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        modalities: List[str],
        patch_size: Optional[Tuple[int, int, int]] = None,
        overlap: float = 0.5,
        transform: Optional[Callable] = None,
        cache_volumes: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.cache_volumes = cache_volumes
        
        self._cache = {}
        self.samples = self._discover_samples()
    
    def _discover_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        for subject_dir in sorted(self.root_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            modality_paths = {}
            for mod in self.modalities:
                matches = list(subject_dir.glob(f'*{mod}*.nii.gz'))
                if matches:
                    modality_paths[mod] = matches[0]
            
            if len(modality_paths) == len(self.modalities):
                samples.append({
                    'subject_id': subject_dir.name,
                    'modality_paths': modality_paths
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        
        volumes = []
        for mod in self.modalities:
            path = sample['modality_paths'][mod]
            if HAS_NIBABEL:
                vol = nib.load(str(path)).get_fdata()
            else:
                raise RuntimeError("nibabel required")
            volumes.append(vol)
        
        volume = np.stack(volumes, axis=0).astype(np.float32)
        
        # extract patch if specified
        if self.patch_size is not None:
            volume = self._random_patch(volume)
        
        item = {
            'image': torch.from_numpy(volume),
            'subject_id': sample['subject_id']
        }
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def _random_patch(self, volume: np.ndarray) -> np.ndarray:
        """extract random patch from volume."""
        _, D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        d_start = random.randint(0, max(0, D - pd))
        h_start = random.randint(0, max(0, H - ph))
        w_start = random.randint(0, max(0, W - pw))
        
        return volume[:, d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]


# dataset registry
DATASET_REGISTRY = {
    'ixi': IXIDataset,
    'oasis': OASISDataset,
    'adni': ADNIDataset,
    'abide': ABIDEDataset,
    'hcp': HCPDataset,
    'tcga_gbm': TCGAGBMDataset,
    'volumetric': VolumetricDataset
}


def create_medical_dataset(
    dataset_name: str,
    root_dir: Union[str, Path],
    **kwargs
) -> Dataset:
    """
    factory function to create datasets.
    
    args:
        dataset_name: name of dataset ('ixi', 'oasis', etc.)
        root_dir: root directory of dataset
        **kwargs: dataset-specific arguments
        
    returns:
        dataset instance
    """
    if dataset_name.lower() not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DATASET_REGISTRY[dataset_name.lower()](root_dir, **kwargs)
