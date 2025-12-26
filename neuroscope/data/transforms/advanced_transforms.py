"""
Image Transformation and Augmentation Pipelines.

Comprehensive transforms for medical image preprocessing
and data augmentation in training pipelines.
"""

from typing import List, Optional, Dict, Tuple, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, map_coordinates
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BaseTransform(ABC):
    """Abstract base class for all transforms."""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    """Compose multiple transforms sequentially."""
    
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self) -> str:
        transform_str = ', '.join(repr(t) for t in self.transforms)
        return f"Compose([{transform_str}])"


# =============================================================================
# Intensity Transforms
# =============================================================================

class IntensityNormalization(BaseTransform):
    """Generic intensity normalization base class."""
    
    def __init__(self, per_channel: bool = True):
        self.per_channel = per_channel


class ZScoreNormalization(IntensityNormalization):
    """
    Z-score (standard) normalization.
    
    Normalizes to zero mean and unit variance.
    """
    
    def __init__(
        self,
        per_channel: bool = True,
        epsilon: float = 1e-8,
        clip_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(per_channel)
        self.epsilon = epsilon
        self.clip_range = clip_range
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._normalize(data[key])
            return data
        return self._normalize(data)
    
    def _normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        
        if is_tensor:
            if self.per_channel:
                dims = tuple(range(1, x.dim()))
                mean = x.mean(dim=dims, keepdim=True)
                std = x.std(dim=dims, keepdim=True)
            else:
                mean = x.mean()
                std = x.std()
            
            x = (x - mean) / (std + self.epsilon)
            
            if self.clip_range:
                x = x.clamp(self.clip_range[0], self.clip_range[1])
        else:
            if self.per_channel:
                axes = tuple(range(1, x.ndim))
                mean = x.mean(axis=axes, keepdims=True)
                std = x.std(axis=axes, keepdims=True)
            else:
                mean = x.mean()
                std = x.std()
            
            x = (x - mean) / (std + self.epsilon)
            
            if self.clip_range:
                x = np.clip(x, self.clip_range[0], self.clip_range[1])
        
        return x


class MinMaxNormalization(IntensityNormalization):
    """
    Min-max normalization to [0, 1] or custom range.
    """
    
    def __init__(
        self,
        per_channel: bool = True,
        output_range: Tuple[float, float] = (0.0, 1.0),
        percentile_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(per_channel)
        self.output_range = output_range
        self.percentile_range = percentile_range
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._normalize(data[key])
            return data
        return self._normalize(data)
    
    def _normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        
        if is_tensor:
            if self.percentile_range:
                # Convert to numpy for percentile
                x_np = x.numpy()
                p_low = np.percentile(x_np, self.percentile_range[0])
                p_high = np.percentile(x_np, self.percentile_range[1])
                x = x.clamp(p_low, p_high)
            
            if self.per_channel:
                dims = tuple(range(1, x.dim()))
                x_min = x.amin(dim=dims, keepdim=True)
                x_max = x.amax(dim=dims, keepdim=True)
            else:
                x_min = x.min()
                x_max = x.max()
            
            x = (x - x_min) / (x_max - x_min + 1e-8)
            x = x * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        else:
            if self.percentile_range:
                p_low = np.percentile(x, self.percentile_range[0])
                p_high = np.percentile(x, self.percentile_range[1])
                x = np.clip(x, p_low, p_high)
            
            if self.per_channel:
                axes = tuple(range(1, x.ndim))
                x_min = x.min(axis=axes, keepdims=True)
                x_max = x.max(axis=axes, keepdims=True)
            else:
                x_min = x.min()
                x_max = x.max()
            
            x = (x - x_min) / (x_max - x_min + 1e-8)
            x = x * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        
        return x


class PercentileNormalization(IntensityNormalization):
    """
    Percentile-based normalization for robust intensity scaling.
    """
    
    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        per_channel: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__(per_channel)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.output_range = output_range
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._normalize(data[key])
            return data
        return self._normalize(data)
    
    def _normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x
        
        if self.per_channel and x_np.ndim > 2:
            normalized = np.zeros_like(x_np)
            for c in range(x_np.shape[0]):
                channel = x_np[c]
                p_low = np.percentile(channel, self.lower_percentile)
                p_high = np.percentile(channel, self.upper_percentile)
                
                clipped = np.clip(channel, p_low, p_high)
                normalized[c] = (clipped - p_low) / (p_high - p_low + 1e-8)
        else:
            p_low = np.percentile(x_np, self.lower_percentile)
            p_high = np.percentile(x_np, self.upper_percentile)
            
            clipped = np.clip(x_np, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
        
        # Scale to output range
        normalized = normalized * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        
        if is_tensor:
            return torch.from_numpy(normalized.astype(np.float32))
        return normalized.astype(np.float32)


class HistogramEqualization(BaseTransform):
    """Histogram equalization for contrast enhancement."""
    
    def __init__(self, per_channel: bool = True, nbins: int = 256):
        self.per_channel = per_channel
        self.nbins = nbins
    
    def __call__(self, data: Union[np.ndarray, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._equalize(data[key])
            return data
        return self._equalize(data)
    
    def _equalize(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return self._equalize_single(x)
        
        if self.per_channel:
            equalized = np.zeros_like(x)
            for c in range(x.shape[0]):
                equalized[c] = self._equalize_single(x[c])
            return equalized
        else:
            return self._equalize_single(x)
    
    def _equalize_single(self, img: np.ndarray) -> np.ndarray:
        # Compute histogram
        hist, bins = np.histogram(img.flatten(), bins=self.nbins, density=True)
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]  # Normalize
        
        # Interpolate
        equalized = np.interp(img.flatten(), bins[:-1], cdf)
        return equalized.reshape(img.shape).astype(np.float32)


class AdaptiveHistogramEqualization(BaseTransform):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    
    def __init__(
        self,
        clip_limit: float = 0.01,
        tile_grid_size: Tuple[int, int] = (8, 8),
        per_channel: bool = True
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.per_channel = per_channel
    
    def __call__(self, data: Union[np.ndarray, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._clahe(data[key])
            return data
        return self._clahe(data)
    
    def _clahe(self, x: np.ndarray) -> np.ndarray:
        try:
            import cv2
            
            # Normalize to 0-255 for OpenCV
            x_min, x_max = x.min(), x.max()
            x_norm = ((x - x_min) / (x_max - x_min + 1e-8) * 255).astype(np.uint8)
            
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit * 255,
                tileGridSize=self.tile_grid_size
            )
            
            if x.ndim == 2:
                result = clahe.apply(x_norm)
            elif self.per_channel:
                result = np.zeros_like(x_norm)
                for c in range(x.shape[0]):
                    result[c] = clahe.apply(x_norm[c])
            else:
                result = clahe.apply(x_norm.reshape(-1)).reshape(x.shape)
            
            # Convert back to original range
            result = result.astype(np.float32) / 255 * (x_max - x_min) + x_min
            return result
            
        except ImportError:
            # Fallback without OpenCV
            return x


# =============================================================================
# Spatial Transforms
# =============================================================================

class RandomCrop(BaseTransform):
    """Random crop of specified size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._crop(data[key])
            if 'segmentation' in data:
                data['segmentation'] = self._crop(data['segmentation'])
            return data
        return self._crop(data)
    
    def _crop(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            h, w = x.shape[-2:]
        else:
            h, w = x.shape[-2:]
        
        th, tw = self.size
        
        if h < th or w < tw:
            return x
        
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        
        if isinstance(x, torch.Tensor):
            return x[..., top:top+th, left:left+tw]
        else:
            return x[..., top:top+th, left:left+tw]


class CenterCrop(BaseTransform):
    """Center crop of specified size."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._crop(data[key])
            if 'segmentation' in data:
                data['segmentation'] = self._crop(data['segmentation'])
            return data
        return self._crop(data)
    
    def _crop(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        h, w = x.shape[-2:]
        th, tw = self.size
        
        top = (h - th) // 2
        left = (w - tw) // 2
        
        if isinstance(x, torch.Tensor):
            return x[..., top:top+th, left:left+tw]
        else:
            return x[..., top:top+th, left:left+tw]


class Resize(BaseTransform):
    """Resize to specified size."""
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        mode: str = 'bilinear',
        align_corners: bool = False
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.mode = mode
        self.align_corners = align_corners
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._resize(data[key])
            if 'segmentation' in data:
                data['segmentation'] = self._resize(
                    data['segmentation'], mode='nearest'
                )
            return data
        return self._resize(data)
    
    def _resize(
        self,
        x: Union[np.ndarray, torch.Tensor],
        mode: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        mode = mode or self.mode
        is_numpy = isinstance(x, np.ndarray)
        
        if is_numpy:
            x = torch.from_numpy(x)
        
        # Add batch dimension if needed
        needs_squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            needs_squeeze = True
        elif x.dim() == 3:
            x = x.unsqueeze(0)
            needs_squeeze = True
        
        x = F.interpolate(
            x.float(),
            size=self.size,
            mode=mode,
            align_corners=self.align_corners if mode != 'nearest' else None
        )
        
        if needs_squeeze:
            x = x.squeeze(0)
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
        
        if is_numpy:
            return x.numpy()
        return x


class RandomFlip(BaseTransform):
    """Random horizontal and/or vertical flip."""
    
    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = False,
        p: float = 0.5
    ):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            
            h_flip = self.horizontal and random.random() < self.p
            v_flip = self.vertical and random.random() < self.p
            
            data[key] = self._flip(data[key], h_flip, v_flip)
            if 'segmentation' in data:
                data['segmentation'] = self._flip(data['segmentation'], h_flip, v_flip)
            return data
        
        h_flip = self.horizontal and random.random() < self.p
        v_flip = self.vertical and random.random() < self.p
        return self._flip(data, h_flip, v_flip)
    
    def _flip(
        self,
        x: Union[np.ndarray, torch.Tensor],
        h_flip: bool,
        v_flip: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            if h_flip:
                x = torch.flip(x, dims=[-1])
            if v_flip:
                x = torch.flip(x, dims=[-2])
        else:
            if h_flip:
                x = np.flip(x, axis=-1).copy()
            if v_flip:
                x = np.flip(x, axis=-2).copy()
        return x


class RandomRotation(BaseTransform):
    """Random rotation by angle in degrees."""
    
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = 10.0,
        p: float = 0.5
    ):
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._rotate(data[key], angle)
            if 'segmentation' in data:
                data['segmentation'] = self._rotate(data['segmentation'], angle, order=0)
            return data
        
        return self._rotate(data, angle)
    
    def _rotate(
        self,
        x: Union[np.ndarray, torch.Tensor],
        angle: float,
        order: int = 1
    ) -> Union[np.ndarray, torch.Tensor]:
        if not HAS_SCIPY:
            return x
        
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        if x.ndim == 2:
            x = ndimage.rotate(x, angle, reshape=False, order=order)
        else:
            rotated = np.zeros_like(x)
            for c in range(x.shape[0]):
                rotated[c] = ndimage.rotate(x[c], angle, reshape=False, order=order)
            x = rotated
        
        if is_tensor:
            return torch.from_numpy(x.astype(np.float32))
        return x.astype(np.float32)


class RandomAffine(BaseTransform):
    """Random affine transformation."""
    
    def __init__(
        self,
        degrees: float = 10.0,
        translate: Optional[Tuple[float, float]] = (0.1, 0.1),
        scale: Optional[Tuple[float, float]] = (0.9, 1.1),
        shear: Optional[float] = 5.0,
        p: float = 0.5
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        # Sample parameters
        angle = random.uniform(-self.degrees, self.degrees)
        
        if self.translate:
            tx = random.uniform(-self.translate[0], self.translate[0])
            ty = random.uniform(-self.translate[1], self.translate[1])
        else:
            tx, ty = 0, 0
        
        if self.scale:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0
        
        if self.shear:
            shear = random.uniform(-self.shear, self.shear)
        else:
            shear = 0
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._affine(data[key], angle, tx, ty, scale, shear)
            if 'segmentation' in data:
                data['segmentation'] = self._affine(
                    data['segmentation'], angle, tx, ty, scale, shear, order=0
                )
            return data
        
        return self._affine(data, angle, tx, ty, scale, shear)
    
    def _affine(
        self,
        x: Union[np.ndarray, torch.Tensor],
        angle: float,
        tx: float,
        ty: float,
        scale: float,
        shear: float,
        order: int = 1
    ) -> Union[np.ndarray, torch.Tensor]:
        if not HAS_SCIPY:
            return x
        
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        # Build affine matrix
        h, w = x.shape[-2:]
        center = (h / 2, w / 2)
        
        # Rotation
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Shear
        shear_rad = np.deg2rad(shear)
        
        # Combined matrix
        matrix = np.array([
            [scale * cos_t, -scale * sin_t + np.tan(shear_rad), tx * w],
            [scale * sin_t, scale * cos_t, ty * h],
            [0, 0, 1]
        ])
        
        # Center transformation
        offset = np.array(center) - np.array(center) @ matrix[:2, :2].T
        
        if x.ndim == 2:
            x = ndimage.affine_transform(x, matrix[:2, :2], offset=offset, order=order)
        else:
            transformed = np.zeros_like(x)
            for c in range(x.shape[0]):
                transformed[c] = ndimage.affine_transform(
                    x[c], matrix[:2, :2], offset=offset, order=order
                )
            x = transformed
        
        if is_tensor:
            return torch.from_numpy(x.astype(np.float32))
        return x.astype(np.float32)


class ElasticDeformation(BaseTransform):
    """Elastic deformation for data augmentation."""
    
    def __init__(
        self,
        alpha: float = 100.0,
        sigma: float = 10.0,
        p: float = 0.5
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            
            # Generate displacement fields once
            shape = data[key].shape[-2:]
            dx, dy = self._generate_displacement(shape)
            
            data[key] = self._deform(data[key], dx, dy)
            if 'segmentation' in data:
                data['segmentation'] = self._deform(data['segmentation'], dx, dy, order=0)
            return data
        
        shape = data.shape[-2:]
        dx, dy = self._generate_displacement(shape)
        return self._deform(data, dx, dy)
    
    def _generate_displacement(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_SCIPY:
            return np.zeros(shape), np.zeros(shape)
        
        dx = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        return dx, dy
    
    def _deform(
        self,
        x: Union[np.ndarray, torch.Tensor],
        dx: np.ndarray,
        dy: np.ndarray,
        order: int = 1
    ) -> Union[np.ndarray, torch.Tensor]:
        if not HAS_SCIPY:
            return x
        
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        shape = x.shape[-2:]
        y, x_coord = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        
        indices = [y + dy, x_coord + dx]
        
        if x.ndim == 2:
            x = map_coordinates(x, indices, order=order, mode='reflect')
        else:
            deformed = np.zeros_like(x)
            for c in range(x.shape[0]):
                deformed[c] = map_coordinates(x[c], indices, order=order, mode='reflect')
            x = deformed
        
        if is_tensor:
            return torch.from_numpy(x.astype(np.float32))
        return x.astype(np.float32)


# =============================================================================
# Augmentation Transforms
# =============================================================================

class RandomNoise(BaseTransform):
    """Add random Gaussian noise."""
    
    def __init__(
        self,
        std_range: Tuple[float, float] = (0.01, 0.05),
        p: float = 0.5
    ):
        self.std_range = std_range
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        std = random.uniform(self.std_range[0], self.std_range[1])
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._add_noise(data[key], std)
            return data
        
        return self._add_noise(data, std)
    
    def _add_noise(
        self,
        x: Union[np.ndarray, torch.Tensor],
        std: float
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * std
            return x + noise
        else:
            noise = np.random.randn(*x.shape) * std
            return (x + noise).astype(np.float32)


class RandomBlur(BaseTransform):
    """Apply random Gaussian blur."""
    
    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 2.0),
        p: float = 0.5
    ):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._blur(data[key], sigma)
            return data
        
        return self._blur(data, sigma)
    
    def _blur(
        self,
        x: Union[np.ndarray, torch.Tensor],
        sigma: float
    ) -> Union[np.ndarray, torch.Tensor]:
        if not HAS_SCIPY:
            return x
        
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        if x.ndim == 2:
            x = gaussian_filter(x, sigma)
        else:
            blurred = np.zeros_like(x)
            for c in range(x.shape[0]):
                blurred[c] = gaussian_filter(x[c], sigma)
            x = blurred
        
        if is_tensor:
            return torch.from_numpy(x.astype(np.float32))
        return x.astype(np.float32)


class RandomBrightnessContrast(BaseTransform):
    """Random brightness and contrast adjustment."""
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (-0.2, 0.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._adjust(data[key], brightness, contrast)
            return data
        
        return self._adjust(data, brightness, contrast)
    
    def _adjust(
        self,
        x: Union[np.ndarray, torch.Tensor],
        brightness: float,
        contrast: float
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            mean = x.mean()
            x = (x - mean) * contrast + mean + brightness
        else:
            mean = x.mean()
            x = ((x - mean) * contrast + mean + brightness).astype(np.float32)
        return x


class RandomGamma(BaseTransform):
    """Random gamma correction."""
    
    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5
    ):
        self.gamma_range = gamma_range
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._gamma(data[key], gamma)
            return data
        
        return self._gamma(data, gamma)
    
    def _gamma(
        self,
        x: Union[np.ndarray, torch.Tensor],
        gamma: float
    ) -> Union[np.ndarray, torch.Tensor]:
        # Normalize to [0, 1] for gamma correction
        x_min = x.min()
        x_max = x.max()
        
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        
        if isinstance(x, torch.Tensor):
            x_gamma = torch.pow(x_norm.clamp(min=0), gamma)
        else:
            x_gamma = np.power(np.clip(x_norm, 0, None), gamma)
        
        # Scale back
        return x_gamma * (x_max - x_min) + x_min


class BiasFieldAugmentation(BaseTransform):
    """Simulate MRI bias field artifacts."""
    
    def __init__(
        self,
        degree: int = 3,
        coefficient_range: Tuple[float, float] = (-0.5, 0.5),
        p: float = 0.5
    ):
        self.degree = degree
        self.coefficient_range = coefficient_range
        self.p = p
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if random.random() > self.p:
            return data
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._add_bias(data[key])
            return data
        
        return self._add_bias(data)
    
    def _add_bias(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x = x.numpy()
        
        h, w = x.shape[-2:]
        
        # Create coordinate grids
        y = np.linspace(-1, 1, h)
        x_coord = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y, x_coord, indexing='ij')
        
        # Generate polynomial bias field
        bias_field = np.zeros((h, w))
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                coef = random.uniform(self.coefficient_range[0], self.coefficient_range[1])
                bias_field += coef * (xx ** i) * (yy ** j)
        
        # Apply multiplicative bias
        bias_field = np.exp(bias_field)
        
        if x.ndim == 2:
            x = x * bias_field
        else:
            x = x * bias_field[np.newaxis, :, :]
        
        if is_tensor:
            return torch.from_numpy(x.astype(np.float32))
        return x.astype(np.float32)


# =============================================================================
# Medical-Specific Transforms
# =============================================================================

class N4BiasFieldCorrection(BaseTransform):
    """N4 bias field correction (requires SimpleITK)."""
    
    def __init__(
        self,
        shrink_factor: int = 4,
        num_iterations: List[int] = None
    ):
        self.shrink_factor = shrink_factor
        self.num_iterations = num_iterations or [50, 50, 50, 50]
    
    def __call__(self, data: Union[np.ndarray, Dict]) -> Any:
        try:
            import SimpleITK as sitk
        except ImportError:
            return data  # Skip if SimpleITK not available
        
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._correct(data[key])
            return data
        
        return self._correct(data)
    
    def _correct(self, x: np.ndarray) -> np.ndarray:
        try:
            import SimpleITK as sitk
        except ImportError:
            return x
        
        if x.ndim == 2:
            return self._correct_single(x)
        else:
            corrected = np.zeros_like(x)
            for c in range(x.shape[0]):
                corrected[c] = self._correct_single(x[c])
            return corrected
    
    def _correct_single(self, img: np.ndarray) -> np.ndarray:
        import SimpleITK as sitk
        
        sitk_img = sitk.GetImageFromArray(img.astype(np.float32))
        
        # Create mask
        mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)
        
        # Shrink for speed
        shrunk = sitk.Shrink(sitk_img, [self.shrink_factor] * sitk_img.GetDimension())
        mask_shrunk = sitk.Shrink(mask, [self.shrink_factor] * mask.GetDimension())
        
        # N4 correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(self.num_iterations)
        
        corrected = corrector.Execute(shrunk, mask_shrunk)
        
        # Get bias field and apply to full resolution
        log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
        corrected_full = sitk_img / sitk.Exp(log_bias)
        
        return sitk.GetArrayFromImage(corrected_full)


class SkullStripping(BaseTransform):
    """Skull stripping using thresholding (simplified)."""
    
    def __init__(self, threshold_percentile: float = 10.0):
        self.threshold_percentile = threshold_percentile
    
    def __call__(self, data: Union[np.ndarray, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._strip(data[key])
            return data
        return self._strip(data)
    
    def _strip(self, x: np.ndarray) -> np.ndarray:
        threshold = np.percentile(x, self.threshold_percentile)
        mask = x > threshold
        return x * mask


class IntensityClipping(BaseTransform):
    """Clip intensity values to percentile range."""
    
    def __init__(
        self,
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
    
    def __call__(self, data: Union[np.ndarray, torch.Tensor, Dict]) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            key = 'image' if 'image' in data else 'source'
            data[key] = self._clip(data[key])
            return data
        return self._clip(data)
    
    def _clip(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.numpy()
        else:
            x_np = x
        
        p_low = np.percentile(x_np, self.lower_percentile)
        p_high = np.percentile(x_np, self.upper_percentile)
        
        if is_tensor:
            return x.clamp(p_low, p_high)
        else:
            return np.clip(x_np, p_low, p_high)


# =============================================================================
# Pipeline Builders
# =============================================================================

def create_train_transforms(
    crop_size: int = 224,
    normalize: str = 'zscore',
    augment: bool = True
) -> Compose:
    """Create training transformation pipeline."""
    transforms = []
    
    # Intensity clipping
    transforms.append(IntensityClipping())
    
    # Normalization
    if normalize == 'zscore':
        transforms.append(ZScoreNormalization(clip_range=(-3, 3)))
    elif normalize == 'minmax':
        transforms.append(MinMaxNormalization(output_range=(-1, 1)))
    elif normalize == 'percentile':
        transforms.append(PercentileNormalization())
    
    # Spatial transforms
    transforms.append(RandomCrop(crop_size))
    
    # Augmentation
    if augment:
        transforms.append(RandomFlip(horizontal=True, vertical=True, p=0.5))
        transforms.append(RandomRotation(degrees=15, p=0.5))
        transforms.append(RandomNoise(std_range=(0.01, 0.03), p=0.3))
        transforms.append(RandomBrightnessContrast(p=0.3))
        transforms.append(RandomGamma(gamma_range=(0.9, 1.1), p=0.3))
    
    return Compose(transforms)


def create_val_transforms(
    crop_size: int = 224,
    normalize: str = 'zscore'
) -> Compose:
    """Create validation transformation pipeline."""
    transforms = [
        IntensityClipping(),
    ]
    
    if normalize == 'zscore':
        transforms.append(ZScoreNormalization(clip_range=(-3, 3)))
    elif normalize == 'minmax':
        transforms.append(MinMaxNormalization(output_range=(-1, 1)))
    elif normalize == 'percentile':
        transforms.append(PercentileNormalization())
    
    transforms.append(CenterCrop(crop_size))
    
    return Compose(transforms)


def create_test_transforms(
    normalize: str = 'zscore'
) -> Compose:
    """Create test transformation pipeline (no cropping)."""
    transforms = [
        IntensityClipping(),
    ]
    
    if normalize == 'zscore':
        transforms.append(ZScoreNormalization(clip_range=(-3, 3)))
    elif normalize == 'minmax':
        transforms.append(MinMaxNormalization(output_range=(-1, 1)))
    elif normalize == 'percentile':
        transforms.append(PercentileNormalization())
    
    return Compose(transforms)
