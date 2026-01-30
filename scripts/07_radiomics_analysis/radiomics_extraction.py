#!/usr/bin/env python3
"""
radiomics feature extraction for harmonization validation.

extracts standardized radiomics features from mri images to assess
whether harmonization preserves clinically relevant imaging biomarkers.

implements feature categories:
- first-order statistics (intensity-based)
- texture features (glcm-based)
- shape features (morphological)

references:
- van griethuysen et al. (2017): computational radiomics system
- zwanenburg et al. (2020): image biomarker standardisation initiative
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import ndimage, stats
import warnings


@dataclass
class RadiomicsConfig:
    """configuration for radiomics feature extraction."""
    # first-order settings
    bin_width: float = 25.0  # intensity bin width for discretization

    # glcm settings
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 3])
    glcm_angles: List[float] = field(default_factory=lambda: [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # mask settings
    intensity_threshold: float = 0.01  # minimum intensity for brain mask
    min_region_size: int = 100  # minimum connected component size


class FirstOrderFeatures:
    """
    first-order (histogram-based) radiomics features.

    these features describe the distribution of voxel intensities
    without considering spatial relationships.
    """

    @staticmethod
    def extract(image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        extract first-order features from image.

        args:
            image: input image array
            mask: optional binary mask (uses non-zero voxels if None)

        returns:
            dictionary of feature names to values
        """
        if mask is None:
            mask = image > 0

        # get masked values
        values = image[mask].flatten()

        if len(values) < 10:
            return {k: 0.0 for k in [
                'mean', 'median', 'std', 'variance', 'skewness', 'kurtosis',
                'min', 'max', 'range', 'iqr', 'mad', 'energy', 'entropy',
                'uniformity', 'percentile_10', 'percentile_90', 'robust_mean'
            ]}

        features = {}

        # basic statistics
        features['mean'] = float(np.mean(values))
        features['median'] = float(np.median(values))
        features['std'] = float(np.std(values))
        features['variance'] = float(np.var(values))

        # higher-order moments
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features['skewness'] = float(stats.skew(values))
            features['kurtosis'] = float(stats.kurtosis(values))

        # range statistics
        features['min'] = float(np.min(values))
        features['max'] = float(np.max(values))
        features['range'] = features['max'] - features['min']

        # robust statistics
        q1, q3 = np.percentile(values, [25, 75])
        features['iqr'] = float(q3 - q1)
        features['mad'] = float(np.median(np.abs(values - features['median'])))

        # energy and entropy
        features['energy'] = float(np.sum(values ** 2))

        # histogram-based entropy
        hist, _ = np.histogram(values, bins=64, density=True)
        hist = hist[hist > 0]  # remove zeros
        features['entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))

        # uniformity (sum of squared probabilities)
        hist_norm = hist / (np.sum(hist) + 1e-10)
        features['uniformity'] = float(np.sum(hist_norm ** 2))

        # percentiles
        features['percentile_10'] = float(np.percentile(values, 10))
        features['percentile_90'] = float(np.percentile(values, 90))

        # robust mean (mean of values within 10-90 percentile)
        robust_mask = (values >= features['percentile_10']) & (values <= features['percentile_90'])
        features['robust_mean'] = float(np.mean(values[robust_mask])) if np.any(robust_mask) else features['mean']

        return features


class GLCMFeatures:
    """
    gray level co-occurrence matrix (glcm) texture features.

    these features capture spatial relationships between neighboring
    voxels and are sensitive to texture patterns.
    """

    @staticmethod
    def compute_glcm(image: np.ndarray, distance: int = 1, angle: float = 0,
                     levels: int = 32) -> np.ndarray:
        """
        compute gray level co-occurrence matrix.

        args:
            image: 2d input image (discretized to levels)
            distance: pixel distance for co-occurrence
            angle: angle in radians for direction
            levels: number of gray levels

        returns:
            normalized glcm matrix
        """
        # discretize image
        image_min, image_max = image.min(), image.max()
        if image_max - image_min < 1e-10:
            return np.eye(levels) / levels

        discretized = ((image - image_min) / (image_max - image_min + 1e-10) * (levels - 1)).astype(int)
        discretized = np.clip(discretized, 0, levels - 1)

        # compute offset based on angle
        dx = int(np.round(distance * np.cos(angle)))
        dy = int(np.round(distance * np.sin(angle)))

        # initialize glcm
        glcm = np.zeros((levels, levels), dtype=np.float64)

        # compute co-occurrences
        rows, cols = discretized.shape
        for i in range(max(0, -dy), min(rows, rows - dy)):
            for j in range(max(0, -dx), min(cols, cols - dx)):
                if discretized[i, j] > 0 or discretized[i + dy, j + dx] > 0:
                    glcm[discretized[i, j], discretized[i + dy, j + dx]] += 1

        # make symmetric
        glcm = glcm + glcm.T

        # normalize
        total = np.sum(glcm)
        if total > 0:
            glcm = glcm / total

        return glcm

    @staticmethod
    def extract_from_glcm(glcm: np.ndarray) -> Dict[str, float]:
        """
        extract haralick texture features from glcm.

        args:
            glcm: normalized co-occurrence matrix

        returns:
            dictionary of texture feature values
        """
        features = {}

        # marginal distributions
        px = np.sum(glcm, axis=1)
        py = np.sum(glcm, axis=0)

        # indices
        levels = glcm.shape[0]
        i_indices, j_indices = np.meshgrid(range(levels), range(levels), indexing='ij')

        # mean and std of marginals
        mu_x = np.sum(i_indices * glcm)
        mu_y = np.sum(j_indices * glcm)
        sigma_x = np.sqrt(np.sum(((i_indices - mu_x) ** 2) * glcm))
        sigma_y = np.sqrt(np.sum(((j_indices - mu_y) ** 2) * glcm))

        # contrast (measure of local variation)
        features['contrast'] = float(np.sum(((i_indices - j_indices) ** 2) * glcm))

        # dissimilarity
        features['dissimilarity'] = float(np.sum(np.abs(i_indices - j_indices) * glcm))

        # homogeneity (inverse difference moment)
        features['homogeneity'] = float(np.sum(glcm / (1 + (i_indices - j_indices) ** 2)))

        # energy (angular second moment)
        features['energy'] = float(np.sum(glcm ** 2))

        # entropy
        glcm_nz = glcm[glcm > 0]
        features['entropy'] = float(-np.sum(glcm_nz * np.log2(glcm_nz + 1e-10)))

        # correlation
        if sigma_x > 0 and sigma_y > 0:
            features['correlation'] = float(
                np.sum((i_indices - mu_x) * (j_indices - mu_y) * glcm) / (sigma_x * sigma_y)
            )
        else:
            features['correlation'] = 0.0

        # cluster shade
        features['cluster_shade'] = float(
            np.sum(((i_indices + j_indices - mu_x - mu_y) ** 3) * glcm)
        )

        # cluster prominence
        features['cluster_prominence'] = float(
            np.sum(((i_indices + j_indices - mu_x - mu_y) ** 4) * glcm)
        )

        return features

    @staticmethod
    def extract(image: np.ndarray, mask: Optional[np.ndarray] = None,
                distances: List[int] = [1], angles: List[float] = [0]) -> Dict[str, float]:
        """
        extract averaged glcm features across distances and angles.
        """
        if mask is not None:
            # apply mask
            image = image.copy()
            image[~mask] = 0

        all_features = []

        for d in distances:
            for a in angles:
                glcm = GLCMFeatures.compute_glcm(image, distance=d, angle=a)
                features = GLCMFeatures.extract_from_glcm(glcm)
                all_features.append(features)

        # average across all glcms
        if not all_features:
            return {}

        averaged = {}
        for key in all_features[0].keys():
            values = [f[key] for f in all_features]
            averaged[f'glcm_{key}'] = float(np.mean(values))

        return averaged


class ShapeFeatures:
    """
    shape-based radiomics features.

    these features describe the morphological characteristics
    of the segmented region.
    """

    @staticmethod
    def extract(mask: np.ndarray, voxel_spacing: Tuple[float, ...] = (1.0, 1.0)) -> Dict[str, float]:
        """
        extract shape features from binary mask.

        args:
            mask: binary mask array
            voxel_spacing: physical spacing of voxels

        returns:
            dictionary of shape feature values
        """
        features = {}

        if not np.any(mask):
            return {
                'area': 0.0, 'perimeter': 0.0, 'compactness': 0.0,
                'eccentricity': 0.0, 'solidity': 0.0, 'extent': 0.0
            }

        # area (number of pixels * voxel area)
        voxel_area = np.prod(voxel_spacing)
        features['area'] = float(np.sum(mask) * voxel_area)

        # perimeter estimation using gradient
        gradient_x = ndimage.sobel(mask.astype(float), axis=0)
        gradient_y = ndimage.sobel(mask.astype(float), axis=1)
        perimeter_img = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        features['perimeter'] = float(np.sum(perimeter_img > 0))

        # compactness (circularity): 4*pi*area / perimeter^2
        if features['perimeter'] > 0:
            features['compactness'] = float(
                4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            )
        else:
            features['compactness'] = 0.0

        # eccentricity using moments
        moments = ndimage.measurements.moments(mask.astype(float))
        if moments[0, 0] > 0:
            cx = moments[1, 0] / moments[0, 0]
            cy = moments[0, 1] / moments[0, 0]

            # central moments
            mu20 = moments[2, 0] / moments[0, 0] - cx ** 2
            mu02 = moments[0, 2] / moments[0, 0] - cy ** 2
            mu11 = moments[1, 1] / moments[0, 0] - cx * cy

            # eigenvalues of covariance matrix
            delta = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
            lambda1 = (mu20 + mu02 + delta) / 2
            lambda2 = (mu20 + mu02 - delta) / 2

            if lambda1 > 0:
                features['eccentricity'] = float(np.sqrt(1 - lambda2 / lambda1))
            else:
                features['eccentricity'] = 0.0
        else:
            features['eccentricity'] = 0.0

        # solidity: area / convex hull area (approximated)
        # using morphological closing as approximation
        closed = ndimage.binary_closing(mask, iterations=5)
        convex_area = np.sum(closed) * voxel_area
        features['solidity'] = float(features['area'] / (convex_area + 1e-10))

        # extent: area / bounding box area
        coords = np.where(mask)
        if len(coords[0]) > 0:
            bbox_area = (coords[0].max() - coords[0].min() + 1) * \
                       (coords[1].max() - coords[1].min() + 1) * voxel_area
            features['extent'] = float(features['area'] / (bbox_area + 1e-10))
        else:
            features['extent'] = 0.0

        return features


class RadiomicsExtractor:
    """
    comprehensive radiomics feature extractor.

    combines first-order, glcm texture, and shape features
    for complete radiomics characterization.
    """

    def __init__(self, config: Optional[RadiomicsConfig] = None):
        self.config = config or RadiomicsConfig()
        self.first_order = FirstOrderFeatures()
        self.glcm = GLCMFeatures()
        self.shape = ShapeFeatures()

    def create_mask(self, image: np.ndarray) -> np.ndarray:
        """create brain mask from image using intensity thresholding."""
        threshold = self.config.intensity_threshold * image.max()
        mask = image > threshold

        # remove small components
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            mask = labeled == largest_component

        return mask

    def extract_slice(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        extract all radiomics features from a single slice.

        args:
            image: 2d image array
            mask: optional binary mask

        returns:
            dictionary of all feature values
        """
        if mask is None:
            mask = self.create_mask(image)

        features = {}

        # first-order features
        fo_features = self.first_order.extract(image, mask)
        features.update({f'fo_{k}': v for k, v in fo_features.items()})

        # glcm features
        glcm_features = self.glcm.extract(
            image, mask,
            distances=self.config.glcm_distances,
            angles=self.config.glcm_angles
        )
        features.update(glcm_features)

        # shape features
        shape_features = self.shape.extract(mask)
        features.update({f'shape_{k}': v for k, v in shape_features.items()})

        return features

    def extract_multimodal(self, images: Dict[str, np.ndarray],
                          mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        extract features from multiple modalities.

        args:
            images: dictionary of modality name to image array
            mask: optional shared mask

        returns:
            dictionary of features prefixed by modality
        """
        all_features = {}

        for modality, image in images.items():
            features = self.extract_slice(image, mask)
            all_features.update({f'{modality}_{k}': v for k, v in features.items()})

        return all_features


def extract_radiomics_from_dataset(
    data_dir: str,
    output_path: str,
    modalities: List[str] = ['t1', 't1gd', 't2', 'flair'],
    max_samples: int = 500
) -> Dict[str, np.ndarray]:
    """
    extract radiomics features from a dataset directory.

    args:
        data_dir: path to preprocessed data
        output_path: path to save extracted features
        modalities: list of modality names
        max_samples: maximum number of samples to process

    returns:
        dictionary of extracted feature arrays
    """
    import os
    import glob
    from pathlib import Path

    data_path = Path(data_dir)
    extractor = RadiomicsExtractor()

    all_features = []
    sample_ids = []

    # find all subject directories or npy files
    subjects = sorted([d for d in data_path.iterdir() if d.is_dir()])[:max_samples]

    print(f'[radiomics] processing {len(subjects)} subjects...')

    for i, subject_dir in enumerate(subjects):
        try:
            # load modalities
            images = {}
            for mod in modalities:
                mod_file = subject_dir / f'{mod}.nii.gz'
                if mod_file.exists():
                    import nibabel as nib
                    img = nib.load(str(mod_file))
                    data = img.get_fdata()
                    # take middle slice
                    mid_slice = data.shape[2] // 2
                    images[mod] = data[:, :, mid_slice]

            if images:
                features = extractor.extract_multimodal(images)
                all_features.append(features)
                sample_ids.append(subject_dir.name)

            if (i + 1) % 20 == 0:
                print(f'[radiomics] processed {i + 1}/{len(subjects)} subjects')

        except Exception as e:
            print(f'[radiomics] error processing {subject_dir}: {e}')
            continue

    if not all_features:
        print('[radiomics] no features extracted!')
        return {}

    # convert to numpy arrays
    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([[f.get(name, 0) for name in feature_names] for f in all_features])

    print(f'[radiomics] extracted {len(feature_names)} features from {len(all_features)} samples')

    # save
    np.save(output_path, feature_matrix)

    return {
        'features': feature_matrix,
        'feature_names': feature_names,
        'sample_ids': sample_ids
    }


if __name__ == '__main__':
    # test extraction
    print('[radiomics] testing feature extraction...')

    # create synthetic test image
    np.random.seed(42)
    test_image = np.random.rand(128, 128) * 100 + 50
    test_image[40:80, 40:80] += 50  # add a region

    extractor = RadiomicsExtractor()
    features = extractor.extract_slice(test_image)

    print(f'[radiomics] extracted {len(features)} features:')
    for name, value in sorted(features.items())[:10]:
        print(f'  {name}: {value:.4f}')
    print('  ...')
