#!/usr/bin/env python3
"""
additional baseline harmonization methods for comparison.

implements classical and simple methods:
- histogram matching
- z-score normalization
- nyul histogram standardization
- intensity range normalization

these serve as baselines to demonstrate the benefit of
deep learning-based harmonization approaches.

references:
- nyul et al. (2000): histogram-based intensity standardization
- shinohara et al. (2014): statistical normalization for multi-site mri
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import ndimage, stats, interpolate


@dataclass
class NormalizationConfig:
    """configuration for normalization methods."""
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    n_bins: int = 256
    target_range: Tuple[float, float] = (0.0, 1.0)


class ZScoreNormalizer:
    """
    z-score normalization (standardization).

    transforms intensities to zero mean and unit variance
    within each subject independently.
    """

    def __init__(self, mask_threshold: float = 0.01):
        self.mask_threshold = mask_threshold

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        apply z-score normalization.

        args:
            image: input image array
            mask: optional brain mask

        returns:
            normalized image
        """
        if mask is None:
            mask = image > (self.mask_threshold * image.max())

        # compute statistics within mask
        values = image[mask]
        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-10:
            return image.copy()

        # normalize
        normalized = (image - mean) / std

        # only apply within mask
        result = np.zeros_like(image)
        result[mask] = normalized[mask]

        return result


class IntensityRangeNormalizer:
    """
    intensity range normalization.

    scales intensities to a target range using percentile clipping.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        apply intensity range normalization.

        args:
            image: input image array
            mask: optional brain mask

        returns:
            normalized image
        """
        if mask is None:
            mask = image > 0

        values = image[mask]

        # percentile clipping
        p_low = np.percentile(values, self.config.percentile_low)
        p_high = np.percentile(values, self.config.percentile_high)

        # clip and scale
        clipped = np.clip(image, p_low, p_high)
        scaled = (clipped - p_low) / (p_high - p_low + 1e-10)

        # map to target range
        target_low, target_high = self.config.target_range
        normalized = scaled * (target_high - target_low) + target_low

        # only apply within mask
        result = np.zeros_like(image)
        result[mask] = normalized[mask]

        return result


class HistogramMatcher:
    """
    histogram matching (specification).

    transforms the histogram of a source image to match
    the histogram of a reference image.
    """

    def __init__(self, n_bins: int = 256):
        self.n_bins = n_bins
        self.reference_hist = None
        self.reference_bins = None

    def fit(self, reference: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        compute reference histogram.

        args:
            reference: reference image
            mask: optional brain mask
        """
        if mask is None:
            mask = reference > 0

        values = reference[mask].flatten()

        # compute histogram
        self.reference_hist, self.reference_bins = np.histogram(
            values, bins=self.n_bins, density=True
        )

        # compute cdf
        self.reference_cdf = np.cumsum(self.reference_hist)
        self.reference_cdf = self.reference_cdf / self.reference_cdf[-1]

    def transform(self, source: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        transform source to match reference histogram.

        args:
            source: source image to transform
            mask: optional brain mask

        returns:
            histogram-matched image
        """
        if self.reference_hist is None:
            raise RuntimeError("must call fit() before transform()")

        if mask is None:
            mask = source > 0

        values = source[mask].flatten()

        # compute source histogram and cdf
        source_hist, source_bins = np.histogram(values, bins=self.n_bins, density=True)
        source_cdf = np.cumsum(source_hist)
        source_cdf = source_cdf / source_cdf[-1]

        # create interpolation function for source cdf
        source_bin_centers = (source_bins[:-1] + source_bins[1:]) / 2
        ref_bin_centers = (self.reference_bins[:-1] + self.reference_bins[1:]) / 2

        # map source values through cdfs
        source_cdf_func = interpolate.interp1d(
            source_bin_centers, source_cdf,
            bounds_error=False, fill_value=(0, 1)
        )
        ref_cdf_inverse = interpolate.interp1d(
            self.reference_cdf, ref_bin_centers,
            bounds_error=False, fill_value=(ref_bin_centers[0], ref_bin_centers[-1])
        )

        # transform values
        source_cdf_values = source_cdf_func(values)
        matched_values = ref_cdf_inverse(source_cdf_values)

        # create result image
        result = np.zeros_like(source)
        result[mask] = matched_values

        return result

    def fit_transform(self, source: np.ndarray, reference: np.ndarray,
                      source_mask: Optional[np.ndarray] = None,
                      reference_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """fit to reference and transform source in one step."""
        self.fit(reference, reference_mask)
        return self.transform(source, source_mask)


class NyulNormalizer:
    """
    nyul histogram standardization.

    piecewise linear histogram transformation based on
    decile landmarks.

    reference:
    - nyul et al. (2000): new variants of a method of mri scale standardization
    """

    def __init__(self, n_landmarks: int = 11):
        self.n_landmarks = n_landmarks
        self.standard_landmarks = None
        self.percentiles = np.linspace(0, 100, n_landmarks)

    def learn_standard(self, images: List[np.ndarray],
                       masks: Optional[List[np.ndarray]] = None):
        """
        learn standard histogram landmarks from training images.

        args:
            images: list of training images
            masks: optional list of brain masks
        """
        if masks is None:
            masks = [None] * len(images)

        all_landmarks = []

        for image, mask in zip(images, masks):
            if mask is None:
                mask = image > 0

            values = image[mask].flatten()
            landmarks = np.percentile(values, self.percentiles)
            all_landmarks.append(landmarks)

        # compute mean landmarks
        self.standard_landmarks = np.mean(all_landmarks, axis=0)

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        normalize image to standard histogram.

        args:
            image: input image
            mask: optional brain mask

        returns:
            normalized image
        """
        if self.standard_landmarks is None:
            raise RuntimeError("must call learn_standard() before normalize()")

        if mask is None:
            mask = image > 0

        values = image[mask].flatten()

        # compute source landmarks
        source_landmarks = np.percentile(values, self.percentiles)

        # create piecewise linear mapping
        mapping = interpolate.interp1d(
            source_landmarks, self.standard_landmarks,
            bounds_error=False,
            fill_value=(self.standard_landmarks[0], self.standard_landmarks[-1])
        )

        # apply mapping
        normalized_values = mapping(values)

        # create result
        result = np.zeros_like(image)
        result[mask] = normalized_values

        return result


class WhiteStripeNormalizer:
    """
    whitestripe normalization.

    normalizes based on the peak of normal-appearing white matter.

    reference:
    - shinohara et al. (2014): statistical normalization techniques
    """

    def __init__(self, peak_width: float = 0.05):
        self.peak_width = peak_width

    def find_white_matter_peak(self, image: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> float:
        """
        find the white matter intensity peak.

        uses kernel density estimation to find the histogram peak.
        """
        if mask is None:
            mask = image > 0

        values = image[mask].flatten()

        # kernel density estimation
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 1000)
        density = kde(x_range)

        # find peak
        peak_idx = np.argmax(density)
        peak_value = x_range[peak_idx]

        return peak_value

    def normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        normalize image using white matter peak.

        args:
            image: input image
            mask: optional brain mask

        returns:
            normalized image
        """
        if mask is None:
            mask = image > 0

        # find white matter peak
        wm_peak = self.find_white_matter_peak(image, mask)

        if wm_peak < 1e-10:
            return image.copy()

        # normalize
        normalized = image / wm_peak

        # only apply within mask
        result = np.zeros_like(image)
        result[mask] = normalized[mask]

        return result


def apply_baseline_harmonization(
    domain_a_images: List[np.ndarray],
    domain_b_images: List[np.ndarray],
    method: str = 'histogram_matching'
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    apply baseline harmonization method to two domains.

    args:
        domain_a_images: list of images from domain a
        domain_b_images: list of images from domain b
        method: harmonization method name

    returns:
        (harmonized_a, harmonized_b) tuple of lists
    """
    if method == 'zscore':
        normalizer = ZScoreNormalizer()
        harmonized_a = [normalizer.normalize(img) for img in domain_a_images]
        harmonized_b = [normalizer.normalize(img) for img in domain_b_images]

    elif method == 'intensity_range':
        normalizer = IntensityRangeNormalizer()
        harmonized_a = [normalizer.normalize(img) for img in domain_a_images]
        harmonized_b = [normalizer.normalize(img) for img in domain_b_images]

    elif method == 'histogram_matching':
        # use domain a as reference
        matcher = HistogramMatcher()
        # compute average reference from domain a
        reference = np.mean(domain_a_images, axis=0)
        matcher.fit(reference)
        harmonized_a = domain_a_images  # reference domain unchanged
        harmonized_b = [matcher.transform(img) for img in domain_b_images]

    elif method == 'nyul':
        normalizer = NyulNormalizer()
        all_images = domain_a_images + domain_b_images
        normalizer.learn_standard(all_images)
        harmonized_a = [normalizer.normalize(img) for img in domain_a_images]
        harmonized_b = [normalizer.normalize(img) for img in domain_b_images]

    elif method == 'whitestripe':
        normalizer = WhiteStripeNormalizer()
        harmonized_a = [normalizer.normalize(img) for img in domain_a_images]
        harmonized_b = [normalizer.normalize(img) for img in domain_b_images]

    else:
        raise ValueError(f"unknown method: {method}")

    return harmonized_a, harmonized_b


def evaluate_baseline_method(
    domain_a_features: np.ndarray,
    domain_b_features: np.ndarray,
    method_name: str = 'baseline'
) -> Dict:
    """
    evaluate baseline method using same metrics as sa-cyclegan.

    args:
        domain_a_features: features from domain a
        domain_b_features: features from domain b
        method_name: name of the method

    returns:
        evaluation metrics dictionary
    """
    def compute_mmd(x, y, sigma=1.0):
        """compute maximum mean discrepancy."""
        n_samples = min(500, len(x), len(y))
        idx_x = np.random.choice(len(x), n_samples, replace=False)
        idx_y = np.random.choice(len(y), n_samples, replace=False)
        x_sub, y_sub = x[idx_x], y[idx_y]

        def rbf_kernel(a, b):
            diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
            return np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))

        k_xx = rbf_kernel(x_sub, x_sub).mean()
        k_yy = rbf_kernel(y_sub, y_sub).mean()
        k_xy = rbf_kernel(x_sub, y_sub).mean()

        return k_xx + k_yy - 2 * k_xy

    def compute_cosine_sim(x, y):
        """compute cosine similarity of distribution means."""
        mu_x = np.mean(x, axis=0)
        mu_y = np.mean(y, axis=0)
        return np.dot(mu_x, mu_y) / (np.linalg.norm(mu_x) * np.linalg.norm(mu_y) + 1e-10)

    mmd = compute_mmd(domain_a_features, domain_b_features)
    cosine = compute_cosine_sim(domain_a_features, domain_b_features)
    mean_diff = np.linalg.norm(np.mean(domain_a_features, axis=0) - np.mean(domain_b_features, axis=0))

    return {
        'method': method_name,
        'mmd': float(mmd),
        'cosine_similarity': float(cosine),
        'mean_difference': float(mean_diff)
    }


if __name__ == '__main__':
    # test baseline methods
    print('[baselines] testing baseline methods...')

    # create synthetic test images
    np.random.seed(42)
    domain_a = [np.random.rand(64, 64) * 100 + 50 for _ in range(5)]
    domain_b = [np.random.rand(64, 64) * 150 + 30 for _ in range(5)]  # different distribution

    for method in ['zscore', 'intensity_range', 'histogram_matching', 'nyul', 'whitestripe']:
        print(f'\n[baselines] testing {method}...')
        try:
            harm_a, harm_b = apply_baseline_harmonization(domain_a, domain_b, method)
            print(f'  domain a: {len(harm_a)} images')
            print(f'  domain b: {len(harm_b)} images')
            print(f'  mean intensity a: {np.mean([img.mean() for img in harm_a]):.2f}')
            print(f'  mean intensity b: {np.mean([img.mean() for img in harm_b]):.2f}')
        except Exception as e:
            print(f'  error: {e}')
