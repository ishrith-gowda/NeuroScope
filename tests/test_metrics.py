"""
unit tests for the result-critical metric / transform functions used across the
ext a + ext c evaluation scripts. these guard the integrity of every reported
number -- a silent bug in a metric would invalidate the paper.

run on the cluster venv (which has torch/scipy/skimage):
    cd ~/neuroscope/code && python -m pytest tests/test_metrics.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "journal_extension" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_kid_avg_means_over_modalities_x1000():
    from generate_ext_a_results import kid_avg

    r = {
        "kid_mean": {
            "A2B_FLAIR": 0.04,
            "A2B_T1": 0.04,
            "A2B_T1ce": 0.04,
            "A2B_T2": 0.04,
            "B2A_FLAIR": 0.20,
            "B2A_T1": 0.20,
            "B2A_T1ce": 0.20,
            "B2A_T2": 0.20,
        }
    }
    assert kid_avg(r, "A2B") == pytest.approx(40.0)
    assert kid_avg(r, "B2A") == pytest.approx(200.0)
    assert kid_avg({}, "A2B") is None


def test_mmd_rbf_zero_for_identical_positive_for_shifted():
    from eval_harmonization_correct import mmd_rbf

    rng = np.random.RandomState(0)
    x = rng.randn(200, 16)
    assert mmd_rbf(x, x.copy()) == pytest.approx(0.0, abs=1e-6)
    assert mmd_rbf(x, x + 5.0) > 0.01


def test_masked_windowed_ssim_identity_is_one():
    from eval_harmonization_correct import masked_windowed_ssim

    rng = np.random.RandomState(1)
    img = rng.rand(4, 64, 64).astype(np.float32)
    img[:, :8, :] = 0.0  # some background (zero) so the brain mask is non-trivial
    assert masked_windowed_ssim(img, img.copy()) == pytest.approx(1.0, abs=1e-4)


def test_modality_uint8_range_shape_dtype():
    import torch
    from multi_domain_eval import modality_uint8

    x = torch.zeros(2, 4, 8, 8)
    x[0] = -1.0
    x[1] = 1.0
    u = modality_uint8(x)
    assert u.shape == (2 * 4, 3, 8, 8)  # each modality -> its own grayscale-rgb image
    assert u.dtype == torch.uint8
    assert int(u[:4].min()) == 0  # -1 -> 0
    assert int(u[4:].max()) == 255  # +1 -> 255


def test_slice_sharpness_edge_sharper_than_flat():
    from boundary_sharpness import slice_sharpness

    flat = np.full((64, 64), 0.5, np.float32)
    edge = flat.copy()
    edge[:, 32:] = 1.0
    g_flat, _ = slice_sharpness(flat)
    g_edge, _ = slice_sharpness(edge)
    assert g_edge > g_flat  # a real edge has higher gradient magnitude than a flat region


def test_paired_reports_positive_delta_when_hybrid_higher():
    from boundary_sharpness import paired

    rng = np.random.RandomState(2)
    cycle = rng.rand(50)
    hybrid = cycle + 0.1  # hybrid strictly higher
    out = paired(hybrid, cycle)
    assert out["n_pairs"] == 50
    assert out["hybrid_minus_cycle"] == pytest.approx(0.1, abs=1e-6)
    assert out["wilcoxon_p"] < 1e-6
