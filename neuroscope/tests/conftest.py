"""shared pytest fixtures for neuroscope test suite."""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# device fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """return available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """always return cpu device for deterministic tests."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# random seed fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_everything():
    """set deterministic seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ---------------------------------------------------------------------------
# tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_2d():
    """2d image batch: (batch=2, channels=4, h=64, w=64)."""
    return torch.randn(2, 4, 64, 64)


@pytest.fixture
def batch_25d():
    """2.5d tri-planar batch: (batch=2, channels=12, h=64, w=64).

    simulates 3 adjacent slices x 4 modalities.
    """
    return torch.randn(2, 12, 64, 64)


@pytest.fixture
def batch_3d():
    """3d volume batch: (batch=2, channels=4, d=32, h=32, w=32)."""
    return torch.randn(2, 4, 32, 32, 32)


@pytest.fixture
def volume_numpy():
    """single mri volume as numpy array: (channels=4, d=64, h=64, w=64)."""
    return np.random.randn(4, 64, 64, 64).astype(np.float32)


@pytest.fixture
def single_modality_volume():
    """single modality mri volume: (1, d=64, h=64, w=64)."""
    return np.random.randn(1, 64, 64, 64).astype(np.float32)


# ---------------------------------------------------------------------------
# temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_experiment_dir(tmp_path):
    """create a temporary experiment directory structure."""
    dirs = {
        "checkpoints": tmp_path / "checkpoints",
        "samples": tmp_path / "samples",
        "logs": tmp_path / "logs",
        "metrics": tmp_path / "metrics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def tmp_data_dir(tmp_path):
    """create temporary data directory with fake nifti-like structure."""
    data_dir = tmp_path / "data"
    for site in ["site_a", "site_b"]:
        site_dir = data_dir / site
        site_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            subject_dir = site_dir / f"subject_{i:03d}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            # create dummy numpy arrays as stand-in for nifti
            for modality in ["t1", "t2", "flair", "t1ce"]:
                vol = np.random.randn(64, 64, 64).astype(np.float32)
                np.save(subject_dir / f"{modality}.npy", vol)
    return data_dir


# ---------------------------------------------------------------------------
# model config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def generator_config():
    """minimal generator configuration for testing."""
    return {
        "input_nc": 12,
        "output_nc": 12,
        "ngf": 32,
        "n_residual_blocks": 3,
        "use_attention": True,
        "use_cbam": False,
    }


@pytest.fixture
def discriminator_config():
    """minimal discriminator configuration for testing."""
    return {
        "input_nc": 12,
        "ndf": 32,
        "n_layers": 2,
    }


@pytest.fixture
def training_config(tmp_path):
    """minimal training configuration for testing."""
    return {
        "experiment_name": "test_run",
        "output_dir": str(tmp_path / "output"),
        "epochs": 2,
        "batch_size": 2,
        "lr_g": 2e-4,
        "lr_d": 2e-4,
        "lambda_cycle": 10.0,
        "lambda_identity": 5.0,
        "use_amp": False,
        "save_interval": 1,
        "log_interval": 1,
    }


# ---------------------------------------------------------------------------
# skip markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring gpu")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "data: marks tests requiring real data")


def pytest_collection_modifyitems(config, items):
    """auto-skip gpu tests when no gpu is available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="no gpu available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
