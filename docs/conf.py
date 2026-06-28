"""sphinx configuration for the SA-CycleGAN-2.5D documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "SA-CycleGAN-2.5D"
author = "Ishrith Gowda, Chunwei Liu"
copyright = "2026, Ishrith Gowda, Chunwei Liu"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_typehints = "description"

# mock heavy/optional runtime deps so autodoc imports do not require them at
# doc-build time (keeps the readthedocs / ci build light and fast).
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "skimage",
    "sklearn",
    "nibabel",
    "SimpleITK",
    "simpleitk",
    "pandas",
    "matplotlib",
    "seaborn",
    "tensorboard",
    "tqdm",
    "yaml",
    "h5py",
    "monai",
    "torchio",
    "cv2",
    "PIL",
]

myst_enable_extensions = ["colon_fence", "deflist"]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

exclude_patterns = [
    "_build",
    "_autosummary",
    "Thumbs.db",
    ".DS_Store",
    # non-documentation content that also lives under docs/
    "website",
    "github-profile",
    "RESEARCHER_PROFILE_SETUP.md",
    "arxiv-comments-update.md",
    "linkedin-update.md",
]

html_theme = "sphinx_rtd_theme"
