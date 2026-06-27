"""setup shim — all package metadata lives in pyproject.toml (PEP 621).

retained only for `pip install -e .` ergonomics. the previous hand-written
setup() duplicated pyproject (with a stale 0.1.0 version and divergent deps)
and listed non-existent scripts/entry-points, which broke editable installs in ci.
"""

from setuptools import setup

setup()
