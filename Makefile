# neuroscope development makefile

.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format typecheck test test-unit test-integration test-all clean pre-commit ci

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

install:  ## install package in editable mode
	pip install -e .

install-dev:  ## install with dev dependencies
	pip install -e ".[dev,docs]"
	pre-commit install

# ---------------------------------------------------------------------------
# code quality
# ---------------------------------------------------------------------------

format:  ## auto-format code with black and isort
	black neuroscope/ journal_extension/ scripts/ tools/ tests/ setup.py
	isort neuroscope/ journal_extension/ scripts/ tools/ tests/ setup.py

lint:  ## run all linters
	ruff check neuroscope/ journal_extension/ scripts/ tools/
	flake8 neuroscope/ journal_extension/ scripts/ tools/

typecheck:  ## run mypy type checking
	mypy neuroscope/

ruff-fix:  ## run ruff with auto-fix
	ruff check --fix neuroscope/ journal_extension/ scripts/ tools/

# ---------------------------------------------------------------------------
# testing
# ---------------------------------------------------------------------------

test:  ## run unit tests
	pytest neuroscope/tests/unit/ -v --tb=short

test-integration:  ## run integration tests
	pytest neuroscope/tests/integration/ -v --tb=short

test-all:  ## run all tests with coverage
	pytest neuroscope/tests/ -v --tb=short --cov=neuroscope --cov-report=term-missing --cov-report=html

test-fast:  ## run tests excluding slow and gpu markers
	pytest neuroscope/tests/ -v --tb=short -m "not slow and not gpu"

# ---------------------------------------------------------------------------
# pre-commit and ci
# ---------------------------------------------------------------------------

pre-commit:  ## run pre-commit on all files
	pre-commit run --all-files

ci: lint typecheck test  ## run full ci pipeline locally

# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------

clean:  ## remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean  ## remove everything including venv
	rm -rf .venv/

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

help:  ## show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
