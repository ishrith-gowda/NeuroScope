# contributing

thanks for your interest in SA-CycleGAN-2.5D. this document describes the development workflow and quality gates for the project.

## development setup

the project targets python 3.11 (see `.python-version`) and uses [`uv`](https://github.com/astral-sh/uv) for reproducible environments (a committed `uv.lock` pins all transitive dependencies).

```bash
# clone
git clone https://github.com/ishrith-gowda/SA-CycleGAN-2.5D.git
cd SA-CycleGAN-2.5D

# create the environment and install the package with dev extras
uv sync                      # or: python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # if not using uv

# install the git hooks
pre-commit install
```

## quality gates

all of the following are enforced in ci and locally via pre-commit:

| tool | purpose | command |
|------|---------|---------|
| `ruff format` | code formatting (line length 100) | `make format` |
| `ruff check` | linting (pyflakes, isort, bugbear, pyupgrade, simplify, …) | `make lint` |
| `mypy` | static type checking | `make typecheck` |
| `bandit` | security scanning | `make security` |
| `pytest` | tests | `make test` |

run the full local gate with `make ci` before opening a pull request.

## git workflow

`main` is protected: it requires a pull request and green ci, and disallows direct pushes (including for admins).

1. branch from `main`: `git checkout -b <type>/<short-description>` (e.g. `fix/console-logger-colors`, `docs/model-card`).
2. make focused commits. commit messages are lowercase, imperative, and prefixed with a type (`fix:`, `feat:`, `docs:`, `ci:`, `chore:`, `refactor:`, `test:`).
3. push and open a pull request against `main`. fill out the pull-request template.
4. ensure ci is green (lint + tests are required checks). resolve all review conversations.
5. merge once green. keep branches small and single-purpose.

## code style

- line length 100; formatting and import order are owned by `ruff` (do not hand-format).
- prefer typed public apis; new code in `neuroscope/models/` and `neuroscope/data/` should be fully type-annotated.
- keep modules focused; avoid files over ~500 lines.
- the canonical research pipeline lives in `journal_extension/scripts/`; the installable library is `neuroscope/`.

## reporting issues

use the issue templates (bug report / feature request). for security-sensitive reports, follow [`SECURITY.md`](SECURITY.md) instead of opening a public issue.
