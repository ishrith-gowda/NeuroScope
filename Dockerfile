# SA-CycleGAN-2.5D — cpu image for development, inference, and ci validation.
# for gpu training, swap the base image, e.g.:
#   FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build toolchain + git (some deps build from source); cleaned in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# copy package metadata + the importable package first for better layer caching
COPY pyproject.toml setup.py README.md ./
COPY neuroscope ./neuroscope

# install cpu torch first (small, ci-friendly), then the package itself
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -e .

# bring in the rest of the project (scripts, journal_extension, configs, ...)
COPY . .

CMD ["python", "-c", "import neuroscope; print('neuroscope import ok')"]
