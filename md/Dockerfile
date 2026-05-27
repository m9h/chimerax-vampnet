# OpenMM + PDBFixer with CUDA on aarch64 for the DGX Spark (NVIDIA GB10).
#
# conda-forge ships an aarch64 build of openmm that includes the CUDA
# platform; the PyPI wheel does not. We install via micromamba to keep
# the image small and reproducible.
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba.
RUN mkdir -p /opt/conda/bin && \
    curl -sLo /tmp/mm.tar.bz2 \
      "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest" && \
    tar -xvjf /tmp/mm.tar.bz2 -C /opt/conda bin/micromamba && \
    rm /tmp/mm.tar.bz2 && \
    /opt/conda/bin/micromamba --version

# Create the openmm env with CUDA 12 support.
# CONDA_OVERRIDE_CUDA forces conda-forge to pull the CUDA-built openmm
# variant even though we're building on a host without a GPU visible
# to the build sandbox.
ENV CONDA_OVERRIDE_CUDA=12.9
RUN micromamba install -y -n base -c conda-forge \
        python=3.13 \
        "openmm=8.5.1" \
        openmmtools \
        pdbfixer \
        "cuda-version>=12.9,<13" \
        numpy \
        scipy \
    && micromamba clean -a -y \
    && python -c "import openmm; from openmm import Platform; \
print('OpenMM', openmm.version.full_version); \
[print('  ', Platform.getPlatform(i).getName()) for i in range(Platform.getNumPlatforms())]"

WORKDIR /workspace
