#!/bin/bash

# Install PyTorch and related packages with pip
pip install --no-cache-dir \
    torch==2.5.0 \
    torchvision \
    torchaudio \
    lightning \
    torchmetrics \
    torchsummary \
    dgl \
    matgl

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install other packages with UV
uv pip install --system \
    loguru \
    ply \
    hydra-core \
    hydra-colorlog \
    omegaconf \
    optuna \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    tensorboard \
    plotly \
    pillow \
    tqdm \
    python-dotenv \
    pymatgen \
    python-box \
    pydantic \
    pyyaml
