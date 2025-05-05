#!/bin/bash

echo "Installing all packages"
pip install "duckdb>=1.2.2" \
    "geopandas>=1.0.1" \
    "ipykernel>=6.29.5" \
    "lightning>=2.5.1.post0" \
    "matplotlib>=3.10.1" \
    "numpy>=2.2.5" \
    "pandas>=2.2.3" \
    "polars>=1.29.0" \
    "pyarrow>=20.0.0" \
    "pympler>=1.1" \
    "returns>=0.25.0" \
    "ruff>=0.11.8" \
    "scikit-learn>=1.6.1" \
    "seaborn>=0.13.2" \
    "tensorboard>=2.19.0" \
    "torch>=2.7.0"

echo "Installing additional packages"
pip install -e .

# Configure Git user
echo "Configuring Git..."
git config --global user.name "CooperBigFoot"
git config --global user.email "nlazaro@student.ethz.ch"

# Print success message
echo "RunPod setup complete. You can now train and push the code to GitHub"
