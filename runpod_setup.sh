#!/bin/bash

echo "Installing all packages"
pip install uv
uv sync

echo "Installing additional packages"
pip install -e .

# Configure Git user
echo "Configuring Git..."
git config --global user.name "CooperBigFoot"
git config --global user.email "nlazaro@student.ethz.ch"

# Print success message
echo "RunPod setup complete. You can now train and push the code to GitHub"
