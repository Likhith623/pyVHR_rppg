#!/bin/bash
# Neuro-Pulse Environment Setup Script
# Creates a conda environment with all required dependencies

set -e

ENV_NAME="neuropulse"

echo "============================================"
echo "  Neuro-Pulse Environment Setup"
echo "============================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment with Python 3.10
echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate the environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation, run:"
echo "  python verify_env.py"
echo ""
