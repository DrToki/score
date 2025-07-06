#!/bin/bash
# download_models.sh
# Simple script to download AF2 model weights

set -e

echo "🔽 Downloading AlphaFold2 model weights..."

# Create directory if it doesn't exist
mkdir -p af2_initial_guess/model_weights
cd af2_initial_guess/model_weights

# Check if model already exists
if [ -f "model_1_ptm.npz" ]; then
    echo "✅ Model weights already exist. Skipping download."
    exit 0
fi

# Download and extract
echo "📦 Downloading model parameters (~4GB)..."
wget -q --show-progress https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar

echo "📂 Extracting model files..."
tar -xf alphafold_params_2022-12-06.tar

# Move the main model file
mv params_model_1_ptm.npz model_1_ptm.npz

# Clean up
rm alphafold_params_2022-12-06.tar
rm -f params_model_*  # Remove other models to save space

echo "✅ Model download complete!"
echo "📁 Model location: af2_initial_guess/model_weights/model_1_ptm.npz"