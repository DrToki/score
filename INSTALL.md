# Installation Guide

## Quick Setup

```bash
# 1. Clone repository
git clone <repository_url>
cd score_all

# 2. Create environment (GPU recommended)
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu

# 3. Test installation
python simple_pipeline.py --pdb test_complex.pdb --no_ipsae
```

## Environment Options

### GPU Environment (Recommended)
```bash
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu
```

### CPU Environment (Slower AF2)
```bash
conda env create -f environment.yml  
conda activate protein_scoring
```

## AF2 Model Weights (Optional)

```bash
# Download AF2 models for full prediction capability
mkdir -p af2_initial_guess/model_weights
cd af2_initial_guess/model_weights
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar
mv params_model_1_ptm.npz model_1_ptm.npz
```

## Verification

```bash
# Test core components
python -c "
import jax
import numpy as np
from simple_structure import SimpleStructure
print('✅ Installation successful')
print(f'JAX devices: {jax.devices()}')
"

# Test pipeline with mock data
python simple_pipeline.py --pdb test_complex.pdb
```

## Troubleshooting

### GPU Issues
```bash
# Check CUDA/GPU
nvidia-smi
python -c "import jax; print(jax.devices())"

# Reinstall JAX for CUDA 11
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Memory Issues
```bash
# Limit GPU memory usage
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

### Import Errors
```bash
# Ensure conda environment is activated
conda activate protein_scoring_gpu

# Check critical imports
python -c "from alphafold.model import config; print('AF2 OK')"
```

## System Requirements

**Minimum**: 4 cores, 8GB RAM, 10GB storage
**Recommended**: 8+ cores, 32GB RAM, NVIDIA GPU (8GB+ VRAM), 50GB storage

## SLURM Setup

```bash
# Test SLURM submission
python submit_slurm.py --pdb complex.pdb --partition gpu --time 4:00:00

# Batch processing
python submit_slurm.py --pdb structures/ --gpu
```

For additional help, check the example test files or create a minimal test structure for debugging.