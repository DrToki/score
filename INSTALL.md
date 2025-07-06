# Installation Guide

Complete setup guide for the AF2 + Rosetta + ipSAE scoring pipeline.

## Prerequisites

### 1. Conda/Mamba
Install conda or mamba (mamba is faster):
```bash
# Option A: Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Option B: Mamba (recommended)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
```

### 2. Rosetta (Will be configured later)
Rosetta XML scripts and configuration will be added when available.
Pipeline works without Rosetta for AF2 + ipSAE scoring.

## Environment Setup

### Option A: CPU-Only (Faster setup, slower AF2)
```bash
# Clone repository
git clone <repository_url>
cd score_all

# Create conda environment
conda env create -f environment.yml
conda activate protein_scoring
```

### Option B: GPU (Recommended for AF2)
```bash
# Clone repository  
git clone <repository_url>
cd score_all

# Create GPU environment
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu

# Verify GPU setup
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

## AF2 Model Weights (Optional)

AF2 model weights will be configured when needed. For now, the pipeline can run with:
- ipSAE interface scoring (works immediately)
- Rosetta scoring (when XML scripts are added)
- Mock AF2 scores for testing

### When AF2 Models Are Needed
```bash
# Create weights directory
mkdir -p af2_initial_guess/model_weights

# Download model 1 (recommended for complexes)
cd af2_initial_guess/model_weights
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar
mv params_model_1_ptm.npz model_1_ptm.npz
```

## Rosetta Setup (Optional)
Rosetta integration will be configured when XML scripts become available.
For now, the pipeline provides AF2 + ipSAE scoring without Rosetta.

## Installation Verification

### 1. Test Pipeline Components
```bash
# Activate environment
conda activate protein_scoring  # or protein_scoring_gpu

# Test imports
python -c "
import jax
import numpy as np
import Bio
from simple_structure import SimpleStructure
print('✅ All imports successful')
print(f'JAX devices: {jax.devices()}')
"
```

### 2. Test with Sample Data
```bash
# Create test PDB (minimal complex)
cat > test_complex.pdb << 'EOF'
ATOM      1  N   ALA A   1      20.154  11.200  19.898  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  11.469  20.801  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.618  11.122  20.168  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.534  10.656  19.033  1.00 20.00           O  
ATOM      5  CB  ALA A   1      19.076  12.892  21.370  1.00 20.00           C  
ATOM      6  N   GLY B   1      30.154  11.200  19.898  1.00 20.00           N  
ATOM      7  CA  GLY B   1      29.030  11.469  20.801  1.00 20.00           C  
ATOM      8  C   GLY B   1      27.618  11.122  20.168  1.00 20.00           C  
ATOM      9  O   GLY B   1      27.534  10.656  19.033  1.00 20.00           O  
TER
END
EOF

# Test AF2 + ipSAE pipeline
python simple_pipeline.py --pdb test_complex.pdb

# Test AF2 only (without ipSAE)
python simple_pipeline.py --pdb test_complex.pdb --no_ipsae
```

## Troubleshooting

### Common Issues

#### 1. JAX/CUDA Issues
```bash
# Check CUDA version
nvidia-smi

# Reinstall JAX for specific CUDA version
pip uninstall jax jaxlib
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 2. AF2 Model Loading Issues
```bash
# Check model files
ls -la af2_initial_guess/model_weights/
# Should contain: model_1_ptm.npz (or other model files)

# Test model loading
python -c "
from alphafold.model import data
params = data.get_model_haiku_params('model_1_ptm', 'af2_initial_guess/model_weights')
print('✅ Model loaded successfully')
"
```

#### 3. Rosetta Not Found
```bash
# Check PATH
echo $PATH | grep rosetta

# Manual specification
python simple_pipeline.py --pdb test.pdb --rosetta_path /full/path/to/rosetta_scripts
```

#### 4. BioPython Structure Issues
```bash
# Test structure loading
python -c "
from simple_structure import SimpleStructure
s = SimpleStructure('test_complex.pdb')
print(f'✅ Structure loaded: {len(s.sequence())} residues')
"
```

### Performance Tips

#### 1. GPU Memory Issues
```bash
# Limit GPU memory for JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Or in Python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
```

#### 2. Parallel Processing
```bash
# Run multiple complexes in parallel
find structures/ -name "*.pdb" | xargs -P 4 -I {} python simple_pipeline.py --pdb {}
```

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ 
- **Storage**: 10GB+ (for AF2 models)
- **OS**: Linux (recommended), macOS, Windows WSL

### Recommended for Production
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 50GB+ SSD
- **OS**: Linux with SLURM

### HPC Considerations
- Check SLURM partition availability
- Verify GPU access and quotas
- Consider module loading in job scripts
- Test job submission limits

## Quick Start Commands

```bash
# Complete setup (GPU)
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu

# Test with ipSAE scoring
python simple_pipeline.py --pdb your_complex.pdb

# SLURM submission
python submit_slurm.py --pdb structures/ --gpu --partition gpu
```