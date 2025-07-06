# Simple AF2 + Rosetta + ipSAE Scoring Pipeline

A simple pipeline for scoring protein complexes using AlphaFold2 (AF2) predictions, Rosetta XML scripts, and ipSAE interface analysis, designed for HPC environments.

## Features

- **Triple Scoring**: AF2 confidence + Rosetta energy + ipSAE interface metrics
- **Simple Design**: Minimal code, easy to understand and modify
- **No PyRosetta Dependency**: Uses BioPython for structure handling  
- **SLURM Ready**: Basic job submission for cluster computing
- **Extensible**: Easy to add XML scripts and flags when available

## Quick Start

### Basic Usage

```bash
# Score a single PDB file (with AF2 + Rosetta + ipSAE)
python simple_pipeline.py --pdb complex.pdb --xml_script placeholder_scoring.xml

# Score multiple PDBs from directory  
python simple_pipeline.py --pdb structures/

# Score without ipSAE interface analysis
python simple_pipeline.py --pdb complex.pdb --no_ipsae

# Submit to SLURM
python submit_slurm.py --pdb complex.pdb --xml_script scoring.xml
```

## Installation

### Quick Setup
```bash
# Clone repository
git clone <repository_url>
cd score_all

# Create conda environment (GPU recommended)
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu

# Test installation (without AF2 models for now)
python simple_pipeline.py --pdb test_complex.pdb
```

### Detailed Installation
See [INSTALL.md](INSTALL.md) for complete setup instructions including:
- CPU vs GPU environment setup
- AF2 model weight download
- Troubleshooting guide
- HPC configuration

## Files

### Core Files
- **`simple_pipeline.py`**: Main scoring script (~200 lines)
- **`af2_no_pyrosetta.py`**: AF2 scoring without PyRosetta  
- **`ipsae_simple.py`**: Simplified ipSAE interface scoring
- **`submit_slurm.py`**: SLURM job submission (~120 lines)

### Integration Files
- **`psae.py`**: Full ipSAE implementation (reference)
- **`placeholder_scoring.xml`**: Template for Rosetta XML protocol
- **`placeholder_flags.txt`**: Template for Rosetta flags

## Output

The pipeline generates a simple CSV file with:
- **AF2 metrics**: pLDDT confidence, PAE interaction scores
- **Rosetta scores**: Energy values from XML protocol  
- **ipSAE metrics**: Interface analysis (ipSAE, pDockQ, LIS scores)
- **Unified score**: Weighted combination of all three

## When XML/Flags are Available

1. Replace `placeholder_scoring.xml` with actual protocol
2. Replace `placeholder_flags.txt` with actual flags
3. Update the `_run_rosetta()` function in `simple_pipeline.py` to use flags

## Design Principles

- **Keep it simple**: Minimal code, easy to understand
- **Extend later**: Add complexity only when needed  
- **Use existing code**: Leverage the working AF2 implementation
- **No over-engineering**: Avoid premature optimization