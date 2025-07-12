# AF2 + Rosetta + ipSAE Scoring Pipeline

Simple pipeline for scoring protein complexes using AlphaFold2 predictions, Rosetta energy functions, and interface analysis.

## Quick Start

```bash
# Score a single complex
python simple_pipeline.py --pdb complex.pdb

# Score multiple structures  
python simple_pipeline.py --pdb structures/

# Submit to SLURM cluster
python submit_slurm.py --pdb complex.pdb
```

## Features

- **AF2 Prediction**: Structure prediction with confidence scores
- **Rosetta Scoring**: Comprehensive binding analysis (33 metrics)
- **Interface Analysis**: ipSAE/pDockQ interface quality assessment
- **Real RMSD**: Structural change quantification
- **PyRosetta-free**: Uses BioPython for structure handling
- **HPC Ready**: SLURM job submission included

## Installation

```bash
# Quick setup
git clone <repository_url>
cd score_all
conda env create -f environment_gpu.yml
conda activate protein_scoring_gpu

# Test installation
python simple_pipeline.py --pdb test_complex.pdb
```

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

## Output

The pipeline produces a CSV file with 33 columns including:

- **AF2 scores**: pLDDT confidence, PAE interaction, structural RMSD
- **Rosetta scores**: Binding energies, surface areas, interface quality
- **Interface scores**: ipSAE, pDockQ, LIS metrics  
- **Combined score**: Weighted unified score

## Key Files

- `simple_pipeline.py` - Main scoring pipeline
- `submit_slurm.py` - SLURM job submission
- `psae.py` - Interface analysis scoring
- `af2_initial_guess/` - AF2 prediction components

## Requirements

- Python 3.8+
- JAX/JAXlib (GPU recommended)
- AlphaFold2 model weights
- Rosetta (optional)

For questions or issues, see the troubleshooting section in INSTALL.md.