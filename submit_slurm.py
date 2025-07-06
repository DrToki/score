#!/usr/bin/env python3
"""
Simple SLURM submission for protein scoring
No over-engineering - just basic job submission
"""

import subprocess
import tempfile
import os
from pathlib import Path
import argparse

def create_slurm_script(pdb_input: str, xml_script: str, job_name: str, 
                       partition: str = "cpu", cpus: int = 8, memory: str = "32G", 
                       time: str = "24:00:00", gpu: bool = False, no_ipsae: bool = False) -> str:
    """Create simple SLURM script"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
"""
    
    if gpu:
        script_content += "#SBATCH --gres=gpu:1\n"
    
    script_content += f"""
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules (adjust for your system)
module load python/3.9
"""
    
    if gpu:
        script_content += "module load cuda/11.8\n"
    
    script_content += f"""
# Run scoring pipeline
cd {Path(__file__).parent}

python simple_pipeline.py \\
    --pdb {pdb_input} \\
"""
    
    if xml_script:
        script_content += f"    --xml_script {xml_script} \\\n"
    
    if no_ipsae:
        script_content += "    --no_ipsae \\\n"
    
    script_content += f"""    --output {job_name}_scores.csv

echo "Job completed: $(date)"
"""
    
    return script_content

def submit_job(script_content: str) -> str:
    """Submit job to SLURM"""
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        # Make executable
        os.chmod(script_path, 0o755)
        
        # Submit job
        result = subprocess.run(['sbatch', script_path], 
                              capture_output=True, text=True, check=True)
        
        # Extract job ID
        job_id = result.stdout.strip().split()[-1]
        
        return job_id
        
    finally:
        # Clean up script file
        os.unlink(script_path)

def main():
    parser = argparse.ArgumentParser(description="Submit protein scoring to SLURM")
    parser.add_argument("--pdb", required=True, help="PDB file or directory")
    parser.add_argument("--xml_script", help="Rosetta XML script")
    parser.add_argument("--job_name", default="protein_scoring", help="Job name")
    parser.add_argument("--partition", default="cpu", help="SLURM partition")
    parser.add_argument("--cpus", type=int, default=8, help="Number of CPUs")
    parser.add_argument("--memory", default="32G", help="Memory requirement")
    parser.add_argument("--time", default="24:00:00", help="Time limit")
    parser.add_argument("--gpu", action="store_true", help="Request GPU")
    parser.add_argument("--no_ipsae", action="store_true", help="Disable ipSAE interface scoring")
    
    args = parser.parse_args()
    
    # Create script
    script = create_slurm_script(
        pdb_input=args.pdb,
        xml_script=args.xml_script,
        job_name=args.job_name,
        partition=args.partition,
        cpus=args.cpus,
        memory=args.memory,
        time=args.time,
        gpu=args.gpu,
        no_ipsae=args.no_ipsae
    )
    
    # Submit job
    try:
        job_id = submit_job(script)
        print(f"Job submitted: {job_id}")
        print(f"Monitor with: squeue -j {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e}")
        print(f"Error output: {e.stderr}")

if __name__ == "__main__":
    main()