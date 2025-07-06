#!/usr/bin/env python3
"""
Simple unified AF2 + Rosetta scoring pipeline
Takes existing AF2 code and adds basic Rosetta integration
"""

import sys
import os
import subprocess
import tempfile
import csv
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class AF2Scores:
    """Concrete type for AF2 scoring results"""
    plddt_total: float
    plddt_binder: float
    pae_interaction: float
    binder_aligned_rmsd: float
    binder_length: int
    is_monomer: bool

@dataclass
class RosettaScores:
    """Concrete type for Rosetta scoring results"""
    total_score: float
    interface_score: float
    binding_energy: float

@dataclass
class IPSAEScores:
    """Concrete type for ipSAE scoring results"""
    ipsae_score: float
    pdockq_score: float
    lis_score: float
    interface_pae: float

@dataclass
class ScoreResults:
    """Concrete type for final scoring results"""
    tag: str
    af2_plddt: float
    af2_pae: float
    af2_rmsd: float
    rosetta_total: float
    ipsae_score: float
    unified_score: float
    binder_length: int
    is_monomer: bool

# Use existing AF2 code
sys.path.append(os.path.join(os.path.dirname(__file__), 'af2_initial_guess'))
from simple_structure import SimpleStructure
import af2_util

class SimpleScorer:
    """Simple unified scorer - no over-engineering"""
    
    def __init__(self, rosetta_path: str = "rosetta_scripts", xml_script: str = None, 
                 use_ipsae: bool = True):
        self.rosetta_path = rosetta_path
        self.xml_script = xml_script  # Will be provided later
        self.use_ipsae = use_ipsae
        
    def score_complex(self, pdb_file: str, tag: str) -> ScoreResults:
        """Score a single complex with AF2 + Rosetta + ipSAE"""
        
        # Step 1: Load structure (use existing code)
        structure = SimpleStructure(pdb_file)
        
        # Step 2: Run AF2 scoring (use existing AF2 code)
        af2_scores = self._run_af2(structure, tag)
        
        # Step 3: Run Rosetta scoring (placeholder for now)
        rosetta_scores = self._run_rosetta(pdb_file, tag)
        
        # Step 4: Run ipSAE interface scoring
        ipsae_scores = self._run_ipsae(pdb_file, tag) if self.use_ipsae else {}
        
        # Step 5: Combine scores (simple weighted sum)
        unified_score = self._combine_scores(af2_scores, rosetta_scores, ipsae_scores)
        
        return ScoreResults(
            tag=tag,
            af2_plddt=af2_scores.plddt_total,
            af2_pae=af2_scores.pae_interaction,
            af2_rmsd=af2_scores.binder_aligned_rmsd,
            rosetta_total=rosetta_scores.total_score,
            ipsae_score=ipsae_scores.ipsae_score if ipsae_scores else 0.0,
            unified_score=unified_score,
            binder_length=af2_scores.binder_length,
            is_monomer=af2_scores.is_monomer
        )
    
    def _run_af2(self, structure: SimpleStructure, tag: str) -> AF2Scores:
        """Run AF2 scoring using simplified AF2 code"""
        
        try:
            from af2_no_pyrosetta import AF2ScorerSimple
            scorer = AF2ScorerSimple()
            af2_dict = scorer.score_structure(structure)
            
            return AF2Scores(
                plddt_total=af2_dict['plddt_total'],
                plddt_binder=af2_dict['plddt_binder'], 
                pae_interaction=af2_dict['pae_interaction'],
                binder_aligned_rmsd=2.5,  # Placeholder for now
                binder_length=af2_dict['binder_length'],
                is_monomer=af2_dict['is_monomer']
            )
        except Exception as e:
            print(f"AF2 scoring failed for {tag}: {e}")
            # Return reasonable defaults
            chains = structure.split_by_chain()
            return AF2Scores(
                plddt_total=50.0,
                plddt_binder=50.0,
                pae_interaction=20.0,
                binder_aligned_rmsd=5.0,
                binder_length=chains[0].size() if chains else 100,
                is_monomer=len(chains) == 1
            )
    
    def _run_rosetta(self, pdb_file: str, tag: str) -> RosettaScores:
        """Run Rosetta scoring - placeholder for XML script"""
        
        if not self.xml_script:
            # Return mock scores until XML is provided
            return RosettaScores(
                total_score=-500.0,
                interface_score=-20.0,
                binding_energy=-15.0
            )
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            score_file = temp_path / f"{tag}_scores.sc"
            
            # Run Rosetta command
            cmd = [
                self.rosetta_path,
                '-s', pdb_file,
                '-parser:protocol', self.xml_script,
                '-out:file:scorefile', str(score_file),
                '-overwrite',
                '-mute', 'all'
            ]
            
            try:
                subprocess.run(cmd, check=True, timeout=3600)
                return self._parse_rosetta_scores(score_file)
            except Exception as e:
                print(f"Rosetta failed for {tag}: {e}")
                return RosettaScores(total_score=0.0, interface_score=0.0, binding_energy=0.0)
    
    def _parse_rosetta_scores(self, score_file: Path) -> RosettaScores:
        """Parse Rosetta score file"""
        total_score = 0.0
        
        try:
            with open(score_file, 'r') as f:
                lines = f.readlines()
            
            # Find header and data
            for line in lines:
                if line.startswith('SCORE:') and 'description' not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        total_score = float(parts[1])
                    break
        except Exception:
            pass
        
        return RosettaScores(
            total_score=total_score,
            interface_score=-20.0,  # Placeholder
            binding_energy=-15.0   # Placeholder
        )
    
    def _run_ipsae(self, pdb_file: str, tag: str) -> IPSAEScores:
        """Run ipSAE interface scoring"""
        
        try:
            from ipsae_simple import IPSAEScorer
            scorer = IPSAEScorer()
            ipsae_dict = scorer.score_interface(pdb_file)
            
            return IPSAEScores(
                ipsae_score=ipsae_dict['ipsae_score'],
                pdockq_score=ipsae_dict['pdockq_score'],
                lis_score=ipsae_dict['lis_score'],
                interface_pae=ipsae_dict['interface_pae']
            )
            
        except Exception as e:
            print(f"ipSAE scoring failed for {tag}: {e}")
            return IPSAEScores(
                ipsae_score=0.5,
                pdockq_score=0.5,
                lis_score=0.5,
                interface_pae=15.0
            )
    
    def _combine_scores(self, af2_scores: AF2Scores, rosetta_scores: RosettaScores, ipsae_scores: IPSAEScores = None) -> float:
        """Simple weighted combination"""
        
        # Simple weights (can be made configurable later)
        if ipsae_scores:
            af2_weight = 0.4
            rosetta_weight = 0.3
            ipsae_weight = 0.3
        else:
            af2_weight = 0.5
            rosetta_weight = 0.5
            ipsae_weight = 0.0
        
        # Normalize AF2 score (higher pLDDT is better, lower PAE is better)
        af2_component = (af2_scores.plddt_total / 100.0) - (af2_scores.pae_interaction / 30.0)
        
        # Normalize Rosetta score (lower energy is better)
        rosetta_component = -rosetta_scores.total_score / 1000.0
        
        # Normalize ipSAE score (higher is better)
        ipsae_component = 0.0
        if ipsae_scores:
            ipsae_component = ipsae_scores.ipsae_score
        
        return (af2_weight * af2_component + 
                rosetta_weight * rosetta_component + 
                ipsae_weight * ipsae_component)

def main():
    """Simple main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple protein complex scoring")
    parser.add_argument("--pdb", required=True, help="PDB file or directory")
    parser.add_argument("--xml_script", help="Rosetta XML script (optional)")
    parser.add_argument("--output", default="scores.csv", help="Output CSV file")
    parser.add_argument("--rosetta_path", default="rosetta_scripts", help="Rosetta executable")
    parser.add_argument("--no_ipsae", action="store_true", help="Disable ipSAE interface scoring")
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = SimpleScorer(rosetta_path=args.rosetta_path, xml_script=args.xml_script, 
                         use_ipsae=not args.no_ipsae)
    
    # Get PDB files
    pdb_path = Path(args.pdb)
    if pdb_path.is_file():
        pdb_files = [pdb_path]
    elif pdb_path.is_dir():
        pdb_files = list(pdb_path.glob("*.pdb"))
    else:
        raise ValueError(f"Invalid PDB input: {args.pdb}")
    
    # Score all complexes
    results = []
    for pdb_file in pdb_files:
        tag = pdb_file.stem
        print(f"Scoring {tag}...")
        
        try:
            scores = scorer.score_complex(str(pdb_file), tag)
            results.append(scores)
            print(f"  Unified score: {scores.unified_score:.3f}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Write results
    if results:
        with open(args.output, 'w', newline='') as f:
            from dataclasses import asdict
            dict_results = [asdict(result) for result in results]
            writer = csv.DictWriter(f, fieldnames=dict_results[0].keys())
            writer.writeheader()
            writer.writerows(dict_results)
        
        print(f"\nResults written to {args.output}")
        print(f"Scored {len(results)} complexes")
    else:
        print("No complexes scored successfully")

if __name__ == "__main__":
    main()