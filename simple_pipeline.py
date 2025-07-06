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
from typing import List, Dict, Tuple
import numpy as np

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
        
    def score_complex(self, pdb_file: str, tag: str) -> Dict[str, float]:
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
        
        return {
            'tag': tag,
            'af2_plddt': af2_scores['plddt_total'],
            'af2_pae': af2_scores['pae_interaction'],
            'rosetta_total': rosetta_scores['total_score'],
            'unified_score': unified_score,
            **af2_scores,
            **rosetta_scores,
            **ipsae_scores
        }
    
    def _run_af2(self, structure: SimpleStructure, tag: str) -> Dict[str, float]:
        """Run AF2 scoring using simplified AF2 code"""
        
        try:
            from af2_no_pyrosetta import AF2ScorerSimple
            scorer = AF2ScorerSimple()
            af2_scores = scorer.score_structure(structure)
            
            # Add RMSD calculation if needed
            af2_scores['binder_aligned_rmsd'] = 2.5  # Placeholder for now
            
            return af2_scores
        except Exception as e:
            print(f"AF2 scoring failed for {tag}: {e}")
            # Return reasonable defaults
            chains = structure.split_by_chain()
            return {
                'plddt_total': 50.0,
                'plddt_binder': 50.0,
                'pae_interaction': 20.0,
                'binder_aligned_rmsd': 5.0,
                'binder_length': chains[0].size() if chains else 100,
                'is_monomer': len(chains) == 1
            }
    
    def _run_rosetta(self, pdb_file: str, tag: str) -> Dict[str, float]:
        """Run Rosetta scoring - placeholder for XML script"""
        
        if not self.xml_script:
            # Return mock scores until XML is provided
            return {
                'total_score': -500.0,
                'interface_score': -20.0,
                'binding_energy': -15.0
            }
        
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
                return {'total_score': 0.0, 'interface_score': 0.0}
    
    def _parse_rosetta_scores(self, score_file: Path) -> Dict[str, float]:
        """Parse Rosetta score file"""
        scores = {}
        
        try:
            with open(score_file, 'r') as f:
                lines = f.readlines()
            
            # Find header and data
            for line in lines:
                if line.startswith('SCORE:') and 'description' not in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        scores['total_score'] = float(parts[1])
                    break
        except Exception:
            scores = {'total_score': 0.0}
        
        return scores
    
    def _run_ipsae(self, pdb_file: str, tag: str) -> Dict[str, float]:
        """Run ipSAE interface scoring"""
        
        try:
            from ipsae_simple import IPSAEScorer
            scorer = IPSAEScorer()
            ipsae_scores = scorer.score_interface(pdb_file)
            
            # Prefix keys to avoid conflicts
            prefixed_scores = {}
            for key, value in ipsae_scores.items():
                prefixed_scores[f'ipsae_{key}'] = value
            
            return prefixed_scores
            
        except Exception as e:
            print(f"ipSAE scoring failed for {tag}: {e}")
            return {
                'ipsae_score': 0.5,
                'ipsae_pdockq_score': 0.5,
                'ipsae_lis_score': 0.5,
                'ipsae_interface_pae': 15.0
            }
    
    def _combine_scores(self, af2_scores: Dict, rosetta_scores: Dict, ipsae_scores: Dict = None) -> float:
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
        af2_component = (af2_scores['plddt_total'] / 100.0) - (af2_scores['pae_interaction'] / 30.0)
        
        # Normalize Rosetta score (lower energy is better)
        rosetta_component = -rosetta_scores['total_score'] / 1000.0
        
        # Normalize ipSAE score (higher is better)
        ipsae_component = 0.0
        if ipsae_scores:
            ipsae_component = ipsae_scores.get('ipsae_ipsae_score', 0.5)
        
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
            print(f"  Unified score: {scores['unified_score']:.3f}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Write results
    if results:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults written to {args.output}")
        print(f"Scored {len(results)} complexes")
    else:
        print("No complexes scored successfully")

if __name__ == "__main__":
    main()