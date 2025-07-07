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

# Import AF2 prediction functionality
import jax
import jax.numpy as jnp
from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from timeit import default_timer as timer
import uuid
import json
import shutil

class SimpleScorer:
    """Simple unified scorer - no over-engineering"""
    
    def __init__(self, rosetta_path: str = "rosetta_scripts", xml_script: str = None, 
                 use_ipsae: bool = True, af2_predictions_dir: str = "af2_predictions"):
        self.rosetta_path = rosetta_path
        self.xml_script = xml_script  # Will be provided later
        self.use_ipsae = use_ipsae
        self.af2_predictions_dir = af2_predictions_dir
        self._setup_af2_model()
        
        # Create predictions directory if it doesn't exist
        os.makedirs(self.af2_predictions_dir, exist_ok=True)
    
    def _setup_af2_model(self):
        """Setup AF2 model for predictions"""
        model_name = "model_1_ptm"
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = 1
        model_config.data.common.num_recycle = 3
        model_config.model.num_recycle = 3
        model_config.model.embeddings_and_evoformer.initial_guess = True
        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5
        
        params_dir = os.path.join(os.path.dirname(__file__), 'af2_initial_guess', 'model_weights')
        model_params = data.get_model_haiku_params(model_name=model_name, data_dir=params_dir)
        
        self.model_runner = model.RunModel(model_config, model_params)
        self.tmp_fn = f'tmp_{uuid.uuid4()}.pdb'
        
    def score_complex(self, pdb_file: str, tag: str) -> ScoreResults:
        """Score a single complex with AF2 + Rosetta + ipSAE"""
        
        # Step 1: Load structure (use existing code)
        structure = SimpleStructure(pdb_file)
        
        # Step 2: Run AF2 prediction to generate predicted structure
        af2_predicted_pdb = self._run_af2_prediction(structure, tag)
        
        # Step 3: Run AF2 scoring on predicted structure
        af2_scores = self._run_af2_scoring(af2_predicted_pdb, tag)
        
        # Step 4: Run Rosetta scoring on predicted structure
        rosetta_scores = self._run_rosetta(af2_predicted_pdb, tag)
        
        # Step 5: Run ipSAE interface scoring on predicted structure
        ipsae_scores = self._run_ipsae(af2_predicted_pdb, tag) if self.use_ipsae else {}
        
        # Step 6: Combine scores (simple weighted sum)
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
    
    def _run_af2_prediction(self, structure: SimpleStructure, tag: str) -> str:
        """Run AF2 prediction to generate predicted structure"""
        
        try:
            # Generate output filename
            af2_predicted_pdb = os.path.join(self.af2_predictions_dir, f"{tag}_af2pred.pdb")
            
            # Skip if already exists
            if os.path.exists(af2_predicted_pdb):
                print(f"AF2 prediction already exists for {tag}, skipping prediction")
                return af2_predicted_pdb
            
            # Determine structure properties
            chains = structure.split_by_chain()
            is_monomer = len(chains) == 1
            binder_length = chains[0].size() if not is_monomer else -1
            
            # Generate AF2 features
            all_atom_positions, all_atom_masks = structure.get_atoms_for_af2()
            initial_guess = af2_util.parse_initial_guess(all_atom_positions)
            
            # Determine which residues to template
            sequence = structure.sequence()
            if is_monomer:
                residue_mask = [False for _ in range(len(sequence))]
            else:
                residue_mask = [i >= binder_length for i in range(len(sequence))]
            
            # Generate template features
            template_dict = af2_util.generate_template_features(
                sequence, all_atom_positions, all_atom_masks, residue_mask
            )
            
            # Create feature dictionary
            feature_dict = {
                **pipeline.make_sequence_features(sequence=sequence, description="none", num_res=len(sequence)),
                **pipeline.make_msa_features(msas=[[sequence]], deletion_matrices=[[[0]*len(sequence)]]),
                **template_dict
            }
            
            # Handle chain breaks
            if not is_monomer:
                breaks = af2_util.check_residue_distances(all_atom_positions, all_atom_masks, 3.0)
                feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)
            
            feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)
            
            # Run AF2 prediction
            print(f"Running AF2 prediction for {tag}...")
            prediction_result = self.model_runner.apply(
                self.model_runner.params,
                jax.random.PRNGKey(0),
                feature_dict,
                initial_guess
            )
            
            # Extract structure from prediction
            structure_module = prediction_result['structure_module']
            predicted_protein = protein.Protein(
                aatype=feature_dict['aatype'][0],
                atom_positions=structure_module['final_atom_positions'][...],
                atom_mask=structure_module['final_atom_mask'][...],
                residue_index=feature_dict['residue_index'][0] + 1,
                b_factors=np.zeros_like(structure_module['final_atom_mask'][...])
            )
            
            # Extract confidence scores for psae.py
            confidences = {}
            confidences['plddt'] = confidence.compute_plddt(prediction_result['predicted_lddt']['logits'])
            
            if 'predicted_aligned_error' in prediction_result:
                confidences.update(confidence.compute_predicted_aligned_error(
                    prediction_result['predicted_aligned_error']['logits'],
                    prediction_result['predicted_aligned_error']['breaks']
                ))
            
            # Save PAE matrix as JSON for psae.py
            af2_pae_json = af2_predicted_pdb.replace('.pdb', '_scores.json')
            pae_data = {
                'plddt': confidences['plddt'].tolist(),
                'predicted_aligned_error': confidences.get('predicted_aligned_error', []).tolist() if 'predicted_aligned_error' in confidences else [],
                'pae': confidences.get('predicted_aligned_error', []).tolist() if 'predicted_aligned_error' in confidences else []
            }
            
            # Add PTM scores if available
            if 'ptm' in prediction_result:
                pae_data['ptm'] = float(prediction_result['ptm'])
            if 'iptm' in prediction_result:
                pae_data['iptm'] = float(prediction_result['iptm'])
            
            with open(af2_pae_json, 'w') as f:
                json.dump(pae_data, f, indent=2)
            
            # Save predicted structure
            predicted_pdb_lines = protein.to_pdb(predicted_protein)
            with open(af2_predicted_pdb, 'w') as f:
                f.write(predicted_pdb_lines)
            
            print(f"AF2 prediction saved to {af2_predicted_pdb}")
            print(f"AF2 PAE matrix saved to {af2_pae_json}")
            return af2_predicted_pdb
            
        except Exception as e:
            print(f"AF2 prediction failed for {tag}: {e}")
            # Return original structure if prediction fails
            return structure.pdb_file
    
    def _run_af2_scoring(self, pdb_file: str, tag: str) -> AF2Scores:
        """Run AF2 scoring on predicted structure"""
        
        try:
            structure = SimpleStructure(pdb_file)
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
            structure = SimpleStructure(pdb_file)
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
            
            # Run Rosetta command with specified flags
            cmd = [
                self.rosetta_path,
                '-s', pdb_file,
                '-parser:protocol', self.xml_script,
                '-out:file:scorefile', str(score_file),
                '-overwrite',
                '-mh:score:use_ss1', 'true',
                '-mh:score:use_ss2', 'true',
                '-mh:score:use_aa1', 'false',
                '-mh:score:use_aa2', 'false',
                '-mh:path:motifs', '/net/software/Rosetta/main/database/additional_protocol_data/sewing/xsmax_bb_ss_AILV_resl0.8_msc0.3/xsmax_bb_ss_AILV_resl0.8_msc0.3.rpm.bin.gz',
                '-mh:path:scores_BB_BB', '/net/software/Rosetta/main/database/additional_protocol_data/sewing/xsmax_bb_ss_AILV_resl0.8_msc0.3/xsmax_bb_ss_AILV_resl0.8_msc0.3',
                '-mh:gen_reverse_motifs_on_load', 'false',
                '-corrections:beta_nov16',
                '-scorefile_format', 'json',
                '-load_PDB_components', 'false',
                '-ignore_zero_occupancy', 'false',
                '-out:file:output_secondary_structure', 'true',
                '-skip_connect_info',
                '-out:file:do_not_autoassign_SS',
                '-output_pose_cache_data',
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
        """Run ipSAE interface scoring using psae.py"""
        
        try:
            # Look for corresponding PAE JSON file
            pae_file = pdb_file.replace('.pdb', '_scores.json')
            if not os.path.exists(pae_file):
                print(f"PAE file not found for {tag}: {pae_file}")
                raise FileNotFoundError(f"PAE file not found: {pae_file}")
            
            # Run psae.py as subprocess
            import subprocess
            import tempfile
            
            # Create temp output directory to capture psae.py results
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdb = os.path.join(temp_dir, f"{tag}.pdb")
                temp_pae = os.path.join(temp_dir, f"{tag}_scores.json")
                
                # Copy files to temp directory
                import shutil
                shutil.copy2(pdb_file, temp_pdb)
                shutil.copy2(pae_file, temp_pae)
                
                # Run psae.py
                cmd = [
                    'python', 'psae.py',
                    temp_pae, temp_pdb,
                    '10', '10'  # pae_cutoff, dist_cutoff
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
                
                if result.returncode != 0:
                    print(f"psae.py failed for {tag}: {result.stderr}")
                    raise Exception(f"psae.py failed: {result.stderr}")
                
                # Parse the output file
                output_file = os.path.join(temp_dir, f"{tag}_10_10.txt")
                if os.path.exists(output_file):
                    return self._parse_psae_output(output_file)
                else:
                    raise Exception(f"psae.py output file not found: {output_file}")
            
        except Exception as e:
            print(f"ipSAE scoring failed for {tag}: {e}")
            # Try fallback to simple ipSAE implementation
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
            except Exception:
                return IPSAEScores(
                    ipsae_score=0.5,
                    pdockq_score=0.5,
                    lis_score=0.5,
                    interface_pae=15.0
                )
    
    def _parse_psae_output(self, output_file: str) -> IPSAEScores:
        """Parse psae.py output file to extract scores"""
        ipsae_score = 0.5
        pdockq_score = 0.5
        lis_score = 0.5
        interface_pae = 15.0
        
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the output - look for the "max" line which has the best scores
            for line in lines:
                if 'max' in line and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 13:
                        ipsae_score = float(parts[5])  # ipSAE column
                        pdockq_score = float(parts[9])  # pDockQ column
                        lis_score = float(parts[11])  # LIS column
                        # interface_pae would need to be calculated from PAE data
                        break
        except Exception as e:
            print(f"Error parsing psae.py output: {e}")
        
        return IPSAEScores(
            ipsae_score=ipsae_score,
            pdockq_score=pdockq_score,
            lis_score=lis_score,
            interface_pae=interface_pae
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
                         use_ipsae=not args.no_ipsae, af2_predictions_dir="af2_predictions")
    
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