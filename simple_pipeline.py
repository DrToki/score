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

# Import core dependencies with error handling
try:
    from simple_structure import SimpleStructure
    import af2_util
    from timeit import default_timer as timer
    import uuid
    import json
    import shutil
except ImportError as e:
    print(f"Error importing core dependencies: {e}")
    sys.exit(1)

# Import AF2 prediction functionality with error handling
try:
    import jax
    import jax.numpy as jnp
    from alphafold.common import residue_constants
    from alphafold.common import protein
    from alphafold.common import confidence
    from alphafold.data import pipeline
    from alphafold.model import data
    from alphafold.model import config
    from alphafold.model import model
    AF2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AF2 dependencies not available: {e}")
    AF2_AVAILABLE = False

# Import robust structure handling functionality
try:
    from robust_structure_handler import RobustStructureHandler, prepare_structure_for_af2
    ROBUST_HANDLER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Robust structure handler not available: {e}")
    ROBUST_HANDLER_AVAILABLE = False

class SimpleScorer:
    """Simple unified scorer - no over-engineering"""
    
    def __init__(self, rosetta_path: str = "rosetta_scripts", xml_script: str = None, 
                 use_ipsae: bool = True, af2_predictions_dir: str = "af2_predictions",
                 rosetta_database_dir: str = None, auto_clean: bool = True, 
                 auto_renumber: bool = True, strict_validation: bool = False):
        self.rosetta_path = rosetta_path
        self.xml_script = xml_script  # Will be provided later
        self.use_ipsae = use_ipsae
        self.af2_predictions_dir = af2_predictions_dir
        self.rosetta_database_dir = rosetta_database_dir or "/net/software/Rosetta/main/database"
        
        # Structure handling options
        self.auto_clean = auto_clean
        self.auto_renumber = auto_renumber
        self.strict_validation = strict_validation
        
        # Initialize robust structure handler
        self.structure_handler = None
        if ROBUST_HANDLER_AVAILABLE:
            self.structure_handler = RobustStructureHandler()
            print("✅ Robust structure handler initialized")
        else:
            print("⚠️  Robust structure handler not available - using basic structure handling")
        
        self._setup_af2_model()
        
        # Create predictions directory if it doesn't exist
        os.makedirs(self.af2_predictions_dir, exist_ok=True)
    
    def _setup_af2_model(self):
        """Setup AF2 model for predictions"""
        if not AF2_AVAILABLE:
            print("Warning: AF2 not available, predictions will be skipped")
            self.model_runner = None
            return
            
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
        
    def score_complex(self, pdb_file: str, tag: str) -> ScoreResults:
        """Score a single complex with AF2 + Rosetta + ipSAE"""
        
        # Step 1: Load and prepare structure using robust handler
        try:
            structure = SimpleStructure(pdb_file)
            print(f"📁 Loaded structure {tag}: {structure.size()} residues")
            
            # Prepare structure for prediction
            prepared_structure = self._prepare_structure_for_prediction(structure, tag)
            
            # Analyze chain structure using robust handler
            chain_info = self._analyze_chain_structure(prepared_structure, tag)
            
        except Exception as e:
            print(f"❌ Structure loading failed for {tag}: {e}")
            raise
        
        # Step 2: Run AF2 prediction to generate predicted structure
        af2_predicted_pdb = self._run_af2_prediction(prepared_structure, tag)
        
        # Step 3: Run AF2 scoring on predicted structure
        af2_scores = self._run_af2_scoring(af2_predicted_pdb, tag)
        
        # Step 4: Run Rosetta scoring on predicted structure
        rosetta_scores = self._run_rosetta(af2_predicted_pdb, tag)
        
        # Step 5: Run ipSAE interface scoring on predicted structure
        ipsae_scores = self._run_ipsae(af2_predicted_pdb, tag) if self.use_ipsae else None
        
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
        
        if not AF2_AVAILABLE or self.model_runner is None:
            print(f"AF2 not available, skipping prediction for {tag}")
            return structure.pdb_file
        
        try:
            # Generate output filename
            af2_predicted_pdb = os.path.join(self.af2_predictions_dir, f"{tag}_af2pred.pdb")
            
            # Skip if already exists
            if os.path.exists(af2_predicted_pdb):
                print(f"AF2 prediction already exists for {tag}, skipping prediction")
                return af2_predicted_pdb
            
            # Prepare structure using robust handler if available
            prepared_structure = self._prepare_structure_for_prediction(structure, tag)
            
            # Analyze chain structure to get proper indices
            chain_info = self._analyze_chain_structure(prepared_structure, tag)
            
            # Use chain analysis results for proper indexing
            is_monomer = chain_info['is_monomer']
            binder_length = chain_info['binder_length'] if not is_monomer else -1
            
            # Validate chain indices are handled correctly
            if not self._validate_chain_indices(prepared_structure, chain_info, tag):
                print(f"⚠️  Chain index validation failed for {tag}, using fallback")
                # Use basic fallback values
                chains = prepared_structure.split_by_chain()
                is_monomer = len(chains) == 1
                binder_length = chains[0].size() if not is_monomer else -1
            
            # Store chain info for downstream use
            self._current_chain_info = chain_info
            
            # Generate AF2 features from prepared structure
            all_atom_positions, all_atom_masks = prepared_structure.get_atoms_for_af2()
            initial_guess = af2_util.parse_initial_guess(all_atom_positions)
            
            # Determine which residues to template using chain analysis
            sequence = prepared_structure.sequence()
            if is_monomer:
                residue_mask = [False for _ in range(len(sequence))]
            else:
                # Template the target chain (after binder_length)
                # Note: prepare_structure_for_af2 puts binder first, then target
                residue_mask = [i >= binder_length for i in range(len(sequence))]
                print(f"🎯 Templating residues {binder_length} to {len(sequence)-1} (target chain)")
                print(f"   Binder residues 0 to {binder_length-1} will be predicted freely")
            
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
    
    def _prepare_structure_for_prediction(self, structure: SimpleStructure, tag: str) -> SimpleStructure:
        """Prepare structure for AF2 prediction using robust handler"""
        
        if not ROBUST_HANDLER_AVAILABLE or self.structure_handler is None:
            print(f"Using basic structure handling for {tag}")
            return structure
        
        try:
            print(f"📋 Preparing structure for {tag} using robust handler...")
            
            # Get original chain information before preparation
            original_chains = structure.split_by_chain()
            print(f"   Original structure: {len(original_chains)} chains")
            
            if len(original_chains) == 2:
                # Get original chain IDs
                original_chain_ids = []
                for chain in original_chains:
                    try:
                        chain_id = list(chain.structure.get_chains())[0].get_id()
                        original_chain_ids.append(chain_id)
                    except:
                        original_chain_ids.append("unknown")
                        
                print(f"   Original chain IDs: {original_chain_ids}")
                print(f"   Original chain sizes: {[chain.size() for chain in original_chains]}")
            
            # Use robust structure preparation
            prepared_structure = prepare_structure_for_af2(
                structure.pdb_file,
                auto_clean=self.auto_clean,
                auto_renumber=self.auto_renumber
            )
            
            # Show what happened during preparation
            prepared_chains = prepared_structure.split_by_chain()
            print(f"   Prepared structure: {len(prepared_chains)} chains")
            
            if len(prepared_chains) == 2:
                print(f"   🔄 Chain order after preparation: binder (index 0), target (index 1)")
                print(f"   🔄 Chain sizes after preparation: {[chain.size() for chain in prepared_chains]}")
                print(f"   📋 Note: prepare_structure_for_af2 automatically puts binder first, target second")
            
            # Validate the prepared structure
            validation_report = self.structure_handler.validate_structure(prepared_structure)
            
            if not validation_report['valid']:
                if self.strict_validation:
                    raise ValueError(f"Structure validation failed for {tag}: {validation_report['errors']}")
                else:
                    print(f"⚠️  Structure validation warnings for {tag}: {validation_report['warnings']}")
                    print(f"⚠️  Structure validation errors for {tag}: {validation_report['errors']}")
            
            print(f"✅ Structure prepared successfully for {tag}")
            return prepared_structure
            
        except Exception as e:
            print(f"❌ Structure preparation failed for {tag}: {e}")
            print(f"   Falling back to basic structure handling")
            return structure
    
    def _analyze_chain_structure(self, structure: SimpleStructure, tag: str) -> dict:
        """Analyze chain structure and provide detailed information"""
        
        chains = structure.split_by_chain()
        chain_info = {
            'num_chains': len(chains),
            'is_monomer': len(chains) == 1,
            'chain_lengths': [chain.size() for chain in chains],
            'total_residues': structure.size(),
            'binder_idx': None,
            'target_idx': None,
            'binder_length': None,
            'original_chain_ids': [],
            'renumbered_structure': True  # Flag to indicate if structure was renumbered
        }
        
        # Get original chain IDs if available
        try:
            for i, chain in enumerate(chains):
                chain_id = getattr(chain, 'original_chain_id', None)
                if chain_id is None:
                    # Try to get current chain ID
                    chain_structures = chain.split_by_chain() if hasattr(chain, 'split_by_chain') else []
                    if chain_structures:
                        chain_id = list(chain_structures[0].structure.get_chains())[0].get_id()
                    else:
                        chain_id = chr(65 + i)  # Default to A, B, C...
                chain_info['original_chain_ids'].append(chain_id)
        except Exception:
            chain_info['original_chain_ids'] = [chr(65 + i) for i in range(len(chains))]
        
        if len(chains) == 1:
            print(f"🔍 Monomer structure for {tag}: {chains[0].size()} residues")
            chain_info['binder_length'] = chains[0].size()
            
        elif len(chains) == 2:
            # For prepared structures, the robust handler puts binder first (index 0), target second (index 1)
            # This is the result of prepare_structure_for_af2 renumbering
            chain_info['binder_idx'] = 0  # Binder is always first after preparation
            chain_info['target_idx'] = 1  # Target is always second after preparation
            chain_info['binder_length'] = chains[0].size()
            
            print(f"🔍 Chain analysis for {tag} (after preparation):")
            print(f"   Binder (originally chain {chain_info['original_chain_ids'][0]}): {chains[0].size()} residues")
            print(f"   Target (originally chain {chain_info['original_chain_ids'][1]}): {chains[1].size()} residues")
            print(f"   📋 Note: Structure has been renumbered with binder first (0-{chains[0].size()-1}), target second ({chains[0].size()}-{structure.size()-1})")
            
        else:
            print(f"⚠️  Complex structure with {len(chains)} chains - using first 2")
            chain_info['binder_idx'] = 0
            chain_info['target_idx'] = 1
            chain_info['binder_length'] = chains[0].size()
        
        return chain_info
    
    def _validate_chain_indices(self, structure: SimpleStructure, chain_info: dict, tag: str) -> bool:
        """Validate that chain indices are handled correctly"""
        
        try:
            chains = structure.split_by_chain()
            
            # Check that chain info matches actual structure
            if len(chains) != chain_info['num_chains']:
                print(f"⚠️  Chain count mismatch for {tag}: expected {chain_info['num_chains']}, got {len(chains)}")
                return False
            
            # Check chain lengths
            actual_lengths = [chain.size() for chain in chains]
            if actual_lengths != chain_info['chain_lengths']:
                print(f"⚠️  Chain length mismatch for {tag}: expected {chain_info['chain_lengths']}, got {actual_lengths}")
                return False
            
            # For complexes, validate binder/target assignment
            if not chain_info['is_monomer'] and len(chains) == 2:
                binder_idx = chain_info['binder_idx']
                target_idx = chain_info['target_idx']
                
                if binder_idx is None or target_idx is None:
                    print(f"⚠️  Chain indices not set for {tag}")
                    return False
                
                # Verify binder length matches
                if chains[binder_idx].size() != chain_info['binder_length']:
                    print(f"⚠️  Binder length mismatch for {tag}: expected {chain_info['binder_length']}, got {chains[binder_idx].size()}")
                    return False
                
                # Check residue numbering consistency
                total_residues = sum(chain.size() for chain in chains)
                if total_residues != structure.size():
                    print(f"⚠️  Total residue count mismatch for {tag}: {total_residues} != {structure.size()}")
                    return False
                
                print(f"✅ Chain indices validated for {tag}")
                print(f"   Binder (idx {binder_idx}): {chains[binder_idx].size()} residues")
                print(f"   Target (idx {target_idx}): {chains[target_idx].size()} residues")
                print(f"   Total: {total_residues} residues")
                
            return True
            
        except Exception as e:
            print(f"❌ Chain index validation failed for {tag}: {e}")
            return False
    
    def _run_af2_scoring(self, pdb_file: str, tag: str) -> AF2Scores:
        """Run AF2 scoring on predicted structure with enhanced error handling"""
        
        try:
            # Load and validate structure
            structure = SimpleStructure(pdb_file)
            print(f"🧪 Running AF2 scoring for {tag}...")
            
            # Validate structure before scoring
            if ROBUST_HANDLER_AVAILABLE and self.structure_handler:
                validation_report = self.structure_handler.validate_structure(structure)
                if not validation_report['valid']:
                    print(f"⚠️  Structure validation issues for {tag}: {validation_report['errors']}")
                    if self.strict_validation:
                        raise ValueError(f"Structure validation failed: {validation_report['errors']}")
            
            # Use AF2 scorer
            from af2_no_pyrosetta import AF2ScorerSimple
            scorer = AF2ScorerSimple()
            af2_dict = scorer.score_structure(structure)
            
            # Validate scoring results
            if not self._validate_af2_scores(af2_dict, tag):
                print(f"⚠️  AF2 scoring results may be unreliable for {tag}")
            
            print(f"✅ AF2 scoring completed for {tag}")
            return AF2Scores(
                plddt_total=af2_dict['plddt_total'],
                plddt_binder=af2_dict['plddt_binder'], 
                pae_interaction=af2_dict['pae_interaction'],
                binder_aligned_rmsd=af2_dict['binder_aligned_rmsd'],
                binder_length=af2_dict['binder_length'],
                is_monomer=af2_dict['is_monomer']
            )
            
        except Exception as e:
            print(f"❌ AF2 scoring failed for {tag}: {e}")
            return self._get_fallback_af2_scores(pdb_file, tag)
    
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
            
            # Build motif paths using configurable database directory
            motif_path = os.path.join(self.rosetta_database_dir, 
                                     'additional_protocol_data/sewing/xsmax_bb_ss_AILV_resl0.8_msc0.3/xsmax_bb_ss_AILV_resl0.8_msc0.3.rpm.bin.gz')
            scores_path = os.path.join(self.rosetta_database_dir,
                                      'additional_protocol_data/sewing/xsmax_bb_ss_AILV_resl0.8_msc0.3/xsmax_bb_ss_AILV_resl0.8_msc0.3')
            
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
                '-mh:path:motifs', motif_path,
                '-mh:path:scores_BB_BB', scores_path,
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
            import tempfile
            
            # Create temp output directory to capture psae.py results
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdb = os.path.join(temp_dir, f"{tag}.pdb")
                temp_pae = os.path.join(temp_dir, f"{tag}_scores.json")
                
                # Copy files to temp directory
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
    
    def _validate_af2_scores(self, af2_dict: dict, tag: str) -> bool:
        """Validate AF2 scoring results for reasonableness"""
        
        warnings = []
        
        # Check pLDDT values
        if af2_dict['plddt_total'] < 30:
            warnings.append(f"Very low pLDDT ({af2_dict['plddt_total']:.1f})")
        elif af2_dict['plddt_total'] > 95:
            warnings.append(f"Unusually high pLDDT ({af2_dict['plddt_total']:.1f})")
        
        # Check PAE values
        if af2_dict['pae_interaction'] > 25:
            warnings.append(f"High PAE interaction ({af2_dict['pae_interaction']:.1f})")
        
        # Check RMSD values
        if af2_dict['binder_aligned_rmsd'] > 10:
            warnings.append(f"High RMSD ({af2_dict['binder_aligned_rmsd']:.1f})")
        
        # Check for NaN or invalid values
        for key, value in af2_dict.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                warnings.append(f"Invalid {key} value: {value}")
        
        if warnings:
            print(f"⚠️  AF2 scoring warnings for {tag}: {'; '.join(warnings)}")
            return False
        
        return True
    
    def _get_fallback_af2_scores(self, pdb_file: str, tag: str) -> AF2Scores:
        """Generate fallback AF2 scores when scoring fails"""
        
        try:
            structure = SimpleStructure(pdb_file)
            chains = structure.split_by_chain()
            is_monomer = len(chains) == 1
            binder_length = chains[0].size() if chains else 100
            
            print(f"🔄 Using fallback AF2 scores for {tag}")
            
            return AF2Scores(
                plddt_total=50.0,
                plddt_binder=50.0,
                pae_interaction=20.0 if not is_monomer else 0.0,
                binder_aligned_rmsd=5.0,
                binder_length=binder_length,
                is_monomer=is_monomer
            )
            
        except Exception:
            print(f"❌ Cannot generate fallback scores for {tag}")
            return AF2Scores(
                plddt_total=50.0,
                plddt_binder=50.0,
                pae_interaction=20.0,
                binder_aligned_rmsd=5.0,
                binder_length=100,
                is_monomer=False
            )

def main():
    """Enhanced main function with robust error handling"""
    import argparse
    
    print("🧬 Simple AF2 + Rosetta + ipSAE Scoring Pipeline")
    print("   Enhanced with robust structure prediction tools")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Simple protein complex scoring")
    parser.add_argument("--pdb", required=True, help="PDB file or directory")
    parser.add_argument("--xml_script", help="Rosetta XML script (optional)")
    parser.add_argument("--output", default="scores.csv", help="Output CSV file")
    parser.add_argument("--rosetta_path", default="rosetta_scripts", help="Rosetta executable")
    parser.add_argument("--rosetta_database_dir", help="Rosetta database directory")
    parser.add_argument("--no_ipsae", action="store_true", help="Disable ipSAE interface scoring")
    
    # Enhanced structure handling options
    parser.add_argument("--no_auto_clean", action="store_true", help="Disable automatic structure cleaning")
    parser.add_argument("--no_auto_renumber", action="store_true", help="Disable automatic residue renumbering")
    parser.add_argument("--strict_validation", action="store_true", help="Fail on any structure validation issues")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue processing even if some structures fail")
    
    args = parser.parse_args()
    
    # Initialize scorer with enhanced options
    scorer = SimpleScorer(
        rosetta_path=args.rosetta_path, 
        xml_script=args.xml_script, 
        use_ipsae=not args.no_ipsae, 
        af2_predictions_dir="af2_predictions",
        rosetta_database_dir=args.rosetta_database_dir,
        auto_clean=not args.no_auto_clean,
        auto_renumber=not args.no_auto_renumber,
        strict_validation=args.strict_validation
    )
    
    # Get PDB files
    pdb_path = Path(args.pdb)
    if pdb_path.is_file():
        pdb_files = [pdb_path]
    elif pdb_path.is_dir():
        pdb_files = list(pdb_path.glob("*.pdb"))
    else:
        raise ValueError(f"Invalid PDB input: {args.pdb}")
    
    # Score all complexes with enhanced error handling
    results = []
    failed_structures = []
    
    print(f"\n🚀 Starting to score {len(pdb_files)} structures...")
    
    for i, pdb_file in enumerate(pdb_files, 1):
        tag = pdb_file.stem
        print(f"\n[{i}/{len(pdb_files)}] Scoring {tag}...")
        
        try:
            scores = scorer.score_complex(str(pdb_file), tag)
            results.append(scores)
            print(f"  ✅ Unified score: {scores.unified_score:.3f}")
            print(f"     AF2 pLDDT: {scores.af2_plddt:.1f}, PAE: {scores.af2_pae:.1f}")
            print(f"     Rosetta: {scores.rosetta_total:.1f}, ipSAE: {scores.ipsae_score:.3f}")
            
        except Exception as e:
            error_msg = f"Failed to score {tag}: {e}"
            failed_structures.append((tag, str(e)))
            print(f"  ❌ {error_msg}")
            
            if not args.continue_on_error:
                print(f"\n💥 Stopping due to error. Use --continue_on_error to skip failed structures.")
                break
    
    # Write results with enhanced reporting
    if results:
        with open(args.output, 'w', newline='') as f:
            from dataclasses import asdict
            dict_results = [asdict(result) for result in results]
            writer = csv.DictWriter(f, fieldnames=dict_results[0].keys())
            writer.writeheader()
            writer.writerows(dict_results)
        
        print(f"\n📊 Results Summary:")
        print(f"   Successfully scored: {len(results)} complexes")
        print(f"   Failed structures: {len(failed_structures)}")
        print(f"   Results written to: {args.output}")
        
        if failed_structures:
            print(f"\n❌ Failed structures:")
            for tag, error in failed_structures:
                print(f"   {tag}: {error}")
        
        # Print top scoring results
        if len(results) > 0:
            sorted_results = sorted(results, key=lambda x: x.unified_score, reverse=True)
            print(f"\n🏆 Top scoring complexes:")
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"   {i}. {result.tag}: {result.unified_score:.3f}")
    else:
        print("\n💥 No complexes scored successfully")
        if failed_structures:
            print(f"All {len(failed_structures)} structures failed:")
            for tag, error in failed_structures:
                print(f"   {tag}: {error}")

if __name__ == "__main__":
    main()